"""
RAG Pipeline
    Main orchestrator that wires all backend modules together.

    Responsibilities:
        1. add_documents  — process files → embed → update VectorStore + BM25
        2. query          — intent analysis → optional web search → retrieval → LLM answer
        3. stream_query   — 
    Designed for future upgrades:
        - Replace VectorStore with a persistent DB (Chroma, Elasticsearch ...)
        - Add session / conversation management by passing history into _build_prompt().
"""

import logging
import re
from pathlib import Path
from typing import Generator, List, Optional

from doc_processor import DocProcessor, Chunk
from legal_chunker import LegalSemanticChunker
from vector_store import VectorStore
from intent_analyze import IntentAnalyzer, SearchScope
from web_search import WebRetriever
from retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    HybridRetriever,
    RecursiveRetriever,
    RetrievalResult,
)
from legal_gkgr import (
    LegalKeyInfoExtractor,
    LegalKnowledgeScorer,
    LegalReviewQueryGenerator,
    ReviewQuery,
)
from legal_reviewer import ErrorAnalyzer, RevisionGenerator
from llm_factory import BaseLLM
from config import cfg

logger = logging.getLogger(__name__)


_ANSWER_PROMPT = """\
You are a professional question-answering assistant.
Answer the user's question based ONLY on the reference content below.
If the content is insufficient, say so honestly — do not use outside knowledge.

[Reference Content]
{context}

[User Question]
{question}

Guidelines:
- Answer in the same language as the question.
- Be thorough, accurate, and well-structured.
- Cite the source tag (e.g. [local: report.pdf] or [web: title]) at the end of each claim.
{extra_instructions}
Answer:"""

_LEGAL_REVIEW_PROMPT = """\
你是一名资深法律审查专家。请基于以下法律法规参考内容，对待审查文书进行专业审查。

[法律法规参考]
{context}

[待审查文书]
{document}

[审查问题]
{query}

审查要求：
- 严格基于参考内容进行分析，不得使用参考内容之外的知识。
- 指出文书中存在的问题，并标注对应的参考依据（如 [参考: 文件名]）。
- 若文书与参考内容一致无问题，明确说明"本条款符合规定"。
- 提供具体的修改建议。

审查结果："""

_EXTRA_TIME_SENSITIVE  = "- Prioritise the most recent information."


class RAGPipeline:
    """
    RAG pipeline.

    Args:
        llm:            A BaseLLM instance (OllamaLLM or OpenAILLM from llm_factory).
        vector_store:   A VectorStore instance (shared with BM25Retriever for consistency).
        doc_processor:  Optional
        web_retriever:  Optional
        use_reranker:   Whether to enable CrossEncoder reranking (default True).
        alpha:          Weight for semantic scores in hybrid fusion (default 0.7).
        max_iterations: Max rounds for recursive retrieval (default 3).
        retrieve_k:     Candidate pool size per retrieval round (default 10).
        rerank_top_k:   Final chunks passed to the LLM after reranking (default 5).
    """

    def __init__(
        self,
        llm:            BaseLLM,
        vector_store:   Optional[VectorStore]  = None,
        doc_processor:  Optional[DocProcessor] = None,
        web_retriever:  Optional[WebRetriever] = None,
        use_reranker:   Optional[bool]  = None,
        alpha:          Optional[float] = None,
        max_iterations: Optional[int]   = None,
        retrieve_k:     Optional[int]   = None,
        rerank_top_k:   Optional[int]   = None,
        use_legal_chunker:  bool  = False,
        legal_chunker_cfg:  Optional[dict] = None,
        use_legal_review:   bool  = False,
        lambda_k:           float = 0.5,
    ):
        if doc_processor:
            self._doc_processor = doc_processor
        elif use_legal_chunker:
            cfg = legal_chunker_cfg or {}
            legal_chunker = LegalSemanticChunker(
                embedding_model_name = cfg.get("embedding_model_name",
                                               "paraphrase-multilingual-MiniLM-L12-v2"),
                max_chunk_length     = cfg.get("max_chunk_length", 512),
                min_chunk_length     = cfg.get("min_chunk_length", 30),
            )
            self._doc_processor = DocProcessor(chunker=legal_chunker)
            logger.info("RAGPipeline: using LegalSemanticChunker")
        else:
            self._doc_processor = DocProcessor()

        # core components
        self._llm           = llm
        self._vs            = vector_store or VectorStore()
        self._web_retriever = web_retriever

        # gkgr
        self._use_legal_review = use_legal_review
        if use_legal_review:
            self._key_extractor   = LegalKeyInfoExtractor(llm_fn=llm)
            self._knowledge_scorer = LegalKnowledgeScorer()
            self._query_generator  = LegalReviewQueryGenerator(llm_fn=llm)
            self._error_analyzer   = ErrorAnalyzer(llm_fn=llm)  
            self._revision_gen     = RevisionGenerator(llm_fn=llm)
            logger.info("RAGPipeline: GKGR legal review mode enabled")
        else:
            self._key_extractor    = None
            self._knowledge_scorer = None
            self._query_generator  = None
            self._error_analyzer   = None                    
            self._revision_gen     = None

        # retriever
        self._bm25    = BM25Retriever()
        reranker      = CrossEncoderReranker() if use_reranker else None
        hybrid        = HybridRetriever(
            vector_store     = self._vs,
            bm25             = self._bm25,
            reranker         = reranker,
            alpha            = alpha,
            knowledge_scorer = self._knowledge_scorer,  # 注入，None 时自动跳过
            lambda_k         = lambda_k,
        )
        self._retriever = RecursiveRetriever(hybrid, self._llm, max_iterations)

        # tuning params
        self._k          = retrieve_k   or cfg.retrieval.retrieve_k
        self._rerank_top_k = rerank_top_k or cfg.retrieval.rerank_top_k

        logger.info(
            f"RAGPipeline ready | "
            f"reranker={'on' if use_reranker else 'off'} | "
            f"web={'on' if web_retriever else 'off'} | "
            f"legal_chunker={'on' if use_legal_chunker else 'off'} | "
            f"legal_review={'on' if use_legal_review else 'off'} | "
            f"alpha={alpha} | lambda_k={lambda_k}"
        )
    
    def add_documents(
        self,
        filepaths: List[str],
        progress_callback=None,
    ) -> dict:
        """
        Process and index a list of files (incremental — existing chunks are kept).

        Args:
            filepaths:         List of absolute or relative file paths.
            progress_callback: Optional callable(current, total, filename, stage)
                               stage is one of: "extract" | "index" | "done"

        Returns:
            {
                "added_files":   int,
                "skipped_files": int,
                "total_chunks":  int,
                "errors":        list,  # [(filename, error_message), ...]
            }
        """
        existing_chunk_ids = set(self._vs._store.keys())
        total_files = len(filepaths)

        # Wrap callback to inject stage="extract"
        def extract_cb(current, total, filename):
            if progress_callback:
                progress_callback(current, total, filename, "extract")

        # DocProcessor handles the loop, progress, and chunk-level dedup
        new_chunks, errors = self._doc_processor.process_batch(
            filepaths,
            existing_chunk_ids = existing_chunk_ids,
            progress_callback  = extract_cb,
        )

        # Count skipped files: files where all chunks already existed
        added_files   = len({c.doc_id for c in new_chunks})
        skipped_files = total_files - added_files - len(errors)

        # Add new chunks to the vector store
        if new_chunks:
            if progress_callback:
                progress_callback(0, 1, f"Embedding {len(new_chunks)} chunks…", "index")
            added = self._vs.add(new_chunks)
            self._rebuild_bm25()
            logger.info(f"Indexed {added} new chunks. Store total: {len(self._vs._store)}")
        else:
            added = 0

        if progress_callback:
            progress_callback(total_files, total_files, "Completed", "done")

        return {
            "added_files":   added_files,
            "skipped_files": max(skipped_files, 0),
            "total_chunks":  added,
            "errors":        errors,
        }
    

    def query(self, question: str, enable_web_search: bool = False) -> str:
        """
        Output the answer(blocking)
        """
        return "".join(self.stream_query(question, enable_web_search))

    def stream_query(
        self,
        question:          str,
        enable_web_search: bool = False,
    ) -> Generator[str, None, None]:
        """
        Output the answer(streaming)
        """

        # check vector store is empty or web search is enabled
        vector_is_empty = self._vs.faiss_index.ntotal == 0
        if vector_is_empty and not enable_web_search:
            raise RuntimeError(
                "Knowledge base is empty. Please add documents first, "
                "or enable web search."
            )
        if vector_is_empty:
            logger.warning("Knowledge base is empty — using web search only")

        # step 1: intent analysis and get the search scope
        scope = self._analyze_intent(question)
        logger.info(f"Intent scope: {scope}")

        # step 2: extract key info
        key_info = None
        if self._use_legal_review and self._key_extractor:
            key_info = self._key_extractor.extract(question)
            logger.info(f"KeyInfo extracted: max={key_info.max_terms}")

        # step 3: optional web search
        web_results: List[RetrievalResult] = []
        if enable_web_search and self._web_retriever:
            web_results = self._web_retriever.search(question)
            logger.info(f"Web search: {len(web_results)} results")

        # step 4: retrieval
        results = self._retriever.search(
            question,
            retrieve_k   = self._k,
            rerank_top_k = self._rerank_top_k,
            web_results  = web_results or None,
            key_info     = key_info,     
        )
        logger.info(f"Retrieved {len(results)} chunks for generation")

        # step 5: build prompt
        prompt = self._build_prompt(question, results, enable_web_search)

        yield from self._llm.stream(prompt)

    
    def review_document(
        self,
        document_chunk:    str,
        max_queries:       int  = 5,
        enable_web_search: bool = False,
    ) -> List[dict]:
        if not self._use_legal_review:
            raise RuntimeError(
                "review_document() requires use_legal_review=True."
            )
        if self._vs.faiss_index.ntotal == 0 and not enable_web_search:
            raise RuntimeError(
                "Knowledge base is empty. Please add legal documents first."
            )

        review_queries = self._query_generator.generate(
            document_chunk, max_queries=max_queries
        )
        logger.info(f"review_document: {len(review_queries)} review queries generated")

        results = []
        for rq in review_queries:
            key_info    = self._key_extractor.extract(rq.query)
            web_results = []
            if enable_web_search and self._web_retriever:
                web_results = self._web_retriever.search(rq.query)

            retrieved = self._retriever.search(
                rq.query,
                k            = self._k,
                rerank_top_k = self._rerank_top_k,
                web_results  = web_results or None,
                key_info     = key_info,
            )
            ref_contents = [r.content for r in retrieved]

            # stage 1: deviation analysis
            analysis = self._error_analyzer.analyze(
                document_chunk   = document_chunk,
                query            = rq.query,
                dimension        = rq.dimension,
                retrieved_chunks = ref_contents,
            )

            # stage 2: revision suggestions + confidence
            revision = self._revision_gen.generate(
                document_chunk  = document_chunk,
                analysis_result = analysis,
            )

            results.append({
                "query":      rq.query,
                "dimension":  rq.dimension,
                "priority":   rq.priority,
                # stage 1 output
                "analysis":       analysis.analysis,
                "error_regions":  analysis.error_regions,
                "references":     ref_contents,
                # stage 2 output
                "revision":    revision.revision_suggestions,
                "confidence":  revision.confidence,
            })
            logger.info(
                f"review_document: [{rq.dimension}] done, "
                f"confidence={revision.confidence}"
            )

        return results

    def stream_review_document(
        self,
        document_chunk:    str,
        max_queries:       int  = 5,
        enable_web_search: bool = False,
    ) -> Generator[dict, None, None]:
        if not self._use_legal_review:
            raise RuntimeError(
                "stream_review_document() requires use_legal_review=True."
            )

        review_queries = self._query_generator.generate(
            document_chunk, max_queries=max_queries
        )

        for rq in review_queries:
            key_info    = self._key_extractor.extract(rq.query)
            web_results = []
            if enable_web_search and self._web_retriever:
                web_results = self._web_retriever.search(rq.query)

            retrieved    = self._retriever.search(
                rq.query,
                k            = self._k,
                rerank_top_k = self._rerank_top_k,
                web_results  = web_results or None,
                key_info     = key_info,
            )
            ref_contents = [r.content for r in retrieved]

            analysis = self._error_analyzer.analyze(
                document_chunk   = document_chunk,
                query            = rq.query,
                dimension        = rq.dimension,
                retrieved_chunks = ref_contents,
            )

            revision = self._revision_gen.generate(
                document_chunk  = document_chunk,
                analysis_result = analysis,
            )

            yield {
                "query":         rq.query,
                "dimension":     rq.dimension,
                "priority":      rq.priority,
                "analysis":      analysis.analysis,
                "error_regions": analysis.error_regions,
                "references":    ref_contents,
                "revision":      revision.revision_suggestions,
                "confidence":    revision.confidence,
            }


    def get_info(self) -> dict:
        """Return a snapshot of the pipeline's current state."""
        return {
            "vector_store":   self._vs.get_info(),
            "bm25_indexed":   len(self._bm25._id_order),
            "web_search":     self._web_retriever is not None,
            "llm":            type(self._llm).__name__,
            "legal_review":   self._use_legal_review,
        }

    def list_documents(self) -> List[str]:
        """
        Return the unique source filenames currently indexed.
        Derived dynamically from VectorStore — no extra state to maintain.
        This list is passed directly to IntentAnalyzer.
        """
        sources = set()
        for info in self._vs._store.values():
            src = info.get("metadata", {}).get("source")
            if src:
                sources.add(src)
        return sorted(sources)


    def _analyze_intent(self, question: str) -> SearchScope:
        """Run IntentAnalyzer using the current document list from VectorStore."""
        doc_list = self.list_documents()
        if not doc_list:
            return SearchScope()   # default global scope
        analyzer = IntentAnalyzer(all_files=doc_list, llm_fn=self._llm)
        return analyzer.analyze(question)


    def _build_prompt(
        self,
        question: str,
        results:  List[RetrievalResult],
        used_web: bool,
    ) -> str:
        """
        Assemble the final prompt for the LLM.

        Added tag metadata to each chunk, e.g. [local: filename.md] or [web: title].
        Extra instructions are added when the query is time-sensitive or sources conflict.
        """
        context_parts: List[str] = []
        for r in results:
            src_type = r.metadata.get("type", "local")
            if src_type == "web":
                title = r.metadata.get("title", "web")
                url   = r.metadata.get("url", "")
                tag   = f"[web: {title}]({url})"
            else:
                src = r.metadata.get("source", "unknown")
                tag = f"[local: {src}]"
            context_parts.append(f"{tag}\n{r.content}")

        context = "\n\n---\n\n".join(context_parts) if context_parts else "(no content retrieved)"

        # extra instructions
        extras: List[str] = []
        if used_web and self._is_time_sensitive(question):
            extras.append(_EXTRA_TIME_SENSITIVE)

        return _ANSWER_PROMPT.format(
            context             = context,
            question            = question,
            extra_instructions  = "\n".join(extras),
        )

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from the current VectorStore contents."""
        ids      = list(self._vs._id_order)
        contents = [self._vs._store[cid]["content"] for cid in ids]
        metas    = [self._vs._store[cid].get("metadata", {}) for cid in ids]
        self._bm25.build(ids, contents, metas)
        logger.info(f"BM25 rebuilt: {len(ids)} docs")

    @staticmethod
    def _is_time_sensitive(question: str) -> bool:
        keywords = ["latest", "recent", "current", "today", "now",
                    "最新", "今年", "当前", "最近", "刚刚"]
        q = question.lower()
        return any(kw in q for kw in keywords)