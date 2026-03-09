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
from llm_factory import BaseLLM

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
        vector_store:   Optional[VectorStore]   = None,
        doc_processor:  Optional[DocProcessor]  = None,
        web_retriever:  Optional[WebRetriever]  = None,
        use_reranker:   bool  = True,
        alpha:          float = 0.7,
        max_iterations: int   = 3,
        retrieve_k:     int   = 10,
        rerank_top_k:   int   = 5,
    ):
        # core components
        self._llm           = llm
        self._vs            = vector_store or VectorStore()
        self._doc_processor = doc_processor or DocProcessor()
        self._web_retriever = web_retriever

        # retrieval stack
        self._bm25     = BM25Retriever()
        reranker       = CrossEncoderReranker() if use_reranker else None
        hybrid         = HybridRetriever(self._vs, self._bm25, reranker, alpha)
        self._retriever = RecursiveRetriever(hybrid, self._llm, max_iterations)

        # tuning parameters
        self._retrieve_k   = retrieve_k
        self._rerank_top_k = rerank_top_k

        logger.info(
            f"RAGPipeline ready | "
            f"reranker={'on' if use_reranker else 'off'} | "
            f"web={'on' if web_retriever else 'off'} | "
            f"alpha={alpha} | max_iter={max_iterations}"
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
            progress_callback: Optional callable(current, total, filename) for UI progress.

        Returns:
            {
                "added_files":   int,
                "skipped_files": int,
                "total_chunks":  int,
                "errors":        list,  # [(filename, error_message), ...]
            }
        """
        existing_chunk_ids = set(self._vs._store.keys())

        # DocProcessor handles the loop, progress, and chunk-level dedup
        new_chunks, errors = self._doc_processor.process_batch(
            filepaths,
            existing_chunk_ids = existing_chunk_ids,
            progress_callback  = progress_callback,
        )

        # Count skipped files: files where all chunks already existed
        total_files    = len(filepaths)
        added_files    = len({c.doc_id for c in new_chunks})
        skipped_files  = total_files - added_files - len(errors)

        # Add new chunks to the vector store
        if new_chunks:
            added = self._vs.add(new_chunks)
            self._rebuild_bm25()
            logger.info(f"Indexed {added} new chunks. Store total: {len(self._vs._store)}")
        else:
            added = 0

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

        # step 2: optional web search
        web_results: List[RetrievalResult] = []
        if enable_web_search and self._web_retriever:
            web_results = self._web_retriever.search(question)
            logger.info(f"Web search: {len(web_results)} results")

        # step 3: retrieval
        results = self._retriever.search(
            question,
            retrieve_k   = self._retrieve_k,
            rerank_top_k = self._rerank_top_k,
            web_results  = web_results or None,
        )
        logger.info(f"Retrieved {len(results)} chunks for generation")

        # step 4: build prompt
        prompt = self._build_prompt(question, results, enable_web_search)

        # step 5: stream LLM response
        yield from self._llm.stream(prompt)


    def get_info(self) -> dict:
        """Return a snapshot of the pipeline's current state."""
        return {
            "vector_store":   self._vs.get_info(),
            "bm25_indexed":   len(self._bm25._id_order),
            "web_search":     self._web_retriever is not None,
            "llm":            type(self._llm).__name__,
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