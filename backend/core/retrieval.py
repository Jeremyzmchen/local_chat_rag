"""
Retrieval
    - BM25Retrieval:         sparse retrieval
    - CrossEncoderRetrieval: reranking
    - HybridRetrieval:       BM25 + semantic + reranking(optional)
    - RecursiveRetrieval:    recursive retrieval, queries refined by LLM
"""


import logging
import threading
import numpy as np
import jieba
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk_id: str
    content:  str
    metadata: Dict[str, Any]
    score:    float                         
    source:   str = "local"


class BM25Retriever:
    """
    Sparse retrieval backed by BM25OKapi
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._id_order: List[str] = []   # map to chunk_id
        self._contents: List[str] = []   
        self._metadatas: List[dict] = []

    def build(self, chunk_ids: List[str], contents: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """
        Build (or rebuild) the BM25 index.

        Args:
            chunk_ids: chunk_id for each document
            contents:  raw text for each document
        """
        if not chunk_ids:
            logger.warning("BM25Retriever.build: empty input, index not built")
            return

        self._id_order = list(chunk_ids)
        self._contents = list(contents)
        self._metadatas = list(metadatas) if metadatas else [{} for _ in chunk_ids]
        tokenized = [list(jieba.cut(doc)) for doc in contents]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(chunk_ids)} documents")

    def search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """
        Return the top-k most relevant chunks for queries.
        Scores are normalised to [0, 1].
        """
        if self._bm25 is None:
            logger.warning("BM25Retriever.search: index is empty")
            return []

        tokens = list(jieba.cut(query))
        raw_scores = self._bm25.get_scores(tokens)

        # Sort and select top-k
        top_indices = np.argsort(raw_scores)[::-1][:k]
        # Filter out negative scores: 
        # BM25 can return negative scores for some queries when:
        # - the query terms are very common in the corpus, leading to low IDF values
        top_indices = [i for i in top_indices if raw_scores[i] > 0]

        if not top_indices:
            logger.warning("BM25Retriever.search: no positive scores")
            return []

        # Encapsulate results
        max_score = raw_scores[top_indices[0]] or 1.0      # avoid /0
        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                chunk_id = self._id_order[idx],
                content  = self._contents[idx],
                metadata = self._metadatas[idx] if self._metadatas else {},
                # normalise score to [0, 1]
                score    = float(raw_scores[idx]) / max_score,
            ))
        return results
    
    def clear(self) -> None:
        self._bm25     = None
        self._id_order = []
        self._contents = []
        self._metadatas = []



class CrossEncoderReranker:
    """
    Reranks a list of candidates using a cross-encoder model.
    Lazily loads the model on first use and uses a lock to stay thread-safe.

    Args:
        model_name: HuggingFace model identifier.
            Default is a multilingual model that works well for Chinese.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",):
        from config import cfg

        self._model_name = model_name or cfg.retrieval.reranker_model
        self._model: Optional[CrossEncoder] = None
        self._lock  = threading.Lock()

    def rerank(self, query: str, results: List[RetrievalResult], 
               top_k: int = 5,) -> List[RetrievalResult]:
        """
        Rerank the given results: FAISS semantic search + BM25 by HybridRetriever.
        """

        if not results:
            logger.warning("CrossEncoderReranker.rerank: empty input")
            return []

        model = self._get_model()
        if model is None:
            logger.warning("CrossEncoderReranker: model unavailable, skipping rerank")
            return results[:top_k]

        pairs  = [[query, r.content] for r in results]
        scores = model.predict(pairs)

        reranked = sorted(
            zip(scores, results),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for s, r in reranked[:top_k]:
            result = RetrievalResult(
                chunk_id = r.chunk_id,
                content  = r.content,
                metadata = r.metadata,
                score    = float(s),
                source   = r.source,
            )
            results.append(result)

        return results
    
    def _get_model(self) -> Optional[CrossEncoder]:
        """Lazy-load with double-checked locking."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        self._model = CrossEncoder(self._model_name)
                        logger.info(f"CrossEncoder loaded: {self._model_name}")
                    except Exception as e:
                        logger.error(f"Failed to load CrossEncoder: {e}")
        return self._model
    

class HybridRetriever:
    """
    Main interface for retrieval.
    Combines dense (VectorStore) and sparse (BM25) retrieval.

    Set up alpha to control the weight of the semantic scores.
        final_score = alpha * semantic_score + (1 - alpha) * bm25_score
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25: BM25Retriever,
        reranker: Optional[CrossEncoderReranker] = None,
        alpha: float = None,
        knowledge_scorer   = None,
        lambda_k: float = 0.5,
    ):
        from config import cfg   

        self._vs = vector_store
        self._bm25 = bm25
        self._reranker = reranker
        self._alpha = alpha if alpha is not None else cfg.retrieval.alpha
        self._knowledge_scorer = knowledge_scorer
        self._lambda_k         = lambda_k


    def search(self, query: str, 
               k: int = 10, 
               rerank_top_k: int = 5,
               web_results: Optional[List[RetrievalResult]] = None,
               key_info = None,
    ) -> List[RetrievalResult]:
        
        """
        Retrieve the most relevant chunks for queries.

        Steps:
            1. Dense retrieval via VectorStore (FAISS)
            2. Sparse retrieval via BM25
            3. Score fusion
            4. knowledge-aware selection
            5. Optional web search
            6. Optional cross-encoder reranking
        """
        # 1. Dense retrieval
        semantic_hits: List[SearchResult] = self._vs.search(query, k=k)

        # 2. Sparse retrieval
        bm25_hits: List[RetrievalResult] = self._bm25.search(query, k=k)

        # 3. Fuse scores
        merged = self._fuse(semantic_hits, bm25_hits)

        # 4. Apply knowledge-aware selection
        if self._knowledge_scorer is not None and key_info is not None:
            merged = self._apply_knowledge_scores(merged, key_info)

        # 5. Insert web results (optional)
        if web_results:
            weighted_web = [
                RetrievalResult(
                    chunk_id = r.chunk_id,
                    content  = r.content,
                    metadata = r.metadata,
                    score    = r.score * 0.9,
                    source   = r.source,
                )
                for r in web_results
            ]
            merged = sorted(merged + weighted_web, key=lambda x: x.score, reverse=True)

        # 6. Rerank (optional)
        if self._reranker and merged:
            merged = self._reranker.rerank(query, merged, top_k=rerank_top_k)
        else:
            merged = merged[:rerank_top_k]

        return merged
    
    def _apply_knowledge_scores(
        self,
        results:  List[RetrievalResult],
        key_info,                           # KeyInfo
    ) -> List[RetrievalResult]:
        """
        对已融合的候选列表叠加知识级分数。

        最终分数 = λ_k · Φ_K + (1 - λ_k) · Φ_S_BM25融合分
        """
        contents = [r.content for r in results]
        k_scores = self._knowledge_scorer.score_chunks(key_info, contents)

        updated = []
        for r, ks in zip(results, k_scores):
            new_score = self._lambda_k * ks + (1 - self._lambda_k) * r.score
            updated.append(RetrievalResult(
                chunk_id = r.chunk_id,
                content  = r.content,
                metadata = r.metadata,
                score    = new_score,
                source   = r.source,
            ))

        updated.sort(key=lambda x: x.score, reverse=True)
        logger.info("HybridRetriever: knowledge-guided scores applied")
        return updated
    
    def _fuse(
        self,
        semantic_hits: List[SearchResult],
        bm25_hits:     List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Merge two ranked lists into one using weighted score fusion.

        Semantic scores: L2 distances from FAISS (lower = better).
            We convert them to similarity with: sim = 1 / (1 + dist)
            so they land in (0, 1] like BM25 scores.
        """
        fused: Dict[str, Dict] = {}

        # --- semantic side ---
        for hit in semantic_hits:
            sim = 1.0 / (1.0 + hit.score)
            fused[hit.chunk_id] = {
                "content":  hit.content,
                "metadata": hit.metadata,
                "score":    self._alpha * sim,
            }

        # --- BM25 side ---
        for hit in bm25_hits:
            if hit.chunk_id in fused:
                fused[hit.chunk_id]["score"] += (1 - self._alpha) * hit.score
            else:
                # BM25-only hit: try to get metadata from VectorStore's internal store
                meta = self._vs.get_metadata(hit.chunk_id)
                fused[hit.chunk_id] = {
                    "content":  hit.content,
                    "metadata": meta,
                    "score":    (1 - self._alpha) * hit.score,
                }

        # sort descending
        sorted_items = sorted(fused.items(), key=lambda x: x[1]["score"], reverse=True)
        
        results = []
        for cid, info in sorted_items:
            result = RetrievalResult(
                chunk_id = cid,
                content = info["content"],
                metadata = info["metadata"],
                score = info["score"],
            )
            results.append(result)

        return results
    


# input : prompt string
# output: response string
LLMCallable = Callable[[str], str]

class RecursiveRetriever:
    """
    Multiple turn recursive retriever
    LLM is used to decide whether the current context is sufficient
    and, if not, to generate a better follow-up query.  The LLM callable is
    injected at construction time so this class stays decoupled from any
    specific model backend (Ollama, SiliconFlow, OpenAI, …).

    Args:
        hybrid_retriever: a HybridRetriever instance
        llm_fn:           callable (prompt: str) -> str
                          Pass None to disable query refinement (single-round).
        max_iterations:   maximum number of retrieval rounds
    """

    # Prompt template for query refinement
    _REFINE_PROMPT = """You are a query optimization assistant. Based on the information below, \
decide whether another search round is needed.

[Original Question]
{initial_query}

[Summary of Retrieved Content]
{context_summary}

Instructions:
1. If the retrieved content is sufficient to answer the original question, \
reply with exactly: NO_FURTHER_QUERY
2. Otherwise, reply with only a single, more precise search query \
(max 50 words, no explanation, respond in the same language as the original question).
"""

    def __init__(self, hybrid_retriever, 
                 llm_fn = None, 
                 max_iterations = None,):
        from config import cfg

        self._retriever = hybrid_retriever
        self._llm_fn = llm_fn
        self._max_iterations = max_iterations if max_iterations is not None else cfg.retrieval.max_iterations

    def search(self, query: str, k: int = 5, rerank_top_k = 5,
               web_results: Optional[List[RetrievalResult]] = None
    ) -> List[RetrievalResult]:
        """
        Iteratively retrieve until the LLM is satisfied or max_iterations is reached.
        """
        current_query = query
        seen_ids: set = set()
        all_results: List[RetrievalResult] = []

        for i in range(self._max_iterations):
            logger.info(f"RecursiveRetriever round {i+1}/{self._max_iterations}: '{current_query}'")

            round_results = self._retriever.search(current_query, 
                                                   k=k, 
                                                   rerank_top_k=rerank_top_k,
                                                   web_results=web_results if i == 0 else None,)

            # deduplicate across rounds
            new_results = [r for r in round_results if r.chunk_id not in seen_ids]
            for r in new_results:
                seen_ids.add(r.chunk_id)
            all_results.extend(new_results)

            if i == self._max_iterations - 1:
                break

            if self._llm_fn is None:
                break

            # LLM decide whether to refine query
            next_query = self._refine_query(query, all_results)
            if next_query is None:
                logger.info("RecursiveRetriever: LLM satisfied, stopping refining query")
                break

            current_query = next_query

        logger.info(f"RecursiveRetriever: returned {len(all_results)} unique chunks")
        return all_results

    def _refine_query(self, initial_query: str, results: List[RetrievalResult],) -> Optional[str]:
        """
        Call the LLM to get a refined query.
        Returns None when the LLM decides no further retrieval is needed.
        """

        # integrate prompt
        # get the first three chunks(200 characters each)
        summary = "\n".join(r.content[:200] for r in results[:3])
        prompt  = self._REFINE_PROMPT.format(
            initial_query = initial_query,
            context_summary = summary,
        )

        try:
            response = self._llm_fn(prompt).strip()
        except Exception as e:
            logger.error(f"RecursiveRetriever: LLM call failed: {e}")
            return None

        if not response or "NO_FURTHER_QUERY" in response:
            return None

        # ensure response is not too long
        if len(response) > 100:
            logger.warning("RecursiveRetriever: LLM returned a long response, ignoring")
            return None

        logger.info(f"RecursiveRetriever: refined query → '{response}'")
        return response