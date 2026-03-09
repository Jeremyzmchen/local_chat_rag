"""
Web Search (Optional)
This module uses the Google Search API to retrieve relevant web pages.
"""

import os
import logging
import requests
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from retrieval import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    title:   str
    url:     str
    snippet: str
    source:  str = "serpapi" 


class SerpAPISearcher:
    """
    Using SerpAPI to search the web.

    Args:
        api_key:      SerpAPI key
        engine:       default "google"
        num_results:  max 5
        timeout:      HTTP request timeout in seconds.
    """

    _ENDPOINT = "https://serpapi.com/search"

    def __init__(
        self,
        api_key:     Optional[str] = None,
        engine:      Optional[str] = None,
        num_results: Optional[int] = None,
        timeout:     Optional[int] = None,
        hl:          Optional[str] = None,
        gl:          Optional[str] = None,
    ):
        from config import cfg
        w = cfg.web_search

        self._api_key     = api_key     or cfg.secrets.serpapi_key or ""
        self._engine      = engine      or w.engine
        self._num_results = num_results or w.num_results
        self._timeout     = timeout     or w.timeout
        self._hl          = hl          or w.hl
        self._gl          = gl          or w.gl

    @property
    def has_serpapi_key(self) -> bool:
        """True when an API key is configured."""
        return bool(self._api_key and self._api_key.strip())
    
    def search(self, query: str) -> List[WebSearchResult]:
        """
        Make web request and execute a web search, convert JSON str into py dict, and return parsed results.
        """
        if not self.has_serpapi_key:
            logger.warning("SerpAPISearcher: no API key configured, skipping web search")
            return []

        try:
            resp = requests.get(
                self._ENDPOINT,
                params={
                    "engine":  self._engine,
                    "q":       query,
                    "api_key": self._api_key,
                    "num":     self._num_results,
                    "hl":      self._hl,
                    "gl":      self._gl,
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return self._parse(resp.json())

        except requests.RequestException as e:
            logger.error(f"SerpAPISearcher: HTTP error — {e}")
            return []
        except Exception as e:
            logger.error(f"SerpAPISearcher: unexpected error — {e}")
            return []
        

    @staticmethod
    def _parse(data: dict) -> List[WebSearchResult]:

        results: List[WebSearchResult] = []

        # Knowledge Graph (pinned answer box at the top)
        if kg := data.get("knowledge_graph"):
            results.append(WebSearchResult(
                title   = kg.get("title", "Knowledge Graph"),
                url     = kg.get("source", {}).get("link", ""),
                snippet = kg.get("description", ""),
                source  = "knowledge_graph",
            ))

        # Organic results
        for item in data.get("organic_results", []):
            results.append(WebSearchResult(
                title   = item.get("title", ""),
                url     = item.get("link", ""),
                snippet = item.get("snippet", ""),
            ))

        return results
    


class WebRetriever:
    """
    Converts WebSearchResult objects into RetrievalResult so they can be
    merged with local results inside HybridRetriever.

    Notes:
        Web results are NOT added to the FAISS / BM25 index.
        They exist only as plain-text context for the current query.
        This avoids polluting the local knowledge base with transient web data
        and sidesteps the FAISS "no delete" limitation.
    """

    def __init__(self, searcher: Optional[SerpAPISearcher] = None):
        self._searcher = searcher or SerpAPISearcher()
        logger.info(f"WebRetriever: using searcher {self._searcher._engine}")

    @property
    def can_search(self) -> bool:
        return self._searcher.has_serpapi_key

    def search(self, query: str) -> List[RetrievalResult]:

        raw: List[WebSearchResult] = self._searcher.search(query)
        if not raw:
            return []

        # Convert raw results into RetrievalResult
        results = []
        total = len(raw)
        for rank, item in enumerate(raw):
            # Build a self-contained text block: title + snippet
            content = f"Title: {item.title}\nSnippet: {item.snippet}"

            # Rank-based score in (0, 1]: first result = 1.0, last ≈ 1/total
            score = 1.0 - (rank / max(total, 1))

            results.append(RetrievalResult(
                chunk_id = f"web_{rank}_{hash(item.url) & 0xFFFFFF}",
                content  = content,
                metadata = {
                    "title":  item.title,
                    "url":    item.url,
                    "source": item.source,
                    "type":   "web",
                },
                score  = score,
                source = "web",
            ))

        logger.info(f"WebRetriever: {len(results)} results for query '{query}'")
        return results