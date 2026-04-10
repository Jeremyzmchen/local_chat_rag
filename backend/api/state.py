"""
Application-level singletons and FastAPI dependency providers.

The RAGPipeline is heavy (loads embedding model, FAISS index, reranker).
We create it once at startup and share it via FastAPI's dependency injection.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singletons (set during lifespan startup)
# ---------------------------------------------------------------------------

_app_state: Dict[str, Any] = {}


def init_state(pipeline, cfg) -> None:
    """Called once from server.py lifespan on startup."""
    _app_state["pipeline"] = pipeline
    _app_state["cfg"]      = cfg
    _app_state["upload_registry"] = {}   # doc_id -> metadata dict


# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------

def get_app_state() -> Dict[str, Any]:
    return _app_state


def get_pipeline():
    return _app_state["pipeline"]


def get_upload_registry() -> Dict[str, Any]:
    return _app_state["upload_registry"]
