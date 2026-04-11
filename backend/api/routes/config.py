"""
Config & Health routes
    GET   /api/health
    GET   /api/config
    PATCH /api/config
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status

from api.models import (
    ConfigPatchRequest,
    ConfigResponse,
    HealthResponse,
    LLMBackend,
    LLMConfigView,
)
from api.state import get_app_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/health", response_model=HealthResponse)
def health_check(app_state: dict = Depends(get_app_state)):
    """
    Return service liveness + current pipeline snapshot.
    Used by the frontend Dashboard to populate the Node_Overview panel.
    """
    pipeline = app_state["pipeline"]
    cfg      = app_state["cfg"]
    info     = pipeline.get_info()

    # Quick Ollama reachability check (only when backend is ollama)
    ollama_alive = False
    if cfg.llm.backend == "ollama":
        try:
            from llm_factory import OllamaLLM
            # reuse the existing LLM instance rather than creating a new one
            llm = pipeline._llm
            if hasattr(llm, "is_available"):
                ollama_alive = llm.is_available()
        except Exception:
            pass

    # Measure rough AI latency with a minimal ping (skip if KB empty)
    ai_latency_ms = None
    try:
        vs = pipeline._vs
        if vs.faiss_index.ntotal > 0:
            t0 = time.time()
            vs.search("test", k=1)
            ai_latency_ms = int((time.time() - t0) * 1000)
    except Exception:
        pass

    return HealthResponse(
        status        = "ok",
        llm_backend   = cfg.llm.backend,
        llm_model     = cfg.llm.active.model,
        ollama_alive  = ollama_alive,
        vector_docs   = info["vector_store"].get("total_chunks", 0),
        bm25_indexed  = info["bm25_indexed"],
        web_search    = info["web_search"],
        legal_review  = info["legal_review"],
        ai_latency_ms = ai_latency_ms,
    )


@router.get("/config", response_model=ConfigResponse)
def get_config(app_state: dict = Depends(get_app_state)):
    """Return the current runtime configuration."""
    cfg      = app_state["cfg"]
    pipeline = app_state["pipeline"]

    return ConfigResponse(
        llm=LLMConfigView(
            backend          = LLMBackend(cfg.llm.backend),
            temperature      = cfg.llm.temperature,
            max_tokens       = cfg.llm.max_tokens,
            ollama_model     = cfg.llm.ollama.model,
            ollama_base_url  = cfg.llm.ollama.base_url,
            openai_model     = cfg.llm.openai.model,
        ),
        web_search_enabled    = cfg.web_search.enabled,
        legal_review_enabled  = pipeline._use_legal_review,
        retrieval={
            "chunk_size":     cfg.retrieval.chunk_size,
            "chunk_overlap":  cfg.retrieval.chunk_overlap,
            "alpha":          cfg.retrieval.alpha,
            "retrieve_k":     cfg.retrieval.retrieve_k,
            "rerank_top_k":   cfg.retrieval.rerank_top_k,
            "max_iterations": cfg.retrieval.max_iterations,
            "use_reranker":   cfg.retrieval.use_reranker,
        },
    )


@router.patch("/config", response_model=ConfigResponse)
def patch_config(
    body:      ConfigPatchRequest,
    app_state: dict = Depends(get_app_state),
):
    """
    Hot-patch runtime configuration.
    Changes to LLM backend / model take effect immediately (a new LLM instance is created).
    Changes to retrieval params require a server restart (logged as warning).
    """
    cfg      = app_state["cfg"]
    pipeline = app_state["pipeline"]
    rebuild_llm = False

    if body.backend is not None:
        cfg.llm.backend = body.backend.value
        rebuild_llm = True

    if body.temperature is not None:
        cfg.llm.temperature = body.temperature
        rebuild_llm = True

    if body.max_tokens is not None:
        cfg.llm.max_tokens = body.max_tokens
        rebuild_llm = True

    if body.ollama_model is not None:
        cfg.llm.ollama.model = body.ollama_model
        rebuild_llm = True

    if body.ollama_base_url is not None:
        cfg.llm.ollama.base_url = body.ollama_base_url
        rebuild_llm = True

    if body.openai_model is not None:
        cfg.llm.openai.model = body.openai_model
        rebuild_llm = True

    if body.enable_web_search is not None:
        cfg.web_search.enabled = body.enable_web_search

    # Rebuild LLM instance inside the pipeline
    if rebuild_llm:
        try:
            from llm_factory import get_llm
            new_llm = get_llm(config=cfg)
            pipeline._llm = new_llm
            # also propagate into sub-components that hold a ref to llm
            if hasattr(pipeline._retriever, "_llm"):
                pipeline._retriever._llm = new_llm
            logger.info(f"LLM rebuilt: backend={cfg.llm.backend} model={cfg.llm.active.model}")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to rebuild LLM: {e}",
            )

    if body.use_legal_review is not None:
        pipeline._use_legal_review = body.use_legal_review
        logger.info(f"Legal review toggled: {body.use_legal_review}")

    return get_config(app_state)
