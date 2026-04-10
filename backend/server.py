"""
DocMind AI — FastAPI Server
Run with:
    cd backend
    uvicorn server:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Path setup — add backend/core to sys.path so all core modules are importable
# without any changes to their internal relative imports.
# ---------------------------------------------------------------------------

_BACKEND_DIR = Path(__file__).parent.resolve()
_CORE_DIR    = _BACKEND_DIR / "core"
for _p in (_BACKEND_DIR, _CORE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Logging setup (before any local imports that use the logger)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream = sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: build the RAGPipeline once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup → yield → Shutdown."""
    logger.info("=== DocMind AI server starting ===")

    from config import cfg
    from llm_factory import get_llm
    from rag_pipeline import RAGPipeline
    from web_search import WebRetriever
    from api.state import init_state

    # Build LLM
    llm = get_llm(config=cfg)
    logger.info(f"LLM ready: backend={cfg.llm.backend} model={cfg.llm.active.model}")

    # Optionally build web retriever
    web_retriever = None
    if cfg.web_search.enabled and cfg.secrets.serpapi_key:
        web_retriever = WebRetriever(
            api_key     = cfg.secrets.serpapi_key,
            engine      = cfg.web_search.engine,
            num_results = cfg.web_search.num_results,
            timeout     = cfg.web_search.timeout,
        )
        logger.info("WebRetriever ready")

    # Build RAGPipeline
    pipeline = RAGPipeline(
        llm           = llm,
        web_retriever = web_retriever,
        use_reranker  = cfg.retrieval.use_reranker,
        alpha         = cfg.retrieval.alpha,
        max_iterations= cfg.retrieval.max_iterations,
        retrieve_k    = cfg.retrieval.retrieve_k,
        rerank_top_k  = cfg.retrieval.rerank_top_k,
        # Legal review OFF by default; toggle via PATCH /api/config
        use_legal_review = False,
    )
    logger.info("RAGPipeline ready")

    # Publish to dependency providers
    init_state(pipeline=pipeline, cfg=cfg)

    logger.info(
        f"=== Server ready | "
        f"host={cfg.server.host} port={cfg.server.port} ==="
    )

    yield  # ← server is running

    logger.info("=== DocMind AI server shutting down ===")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "DocMind AI API",
    description = "Legal document analysis powered by RAG",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# CORS — allow the Vite dev server and production origins
# sys.path already contains core/, so this import works at module level.
from config import cfg as _cfg   # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins     = _cfg.server.cors_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from api.routes.documents import router as documents_router  # noqa: E402
from api.routes.query     import router as query_router      # noqa: E402
from api.routes.config    import router as config_router     # noqa: E402

app.include_router(documents_router)
app.include_router(query_router)
app.include_router(config_router)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({"message": "DocMind AI API", "docs": "/docs"})


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

from fastapi import Request  # noqa: E402
from fastapi.responses import JSONResponse as _JSONResponse  # noqa: E402


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return _JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )
