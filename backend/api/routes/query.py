"""
Query / Review routes
    POST /api/query   — RAG Q&A (SSE streaming)
    POST /api/review  — Legal document review (SSE streaming)
"""
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.models import (
    QueryRequest,
    ReviewFinding,
    ReviewRequest,
    SSEDone,
    SSEError,
    SSEReviewChunk,
    SSEToken,
)
from api.state import get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_query(pipeline, question: str, enable_web: bool) -> AsyncGenerator[str, None]:
    """Wrap RAGPipeline.stream_query() into async SSE generator."""
    try:
        for token in pipeline.stream_query(question, enable_web_search=enable_web):
            yield _sse(SSEToken(content=token).model_dump())
        yield _sse(SSEDone().model_dump())
    except RuntimeError as e:
        yield _sse(SSEError(message=str(e)).model_dump())
    except Exception as e:
        logger.error(f"stream_query error: {e}", exc_info=True)
        yield _sse(SSEError(message="Internal server error during query.").model_dump())


async def _stream_review(
    pipeline, document_chunk: str, max_queries: int, enable_web: bool
) -> AsyncGenerator[str, None]:
    """Wrap RAGPipeline.stream_review_document() into async SSE generator."""
    try:
        for finding in pipeline.stream_review_document(
            document_chunk    = document_chunk,
            max_queries       = max_queries,
            enable_web_search = enable_web,
        ):
            chunk = SSEReviewChunk(
                finding=ReviewFinding(**finding)
            )
            yield _sse(chunk.model_dump())
        yield _sse(SSEDone().model_dump())
    except RuntimeError as e:
        yield _sse(SSEError(message=str(e)).model_dump())
    except Exception as e:
        logger.error(f"stream_review error: {e}", exc_info=True)
        yield _sse(SSEError(message="Internal server error during review.").model_dump())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/query")
async def rag_query(
    body:     QueryRequest,
    pipeline  = Depends(get_pipeline),
):
    """
    Stream a RAG answer for the given question.
    Response is text/event-stream (SSE).

    Event types:
      { "type": "token",  "content": "..." }
      { "type": "done" }
      { "type": "error",  "message": "..." }
    """
    return StreamingResponse(
        _stream_query(pipeline, body.question, body.enable_web_search),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


@router.post("/review")
async def legal_review(
    body:     ReviewRequest,
    pipeline  = Depends(get_pipeline),
):
    """
    Stream legal review findings for a document chunk.
    Response is text/event-stream (SSE).

    Event types:
      { "type": "review_chunk", "finding": { ... } }
      { "type": "done" }
      { "type": "error",        "message": "..." }
    """
    if not pipeline._use_legal_review:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Legal review mode is not enabled. Set use_legal_review=true in config.",
        )

    return StreamingResponse(
        _stream_review(pipeline, body.document_chunk, body.max_queries, body.enable_web_search),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
