"""
Document routes
    POST   /api/documents/upload
    GET    /api/documents
    DELETE /api/documents/{doc_id}
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse

from api.models import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    ProcessState,
    ReviewProtocol,
    UploadResponse,
)
from api.state import get_pipeline, get_upload_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Allowed MIME / extensions
_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".xlsx", ".xls", ".pptx"}
_MAX_FILE_SIZE_MB   = 50


def _validate_file(file: UploadFile) -> None:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}",
        )


@router.post("/upload", status_code=status.HTTP_200_OK)
async def upload_documents(
    files:    List[UploadFile]    = File(...),
    protocol: ReviewProtocol      = Form(ReviewProtocol.compliance),
    pipeline                      = Depends(get_pipeline),
    registry: dict                = Depends(get_upload_registry),
):
    """
    Upload one or more documents and stream indexing progress via SSE.

    SSE event types:
      {"type": "progress", "stage": "extract"|"index", "current": int, "total": int, "filename": str}
      {"type": "done",     "result": UploadResponse}
      {"type": "error",    "detail": str}
    """
    for f in files:
        _validate_file(f)

    tmp_dir   = tempfile.mkdtemp(prefix="docmind_")
    tmp_paths: List[str] = []

    # Read all files into temp dir before streaming (must be done before generator starts)
    for upload in files:
        content = await upload.read()
        if len(content) > _MAX_FILE_SIZE_MB * 1024 * 1024:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"'{upload.filename}' exceeds {_MAX_FILE_SIZE_MB} MB limit.",
            )
        dest = os.path.join(tmp_dir, upload.filename)
        with open(dest, "wb") as fh:
            fh.write(content)
        tmp_paths.append(dest)

    def _sse(event: dict) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    def generate():
        try:
            def progress_callback(current, total, filename, stage):
                yield_val = _sse({
                    "type":     "progress",
                    "stage":    stage,
                    "current":  current,
                    "total":    total,
                    "filename": filename,
                })
                # store in nonlocal list so the generator can yield it
                _events.append(yield_val)

            _events: List[str] = []

            # We can't yield directly from a callback, so we use a polling approach:
            # run add_documents in a thread and drain _events between polls.
            import threading
            result_holder: List[dict] = []
            exc_holder:    List[Exception] = []

            def run():
                try:
                    result_holder.append(
                        pipeline.add_documents(tmp_paths, progress_callback=progress_callback)
                    )
                except Exception as e:
                    exc_holder.append(e)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()

            while thread.is_alive():
                while _events:
                    yield _events.pop(0)
                thread.join(timeout=0.1)

            # drain any remaining events
            while _events:
                yield _events.pop(0)

            if exc_holder:
                yield _sse({"type": "error", "detail": str(exc_holder[0])})
                return

            result = result_holder[0]

            # Build document info list
            doc_infos: List[dict] = []
            vs_store  = pipeline._vs._store
            seen_docs: set = set()

            for chunk_id, info in vs_store.items():
                meta   = info.get("metadata", {})
                source = meta.get("source", "unknown")
                doc_id = meta.get("doc_id", source)

                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)

                if doc_id not in registry:
                    registry[doc_id] = {
                        "filename":        source,
                        "file_type":       meta.get("file_type", ""),
                        "review_protocol": protocol,
                        "created_at":      time.time(),
                        "risk_score":      None,
                    }

                entry = registry[doc_id]
                doc_infos.append({
                    "doc_id":          doc_id,
                    "filename":        entry["filename"],
                    "file_type":       entry.get("file_type", ""),
                    "total_chunks":    meta.get("total_chunks", 0),
                    "char_count":      len(info.get("content", "")),
                    "created_at":      entry["created_at"],
                    "state":           ProcessState.completed,
                    "risk_score":      entry.get("risk_score"),
                    "review_protocol": entry.get("review_protocol"),
                })

            yield _sse({
                "type": "done",
                "result": {
                    "added_files":   result["added_files"],
                    "skipped_files": result["skipped_files"],
                    "total_chunks":  result["total_chunks"],
                    "errors":        result["errors"],
                    "documents":     doc_infos,
                },
            })

        except Exception as e:
            logger.error(f"Upload SSE error: {e}")
            yield _sse({"type": "error", "detail": str(e)})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    pipeline = Depends(get_pipeline),
    registry: dict = Depends(get_upload_registry),
):
    """Return all documents currently indexed in the knowledge base."""
    vs_store = pipeline._vs._store
    seen_docs: dict = {}

    for chunk_id, info in vs_store.items():
        meta   = info.get("metadata", {})
        source = meta.get("source", "unknown")
        doc_id = meta.get("doc_id", source)

        if doc_id in seen_docs:
            seen_docs[doc_id]["total_chunks"] += 1
        else:
            entry = registry.get(doc_id, {})
            seen_docs[doc_id] = {
                "doc_id":          doc_id,
                "filename":        entry.get("filename", source),
                "file_type":       entry.get("file_type", meta.get("file_type", "")),
                "total_chunks":    1,
                "char_count":      len(info.get("content", "")),
                "created_at":      entry.get("created_at", meta.get("created_at", time.time())),
                "state":           ProcessState.completed,
                "risk_score":      entry.get("risk_score"),
                "review_protocol": entry.get("review_protocol"),
            }

    docs = [DocumentInfo(**d) for d in seen_docs.values()]
    docs.sort(key=lambda d: d.created_at, reverse=True)

    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete("/{doc_id}", response_model=DeleteResponse)
def delete_document(
    doc_id:   str,
    pipeline  = Depends(get_pipeline),
    registry: dict = Depends(get_upload_registry),
):
    """Remove a document and all its chunks from the knowledge base."""
    vs_store = pipeline._vs._store

    # Find chunk IDs that belong to this doc.
    # _store[cid]["metadata"] is chunk.to_dict() which has "doc_id" and "source" at top level.
    to_delete = [
        cid for cid, info in vs_store.items()
        if info.get("metadata", {}).get("doc_id") == doc_id
        or info.get("metadata", {}).get("source") == doc_id
        or cid.startswith(doc_id + "_")   # chunk_id format: {doc_id}_{index}
    ]

    if not to_delete:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found in knowledge base.",
        )

    # VectorStore delete
    try:
        pipeline._vs.delete(to_delete)
        pipeline._rebuild_bm25()
    except Exception as e:
        logger.error(f"Delete failed for doc_id={doc_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {e}",
        )

    # Remove from registry
    registry.pop(doc_id, None)

    logger.info(f"Deleted doc_id={doc_id}, removed {len(to_delete)} chunks")
    return DeleteResponse(
        deleted = True,
        doc_id  = doc_id,
        message = f"Removed {len(to_delete)} chunks.",
    )
