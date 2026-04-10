"""
Pydantic schemas for all request / response bodies.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReviewProtocol(str, Enum):
    compliance    = "compliance"
    risk          = "risk"
    due_diligence = "due_diligence"

class LLMBackend(str, Enum):
    ollama = "ollama"
    openai = "openai"

class ProcessState(str, Enum):
    pending    = "pending"
    processing = "processing"
    completed  = "completed"
    error      = "error"

class RiskLevel(str, Enum):
    low      = "low"
    moderate = "moderate"
    elevated = "elevated"
    high     = "high"


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class DocumentInfo(BaseModel):
    doc_id:          str
    filename:        str
    file_type:       str
    total_chunks:    int
    char_count:      int
    created_at:      float
    state:           ProcessState = ProcessState.completed
    risk_score:      Optional[int]  = None
    review_protocol: Optional[ReviewProtocol] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total:     int


class UploadResponse(BaseModel):
    added_files:   int
    skipped_files: int
    total_chunks:  int
    errors:        List[List[str]] = Field(default_factory=list)
    documents:     List[DocumentInfo] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    deleted: bool
    doc_id:  str
    message: str


# ---------------------------------------------------------------------------
# Query / Chat
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question:          str  = Field(..., min_length=1, max_length=4096)
    enable_web_search: bool = False


class ReviewRequest(BaseModel):
    document_chunk:    str  = Field(..., min_length=1)
    max_queries:       int  = Field(5, ge=1, le=20)
    enable_web_search: bool = False


# SSE event envelopes
class SSEToken(BaseModel):
    type:    str = "token"
    content: str


class SSEDone(BaseModel):
    type: str = "done"


class SSEError(BaseModel):
    type:    str = "error"
    message: str


class ReviewFinding(BaseModel):
    query:         str
    dimension:     str
    priority:      int
    analysis:      str
    error_regions: List[str]
    references:    List[str]
    revision:      str
    confidence:    float


class SSEReviewChunk(BaseModel):
    type:    str = "review_chunk"
    finding: ReviewFinding


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class LLMConfigView(BaseModel):
    backend:     LLMBackend
    temperature: float
    max_tokens:  int
    ollama_model:   str
    ollama_base_url: str
    openai_model:   str


class ConfigPatchRequest(BaseModel):
    backend:          Optional[LLMBackend] = None
    temperature:      Optional[float]      = Field(None, ge=0.0, le=2.0)
    max_tokens:       Optional[int]        = Field(None, ge=64, le=8192)
    ollama_model:     Optional[str]        = None
    ollama_base_url:  Optional[str]        = None
    openai_model:     Optional[str]        = None
    enable_web_search: Optional[bool]      = None
    use_legal_review:  Optional[bool]      = None


class ConfigResponse(BaseModel):
    llm:              LLMConfigView
    web_search_enabled: bool
    legal_review_enabled: bool
    retrieval:        Dict[str, Any]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status:       str  = "ok"
    llm_backend:  str
    llm_model:    str
    ollama_alive: bool
    vector_docs:  int
    bm25_indexed: int
    web_search:   bool
    legal_review: bool
    ai_latency_ms: Optional[int] = None
