# DocMind AI — Legal Document Intelligence Platform

> A full-stack RAG-powered legal document review system with GKGR (Guided Knowledge-Graph Retrieval) framework, hybrid retrieval, and a Forensic Terminal dark UI.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v4-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-black?logo=anthropic&logoColor=white)](https://claude.ai/code)

---

## Overview

DocMind AI is a local-first, AI-powered legal document analysis platform. Upload contracts, regulations, or case files and get structured risk assessments, clause annotations, and amendment suggestions — all streamed in real time via SSE. The system runs entirely on-premise using Ollama local LLMs, or can be pointed at OpenAI for cloud inference.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DOCMIND AI — SYSTEM OVERVIEW                 │
└─────────────────────────────────────────────────────────────────────┘

  Browser / Client
  ┌─────────────────────────────────────────────────┐
  │  React 18 + Vite + TailwindCSS v4               │
  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
  │  │Dashboard │  │ Upload   │  │ Report/Review│  │
  │  └──────────┘  └──────────┘  └──────────────┘  │
  │  ┌──────────────────────────────────────────┐   │
  │  │  Zustand stores  │  SSE client (fetch)   │   │
  │  └──────────────────────────────────────────┘   │
  └─────────────────────┬───────────────────────────┘
                        │ HTTP / SSE (/api/*)
                        ▼
  ┌─────────────────────────────────────────────────┐
  │  FastAPI  (server.py)                           │
  │  ┌─────────────────────────────────────────┐   │
  │  │  /api/documents  /api/query  /api/review │   │
  │  │  /api/config     /api/health             │   │
  │  └──────────────┬──────────────────────────┘   │
  │                 │ RAGPipeline singleton          │
  └─────────────────┼───────────────────────────────┘
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
  ┌──────────────┐        ┌──────────────────────┐
  │  VectorStore  │        │  BM25Retriever        │
  │  (FAISS + E5) │        │  (rank_bm25)          │
  └──────┬───────┘        └──────────┬───────────┘
         │     Hybrid RRF merge       │
         └───────────┬───────────────┘
                     ▼
          ┌──────────────────────┐
          │  CrossEncoder Rerank  │
          │  (MiniLM-L-6-v2)     │
          └──────────┬───────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
  ┌───────────────┐    ┌──────────────────────────┐
  │  Query Route   │    │   GKGR Legal Review       │
  │  (streaming)   │    │   ┌────────────────────┐  │
  └───────┬───────┘    │   │LegalSemanticChunker│  │
          │             │   │LegalKeyInfoExtract │  │
          ▼             │   │LegalKnowledgeScorer│  │
  ┌───────────────┐    │   └────────────────────┘  │
  │  LLM Backend  │    │   ┌────────────────────┐  │
  │  ┌──────────┐ │    │   │ Findings SSE stream │  │
  │  │ Ollama   │ │    │   └────────────────────┘  │
  │  │ OpenAI   │ │    └──────────────────────────┘
  │  └──────────┘ │
  └───────────────┘
```

---

## GKGR Pipeline Flow

```
  Document Input
       │
       ▼
  ┌─────────────────────────────────────────────────────┐
  │  LegalSemanticChunker                               │
  │  · Clause-boundary-aware splitting                  │
  │  · Metadata injection (article, section, type)      │
  └───────────────────────┬─────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │  LegalKeyInfoExtractor  (3-level term extraction)   │
  │  ┌──────────────────────────────────────────────┐   │
  │  │  MAX  — primary legal obligations & rights   │   │
  │  │  MID  — conditional clauses & exceptions     │   │
  │  │  LIT  — literal citations & defined terms    │   │
  │  └──────────────────────────────────────────────┘   │
  └───────────────────────┬─────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │  LegalKnowledgeScorer                               │
  │  · Term significance score                          │
  │  · Rarity index (IDF over corpus)                   │
  │  · Coherence index (inter-chunk semantic density)   │
  └───────────────────────┬─────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │  HybridRetriever  (GKGR-weighted)                   │
  │  · FAISS dense  +  BM25 sparse  →  RRF merge        │
  │  · GKGR scores boost legally significant chunks     │
  │  · CrossEncoder final reranking                     │
  └───────────────────────┬─────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │  LegalReviewer  (SSE streaming)                     │
  │  · Clause-by-clause risk analysis                   │
  │  · Structured findings: severity / article / text   │
  │  · LLM-generated amendment suggestions              │
  └─────────────────────────────────────────────────────┘
```

---

## Data Flow — Query Request

```
  User types question
         │
         ▼
  POST /api/query  ──►  IntentAnalyzer  ──►  query rewrite
         │
         ▼
  HybridRetriever.retrieve(query, top_k=12)
    ├─ FAISS ANN search (dense E5 embeddings)
    ├─ BM25 lexical search
    └─ RRF score merge → CrossEncoder rerank → top 5

         │  context chunks
         ▼
  LLM.stream(system_prompt + context + question)
         │
         ▼  SSE events
  {"type":"token","content":"…"}  ×N
  {"type":"done"}
```

---

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| **Frontend** | React 18, TypeScript, Vite | SPA, no SSR |
| **Styling** | TailwindCSS v4 (`@theme {}`) | "Forensic Terminal" design system |
| **State** | Zustand | Log store + Toast store |
| **Routing** | React Router v6 | Hash-based SPA routes |
| **Backend** | FastAPI 0.111, Python 3.11 | Async, lifespan startup |
| **Embeddings** | `intfloat/multilingual-e5-base` | via sentence-transformers |
| **Vector DB** | FAISS (CPU) | Persisted to Docker volume |
| **Sparse Retrieval** | rank-bm25 | BM25Okapi |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder |
| **LLM (local)** | Ollama | `llama3.2` default |
| **LLM (cloud)** | OpenAI API | `gpt-4o-mini` default |
| **Web Search** | SerpAPI | Optional; `SERPAPI_KEY` required |
| **Serving** | nginx:alpine | SSE buffering disabled |
| **Containerization** | Docker Compose | Multi-stage build |

---

## Directory Structure

```
local_chat_rag/
├── backend/
│   ├── server.py                  # FastAPI app entry, lifespan, CORS
│   ├── requirements_server.txt    # Production Python dependencies
│   ├── Dockerfile
│   ├── api/
│   │   ├── models.py              # All Pydantic schemas & enums
│   │   ├── state.py               # Singleton state + dependency providers
│   │   └── routes/
│   │       ├── documents.py       # Upload / list / delete documents
│   │       ├── query.py           # /query and /review SSE endpoints
│   │       └── config.py          # GET/PATCH config, health check
│   └── core/
│       ├── rag_pipeline.py        # RAGPipeline orchestrator
│       ├── retrieval.py           # HybridRetriever, RecursiveRetriever
│       ├── vector_store.py        # FAISS store with delete support
│       ├── doc_processor.py       # PDF / text ingestion & chunking
│       ├── llm_factory.py         # Ollama / OpenAI LLM factory
│       ├── config.py              # AppConfig (pydantic-settings)
│       ├── legal_chunker.py       # LegalSemanticChunker
│       ├── legal_gkgr.py          # LegalKeyInfoExtractor + Scorer
│       ├── legal_reviewer.py      # LegalReviewer (SSE findings)
│       ├── intent_analyze.py      # Query intent classification
│       └── web_search.py          # SerpAPI integration
│
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf                 # Reverse proxy + SSE config
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── src/
│       ├── api/client.ts          # fetch-based API client + SSE generator
│       ├── store/
│       │   ├── logStore.ts        # Terminal log (Zustand)
│       │   └── toastStore.ts      # Toast notifications (Zustand)
│       ├── components/
│       │   ├── layout/
│       │   │   ├── AppLayout.tsx  # Root shell, CMD+K handler
│       │   │   ├── SideNavBar.tsx
│       │   │   ├── TopAppBar.tsx  # Health ping, search trigger
│       │   │   └── TerminalLog.tsx
│       │   └── ui/
│       │       ├── CommandPalette.tsx
│       │       ├── ConfigPanel.tsx
│       │       └── ToastContainer.tsx
│       └── pages/
│           ├── Dashboard.tsx      # Document table + risk overview
│           ├── Upload.tsx         # Drag-drop upload + protocol select
│           ├── Report.tsx         # Legal review + findings panel
│           └── Review.tsx         # Diff viewer + amendment redlines
│
└── docker-compose.yml
```

---

## Features

### Document Management
- Drag-and-drop multi-file upload (PDF, TXT, DOCX)
- Per-document delete with FAISS index rebuild
- Metadata table: file size, chunk count, upload date
- Risk score overview with color-coded bars

### AI Query
- Streaming Q&A over ingested documents
- Hybrid dense + sparse retrieval with RRF fusion
- CrossEncoder reranking for precision
- Web search augmentation (optional, SerpAPI)
- Intent analysis for query routing

### Legal Review (GKGR)
- Clause-level risk analysis with severity ratings (CRITICAL / HIGH / MEDIUM / LOW)
- Three-level legal term extraction (obligation, conditional, literal)
- GKGR knowledge scoring boosts legally significant passages
- Structured findings: article reference, risk description, suggested fix
- Real-time SSE streaming of findings as they are generated

### Amendment Redline
- Side-by-side diff view (original vs. amended)
- LLM-generated amendment suggestions per finding
- Multi-finding selector with confidence bars
- Color-coded diff lines (removed=red, added=blue)

### Configuration (Runtime Hot-Patch)
- Switch LLM backend (Ollama ↔ OpenAI) without restart
- Update model name, base URL, temperature, max_tokens
- Toggle web search and legal review mode

### UI / UX
- "Forensic Terminal" dark design system (custom TailwindCSS v4 theme)
- Command palette (CMD+K) with keyboard navigation
- Persistent terminal log panel
- Toast notifications with auto-dismiss
- Health indicator with live latency display

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/documents/upload` | Upload documents (multipart/form-data) |
| `GET` | `/api/documents` | List all indexed documents |
| `DELETE` | `/api/documents/{doc_id}` | Delete document + rebuild index |
| `POST` | `/api/query` | Stream Q&A (SSE) |
| `POST` | `/api/review` | Stream legal review findings (SSE) |
| `GET` | `/api/config` | Get current runtime config |
| `PATCH` | `/api/config` | Hot-patch LLM/feature config |
| `GET` | `/api/health` | Health + pipeline status |

### SSE Event Types

```jsonc
// Token stream (query & review)
{ "type": "token", "content": "…" }

// Stream complete
{ "type": "done" }

// Error
{ "type": "error", "detail": "…" }

// Legal review finding (review endpoint only)
{ "type": "review_chunk", "finding": {
    "article":   "第3条",
    "severity":  "HIGH",
    "risk":      "Liability clause lacks cap on damages",
    "suggestion": "Add: '…total liability shall not exceed…'"
}}
```

---

## Quickstart

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Docker mode)
- [Ollama](https://ollama.com/) with `llama3.2` pulled (for local LLM)
- Node 20+ and Python 3.11+ (for dev mode)

### Option A — Docker Compose (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/Jeremyzmchen/Legal-Document-Review-System.git
cd Legal-Document-Review-System

# 2. Create your secrets file
cp backend/core/.env.example backend/core/dev.env
#    Edit dev.env: set SERPAPI_KEY and/or OPENAI_API_KEY

# 3. Pull your local model (if using Ollama)
ollama pull llama3.2

# 4. Start everything
docker compose up --build

# 5. Open http://localhost:5173
```

### Option B — Development Mode

```bash
# Terminal 1 — Backend
cd backend
pip install -r requirements_server.txt
uvicorn server:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
# Vite dev server proxies /api → localhost:8000
# Open http://localhost:5173
```

### Environment Variables

Create `backend/core/dev.env`:

```env
# Optional — leave blank to use only local Ollama
OPENAI_API_KEY=sk-...

# Optional — required only if web search is enabled
SERPAPI_KEY=...

# LLM defaults (can be changed at runtime via ConfigPanel)
LLM_BACKEND=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://host.docker.internal:11434
OPENAI_MODEL=gpt-4o-mini
TEMPERATURE=0.7
MAX_TOKENS=1536
```

> **Note on Ollama inside Docker**: use `host.docker.internal:11434` as the base URL so the backend container can reach Ollama running on your host machine.

---

## Design System

The frontend uses a custom "Forensic Terminal" design language built with TailwindCSS v4's `@theme {}` CSS block (no `tailwind.config.js`).

Key tokens:

| Token | Value | Usage |
|-------|-------|-------|
| `--color-surface-primary` | `#0e0e0e` | Main backgrounds |
| `--color-accent-blue` | `#adc6ff` | Interactive elements, headings |
| `--color-accent-amber` | `#f7c6a3` | Warnings |
| `--color-accent-red` | `#ff8a80` | Critical risk, errors |
| `--color-accent-green` | `#81c995` | Success, safe clauses |
| `--font-headline` | Space Grotesk | Section headers |
| `--font-mono` | JetBrains Mono | All data / terminal output |

---

## Co-Development Notice

> The frontend UI of DocMind AI was **co-developed with [Claude Code](https://claude.ai/code)** (Anthropic's AI coding assistant). Claude Code contributed to the complete React component architecture, TailwindCSS v4 design system implementation, SSE streaming client, Zustand store design, and Docker/nginx configuration throughout this project.

---

## License

MIT — see [LICENSE](LICENSE) for details.
