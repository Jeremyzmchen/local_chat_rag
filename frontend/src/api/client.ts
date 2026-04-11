/**
 * DocMind AI — API Client
 * All requests go to /api (proxied to http://localhost:8000 in dev)
 */

const BASE = '/api'

// ── Types (mirrors backend api/models.py) ──────────────────────────────────

export interface HealthResponse {
  status:        string
  llm_backend:   string
  llm_model:     string
  ollama_alive:  boolean
  vector_docs:   number
  bm25_indexed:  number
  web_search:    boolean
  legal_review:  boolean
  ai_latency_ms: number | null
}

export interface DocumentInfo {
  doc_id:          string
  filename:        string
  file_type:       string
  total_chunks:    number
  char_count:      number
  created_at:      number
  state:           'pending' | 'processing' | 'completed' | 'error'
  risk_score:      number | null
  review_protocol: string | null
}

export interface DocumentListResponse {
  documents: DocumentInfo[]
  total:     number
}

export interface UploadResponse {
  added_files:   number
  skipped_files: number
  total_chunks:  number
  errors:        string[][]
  documents:     DocumentInfo[]
}

export interface DeleteResponse {
  deleted: boolean
  doc_id:  string
  message: string
}

export interface ReviewFinding {
  query:         string
  dimension:     string
  priority:      number
  analysis:      string
  error_regions: string[]
  references:    string[]
  revision:      string
  confidence:    number
}

export interface ConfigResponse {
  llm: {
    backend:         string
    temperature:     number
    max_tokens:      number
    ollama_model:    string
    ollama_base_url: string
    openai_model:    string
  }
  web_search_enabled:   boolean
  legal_review_enabled: boolean
  retrieval: Record<string, unknown>
}

// ── SSE helpers ────────────────────────────────────────────────────────────

export type SSEEvent =
  | { type: 'token';        content: string }
  | { type: 'done' }
  | { type: 'error';        message: string }
  | { type: 'review_chunk'; finding: ReviewFinding }

export type UploadSSEEvent =
  | { type: 'progress'; stage: 'extract' | 'index' | 'done'; current: number; total: number; filename: string }
  | { type: 'done';     result: UploadResponse }
  | { type: 'error';    detail: string }

/**
 * Open an SSE stream via fetch + ReadableStream.
 * Calls onEvent for each parsed SSE message, onDone when stream closes.
 */
export async function openSSE(
  url:     string,
  body:    unknown,
  onEvent: (e: SSEEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
    signal,
  })

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`HTTP ${res.status}: ${text}`)
  }

  const reader  = res.body!.getReader()
  const decoder = new TextDecoder()
  let   buffer  = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const raw = line.slice(6).trim()
      if (!raw) continue
      try {
        const event = JSON.parse(raw) as SSEEvent
        onEvent(event)
      } catch {
        // malformed line — skip
      }
    }
  }
}

// ── REST helpers ───────────────────────────────────────────────────────────

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`)
  return res.json()
}

async function del<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`DELETE ${path} → ${res.status}`)
  return res.json()
}

async function patch<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method:  'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`PATCH ${path} → ${res.status}`)
  return res.json()
}

// ── API surface ────────────────────────────────────────────────────────────

export const api = {
  // Health & Config
  health: ()                         => get<HealthResponse>('/health'),
  getConfig: ()                      => get<ConfigResponse>('/config'),
  patchConfig: (body: Partial<ConfigResponse['llm'] & {
    enable_web_search?: boolean
    use_legal_review?:  boolean
  }>)                                => patch<ConfigResponse>('/config', body),

  // Documents
  listDocuments: ()                  => get<DocumentListResponse>('/documents'),
  deleteDocument: (docId: string)    => del<DeleteResponse>(`/documents/${docId}`),

  uploadDocuments: (
    files:    File[],
    protocol: string = 'compliance',
    onEvent?: (e: UploadSSEEvent) => void,
    signal?:  AbortSignal,
  ): Promise<UploadResponse> => {
    const form = new FormData()
    files.forEach(f => form.append('files', f))
    form.append('protocol', protocol)

    return fetch(`${BASE}/documents/upload`, { method: 'POST', body: form, signal })
      .then(async r => {
        if (!r.ok) return r.text().then(t => Promise.reject(new Error(t)))

        const reader  = r.body!.getReader()
        const decoder = new TextDecoder()
        let   buffer  = ''

        while (true) {
          const { value, done } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            const raw = line.slice(6).trim()
            if (!raw) continue
            try {
              const event = JSON.parse(raw) as UploadSSEEvent
              onEvent?.(event)
              if (event.type === 'done') return event.result
              if (event.type === 'error') throw new Error(event.detail)
            } catch (e) {
              if (e instanceof SyntaxError) continue
              throw e
            }
          }
        }
        throw new Error('Upload stream ended without a done event')
      })
  },

  // Query (SSE)
  streamQuery: (
    question:          string,
    enableWebSearch:   boolean,
    onEvent:           (e: SSEEvent) => void,
    signal?:           AbortSignal,
  ) =>
    openSSE(
      `${BASE}/query`,
      { question, enable_web_search: enableWebSearch },
      onEvent,
      signal,
    ),

  // Legal Review (SSE)
  streamReview: (
    documentChunk:    string,
    maxQueries:       number,
    enableWebSearch:  boolean,
    onEvent:          (e: SSEEvent) => void,
    signal?:          AbortSignal,
  ) =>
    openSSE(
      `${BASE}/review`,
      {
        document_chunk:    documentChunk,
        max_queries:       maxQueries,
        enable_web_search: enableWebSearch,
      },
      onEvent,
      signal,
    ),
}
