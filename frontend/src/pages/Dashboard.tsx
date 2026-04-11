import { useEffect, useState } from 'react'
import { useNavigate }         from 'react-router-dom'
import { api } from '../api/client'
import type { DocumentInfo, HealthResponse } from '../api/client'
import { useLogStore }         from '../store/logStore'

// ── Risk colour helper ───────────────────────────────────────────────────────
function riskColor(score: number | null) {
  if (score === null) return 'text-zinc-500'
  if (score >= 80) return 'text-[#ffb4ab]'
  if (score >= 50) return 'text-[#ffb786]'
  return 'text-[#adc6ff]'
}
function riskBarColor(score: number | null) {
  if (score === null) return 'bg-zinc-700'
  if (score >= 80) return 'bg-[#ffb4ab]'
  if (score >= 50) return 'bg-[#ffb786]'
  return 'bg-[#adc6ff]'
}

// ── State chip ───────────────────────────────────────────────────────────────
function StateChip({ state }: { state: DocumentInfo['state'] }) {
  const map: Record<string, string> = {
    completed:  'bg-[#304671] text-[#9fb5e7]',
    processing: 'bg-[#df7412]/20 text-[#ffb786]',
    pending:    'bg-zinc-800 text-zinc-400',
    error:      'bg-[#93000a]/20 text-[#ffb4ab]',
  }
  return (
    <span className={`px-2 py-0.5 text-[10px] font-mono font-bold uppercase tracking-tighter ${map[state] ?? map.pending}`}>
      {state}
    </span>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function Dashboard() {
  const navigate  = useNavigate()
  const addLog    = useLogStore(s => s.addLog)

  const [docs,   setDocs]   = useState<DocumentInfo[]>([])
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.listDocuments(), api.health()])
      .then(([docRes, healthRes]) => {
        setDocs(docRes.documents)
        setHealth(healthRes)
        addLog(`Loaded ${docRes.total} documents | AI latency: ${healthRes.ai_latency_ms ?? '—'}ms`)
      })
      .catch(e => addLog(`Failed to load dashboard: ${e.message}`, 'error'))
      .finally(() => setLoading(false))
  }, [])

  const handleDelete = async (docId: string, filename: string) => {
    try {
      await api.deleteDocument(docId)
      setDocs(prev => prev.filter(d => d.doc_id !== docId))
      addLog(`Deleted document: ${filename}`, 'warn')
    } catch (e: any) {
      addLog(`Delete failed: ${e.message}`, 'error')
    }
  }

  const systemIntegrity = health
    ? Math.round((health.vector_docs > 0 ? 92 : 60))
    : null

  return (
    <div className="flex flex-1 h-full overflow-hidden">

      {/* ── Main content ── */}
      <div className="flex-1 bg-[#1c1b1b] p-6 flex flex-col overflow-hidden">

        {/* Header */}
        <section className="flex justify-between items-end mb-6 shrink-0">
          <div>
            <h1 className="font-headline font-black text-4xl text-[#e5e2e1] uppercase tracking-tighter">
              Terminal_Dashboard
            </h1>
            <p className="text-[#c2c6d6] font-mono text-[10px] uppercase tracking-widest mt-1">
              Status: Operational
              {health && ` // AI Latency: ${health.ai_latency_ms ?? '—'}ms`}
              {health && ` // LLM: ${health.llm_model}`}
            </p>
          </div>
          <button
            onClick={() => navigate('/upload')}
            className="bg-gradient-to-br from-[#adc6ff] to-[#4d8eff] text-[#002e6a] font-headline font-bold uppercase text-xs px-6 py-3 flex items-center gap-2 hover:brightness-110 transition-all shadow-[0_0_15px_rgba(173,198,255,0.2)]"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>upload</span>
            Upload Document
          </button>
        </section>

        {/* Table */}
        <div className="bg-[#20201f] border border-[#424754]/10 flex-1 flex flex-col overflow-hidden">
          <div className="px-4 py-2.5 bg-[#353535]/30 border-b border-[#424754]/10 flex justify-between items-center shrink-0">
            <span className="font-mono text-[10px] uppercase text-zinc-500 tracking-tighter">
              Active_Reviews_Queue
            </span>
            <span className="font-mono text-[10px] uppercase text-[#adc6ff] tracking-tighter">
              Results: {docs.length} Documents
            </span>
          </div>

          <div className="overflow-auto flex-1">
            {loading ? (
              <div className="flex items-center justify-center h-32 font-mono text-[11px] text-zinc-600 uppercase tracking-widest">
                <span className="animate-pulse">Scanning knowledge base...</span>
              </div>
            ) : docs.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-32 gap-2">
                <span className="material-symbols-outlined text-zinc-700" style={{ fontSize: 36 }}>folder_open</span>
                <span className="font-mono text-[11px] text-zinc-600 uppercase tracking-widest">
                  No documents indexed — upload one to begin
                </span>
              </div>
            ) : (
              <table className="w-full border-collapse text-left">
                <thead className="sticky top-0 bg-[#20201f] z-10 border-b border-[#424754]/20">
                  <tr className="font-mono text-[10px] uppercase text-zinc-400 tracking-tighter">
                    <th className="p-4 font-normal">File_Identifier</th>
                    <th className="p-4 font-normal">Protocol</th>
                    <th className="p-4 font-normal">Chunks</th>
                    <th className="p-4 font-normal">Process_State</th>
                    <th className="p-4 font-normal text-right">Risk_Vector</th>
                    <th className="p-4 font-normal text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-xs text-[#c2c6d6]">
                  {docs.map((doc, i) => (
                    <tr
                      key={doc.doc_id}
                      className={`hover:bg-[#2a2a2a] transition-colors group cursor-pointer ${i % 2 === 1 ? 'bg-[#1c1b1b]/50' : ''}`}
                      onClick={() => navigate(`/report/${doc.doc_id}`)}
                    >
                      <td className="p-4 border-b border-[#424754]/5">
                        <div className="flex items-center gap-3">
                          <span className="material-symbols-outlined text-zinc-600 group-hover:text-[#adc6ff] transition-colors" style={{ fontSize: 18 }}>
                            description
                          </span>
                          <span className="text-[#e5e2e1] truncate max-w-[180px]">{doc.filename}</span>
                        </div>
                      </td>
                      <td className="p-4 border-b border-[#424754]/5 capitalize">
                        {doc.review_protocol ?? '—'}
                      </td>
                      <td className="p-4 border-b border-[#424754]/5">
                        {doc.total_chunks}
                      </td>
                      <td className="p-4 border-b border-[#424754]/5">
                        <StateChip state={doc.state} />
                      </td>
                      <td className="p-4 border-b border-[#424754]/5 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <span className={`font-bold ${riskColor(doc.risk_score)}`}>
                            {doc.risk_score !== null ? String(doc.risk_score).padStart(2, '0') : '—'}
                          </span>
                          <div className="w-12 h-1 bg-zinc-800 overflow-hidden">
                            <div
                              className={`h-full ${riskBarColor(doc.risk_score)} transition-all`}
                              style={{ width: `${doc.risk_score ?? 0}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className="p-4 border-b border-[#424754]/5 text-right">
                        <button
                          onClick={e => { e.stopPropagation(); handleDelete(doc.doc_id, doc.filename) }}
                          className="text-zinc-600 hover:text-[#ffb4ab] transition-colors"
                        >
                          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>delete</span>
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>

      {/* ── Right panel: Risk Inspector ── */}
      <aside className="w-72 bg-[#20201f] border-l border-[#424754]/10 p-5 flex flex-col gap-5 shrink-0 overflow-y-auto">

        {/* System integrity gauge */}
        <div className="space-y-2">
          <h3 className="font-headline font-bold text-xs uppercase tracking-widest text-zinc-400">
            Node_Overview
          </h3>
          <div className="h-28 bg-[#1c1b1b] border border-[#424754]/20 flex flex-col items-center justify-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-[#adc6ff]/5 to-transparent pointer-events-none" />
            <span className="text-4xl font-headline font-black text-[#adc6ff] phosphor-glow">
              {systemIntegrity !== null ? `${systemIntegrity}%` : '—'}
            </span>
            <span className="font-mono text-[10px] uppercase text-zinc-500 mt-1">
              System_Integrity
            </span>
          </div>
        </div>

        {/* AI Insights */}
        <div className="space-y-3">
          <h3 className="font-headline font-bold text-xs uppercase tracking-widest text-zinc-400">
            AI_Insights
          </h3>
          {health ? (
            <>
              <div className="p-3 bg-[#2a2a2a] border-l-2 border-[#adc6ff]">
                <p className="font-mono text-[11px] leading-relaxed text-[#c2c6d6]">
                  <span className="text-[#adc6ff]">DETECTED: </span>
                  {health.vector_docs} chunks indexed across {docs.length} documents.
                  LLM backend: {health.llm_backend.toUpperCase()}.
                </p>
              </div>
              {!health.ollama_alive && health.llm_backend === 'ollama' && (
                <div className="p-3 bg-[#2a2a2a] border-l-2 border-[#ffb786]">
                  <p className="font-mono text-[11px] leading-relaxed text-[#c2c6d6]">
                    <span className="text-[#ffb786]">WARNING: </span>
                    Ollama service is unreachable. Check that Ollama is running on port 11434.
                  </p>
                </div>
              )}
            </>
          ) : (
            <div className="p-3 bg-[#2a2a2a] border-l-2 border-zinc-700">
              <p className="font-mono text-[11px] text-zinc-600 animate-pulse">Connecting to AI core...</p>
            </div>
          )}
        </div>

        {/* System stats */}
        {health && (
          <div className="mt-auto space-y-2">
            {[
              { label: 'VECTOR_CHUNKS', value: health.vector_docs, max: Math.max(health.vector_docs, 100) },
              { label: 'BM25_INDEX',    value: health.bm25_indexed, max: Math.max(health.bm25_indexed, 100) },
            ].map(({ label, value, max }) => (
              <div key={label}>
                <div className="flex justify-between font-mono text-[10px] uppercase text-zinc-500 mb-1">
                  <span>{label}</span>
                  <span>{value}</span>
                </div>
                <div className="w-full h-1 bg-zinc-800">
                  <div
                    className="h-full bg-[#adc6ff]/40"
                    style={{ width: `${Math.min((value / max) * 100, 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </aside>
    </div>
  )
}
