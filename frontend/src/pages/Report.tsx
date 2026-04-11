import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams }       from 'react-router-dom'
import { api }                          from '../api/client'
import type { ReviewFinding }           from '../api/client'
import { useLogStore }                  from '../store/logStore'

// ── Severity chip ─────────────────────────────────────────────────────────────
function SeverityChip({ priority }: { priority: number }) {
  if (priority >= 3) return (
    <span className="text-[8px] font-mono bg-[#93000a] text-[#ffdad6] px-2 py-0.5 font-bold uppercase">HIGH_SEVERITY</span>
  )
  if (priority === 2) return (
    <span className="text-[8px] font-mono bg-[#df7412] text-[#461f00] px-2 py-0.5 font-bold uppercase">MED_SEVERITY</span>
  )
  return (
    <span className="text-[8px] font-mono bg-[#4d8eff] text-[#00285d] px-2 py-0.5 font-bold uppercase">LOW_SEVERITY</span>
  )
}

// ── Finding card ──────────────────────────────────────────────────────────────
function FindingCard({ finding, index, onDiff }: {
  finding: ReviewFinding
  index:   number
  onDiff:  (f: ReviewFinding) => void
}) {
  const borderColor = finding.priority >= 3 ? 'border-[#93000a]/30' : finding.priority === 2 ? 'border-[#df7412]/30' : 'border-[#4d8eff]/30'
  const id = `LR-${String(index + 1).padStart(3, '0')}`

  return (
    <div className={`bg-[#353535]/40 border ${borderColor} p-4 relative overflow-hidden group hover:brightness-110 transition-all`}>
      <div className="flex items-center justify-between mb-2">
        <SeverityChip priority={finding.priority} />
        <span className="text-[10px] font-mono text-zinc-500">ID: #{id}</span>
      </div>

      {/* Error regions */}
      {finding.error_regions.length > 0 && (
        <p className="text-xs font-mono text-[#c2c6d6] mb-2 leading-tight">
          "<span className={finding.priority >= 3 ? 'text-[#ffb4ab]' : finding.priority === 2 ? 'text-[#ffb786]' : 'text-[#adc6ff]'}>
            {finding.error_regions[0]}
          </span>"
        </p>
      )}

      <div className="pt-2 border-t border-zinc-800">
        <p className="text-[10px] font-mono text-zinc-400 uppercase mb-1">Dimension: {finding.dimension}</p>
        <p className="text-[10px] font-mono text-[#adc6ff]">{finding.revision.slice(0, 120)}{finding.revision.length > 120 ? '...' : ''}</p>
      </div>

      <div className="mt-2 flex items-center justify-between">
        <span className="text-[9px] font-mono text-zinc-600">Confidence: {(finding.confidence * 100).toFixed(0)}%</span>
        <button
          onClick={() => onDiff(finding)}
          className="text-[9px] font-mono text-[#adc6ff] hover:text-white transition-colors flex items-center gap-1"
        >
          <span className="material-symbols-outlined" style={{ fontSize: 12 }}>difference</span>
          VIEW DIFF
        </button>
      </div>
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────
export default function Report() {
  const { id }   = useParams<{ id: string }>()
  const navigate  = useNavigate()
  const addLog    = useLogStore(s => s.addLog)

  const [docText,   setDocText]   = useState<string>('')
  const [findings,  setFindings]  = useState<ReviewFinding[]>([])
  const [scanning,  setScanning]  = useState(false)
  const [done,      setDone]      = useState(false)
  const [riskScore, setRiskScore] = useState<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Load document text from vector store metadata (best-effort)
  useEffect(() => {
    if (!id) return
    // We don't have a direct "get document text" endpoint yet,
    // so show a placeholder and let user paste / we use doc_id as label
    setDocText(`Document: ${id}\n\nClick "Run AI Scan" to begin legal review analysis.`)
  }, [id])

  const runScan = async () => {
    if (!docText || scanning) return
    abortRef.current?.abort()
    abortRef.current = new AbortController()

    setScanning(true)
    setFindings([])
    setDone(false)
    addLog(`Starting legal review scan for doc: ${id}`)

    try {
      await api.streamReview(
        docText,
        5,
        false,
        (event) => {
          if (event.type === 'review_chunk') {
            setFindings(prev => [...prev, event.finding])
            addLog(`[SCAN] Found issue: ${event.finding.dimension} (priority ${event.finding.priority})`, 'warn')
          } else if (event.type === 'done') {
            setDone(true)
            addLog('Legal review scan complete.')
          } else if (event.type === 'error') {
            addLog(`Scan error: ${event.message}`, 'error')
          }
        },
        abortRef.current.signal,
      )
    } catch (e: any) {
      if (e.name !== 'AbortError') addLog(`Scan failed: ${e.message}`, 'error')
    } finally {
      setScanning(false)
    }

    // Derive risk score from findings
    setRiskScore(findings.length ? Math.min(findings.reduce((a, f) => a + f.priority * 25, 0), 100) : 0)
  }

  // Outline sections derived from findings
  const dimensions = [...new Set(findings.map(f => f.dimension))]

  return (
    <div className="flex h-full overflow-hidden">

      {/* ── Left: Document outline ── */}
      <aside className="w-56 bg-[#20201f] flex flex-col border-r border-[#424754]/10 shrink-0">
        <div className="p-4 border-b border-zinc-800/50">
          <h2 className="text-[10px] font-mono font-bold text-zinc-500 uppercase tracking-widest">
            Document_Structure
          </h2>
        </div>
        <nav className="flex-1 overflow-y-auto py-2">
          {dimensions.length === 0 ? (
            <p className="px-4 py-3 font-mono text-[10px] text-zinc-700 uppercase">Run scan to see structure</p>
          ) : dimensions.map((dim, i) => {
            const hasHigh = findings.some(f => f.dimension === dim && f.priority >= 3)
            const hasMed  = findings.some(f => f.dimension === dim && f.priority === 2)
            const dotColor = hasHigh ? 'bg-[#ffb4ab]' : hasMed ? 'bg-[#ffb786]' : 'bg-[#adc6ff]'
            return (
              <a
                key={dim}
                href={`#dim-${i}`}
                className="flex items-center justify-between px-4 py-2 text-xs font-mono text-[#c2c6d6] hover:bg-zinc-800 group"
              >
                <span className="truncate">{String(i + 1).padStart(2, '0')}. {dim.toUpperCase()}</span>
                <div className={`w-2 h-2 rounded-full shrink-0 ${dotColor}`} />
              </a>
            )
          })}
        </nav>
        <div className="p-4 bg-[#1c1b1b] space-y-1.5">
          {[['bg-[#ffb4ab]', 'CRITICAL_RISK'], ['bg-[#ffb786]', 'MODERATE'], ['bg-[#adc6ff]', 'LOW_RISK']].map(([c, l]) => (
            <div key={l} className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${c}`} />
              <span className="text-[9px] font-mono text-zinc-500">{l}</span>
            </div>
          ))}
        </div>
      </aside>

      {/* ── Center: Document viewer ── */}
      <section className="flex-1 bg-[#1c1b1b] flex flex-col overflow-hidden">
        {/* Toolbar */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-zinc-800/50 shrink-0">
          <div className="flex items-center gap-3">
            <nav className="flex items-center gap-1 text-[10px] font-mono text-zinc-500 uppercase">
              <button onClick={() => navigate('/')} className="hover:text-[#adc6ff]">Documents</button>
              <span>/</span>
              <span className="text-[#d8e2ff]">{id}</span>
            </nav>
          </div>
          <div className="flex items-center gap-2">
            {riskScore !== null && (
              <div className="flex items-center gap-2 bg-[#0e0e0e] px-3 py-1 border border-[#424754]/10">
                <div className="relative w-7 h-7">
                  <svg className="w-full h-full -rotate-90" viewBox="0 0 28 28">
                    <circle cx="14" cy="14" r="12" fill="transparent" stroke="#353535" strokeWidth="3"/>
                    <circle
                      cx="14" cy="14" r="12" fill="transparent"
                      stroke={riskScore >= 70 ? '#ffb786' : '#adc6ff'}
                      strokeWidth="3"
                      strokeDasharray={`${(riskScore / 100) * 75} 75`}
                    />
                  </svg>
                  <span className="absolute inset-0 flex items-center justify-center font-mono text-[9px] font-bold">{riskScore}</span>
                </div>
                <div>
                  <div className="font-mono text-[9px] text-zinc-500 uppercase leading-none">RISK</div>
                  <div className={`font-mono text-[9px] uppercase leading-none ${riskScore >= 70 ? 'text-[#ffb786]' : 'text-[#adc6ff]'}`}>
                    {riskScore >= 70 ? 'ELEVATED' : 'NORMAL'}
                  </div>
                </div>
              </div>
            )}
            <button
              onClick={runScan}
              disabled={scanning}
              className="bg-gradient-to-br from-[#adc6ff] to-[#4d8eff] text-[#002e6a] font-mono text-[10px] font-bold uppercase px-4 py-2 flex items-center gap-1.5 hover:brightness-110 transition-all disabled:opacity-50 shadow-[0_0_10px_rgba(77,142,255,0.3)]"
            >
              <span className={`material-symbols-outlined ${scanning ? 'animate-spin' : ''}`} style={{ fontSize: 14 }}>
                {scanning ? 'autorenew' : 'psychology'}
              </span>
              {scanning ? 'Scanning...' : 'Run_AI_Scan'}
            </button>
          </div>
        </div>

        {/* Document text */}
        <div className="flex-1 overflow-y-auto p-10">
          <div className="max-w-2xl mx-auto space-y-5">
            <h1 className="text-lg font-bold font-headline text-white uppercase tracking-tight">
              {id}
            </h1>
            <textarea
              value={docText}
              onChange={e => setDocText(e.target.value)}
              className="w-full h-96 bg-transparent border border-[#424754]/20 font-mono text-sm text-[#c2c6d6] leading-relaxed p-4 resize-none focus:outline-none focus:border-[#adc6ff]/40"
              placeholder="Paste document text here, then click Run AI Scan..."
            />
            {scanning && (
              <div className="text-[10px] font-mono text-[#adc6ff] animate-pulse uppercase tracking-widest">
                AI scanning document... {findings.length} issues found so far
              </div>
            )}
          </div>
        </div>

        {/* Status bar */}
        <div className="h-7 bg-zinc-900 border-t border-[#424754]/10 flex items-center px-4 justify-between font-mono text-[9px] text-zinc-500 uppercase tracking-widest shrink-0">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <span className={`w-2 h-2 rounded-full ${scanning ? 'bg-[#adc6ff] animate-pulse' : done ? 'bg-green-500' : 'bg-zinc-700'}`} />
              {scanning ? 'AI_SCANNING' : done ? 'SCAN_COMPLETE' : 'READY'}
            </span>
            <span>{docText.length} CHARS</span>
          </div>
          <span>{findings.length} FINDINGS</span>
        </div>
      </section>

      {/* ── Right: AI Findings ── */}
      <aside className="w-72 bg-[#20201f] flex flex-col border-l border-[#424754]/10 shrink-0">
        <div className="p-4 border-b border-zinc-800/50 flex items-center justify-between shrink-0">
          <h2 className="text-[10px] font-mono font-bold text-zinc-500 uppercase tracking-widest">
            AI_Analysis_Log
          </h2>
          {scanning && (
            <span className="text-[10px] font-mono text-[#adc6ff] bg-[#adc6ff]/10 px-2 py-0.5">LIVE_SCAN</span>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {findings.length === 0 && !scanning && (
            <p className="font-mono text-[10px] text-zinc-700 uppercase">No findings yet — run scan</p>
          )}
          {findings.map((f, i) => (
            <FindingCard
              key={i}
              finding={f}
              index={i}
              onDiff={finding => navigate(`/review/${id}`, { state: { finding } })}
            />
          ))}
        </div>

        {findings.length > 0 && (
          <div className="p-4 border-t border-zinc-800/50 shrink-0">
            <button
              onClick={() => navigate(`/review/${id}`, { state: { findings } })}
              className="w-full bg-[#adc6ff] text-[#002e6a] py-2 font-mono text-[10px] font-bold uppercase tracking-widest hover:bg-[#4d8eff] transition-colors relative group"
            >
              Generate_Amendment_Redline
              <div className="absolute inset-0 border border-[#adc6ff] opacity-0 group-hover:opacity-100 animate-pulse pointer-events-none" />
            </button>
          </div>
        )}
      </aside>
    </div>
  )
}
