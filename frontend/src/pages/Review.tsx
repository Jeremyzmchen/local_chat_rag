import { useLocation, useNavigate, useParams } from 'react-router-dom'
import type { ReviewFinding } from '../api/client'
import { useLogStore }   from '../store/logStore'
import { useState }      from 'react'

// ── Split text into diff lines using error_regions ──────────────────────────
function buildDiff(original: string, finding: ReviewFinding) {
  const lines = original.split('\n').filter(Boolean)
  const errorSet = new Set(finding.error_regions.map(r => r.trim()))

  return lines.map(line => {
    const trimmed = line.trim()
    const isRemoved = [...errorSet].some(e => trimmed.includes(e.slice(0, 20)))
    return { text: line, removed: isRemoved }
  })
}

function buildRevisionLines(finding: ReviewFinding) {
  return finding.revision.split('\n').filter(Boolean)
}

// ── Line component ────────────────────────────────────────────────────────────
function DiffLine({
  lineNo, text, type,
}: { lineNo: number | string; text: string; type: 'removed' | 'added' | 'context' }) {
  const bg =
    type === 'removed' ? 'bg-[#93000a]/10' :
    type === 'added'   ? 'bg-[#4d8eff]/10' : ''
  const textColor =
    type === 'removed' ? 'text-[#ffb4ab]/80' :
    type === 'added'   ? 'text-[#adc6ff]/80' : 'text-[#c2c6d6]'
  const lineColor =
    type === 'removed' ? 'text-[#ffb4ab]/40' :
    type === 'added'   ? 'text-[#adc6ff]/40' : 'text-zinc-600'

  return (
    <div className={`flex hover:bg-white/5 transition-colors ${bg}`}>
      <span className={`shrink-0 w-10 text-right pr-3 font-mono text-[11px] select-none ${lineColor}`}>
        {lineNo}
      </span>
      <pre className={`pl-2 font-mono text-[12px] leading-6 whitespace-pre-wrap break-all ${textColor}`}>
        {text}
      </pre>
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────
export default function Review() {
  const { id }    = useParams<{ id: string }>()
  const location  = useLocation()
  const navigate  = useNavigate()
  const addLog    = useLogStore(s => s.addLog)

  // Accept either single finding or array passed via navigation state
  const state     = location.state as { finding?: ReviewFinding; findings?: ReviewFinding[] } | null
  const findings: ReviewFinding[] = state?.findings ?? (state?.finding ? [state.finding] : [])
  const [activeIdx, setActiveIdx] = useState(0)

  const finding   = findings[activeIdx]

  const originalLines = finding ? buildDiff(finding.analysis, finding)      : []
  const revisionLines = finding ? buildRevisionLines(finding)                : []

  const handleAccept = () => {
    addLog(`[✓] Revision accepted for: ${finding?.dimension ?? id}`)
    navigate(`/report/${id}`)
  }

  if (!finding) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-3">
          <span className="material-symbols-outlined text-zinc-700" style={{ fontSize: 48 }}>difference</span>
          <p className="font-mono text-[11px] text-zinc-600 uppercase tracking-widest">
            No revision data — go back to report and run a scan first
          </p>
          <button onClick={() => navigate(`/report/${id}`)} className="text-[#adc6ff] font-mono text-[10px] uppercase hover:underline">
            ← Back to Report
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full overflow-hidden">

      {/* ── Header strip ── */}
      <div className="absolute top-14 left-16 right-0 z-10 bg-[#1c1b1b] border-b border-[#424754]/10 px-6 py-3">
        <div className="flex items-center justify-between">
          <div>
            <nav className="flex items-center gap-1 text-[10px] font-mono text-zinc-500 uppercase mb-1">
              <button onClick={() => navigate('/')} className="hover:text-[#adc6ff]">Documents</button>
              <span>/</span>
              <button onClick={() => navigate(`/report/${id}`)} className="hover:text-[#adc6ff]">{id}</button>
              <span>/</span>
              <span className="text-[#d8e2ff]">Diff_View</span>
            </nav>
            <h1 className="font-headline text-lg font-black text-[#e5e2e1] tracking-tighter uppercase">
              {finding.dimension}: Revision Suggestion
            </h1>
            <p className="font-mono text-[10px] text-zinc-400 mt-0.5">
              QUERY: "{finding.query.slice(0, 80)}{finding.query.length > 80 ? '...' : ''}"
            </p>
          </div>
          <div className="flex gap-2">
            {/* Finding selector (if multiple) */}
            {findings.length > 1 && findings.map((_f, i) => (
              <button
                key={i}
                onClick={() => setActiveIdx(i)}
                className={`font-mono text-[9px] uppercase px-2 py-1 border transition-colors ${i === activeIdx ? 'border-[#adc6ff] text-[#adc6ff]' : 'border-zinc-700 text-zinc-500 hover:border-zinc-500'}`}
              >
                #{i + 1}
              </button>
            ))}
            <button
              className="bg-[#353535] px-3 py-1.5 text-[10px] font-mono font-bold text-[#c2c6d6] flex items-center gap-1.5 hover:bg-zinc-700 transition-colors"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 12 }}>download</span>
              EXPORT RAW
            </button>
            <button
              onClick={handleAccept}
              className="bg-gradient-to-br from-[#adc6ff] to-[#4d8eff] px-3 py-1.5 text-[10px] font-mono font-bold text-[#002e6a] flex items-center gap-1.5 hover:brightness-110 transition-all shadow-[0_0_10px_rgba(77,142,255,0.3)]"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 12 }}>check_circle</span>
              ACCEPT REVISION
            </button>
          </div>
        </div>
      </div>

      {/* Spacer for fixed header */}
      <div className="w-full" style={{ paddingTop: '100px' }}>
        <div className="flex h-[calc(100vh-214px)] overflow-hidden">

          {/* ── Side metadata rail ── */}
          <div className="w-10 border-r border-[#424754]/10 bg-[#20201f] flex flex-col items-center py-4 gap-4 shrink-0">
            <div className="relative group cursor-help">
              <div className={`w-7 h-7 rounded-full border flex items-center justify-center font-mono text-[9px] font-bold
                ${finding.priority >= 3 ? 'border-[#ffb786]/40 text-[#ffb786]' : 'border-[#adc6ff]/40 text-[#adc6ff]'}`}>
                {Math.round(finding.confidence * 100)}
              </div>
            </div>
            <div className="h-px w-3 bg-[#424754]/30" />
            {['history', 'comment', 'bookmark'].map(icon => (
              <button key={icon} className="text-zinc-600 hover:text-[#adc6ff] transition-colors">
                <span className="material-symbols-outlined" style={{ fontSize: 16 }}>{icon}</span>
              </button>
            ))}
          </div>

          {/* ── Diff editor ── */}
          <div className="flex-1 bg-[#0e0e0e] flex flex-col overflow-hidden">
            {/* Diff header bar */}
            <div className="flex border-b border-[#424754]/10 bg-[#1c1b1b] font-mono text-[10px] tracking-widest text-zinc-500 shrink-0">
              <div className="flex-1 px-4 py-2 border-r border-[#424754]/10 flex items-center justify-between">
                <span>ORIGINAL_SOURCE</span>
                <span className="text-[#ffb4ab]/70">
                  -{originalLines.filter(l => l.removed).length} DELETIONS
                </span>
              </div>
              <div className="flex-1 px-4 py-2 flex items-center justify-between bg-[#adc6ff]/5">
                <span className="text-[#adc6ff]">AI_REVISION_v1.0</span>
                <span className="text-[#adc6ff]/70">
                  +{revisionLines.length} ADDITIONS
                </span>
              </div>
            </div>

            {/* Side-by-side diff */}
            <div className="flex flex-1 overflow-y-auto">
              {/* Left: original */}
              <div className="flex-1 border-r border-[#424754]/10">
                {originalLines.map((line, i) => (
                  <DiffLine
                    key={i}
                    lineNo={i + 1}
                    text={line.text}
                    type={line.removed ? 'removed' : 'context'}
                  />
                ))}
              </div>
              {/* Right: revision */}
              <div className="flex-1">
                {revisionLines.map((line, i) => (
                  <DiffLine key={i} lineNo={i + 1} text={line} type="added" />
                ))}
              </div>
            </div>

            {/* Footer status bar */}
            <div className="h-7 bg-zinc-900 border-t border-[#424754]/10 flex items-center px-4 justify-between font-mono text-[9px] text-zinc-500 uppercase tracking-widest shrink-0">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-[#adc6ff] animate-pulse" />
                  AI_ENGINE_READY
                </span>
                <span>UTF-8</span>
              </div>
              <div className="flex items-center gap-4">
                <span>DIFF_MODE: SIDE_BY_SIDE</span>
                <span className="text-[#adc6ff]">CONFIDENCE: {(finding.confidence * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>

          {/* ── Right: AI Analysis Log ── */}
          <div className="w-72 bg-[#20201f] border-l border-[#424754]/10 flex flex-col p-4 shrink-0 overflow-y-auto">
            <h3 className="font-headline text-xs font-bold text-zinc-400 mb-4 tracking-wider uppercase">
              AI Analysis Log
            </h3>

            <div className="space-y-4">
              <div className="p-3 bg-[#1c1b1b] border border-[#424754]/20">
                <div className="flex items-center gap-2 mb-2">
                  <span className="material-symbols-outlined text-[#adc6ff]" style={{ fontSize: 14 }}>auto_awesome</span>
                  <span className="font-mono text-[10px] font-bold text-[#e5e2e1]">REASONING_ENGINE</span>
                </div>
                <p className="font-mono text-[11px] text-zinc-400 leading-relaxed">
                  {finding.analysis.slice(0, 200)}{finding.analysis.length > 200 ? '...' : ''}
                </p>
              </div>

              {finding.error_regions.length > 0 && (
                <div className="p-3 bg-[#1c1b1b] border border-[#424754]/20">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="material-symbols-outlined text-[#ffb786]" style={{ fontSize: 14 }}>warning</span>
                    <span className="font-mono text-[10px] font-bold text-[#ffb786]">LEGAL_RISK_FOUND</span>
                  </div>
                  <ul className="space-y-1">
                    {finding.error_regions.slice(0, 3).map((r, i) => (
                      <li key={i} className="font-mono text-[11px] text-zinc-400 leading-relaxed">
                        "{r.slice(0, 80)}{r.length > 80 ? '...' : ''}"
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {finding.references.length > 0 && (
                <div>
                  <h3 className="font-headline text-xs font-bold text-zinc-400 mb-2 tracking-wider uppercase">
                    References
                  </h3>
                  <div className="flex flex-wrap gap-1.5">
                    {finding.references.slice(0, 4).map((_r, i) => (
                      <span key={i} className="bg-[#304671] px-2 py-0.5 text-[9px] font-mono text-[#9fb5e7]">
                        REF_{i + 1}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Confidence score */}
            <div className="mt-auto pt-4 border-t border-zinc-800">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-[#adc6ff]/20 flex items-center justify-center">
                  <span className="material-symbols-outlined text-[#adc6ff]" style={{ fontSize: 18 }}>psychology</span>
                </div>
                <div>
                  <div className="font-headline text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Confidence</div>
                  <div className="font-mono text-xl font-black text-[#e5e2e1]">
                    {(finding.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
              <div className="mt-2 w-full bg-zinc-800 h-1 overflow-hidden">
                <div
                  className="bg-[#adc6ff] h-full shadow-[0_0_8px_#adc6ff]"
                  style={{ width: `${finding.confidence * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
