import { useState, useEffect } from 'react'
import { api } from '../../api/client'

interface Props {
  onCmdK: () => void
}

export default function TopAppBar({ onCmdK }: Props) {
  const [latency, setLatency] = useState<number | null>(null)

  useEffect(() => {
    const poll = () =>
      api.health().then(h => setLatency(h.ai_latency_ms ?? null)).catch(() => {})
    poll()
    const id = setInterval(poll, 30_000)
    return () => clearInterval(id)
  }, [])

  return (
    <header className="fixed top-0 left-0 right-0 h-14 z-50 bg-zinc-950/80 backdrop-blur-md border-b border-zinc-800/50 shadow-2xl shadow-black/50">
      <div className="flex items-center justify-between h-full pl-20 pr-6">

        {/* Brand */}
        <div className="flex items-center gap-4">
          <span
            className="text-xl font-black tracking-tighter uppercase phosphor-glow"
            style={{ fontFamily: 'Space Grotesk', color: '#adc6ff' }}
          >
            DocMind AI
          </span>
          {latency !== null && (
            <>
              <div className="h-4 w-px bg-zinc-700" />
              <span className="font-mono text-[10px] text-zinc-500 tracking-widest uppercase">
                AI Latency: {latency}ms
              </span>
            </>
          )}
        </div>

        {/* Search — opens CMD+K */}
        <div className="flex-1 max-w-xl mx-8">
          <button
            onClick={onCmdK}
            className="w-full flex items-center gap-3 bg-[#0e0e0e] border-b-2 border-[#8c909f] hover:border-[#adc6ff] py-2 pl-3 pr-4 transition-colors group"
          >
            <span className="material-symbols-outlined text-zinc-500 group-hover:text-[#adc6ff] transition-colors" style={{ fontSize: 18 }}>
              search
            </span>
            <span className="font-mono text-xs text-zinc-700 uppercase tracking-wider">
              CMD+K TO SEARCH DOCUMENT REPOSITORY...
            </span>
            <div className="ml-auto flex items-center gap-1">
              <kbd className="font-mono text-[9px] bg-zinc-800 border border-zinc-700 px-1.5 py-0.5 text-zinc-600">⌘K</kbd>
            </div>
          </button>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-4">
          <button className="text-zinc-400 hover:text-[#adc6ff] transition-colors">
            <span className="material-symbols-outlined" style={{ fontSize: 22 }}>notifications</span>
          </button>
          <button onClick={onCmdK} className="text-zinc-400 hover:text-[#adc6ff] transition-colors">
            <span className="material-symbols-outlined" style={{ fontSize: 22 }}>terminal</span>
          </button>
          <div className="h-8 w-8 bg-zinc-800 border border-zinc-700 flex items-center justify-center">
            <span className="material-symbols-outlined text-zinc-400" style={{ fontSize: 20 }}>account_circle</span>
          </div>
        </div>

      </div>
    </header>
  )
}
