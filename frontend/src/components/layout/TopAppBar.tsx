import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../../api/client'

export default function TopAppBar() {
  const navigate = useNavigate()
  const [latency, setLatency] = useState<number | null>(null)
  const [search, setSearch]   = useState('')

  // Poll AI latency from /api/health every 30s
  useEffect(() => {
    const fetch = () =>
      api.health().then(h => setLatency(h.ai_latency_ms ?? null)).catch(() => {})
    fetch()
    const id = setInterval(fetch, 30_000)
    return () => clearInterval(id)
  }, [])

  const handleSearch = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && search.trim()) {
      navigate(`/?q=${encodeURIComponent(search.trim())}`)
      setSearch('')
    }
  }

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

        {/* Search */}
        <div className="flex-1 max-w-xl mx-8">
          <div className="relative group">
            <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500 group-focus-within:text-[#adc6ff] transition-colors"
              style={{ fontSize: 18 }}>
              search
            </span>
            <input
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              onKeyDown={handleSearch}
              placeholder="CMD+K TO SEARCH DOCUMENT REPOSITORY..."
              className="w-full bg-[#0e0e0e] border-0 border-b-2 border-[#8c909f] focus:border-[#adc6ff] outline-none text-xs font-mono text-on-surface py-2 pl-10 pr-4 placeholder:text-zinc-700 transition-colors"
            />
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-4">
          <button className="text-zinc-400 hover:text-[#adc6ff] transition-colors">
            <span className="material-symbols-outlined" style={{ fontSize: 22 }}>notifications</span>
          </button>
          <button className="text-zinc-400 hover:text-[#adc6ff] transition-colors">
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
