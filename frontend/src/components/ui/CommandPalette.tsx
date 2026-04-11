import { useEffect, useRef, useState } from 'react'
import { useNavigate }                  from 'react-router-dom'
import { api }                          from '../../api/client'

interface Command {
  id:       string
  label:    string
  category: string
  icon:     string
  action:   () => void
}

interface Props {
  open:    boolean
  onClose: () => void
}

export default function CommandPalette({ open, onClose }: Props) {
  const navigate    = useNavigate()
  const [query, setQuery]     = useState('')
  const [docs,  setDocs]      = useState<{ doc_id: string; filename: string }[]>([])
  const [activeIdx, setActive] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  // Load documents for search
  useEffect(() => {
    if (!open) return
    api.listDocuments()
      .then(r => setDocs(r.documents.map(d => ({ doc_id: d.doc_id, filename: d.filename }))))
      .catch(() => {})
  }, [open])

  // Focus input when opened
  useEffect(() => {
    if (open) {
      setQuery('')
      setActive(0)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [open])

  const staticCommands: Command[] = [
    { id: 'dashboard', label: 'Go to Dashboard',      category: 'Navigation', icon: 'dashboard',   action: () => navigate('/') },
    { id: 'upload',    label: 'Upload Document',       category: 'Navigation', icon: 'upload_file', action: () => navigate('/upload') },
    { id: 'documents', label: 'View All Documents',    category: 'Navigation', icon: 'description', action: () => navigate('/documents') },
  ]

  const docCommands: Command[] = docs.map(d => ({
    id:       d.doc_id,
    label:    d.filename,
    category: 'Documents',
    icon:     'description',
    action:   () => navigate(`/report/${d.doc_id}`),
  }))

  const allCommands = [...staticCommands, ...docCommands]

  const filtered = query.trim()
    ? allCommands.filter(c =>
        c.label.toLowerCase().includes(query.toLowerCase()) ||
        c.category.toLowerCase().includes(query.toLowerCase())
      )
    : allCommands

  const run = (cmd: Command) => {
    cmd.action()
    onClose()
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape')     { onClose(); return }
    if (e.key === 'ArrowDown')  { e.preventDefault(); setActive(i => Math.min(i + 1, filtered.length - 1)) }
    if (e.key === 'ArrowUp')    { e.preventDefault(); setActive(i => Math.max(i - 1, 0)) }
    if (e.key === 'Enter' && filtered[activeIdx]) { run(filtered[activeIdx]) }
  }

  if (!open) return null

  // Group by category
  const groups: Record<string, Command[]> = {}
  filtered.forEach(c => {
    groups[c.category] = groups[c.category] ?? []
    groups[c.category].push(c)
  })

  let flatIdx = -1

  return (
    <div className="fixed inset-0 z-[300] flex items-start justify-center pt-24 bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div
        className="w-full max-w-xl bg-[#20201f] border border-[#424754]/30 shadow-2xl shadow-black/60 overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-[#424754]/20">
          <span className="material-symbols-outlined text-zinc-500" style={{ fontSize: 18 }}>search</span>
          <input
            ref={inputRef}
            value={query}
            onChange={e => { setQuery(e.target.value); setActive(0) }}
            onKeyDown={onKeyDown}
            placeholder="TYPE A COMMAND OR SEARCH DOCUMENTS..."
            className="flex-1 bg-transparent font-mono text-xs text-[#e5e2e1] placeholder:text-zinc-700 outline-none uppercase tracking-wider"
          />
          <kbd className="font-mono text-[9px] bg-zinc-800 border border-zinc-700 px-1.5 py-0.5 text-zinc-500">ESC</kbd>
        </div>

        {/* Results */}
        <div className="max-h-80 overflow-y-auto">
          {filtered.length === 0 ? (
            <div className="px-4 py-8 text-center font-mono text-[11px] text-zinc-700 uppercase tracking-widest">
              No results found
            </div>
          ) : (
            Object.entries(groups).map(([category, cmds]) => (
              <div key={category}>
                <div className="px-4 py-1.5 font-mono text-[9px] uppercase tracking-widest text-zinc-600 bg-[#1c1b1b]">
                  {category}
                </div>
                {cmds.map(cmd => {
                  flatIdx++
                  const idx = flatIdx
                  const isActive = idx === activeIdx
                  return (
                    <button
                      key={cmd.id}
                      onClick={() => run(cmd)}
                      onMouseEnter={() => setActive(idx)}
                      className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${isActive ? 'bg-[#adc6ff]/10' : 'hover:bg-[#2a2a2a]'}`}
                    >
                      <span
                        className={`material-symbols-outlined shrink-0 transition-colors ${isActive ? 'text-[#adc6ff]' : 'text-zinc-600'}`}
                        style={{ fontSize: 16 }}
                      >
                        {cmd.icon}
                      </span>
                      <span className={`font-mono text-xs transition-colors ${isActive ? 'text-[#e5e2e1]' : 'text-[#c2c6d6]'}`}>
                        {cmd.label}
                      </span>
                      {isActive && (
                        <span className="ml-auto font-mono text-[9px] text-zinc-600 flex items-center gap-1">
                          <kbd className="bg-zinc-800 border border-zinc-700 px-1">↵</kbd>
                        </span>
                      )}
                    </button>
                  )
                })}
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2 border-t border-[#424754]/20 flex items-center gap-4 font-mono text-[9px] text-zinc-600">
          <span className="flex items-center gap-1">
            <kbd className="bg-zinc-800 border border-zinc-700 px-1">↑↓</kbd> navigate
          </span>
          <span className="flex items-center gap-1">
            <kbd className="bg-zinc-800 border border-zinc-700 px-1">↵</kbd> select
          </span>
          <span className="flex items-center gap-1">
            <kbd className="bg-zinc-800 border border-zinc-700 px-1">esc</kbd> close
          </span>
        </div>
      </div>
    </div>
  )
}
