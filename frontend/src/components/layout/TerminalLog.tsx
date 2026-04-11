import { useEffect, useRef } from 'react'
import { useLogStore } from '../../store/logStore'

export default function TerminalLog() {
  const logs    = useLogStore(s => s.logs)
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new log
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  return (
    <div className="h-24 bg-[#0e0e0e] border-t border-[#393939] flex flex-col font-mono text-[10px] overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 px-3 py-1 border-b border-zinc-800/50">
        <div className="flex gap-1.5">
          <div className="w-2 h-2 rounded-full bg-[#ffb4ab]/40" />
          <div className="w-2 h-2 rounded-full bg-[#ffb786]/40" />
          <div className="w-2 h-2 rounded-full bg-[#adc6ff]/40" />
        </div>
        <span className="text-zinc-600 uppercase tracking-widest">Session_Terminal</span>
        <div className="ml-auto flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-[#adc6ff] animate-pulse shadow-[0_0_6px_#adc6ff]" />
          <span className="text-zinc-600">LIVE</span>
        </div>
      </div>

      {/* Log lines */}
      <div className="flex-1 overflow-y-auto px-3 py-1 space-y-0.5">
        {logs.length === 0 && (
          <div className="flex gap-2 text-zinc-700">
            <span>[SESSION_INIT]</span>
            <span>...</span>
            <span>[ENCRYPTED_HANDSHAKE_SUCCESS]</span>
          </div>
        )}
        {logs.map((log, i) => (
          <div key={i} className="flex gap-2">
            <span className="text-zinc-600 shrink-0">{log.time}</span>
            <span className={logColor(log.level)}>{log.message}</span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

function logColor(level: 'info' | 'warn' | 'error') {
  if (level === 'warn')  return 'text-[#ffb786]/70'
  if (level === 'error') return 'text-[#ffb4ab]/70'
  return 'text-[#adc6ff]/60'
}
