import { useCallback, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api }         from '../api/client'
import { useLogStore } from '../store/logStore'

type Protocol = 'compliance' | 'risk' | 'due_diligence'

const PROTOCOLS: { value: Protocol; label: string; color: string }[] = [
  { value: 'compliance',    label: 'Compliance',    color: '#adc6ff' },
  { value: 'risk',          label: 'Risk',          color: '#ffb786' },
  { value: 'due_diligence', label: 'Due Diligence', color: '#b1c6f9' },
]

const ALLOWED = new Set(['.pdf', '.txt', '.md', '.docx', '.xlsx', '.xls', '.pptx'])

function ext(name: string) {
  return name.slice(name.lastIndexOf('.')).toLowerCase()
}

export default function Upload() {
  const navigate = useNavigate()
  const addLog   = useLogStore(s => s.addLog)

  const [files,    setFiles]    = useState<File[]>([])
  const [protocol, setProtocol] = useState<Protocol>('compliance')
  const [dragging, setDragging] = useState(false)
  const [status,   setStatus]   = useState<'idle' | 'uploading' | 'done' | 'error'>('idle')
  const [progress, setProgress] = useState<string[]>([])
  const inputRef = useRef<HTMLInputElement>(null)

  // ── File helpers ─────────────────────────────────────────────────────────
  const addFiles = (incoming: File[]) => {
    const valid = incoming.filter(f => ALLOWED.has(ext(f.name)))
    const invalid = incoming.filter(f => !ALLOWED.has(ext(f.name)))
    invalid.forEach(f => addLog(`Rejected: ${f.name} (unsupported type)`, 'warn'))
    setFiles(prev => {
      const names = new Set(prev.map(f => f.name))
      return [...prev, ...valid.filter(f => !names.has(f.name))]
    })
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    addFiles(Array.from(e.dataTransfer.files))
  }, [])

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) addFiles(Array.from(e.target.files))
  }

  const removeFile = (name: string) =>
    setFiles(prev => prev.filter(f => f.name !== name))

  // ── Upload ────────────────────────────────────────────────────────────────
  const execute = async () => {
    if (!files.length) return
    setStatus('uploading')
    setProgress([])

    const push = (msg: string) => {
      setProgress(p => [...p, msg])
      addLog(msg)
    }

    push(`[→] Ingesting ${files.length} file(s) with protocol: ${protocol}`)

    try {
      push('[→] Uploading files to server...')
      const res = await api.uploadDocuments(files, protocol)
      push(`[✓] Chunking complete — ${res.total_chunks} segments`)
      push(`[✓] Indexed ${res.added_files} file(s), skipped ${res.skipped_files}`)

      if (res.errors.length) {
        res.errors.forEach(([f, msg]) => push(`[!] ${f}: ${msg}`))
      }

      push('[✓] Knowledge base updated. Redirecting to dashboard...')
      setStatus('done')
      setTimeout(() => navigate('/'), 1500)
    } catch (e: any) {
      push(`[✗] Upload failed: ${e.message}`)
      setStatus('error')
      addLog(`Upload error: ${e.message}`, 'error')
    }
  }

  const isUploading = status === 'uploading'

  return (
    <div className="flex items-center justify-center min-h-full bg-[#131313]/60 p-8">
      <div className="w-full max-w-2xl bg-[#0e0e0e] border border-[#424754]/20 shadow-2xl flex flex-col overflow-hidden">

        {/* Modal header */}
        <header className="bg-zinc-950/80 backdrop-blur-md px-6 py-4 flex justify-between items-center border-b border-zinc-800/50">
          <div className="flex items-center gap-3">
            <span className="material-symbols-outlined text-[#adc6ff]" style={{ fontVariationSettings: "'FILL' 1" }}>
              upload_file
            </span>
            <h2 className="font-headline font-bold uppercase tracking-widest text-sm text-[#adc6ff]">
              Ingest_Document.sys
            </h2>
          </div>
          <button onClick={() => navigate('/')} className="text-[#c2c6d6] hover:text-[#adc6ff] transition-colors">
            <span className="material-symbols-outlined">close</span>
          </button>
        </header>

        <div className="p-8 space-y-7">

          {/* Drop zone */}
          <div className="relative group">
            {dragging && (
              <div className="absolute -inset-0.5 bg-[#adc6ff]/20 blur opacity-100 transition duration-300 pointer-events-none" />
            )}
            <div
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              className={`relative flex flex-col items-center justify-center border-2 border-dashed py-12 px-6 cursor-pointer transition-all
                ${dragging ? 'border-[#adc6ff]/70 bg-[#adc6ff]/5' : 'border-[#424754]/40 bg-[#1c1b1b] hover:bg-[#20201f]'}`}
            >
              <span
                className={`material-symbols-outlined mb-3 transition-colors ${dragging ? 'text-[#adc6ff]' : 'text-[#8c909f]'}`}
                style={{ fontSize: 48, fontVariationSettings: "'wght' 200" }}
              >
                cloud_upload
              </span>
              <p className="font-mono text-xs text-[#c2c6d6] uppercase tracking-tighter text-center">
                Drop legal artifacts here or{' '}
                <span className="text-[#adc6ff] font-bold">browse_file</span>
              </p>
              <p className="font-mono text-[10px] text-zinc-600 mt-1.5 uppercase">
                PDF, DOCX, TXT, MD, XLSX, PPTX (MAX 50MB)
              </p>
              <input
                ref={inputRef}
                type="file"
                multiple
                accept=".pdf,.txt,.md,.docx,.xlsx,.xls,.pptx"
                className="hidden"
                onChange={onInputChange}
              />
            </div>
          </div>

          {/* File list */}
          {files.length > 0 && (
            <div className="space-y-1.5">
              {files.map(f => (
                <div key={f.name} className="flex items-center justify-between bg-[#1c1b1b] px-3 py-2 border border-[#424754]/20">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="material-symbols-outlined text-zinc-500 shrink-0" style={{ fontSize: 16 }}>description</span>
                    <span className="font-mono text-[11px] text-[#e5e2e1] truncate">{f.name}</span>
                    <span className="font-mono text-[10px] text-zinc-600 shrink-0">
                      {(f.size / 1024).toFixed(0)}KB
                    </span>
                  </div>
                  <button onClick={() => removeFile(f.name)} className="text-zinc-600 hover:text-[#ffb4ab] transition-colors ml-3 shrink-0">
                    <span className="material-symbols-outlined" style={{ fontSize: 14 }}>close</span>
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Protocol selector */}
          <div className="space-y-3">
            <h3 className="font-mono text-[10px] uppercase text-zinc-500 tracking-[0.2em] border-b border-[#424754]/20 pb-2">
              Select_Review_Protocol
            </h3>
            <div className="grid grid-cols-3 gap-3">
              {PROTOCOLS.map(p => (
                <label
                  key={p.value}
                  className={`flex items-center p-3 cursor-pointer bg-[#20201f] border transition-all
                    ${protocol === p.value ? `border-[${p.color}]/40` : 'border-transparent hover:border-[#424754]/40'}`}
                >
                  <input
                    type="radio"
                    name="protocol"
                    value={p.value}
                    checked={protocol === p.value}
                    onChange={() => setProtocol(p.value)}
                    className="sr-only"
                  />
                  <div className="w-full">
                    <div className="flex justify-between items-center mb-1">
                      <span
                        className="font-mono text-[11px] uppercase tracking-tighter"
                        style={{ color: protocol === p.value ? p.color : '#c2c6d6' }}
                      >
                        {p.label}
                      </span>
                      {protocol === p.value && (
                        <span className="material-symbols-outlined" style={{ fontSize: 14, color: p.color }}>check_circle</span>
                      )}
                    </div>
                    <div className="h-0.5 w-full bg-[#424754]/20 overflow-hidden">
                      <div
                        className="h-full transition-transform duration-300"
                        style={{
                          backgroundColor: p.color,
                          opacity: 0.5,
                          transform: protocol === p.value ? 'translateX(0)' : 'translateX(-100%)',
                        }}
                      />
                    </div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Terminal progress log */}
          <div className="bg-[#0e0e0e] border border-[#424754]/30 overflow-hidden">
            <div className="bg-[#393939]/10 px-4 py-2 flex items-center justify-between border-b border-[#424754]/20">
              <div className="flex gap-1.5">
                <div className="w-2 h-2 rounded-full bg-[#ffb4ab]/40" />
                <div className="w-2 h-2 rounded-full bg-[#ffb786]/40" />
                <div className="w-2 h-2 rounded-full bg-[#adc6ff]/40" />
              </div>
              <span className="font-mono text-[9px] text-zinc-500 uppercase tracking-widest">
                Execution_Thread
              </span>
            </div>
            <div className="p-4 font-mono text-[11px] h-28 overflow-y-auto space-y-1.5">
              {progress.length === 0 ? (
                <span className="text-zinc-700 uppercase tracking-widest">Awaiting execution...</span>
              ) : (
                progress.map((line, i) => (
                  <div key={i} className={`${line.startsWith('[✓]') ? 'text-[#adc6ff]/80' : line.startsWith('[✗]') || line.startsWith('[!]') ? 'text-[#ffb4ab]/80' : 'text-[#c2c6d6]/60'}`}>
                    {line}
                  </div>
                ))
              )}
              {isUploading && (
                <div className="text-zinc-600 animate-pulse uppercase">Processing...</div>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between pt-2 border-t border-[#424754]/10">
            <div className="flex items-center gap-2">
              <div className={`w-1.5 h-1.5 rounded-full ${isUploading ? 'bg-[#adc6ff] animate-pulse shadow-[0_0_8px_#adc6ff]' : 'bg-zinc-700'}`} />
              <span className="font-mono text-[10px] text-[#c2c6d6] uppercase tracking-tighter">
                AI Core: {isUploading ? 'PROCESSING' : 'ONLINE'}
              </span>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => navigate('/')}
                disabled={isUploading}
                className="px-5 py-2 bg-[#2a2a2a] hover:bg-[#353535] text-[#c2c6d6] font-mono text-[11px] uppercase tracking-widest transition-all disabled:opacity-40"
              >
                Cancel
              </button>
              <button
                onClick={execute}
                disabled={!files.length || isUploading}
                className="px-7 py-2 bg-gradient-to-br from-[#adc6ff] to-[#4d8eff] text-[#002e6a] font-mono text-[11px] font-bold uppercase tracking-widest hover:brightness-110 transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-[0_0_15px_rgba(173,198,255,0.2)]"
              >
                {isUploading ? 'Processing...' : 'Execute_Analysis'}
              </button>
            </div>
          </div>
        </div>

        {/* Footer meta */}
        <footer className="bg-[#353535] px-6 py-1.5 border-t border-[#424754]/30">
          <p className="font-mono text-[8px] text-zinc-500 uppercase flex justify-between">
            <span>kernel: docmind_v1.0</span>
            <span>entropy: optimized</span>
            <span>protocol: {protocol}</span>
          </p>
        </footer>
      </div>
    </div>
  )
}
