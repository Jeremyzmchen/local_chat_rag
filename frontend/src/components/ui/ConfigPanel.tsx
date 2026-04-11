import { useEffect, useState } from 'react'
import { api }                  from '../../api/client'
import { useToastStore }        from '../../store/toastStore'

interface Props {
  open:    boolean
  onClose: () => void
}

export default function ConfigPanel({ open, onClose }: Props) {
  const toast = useToastStore(s => s.push)

  const [loading,  setLoading]  = useState(false)
  const [saving,   setSaving]   = useState(false)

  // Local form state
  const [backend,       setBackend]       = useState('ollama')
  const [ollamaModel,   setOllamaModel]   = useState('llama3.2')
  const [ollamaUrl,     setOllamaUrl]     = useState('http://localhost:11434')
  const [openaiModel,   setOpenaiModel]   = useState('gpt-4o-mini')
  const [temperature,   setTemperature]   = useState(0.7)
  const [maxTokens,     setMaxTokens]     = useState(1536)
  const [webSearch,     setWebSearch]     = useState(false)
  const [legalReview,   setLegalReview]   = useState(false)

  useEffect(() => {
    if (!open) return
    setLoading(true)
    api.getConfig()
      .then(c => {
        setBackend(c.llm.backend)
        setOllamaModel(c.llm.ollama_model)
        setOllamaUrl(c.llm.ollama_base_url)
        setOpenaiModel(c.llm.openai_model)
        setTemperature(c.llm.temperature)
        setMaxTokens(c.llm.max_tokens)
        setWebSearch(c.web_search_enabled)
        setLegalReview(c.legal_review_enabled)
      })
      .catch(() => toast('Failed to load config', 'error'))
      .finally(() => setLoading(false))
  }, [open])

  const save = async () => {
    setSaving(true)
    try {
      await api.patchConfig({
        backend:           backend as any,
        temperature,
        max_tokens:        maxTokens,
        ollama_model:      ollamaModel,
        ollama_base_url:   ollamaUrl,
        openai_model:      openaiModel,
        enable_web_search: webSearch,
        use_legal_review:  legalReview,
      } as any)
      toast('Configuration saved', 'success')
      onClose()
    } catch (e: any) {
      toast(`Save failed: ${e.message}`, 'error')
    } finally {
      setSaving(false)
    }
  }

  if (!open) return null

  return (
    <div className="fixed inset-0 z-[300] flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div
        className="w-full max-w-lg bg-[#0e0e0e] border border-[#424754]/20 shadow-2xl overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <header className="bg-zinc-950/80 px-6 py-4 flex items-center justify-between border-b border-zinc-800/50">
          <div className="flex items-center gap-3">
            <span className="material-symbols-outlined text-[#adc6ff]" style={{ fontSize: 18 }}>settings</span>
            <h2 className="font-headline font-bold uppercase tracking-widest text-sm text-[#adc6ff]">
              Runtime_Config.sys
            </h2>
          </div>
          <button onClick={onClose} className="text-zinc-500 hover:text-[#adc6ff] transition-colors">
            <span className="material-symbols-outlined">close</span>
          </button>
        </header>

        {loading ? (
          <div className="p-8 flex items-center justify-center font-mono text-[11px] text-zinc-600 uppercase animate-pulse">
            Loading config...
          </div>
        ) : (
          <div className="p-6 space-y-6">

            {/* LLM Backend */}
            <div className="space-y-3">
              <h3 className="font-mono text-[10px] uppercase text-zinc-500 tracking-[0.2em] border-b border-[#424754]/20 pb-2">
                LLM_Backend
              </h3>
              <div className="grid grid-cols-2 gap-3">
                {(['ollama', 'openai'] as const).map(b => (
                  <button
                    key={b}
                    onClick={() => setBackend(b)}
                    className={`p-3 border font-mono text-xs uppercase tracking-wider transition-colors text-left
                      ${backend === b ? 'border-[#adc6ff]/50 bg-[#adc6ff]/5 text-[#adc6ff]' : 'border-[#424754]/20 bg-[#1c1b1b] text-zinc-500 hover:border-[#424754]/50'}`}
                  >
                    <div className="flex items-center justify-between">
                      <span>{b}</span>
                      {backend === b && <span className="material-symbols-outlined" style={{ fontSize: 14 }}>check_circle</span>}
                    </div>
                    <div className="text-[9px] mt-1 text-zinc-600">
                      {b === 'ollama' ? 'Local — free, private' : 'Cloud — requires API key'}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Model settings */}
            <div className="space-y-3">
              <h3 className="font-mono text-[10px] uppercase text-zinc-500 tracking-[0.2em] border-b border-[#424754]/20 pb-2">
                Model_Settings
              </h3>
              {backend === 'ollama' ? (
                <div className="space-y-3">
                  <Field label="OLLAMA_MODEL"   value={ollamaModel} onChange={setOllamaModel} placeholder="llama3.2" />
                  <Field label="OLLAMA_BASE_URL" value={ollamaUrl}   onChange={setOllamaUrl}   placeholder="http://localhost:11434" />
                </div>
              ) : (
                <Field label="OPENAI_MODEL" value={openaiModel} onChange={setOpenaiModel} placeholder="gpt-4o-mini" />
              )}

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="font-mono text-[10px] uppercase text-zinc-500 tracking-wider block mb-1">TEMPERATURE</label>
                  <input
                    type="number" min={0} max={2} step={0.1}
                    value={temperature}
                    onChange={e => setTemperature(parseFloat(e.target.value))}
                    className="w-full bg-[#1c1b1b] border-b-2 border-[#8c909f] focus:border-[#adc6ff] outline-none font-mono text-xs text-[#e5e2e1] px-2 py-1.5 transition-colors"
                  />
                </div>
                <div>
                  <label className="font-mono text-[10px] uppercase text-zinc-500 tracking-wider block mb-1">MAX_TOKENS</label>
                  <input
                    type="number" min={64} max={8192} step={128}
                    value={maxTokens}
                    onChange={e => setMaxTokens(parseInt(e.target.value))}
                    className="w-full bg-[#1c1b1b] border-b-2 border-[#8c909f] focus:border-[#adc6ff] outline-none font-mono text-xs text-[#e5e2e1] px-2 py-1.5 transition-colors"
                  />
                </div>
              </div>
            </div>

            {/* Feature toggles */}
            <div className="space-y-3">
              <h3 className="font-mono text-[10px] uppercase text-zinc-500 tracking-[0.2em] border-b border-[#424754]/20 pb-2">
                Feature_Flags
              </h3>
              <Toggle label="WEB_SEARCH"     desc="Enable SerpAPI web retrieval" value={webSearch}   onChange={setWebSearch} />
              <Toggle label="LEGAL_REVIEW"   desc="Enable GKGR legal review mode (requires restart)" value={legalReview} onChange={setLegalReview} />
            </div>

            {/* Actions */}
            <div className="flex justify-end gap-3 pt-2 border-t border-[#424754]/10">
              <button
                onClick={onClose}
                className="px-5 py-2 bg-[#2a2a2a] hover:bg-[#353535] text-[#c2c6d6] font-mono text-[11px] uppercase tracking-widest transition-all"
              >
                Cancel
              </button>
              <button
                onClick={save}
                disabled={saving}
                className="px-6 py-2 bg-gradient-to-br from-[#adc6ff] to-[#4d8eff] text-[#002e6a] font-mono text-[11px] font-bold uppercase tracking-widest hover:brightness-110 transition-all disabled:opacity-50 shadow-[0_0_10px_rgba(173,198,255,0.2)]"
              >
                {saving ? 'Saving...' : 'Apply_Config'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Field({ label, value, onChange, placeholder }: {
  label: string; value: string; onChange: (v: string) => void; placeholder?: string
}) {
  return (
    <div>
      <label className="font-mono text-[10px] uppercase text-zinc-500 tracking-wider block mb-1">{label}</label>
      <input
        type="text"
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full bg-[#1c1b1b] border-b-2 border-[#8c909f] focus:border-[#adc6ff] outline-none font-mono text-xs text-[#e5e2e1] px-2 py-1.5 transition-colors placeholder:text-zinc-700"
      />
    </div>
  )
}

function Toggle({ label, desc, value, onChange }: {
  label: string; desc: string; value: boolean; onChange: (v: boolean) => void
}) {
  return (
    <div
      className="flex items-center justify-between p-3 bg-[#1c1b1b] border border-[#424754]/20 cursor-pointer hover:border-[#424754]/40 transition-colors"
      onClick={() => onChange(!value)}
    >
      <div>
        <div className="font-mono text-[11px] uppercase text-[#c2c6d6] tracking-wider">{label}</div>
        <div className="font-mono text-[9px] text-zinc-600 mt-0.5">{desc}</div>
      </div>
      <div className={`w-10 h-5 relative transition-colors ${value ? 'bg-[#4d8eff]/60' : 'bg-zinc-800'}`}>
        <div className={`absolute top-0.5 w-4 h-4 bg-white transition-all ${value ? 'left-5' : 'left-0.5'}`} />
      </div>
    </div>
  )
}
