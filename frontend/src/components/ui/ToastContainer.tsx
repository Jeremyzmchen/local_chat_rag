import { useToastStore } from '../../store/toastStore'
import type { Toast }    from '../../store/toastStore'

function ToastItem({ toast }: { toast: Toast }) {
  const remove = useToastStore(s => s.remove)

  const styles: Record<Toast['level'], { border: string; icon: string; color: string }> = {
    info:    { border: 'border-l-[#adc6ff]',  icon: 'info',          color: 'text-[#adc6ff]' },
    success: { border: 'border-l-[#adc6ff]',  icon: 'check_circle',  color: 'text-[#adc6ff]' },
    warn:    { border: 'border-l-[#ffb786]',  icon: 'warning',       color: 'text-[#ffb786]' },
    error:   { border: 'border-l-[#ffb4ab]',  icon: 'error',         color: 'text-[#ffb4ab]' },
  }
  const s = styles[toast.level]

  return (
    <div className={`flex items-start gap-3 bg-[#20201f] border border-[#424754]/30 border-l-2 ${s.border} px-4 py-3 shadow-2xl shadow-black/50 min-w-[280px] max-w-sm`}>
      <span className={`material-symbols-outlined shrink-0 ${s.color}`} style={{ fontSize: 16 }}>
        {s.icon}
      </span>
      <p className="font-mono text-[11px] text-[#c2c6d6] leading-relaxed flex-1">{toast.message}</p>
      <button onClick={() => remove(toast.id)} className="text-zinc-600 hover:text-zinc-400 transition-colors shrink-0">
        <span className="material-symbols-outlined" style={{ fontSize: 14 }}>close</span>
      </button>
    </div>
  )
}

export default function ToastContainer() {
  const toasts = useToastStore(s => s.toasts)
  if (!toasts.length) return null

  return (
    <div className="fixed bottom-20 right-6 z-[200] flex flex-col gap-2 items-end">
      {toasts.map(t => <ToastItem key={t.id} toast={t} />)}
    </div>
  )
}
