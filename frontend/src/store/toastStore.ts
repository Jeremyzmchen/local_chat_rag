import { create } from 'zustand'

export interface Toast {
  id:      string
  message: string
  level:   'info' | 'warn' | 'error' | 'success'
}

interface ToastStore {
  toasts: Toast[]
  push:   (message: string, level?: Toast['level']) => void
  remove: (id: string) => void
}

let seq = 0

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],
  push: (message, level = 'info') => {
    const id = `t${++seq}`
    set(s => ({ toasts: [...s.toasts, { id, message, level }] }))
    // Auto-dismiss after 4s
    setTimeout(() => set(s => ({ toasts: s.toasts.filter(t => t.id !== id) })), 4000)
  },
  remove: (id) => set(s => ({ toasts: s.toasts.filter(t => t.id !== id) })),
}))
