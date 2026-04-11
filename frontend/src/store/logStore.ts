import { create } from 'zustand'

export interface LogEntry {
  time:    string
  level:   'info' | 'warn' | 'error'
  message: string
}

interface LogStore {
  logs: LogEntry[]
  addLog: (message: string, level?: LogEntry['level']) => void
  clear:  () => void
}

function now() {
  return new Date().toLocaleTimeString('en-GB', { hour12: false })
}

export const useLogStore = create<LogStore>((set) => ({
  logs: [],
  addLog: (message, level = 'info') =>
    set(s => ({
      logs: [...s.logs.slice(-99), { time: now(), level, message }],
    })),
  clear: () => set({ logs: [] }),
}))
