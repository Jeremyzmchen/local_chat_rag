import { useEffect, useState } from 'react'
import { Outlet }               from 'react-router-dom'
import SideNavBar               from './SideNavBar'
import TopAppBar                from './TopAppBar'
import TerminalLog              from './TerminalLog'
import ToastContainer           from '../ui/ToastContainer'
import CommandPalette           from '../ui/CommandPalette'
import ConfigPanel              from '../ui/ConfigPanel'

export default function AppLayout() {
  const [cmdOpen,    setCmdOpen]    = useState(false)
  const [configOpen, setConfigOpen] = useState(false)

  // Global CMD+K shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setCmdOpen(prev => !prev)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  return (
    <div className="flex h-screen overflow-hidden bg-[#131313]">
      <SideNavBar onSettingsClick={() => setConfigOpen(true)} />

      <div className="flex flex-col flex-1 pl-16">
        <TopAppBar onCmdK={() => setCmdOpen(true)} />

        {/* Page content */}
        <main className="flex-1 pt-14 overflow-hidden flex flex-col">
          <div className="flex-1 overflow-auto">
            <Outlet />
          </div>
          <TerminalLog />
        </main>
      </div>

      {/* Global overlays */}
      <ToastContainer />
      <CommandPalette open={cmdOpen}    onClose={() => setCmdOpen(false)} />
      <ConfigPanel    open={configOpen} onClose={() => setConfigOpen(false)} />

      {/* CMD+K hint bar */}
      {!cmdOpen && !configOpen && (
        <div className="fixed bottom-28 left-1/2 -translate-x-1/2 bg-[#353535]/80 backdrop-blur border border-[#424754]/30 px-4 py-2 flex items-center gap-4 shadow-2xl z-50 pointer-events-none">
          <div className="flex items-center gap-1">
            <kbd className="font-mono text-[10px] bg-zinc-800 px-1.5 border border-zinc-700 text-zinc-400">CMD</kbd>
            <span className="font-mono text-[10px] text-zinc-400">+</span>
            <kbd className="font-mono text-[10px] bg-zinc-800 px-1.5 border border-zinc-700 text-zinc-400">K</kbd>
          </div>
          <div className="w-px h-4 bg-zinc-700" />
          <span className="font-mono text-[10px] text-zinc-400 uppercase tracking-widest">Search_Clauses_Or_Commands</span>
        </div>
      )}
    </div>
  )
}
