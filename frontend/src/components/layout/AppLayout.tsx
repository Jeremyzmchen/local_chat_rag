import { Outlet } from 'react-router-dom'
import SideNavBar  from './SideNavBar'
import TopAppBar   from './TopAppBar'
import TerminalLog from './TerminalLog'

export default function AppLayout() {
  return (
    <div className="flex h-screen overflow-hidden bg-[#131313]">
      <SideNavBar />

      <div className="flex flex-col flex-1 pl-16">
        <TopAppBar />

        {/* Page content */}
        <main className="flex-1 pt-14 overflow-hidden flex flex-col">
          <div className="flex-1 overflow-auto">
            <Outlet />
          </div>
          <TerminalLog />
        </main>
      </div>
    </div>
  )
}
