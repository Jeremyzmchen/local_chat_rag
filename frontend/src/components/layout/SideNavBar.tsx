import { useNavigate, useLocation } from 'react-router-dom'

interface NavItem {
  icon: string
  path: string
  label: string
}

const NAV_ITEMS: NavItem[] = [
  { icon: 'dashboard',   path: '/',          label: 'Dash'   },
  { icon: 'description', path: '/documents', label: 'Docs'   },
  { icon: 'upload_file', path: '/upload',    label: 'Upload' },
  { icon: 'analytics',   path: '/report',    label: 'Report' },
  { icon: 'settings',    path: '/settings',  label: 'Set'    },
]

export default function SideNavBar() {
  const navigate  = useNavigate()
  const { pathname } = useLocation()

  const isActive = (path: string) =>
    path === '/' ? pathname === '/' : pathname.startsWith(path)

  return (
    <nav className="fixed left-0 top-0 w-16 h-screen bg-zinc-900 flex flex-col items-center py-4 z-60">
      {/* Logo */}
      <div className="mb-6">
        <span
          className="material-symbols-outlined phosphor-glow"
          style={{ color: '#adc6ff', fontSize: 28 }}
        >
          terminal
        </span>
      </div>

      {/* Nav items */}
      <div className="flex flex-col items-center w-full gap-1">
        {NAV_ITEMS.map(({ icon, path, label }) => {
          const active = isActive(path)
          return (
            <button
              key={path}
              onClick={() => navigate(path)}
              className={[
                'flex flex-col items-center py-3 w-full transition-colors duration-150',
                active
                  ? 'text-[#adc6ff] border-l-2 border-[#adc6ff] bg-[#adc6ff]/10'
                  : 'text-zinc-500 hover:text-[#adc6ff] hover:bg-zinc-800',
              ].join(' ')}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 22 }}>
                {icon}
              </span>
              <span className="font-mono text-[8px] uppercase tracking-tighter mt-1">
                {label}
              </span>
            </button>
          )
        })}
      </div>
    </nav>
  )
}
