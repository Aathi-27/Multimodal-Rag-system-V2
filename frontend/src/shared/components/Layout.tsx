import { useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import {
  MessageSquare,
  Upload,
  Database,
  Clock,
  Activity,
  LogOut,
  Menu,
  ShieldAlert,
  FlaskConical,
} from 'lucide-react';

const NAV_SECTIONS = [
  {
    label: 'MAIN',
    items: [
      { to: '/', label: 'Chat', icon: MessageSquare },
      { to: '/upload', label: 'Upload', icon: Upload },
    ],
  },
  {
    label: 'DATA',
    items: [
      { to: '/knowledge', label: 'Knowledge Base', icon: Database },
      { to: '/history', label: 'Query History', icon: Clock },
    ],
  },
  {
    label: 'MONITORING',
    items: [
      { to: '/status', label: 'System', icon: Activity },
    ],
  },
  {
    label: 'RESEARCH LAB',
    items: [
      { to: '/diagnosis', label: 'Diagnosis', icon: ShieldAlert },
      { to: '/experiments', label: 'Experiments', icon: FlaskConical },
    ],
  },
];

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { authEnabled, logout } = useAuth();

  return (
    <div className="flex h-screen overflow-hidden bg-slate-950">
      {/* ── Mobile overlay ──────────────────────────────────────────── */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ── Sidebar ─────────────────────────────────────────────────── */}
      <aside
        className={`
          fixed inset-y-0 left-0 z-40 w-64 bg-slate-900 border-r border-slate-800/80
          flex flex-col transform transition-transform duration-200 ease-in-out
          lg:static lg:translate-x-0
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        `}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-5 h-16 border-b border-slate-800/80 flex-shrink-0">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold text-sm">
            R
          </div>
          <div>
            <h1 className="text-sm font-semibold text-slate-100 tracking-tight">Offline RAG</h1>
            <p className="text-[11px] text-slate-500 font-medium">v1.0.0</p>
          </div>
        </div>

        {/* Nav sections */}
        <nav className="flex-1 px-3 py-4 space-y-6 overflow-y-auto">
          {NAV_SECTIONS.map((section) => (
            <div key={section.label}>
              <p className="px-3 mb-2 text-[10px] font-semibold tracking-widest text-slate-500 uppercase">
                {section.label}
              </p>
              <div className="space-y-0.5">
                {section.items.map(({ to, label, icon: Icon }) => (
                  <NavLink
                    key={to}
                    to={to}
                    end={to === '/'}
                    onClick={() => setSidebarOpen(false)}
                    className={({ isActive }) =>
                      `group flex items-center gap-3 px-3 py-2 rounded-r-lg text-[13px] font-medium
                       transition-all duration-200 relative ${
                        isActive
                          ? 'bg-slate-800/80 text-blue-400 border-l-2 border-blue-500 pl-[10px]'
                          : 'text-slate-400 hover:bg-slate-800/40 hover:text-slate-200 hover:pl-[14px]'
                      }`
                    }
                  >
                    <Icon className="w-4 h-4 flex-shrink-0" />
                    {label}
                  </NavLink>
                ))}
              </div>
            </div>
          ))}
        </nav>

        {/* Auth logout (only if enabled) */}
        {authEnabled && (
          <div className="flex-shrink-0 p-3 border-t border-slate-800/80">
            <button
              onClick={logout}
              className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-[13px]
                         text-slate-400 hover:bg-slate-800/50 hover:text-slate-200 transition-colors"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>
          </div>
        )}
      </aside>

      {/* ── Main content area ───────────────────────────────────────── */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Mobile header */}
        <header className="flex items-center h-14 px-4 border-b border-slate-800/80 bg-slate-900/80 backdrop-blur-sm lg:hidden">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 -ml-2 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-slate-200"
          >
            <Menu className="w-5 h-5" />
          </button>
          <h1 className="ml-3 text-sm font-semibold text-slate-100">Offline RAG</h1>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-hidden">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
