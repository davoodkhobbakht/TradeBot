import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Bot, Settings, TrendingUp, FlaskConical, BarChart3, ScrollText, Activity } from 'lucide-react'
import clsx from 'clsx'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/bot', icon: Bot, label: 'Bot Control' },
  { to: '/config', icon: Settings, label: 'Configuration' },
  { to: '/positions', icon: TrendingUp, label: 'Positions' },
  { to: '/backtest', icon: FlaskConical, label: 'Backtest' },
  { to: '/reports', icon: BarChart3, label: 'Reports' },
  { to: '/logs', icon: ScrollText, label: 'Logs' },
]

export function Sidebar() {
  return (
    <aside className="w-64 bg-dark-900 border-r border-dark-700 flex flex-col h-screen fixed left-0 top-0">
      <div className="p-6 border-b border-dark-700">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-primary-600 rounded-lg flex items-center justify-center"><Activity className="w-6 h-6 text-white" /></div>
          <div><h1 className="text-lg font-bold text-white">TradeBot</h1><p className="text-xs text-dark-400">Testnet Trader V0.2</p></div>
        </div>
      </div>
      <nav className="flex-1 p-4 space-y-1">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => clsx('flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200', isActive ? 'bg-primary-600/20 text-primary-400 border border-primary-600/30' : 'text-dark-400 hover:text-white hover:bg-dark-800')}>
            <Icon className="w-5 h-5" />{label}
          </NavLink>
        ))}
      </nav>
      <div className="p-4 border-t border-dark-700">
        <div className="flex items-center gap-2 text-xs text-dark-500">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse-green" /><span>API Connected</span>
        </div>
      </div>
    </aside>
  )
}
