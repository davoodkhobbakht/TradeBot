import { useBotStatus } from '../../hooks/useBotStatus'
import { Bot, Wifi } from 'lucide-react'
import clsx from 'clsx'

export function Header() {
  const { status } = useBotStatus()
  return (
    <header className="h-16 bg-dark-900/50 backdrop-blur-sm border-b border-dark-700 flex items-center justify-between px-6 sticky top-0 z-10">
      <h2 className="text-lg font-semibold text-white">TradeBot Dashboard</h2>
      <div className="flex items-center gap-6">
        {status && (
          <div className="flex items-center gap-2">
            <Bot className={clsx('w-4 h-4', status.status === 'running' ? 'text-green-400' : 'text-dark-500')} />
            <span className={clsx('text-sm font-medium', status.status === 'running' ? 'text-green-400' : 'text-dark-400')}>
              {status.status === 'running' ? `Running (${status.mode})` : 'Stopped'}
            </span>
          </div>
        )}
        <div className="flex items-center gap-2 text-dark-400"><Wifi className="w-4 h-4 text-green-400" /><span className="text-xs">Connected</span></div>
      </div>
    </header>
  )
}
