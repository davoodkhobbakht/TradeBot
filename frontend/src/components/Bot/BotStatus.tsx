import { Activity, Clock, Cpu, Hash, Zap } from 'lucide-react'
import type { BotStatus as BotStatusType } from '../../types'
import clsx from 'clsx'

export function BotStatus({ status }: { status: BotStatusType | null }) {
  if (!status) return <div className="bg-dark-900 border border-dark-700 rounded-xl p-6"><h3 className="text-lg font-semibold text-white mb-4">Bot Status</h3><p className="text-dark-400">Loading...</p></div>
  const isRunning = status.status === 'running'
  return (
    <div className="bg-dark-900 border border-dark-700 rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-white">Bot Status</h3>
        <div className={clsx('flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium', isRunning ? 'bg-green-500/20 text-green-400' : 'bg-dark-700 text-dark-400')}>
          <div className={clsx('w-2 h-2 rounded-full', isRunning ? 'bg-green-400 animate-pulse-green' : 'bg-dark-500')} />{isRunning ? 'Running' : 'Stopped'}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-dark-800/50 rounded-lg"><div className="flex items-center gap-2 mb-2"><Activity className="w-4 h-4 text-primary-400" /><span className="text-xs text-dark-400">Status</span></div><p className={clsx('text-lg font-bold', isRunning ? 'text-green-400' : 'text-dark-400')}>{status.status.toUpperCase()}</p></div>
        <div className="p-4 bg-dark-800/50 rounded-lg"><div className="flex items-center gap-2 mb-2"><Zap className="w-4 h-4 text-yellow-400" /><span className="text-xs text-dark-400">Mode</span></div><p className="text-lg font-bold text-white">{status.mode || 'None'}</p></div>
        <div className="p-4 bg-dark-800/50 rounded-lg"><div className="flex items-center gap-2 mb-2"><Hash className="w-4 h-4 text-purple-400" /><span className="text-xs text-dark-400">PID</span></div><p className="text-lg font-bold text-white">{status.pid || '—'}</p></div>
        <div className="p-4 bg-dark-800/50 rounded-lg"><div className="flex items-center gap-2 mb-2"><Clock className="w-4 h-4 text-cyan-400" /><span className="text-xs text-dark-400">Uptime</span></div><p className="text-lg font-bold text-white">{status.uptime || '—'}</p></div>
      </div>
    </div>
  )
}
