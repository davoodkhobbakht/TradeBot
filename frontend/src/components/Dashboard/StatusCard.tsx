import { Bot, Play, Square, Clock, Cpu } from 'lucide-react'
import clsx from 'clsx'
import type { BotStatus } from '../../types'

export function StatusCard({ status, onStart, onStop }: { status: BotStatus | null; onStart: () => void; onStop: () => void }) {
  const isRunning = status?.status === 'running'
  return (
    <div className="bg-dark-900 border border-dark-700 rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-dark-400 uppercase tracking-wider">Bot Status</h3>
        <div className={clsx('flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium', isRunning ? 'bg-green-500/20 text-green-400' : 'bg-dark-700 text-dark-400')}>
          <div className={clsx('w-2 h-2 rounded-full', isRunning ? 'bg-green-400 animate-pulse-green' : 'bg-dark-500')} />
          {isRunning ? 'Running' : 'Stopped'}
        </div>
      </div>
      <div className="space-y-3">
        <div className="flex items-center gap-3"><Bot className="w-5 h-5 text-primary-400" /><div><p className="text-sm text-dark-400">Mode</p><p className="text-lg font-semibold text-white">{status?.mode || 'None'}</p></div></div>
        <div className="flex items-center gap-3"><Cpu className="w-5 h-5 text-primary-400" /><div><p className="text-sm text-dark-400">PID</p><p className="text-lg font-semibold text-white">{status?.pid || '—'}</p></div></div>
        <div className="flex items-center gap-3"><Clock className="w-5 h-5 text-primary-400" /><div><p className="text-sm text-dark-400">Uptime</p><p className="text-lg font-semibold text-white">{status?.uptime || '—'}</p></div></div>
      </div>
      <div className="mt-6 flex gap-3">
        <button onClick={onStart} disabled={isRunning} className={clsx('flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all', isRunning ? 'bg-dark-700 text-dark-500 cursor-not-allowed' : 'bg-green-600 hover:bg-green-500 text-white')}><Play className="w-4 h-4" />Start</button>
        <button onClick={onStop} disabled={!isRunning} className={clsx('flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all', !isRunning ? 'bg-dark-700 text-dark-500 cursor-not-allowed' : 'bg-red-600 hover:bg-red-500 text-white')}><Square className="w-4 h-4" />Stop</button>
      </div>
    </div>
  )
}
