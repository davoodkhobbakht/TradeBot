import { useState } from 'react'
import { Play, Square, Loader2, AlertCircle } from 'lucide-react'
import { botApi } from '../../api/client'
import clsx from 'clsx'
import type { BotStatus, StartRequest } from '../../types'

const modes = [
  { value: 'live', label: 'Live Trading', description: 'Real-time trading on testnet' },
  { value: 'enhanced', label: 'Enhanced Backtest', description: 'ML-powered backtesting' },
  { value: 'simple', label: 'Simple Backtest', description: 'Basic strategy backtesting' },
  { value: 'train', label: 'Train Models', description: 'Train ML models on historical data' },
  { value: 'validate', label: 'Validate', description: 'Validate strategy performance' },
]

export function BotControls({ status, onStatusChange }: { status: BotStatus | null; onStatusChange: () => void }) {
  const [selectedMode, setSelectedMode] = useState<string>('live')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const isRunning = status?.status === 'running'

  const handleStart = async () => {
    setLoading(true); setError(null); setSuccess(null)
    try {
      const result = await botApi.start({ mode: selectedMode as StartRequest['mode'] })
      if (result.status === 'started') { setSuccess(`Bot started in ${selectedMode} mode (PID: ${result.pid})`); onStatusChange() }
      else { setError(`Failed to start: ${result.status}`) }
    } catch { setError('Failed to communicate with bot API') }
    finally { setLoading(false) }
  }

  const handleStop = async () => {
    setLoading(true); setError(null); setSuccess(null)
    try {
      const result = await botApi.stop()
      if (result.status === 'stopped') { setSuccess('Bot stopped successfully'); onStatusChange() }
      else { setError(`Failed to stop: ${result.status}`) }
    } catch { setError('Failed to communicate with bot API') }
    finally { setLoading(false) }
  }

  return (
    <div className="bg-dark-900 border border-dark-700 rounded-xl p-6">
      <h3 className="text-lg font-semibold text-white mb-6">Bot Controls</h3>
      {error && <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-400 text-sm"><AlertCircle className="w-4 h-4" />{error}</div>}
      {success && <div className="mb-4 p-3 bg-green-500/10 border border-green-500/30 rounded-lg text-green-400 text-sm">{success}</div>}
      <div className="mb-6">
        <label className="block text-sm font-medium text-dark-400 mb-3">Trading Mode</label>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {modes.map(({ value, label, description }) => (
            <button key={value} onClick={() => setSelectedMode(value)} disabled={isRunning} className={clsx('p-4 rounded-lg border text-left transition-all', selectedMode === value ? 'border-primary-500 bg-primary-500/10' : 'border-dark-700 bg-dark-800 hover:border-dark-600', isRunning && 'opacity-50 cursor-not-allowed')}>
              <p className="text-sm font-medium text-white">{label}</p><p className="text-xs text-dark-400 mt-1">{description}</p>
            </button>
          ))}
        </div>
      </div>
      <div className="flex gap-4">
        <button onClick={handleStart} disabled={isRunning || loading} className={clsx('flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-lg text-sm font-medium transition-all', isRunning || loading ? 'bg-dark-700 text-dark-500 cursor-not-allowed' : 'bg-green-600 hover:bg-green-500 text-white')}>
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}Start Bot
        </button>
        <button onClick={handleStop} disabled={!isRunning || loading} className={clsx('flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-lg text-sm font-medium transition-all', !isRunning || loading ? 'bg-dark-700 text-dark-500 cursor-not-allowed' : 'bg-red-600 hover:bg-red-500 text-white')}>
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Square className="w-4 h-4" />}Stop Bot
        </button>
      </div>
    </div>
  )
}
