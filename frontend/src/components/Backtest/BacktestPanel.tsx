import { useState } from 'react'
import { FlaskConical, Zap, Loader2, CheckCircle, XCircle } from 'lucide-react'
import { backtestApi } from '../../api/client'
import clsx from 'clsx'

export function BacktestPanel() {
  const [running, setRunning] = useState<'simple' | 'enhanced' | null>(null)
  const [result, setResult] = useState<unknown>(null)
  const [error, setError] = useState<string | null>(null)

  const runBacktest = async (type: 'simple' | 'enhanced') => {
    setRunning(type); setError(null); setResult(null)
    try {
      if (type === 'simple') await backtestApi.runSimple()
      else await backtestApi.runEnhanced()
      
      const pollInterval = setInterval(async () => {
        try {
          const res = await backtestApi.getResult()
          if (res.type === type && (res.data || res.error)) {
            clearInterval(pollInterval); setRunning(null)
            if (res.error) setError(res.error)
            else setResult(res.data)
          }
        } catch { clearInterval(pollInterval); setRunning(null); setError('Failed to fetch backtest results') }
      }, 3000)
    } catch { setRunning(null); setError('Failed to start backtest') }
  }

  return (
    <div className="space-y-6">
      <div className="bg-dark-900 border border-dark-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Run Backtest</h3>
        <p className="text-sm text-dark-400 mb-6">Run backtests on historical data to evaluate strategy performance.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <button onClick={() => runBacktest('simple')} disabled={running !== null} className={clsx('p-6 rounded-lg border text-left transition-all', running === 'simple' ? 'border-primary-500 bg-primary-500/10' : 'border-dark-700 bg-dark-800 hover:border-dark-600', running !== null && running !== 'simple' && 'opacity-50 cursor-not-allowed')}>
            <div className="flex items-center gap-3 mb-3"><FlaskConical className="w-6 h-6 text-blue-400" /><h4 className="text-lg font-medium text-white">Simple Backtest</h4></div>
            <p className="text-sm text-dark-400">Run basic strategy backtesting without ML models.</p>
            {running === 'simple' && <div className="mt-3 flex items-center gap-2 text-primary-400 text-sm"><Loader2 className="w-4 h-4 animate-spin" />Running...</div>}
          </button>
          <button onClick={() => runBacktest('enhanced')} disabled={running !== null} className={clsx('p-6 rounded-lg border text-left transition-all', running === 'enhanced' ? 'border-primary-500 bg-primary-500/10' : 'border-dark-700 bg-dark-800 hover:border-dark-600', running !== null && running !== 'enhanced' && 'opacity-50 cursor-not-allowed')}>
            <div className="flex items-center gap-3 mb-3"><Zap className="w-6 h-6 text-yellow-400" /><h4 className="text-lg font-medium text-white">Enhanced Backtest</h4></div>
            <p className="text-sm text-dark-400">Run ML-powered backtesting with Random Forest models.</p>
            {running === 'enhanced' && <div className="mt-3 flex items-center gap-2 text-primary-400 text-sm"><Loader2 className="w-4 h-4 animate-spin" />Running...</div>}
          </button>
        </div>
      </div>
      {error && <div className="bg-dark-900 border border-red-500/30 rounded-xl p-6"><div className="flex items-center gap-2 text-red-400"><XCircle className="w-5 h-5" /><h4 className="font-medium">Backtest Error</h4></div><p className="text-sm text-dark-400 mt-2">{error}</p></div>}
      {result && <div className="bg-dark-900 border border-green-500/30 rounded-xl p-6"><div className="flex items-center gap-2 text-green-400 mb-4"><CheckCircle className="w-5 h-5" /><h4 className="font-medium">Backtest Results</h4></div><pre className="bg-dark-800 p-4 rounded-lg text-sm text-dark-300 overflow-x-auto">{JSON.stringify(result, null, 2)}</pre></div>}
    </div>
  )
}
