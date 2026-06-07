import { useEffect, useState } from 'react'
import { TrendingUp, TrendingDown, RefreshCw, Loader2 } from 'lucide-react'
import { positionsApi } from '../../api/client'
import type { Position } from '../../types'
import clsx from 'clsx'

export function PositionsTable() {
  const [positions, setPositions] = useState<Position[]>([])
  const [loading, setLoading] = useState(true)

  const fetchPositions = async () => {
    setLoading(true)
    try { setPositions(await positionsApi.get()) } catch (err) { console.error('Failed to fetch positions:', err) }
    finally { setLoading(false) }
  }

  useEffect(() => { fetchPositions(); const interval = setInterval(fetchPositions, 10000); return () => clearInterval(interval) }, [])

  if (loading && positions.length === 0) return <div className="bg-dark-900 border border-dark-700 rounded-xl p-6 flex items-center justify-center h-64"><Loader2 className="w-8 h-8 animate-spin text-primary-400" /></div>

  return (
    <div className="bg-dark-900 border border-dark-700 rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-white">Open Positions</h3>
        <button onClick={fetchPositions} className="flex items-center gap-2 px-3 py-1.5 bg-dark-800 hover:bg-dark-700 text-dark-400 hover:text-white rounded-lg text-sm transition-all"><RefreshCw className="w-4 h-4" />Refresh</button>
      </div>
      {positions.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-48 text-dark-500"><TrendingUp className="w-12 h-12 mb-2" /><p>No open positions</p></div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead><tr className="border-b border-dark-700">
              <th className="text-left py-3 px-4 text-xs font-medium text-dark-400 uppercase tracking-wider">Symbol</th>
              <th className="text-left py-3 px-4 text-xs font-medium text-dark-400 uppercase tracking-wider">Side</th>
              <th className="text-right py-3 px-4 text-xs font-medium text-dark-400 uppercase tracking-wider">Entry Price</th>
              <th className="text-right py-3 px-4 text-xs font-medium text-dark-400 uppercase tracking-wider">Current Price</th>
              <th className="text-right py-3 px-4 text-xs font-medium text-dark-400 uppercase tracking-wider">PnL %</th>
            </tr></thead>
            <tbody>
              {positions.map((pos, idx) => (
                <tr key={idx} className="border-b border-dark-800 hover:bg-dark-800/50 transition-colors">
                  <td className="py-3 px-4"><span className="font-medium text-white">{pos.symbol}</span></td>
                  <td className="py-3 px-4"><span className={clsx('inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium', pos.side === 'LONG' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400')}>{pos.side === 'LONG' ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}{pos.side}</span></td>
                  <td className="py-3 px-4 text-right text-dark-300">${pos.entry.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                  <td className="py-3 px-4 text-right text-white font-medium">${pos.current.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                  <td className="py-3 px-4 text-right"><span className={clsx('font-medium', pos.pnl_pct >= 0 ? 'text-green-400' : 'text-red-400')}>{pos.pnl_pct >= 0 ? '+' : ''}{pos.pnl_pct.toFixed(2)}%</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
