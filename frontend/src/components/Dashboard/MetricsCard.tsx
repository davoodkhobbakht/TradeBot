import { TrendingUp, TrendingDown, Target, BarChart3, DollarSign, Award } from 'lucide-react'
import type { PerformanceReport } from '../../types'

export function MetricsCard({ report }: { report: PerformanceReport | null }) {
  if (!report) return <div className="bg-dark-900 border border-dark-700 rounded-xl p-6"><h3 className="text-sm font-medium text-dark-400 uppercase tracking-wider mb-4">Performance Metrics</h3><div className="flex items-center justify-center h-32 text-dark-500">No data available</div></div>
  
  const metrics = [
    { label: 'Total Return', value: `${report.return_pct.toFixed(2)}%`, icon: TrendingUp, color: report.return_pct >= 0 ? 'text-green-400' : 'text-red-400', bgColor: report.return_pct >= 0 ? 'bg-green-500/10' : 'bg-red-500/10' },
    { label: 'Win Rate', value: `${report.win_rate}%`, icon: Target, color: 'text-blue-400', bgColor: 'bg-blue-500/10' },
    { label: 'Max Drawdown', value: `${report.max_drawdown.toFixed(2)}%`, icon: TrendingDown, color: 'text-orange-400', bgColor: 'bg-orange-500/10' },
    { label: 'Total Trades', value: report.total_trades.toString(), icon: BarChart3, color: 'text-purple-400', bgColor: 'bg-purple-500/10' },
    { label: 'Profit Factor', value: report.profit_factor.toFixed(2), icon: DollarSign, color: 'text-emerald-400', bgColor: 'bg-emerald-500/10' },
    { label: 'Sharpe Ratio', value: report.sharpe_ratio.toFixed(2), icon: Award, color: 'text-cyan-400', bgColor: 'bg-cyan-500/10' },
  ]
  return (
    <div className="bg-dark-900 border border-dark-700 rounded-xl p-6">
      <h3 className="text-sm font-medium text-dark-400 uppercase tracking-wider mb-4">Performance Metrics</h3>
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        {metrics.map(({ label, value, icon: Icon, color, bgColor }) => (
          <div key={label} className="flex items-center gap-3 p-3 rounded-lg bg-dark-800/50">
            <div className={`w-10 h-10 rounded-lg ${bgColor} flex items-center justify-center`}><Icon className={`w-5 h-5 ${color}`} /></div>
            <div><p className="text-xs text-dark-400">{label}</p><p className={`text-lg font-bold ${color}`}>{value}</p></div>
          </div>
        ))}
      </div>
    </div>
  )
}
