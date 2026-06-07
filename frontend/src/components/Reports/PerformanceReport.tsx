import { useEffect, useState } from 'react'
import { TrendingUp, TrendingDown, Target, BarChart3, DollarSign, Award, Loader2, RefreshCw } from 'lucide-react'
import { reportsApi } from '../../api/client'
import type { PerformanceReport as ReportType } from '../../types'
import clsx from 'clsx'

export function PerformanceReport() {
  const [report, setReport] = useState<ReportType | null>(null)
  const [loading, setLoading] = useState(true)

  const fetchReport = async () => {
    setLoading(true)
    try { setReport(await reportsApi.getLatest()) } catch (err) { console.error('Failed to fetch report:', err) }
    finally { setLoading(false) }
  }

  useEffect(() => { fetchReport() }, [])

  if (loading) return <div className="bg-dark-900 border border-dark-700 rounded-xl p-6 flex items-center justify-center h-64"><Loader2 className="w-8 h-8 animate-spin text-primary-400" /></div>
  if (!report) return <div className="bg-dark-900 border border-dark-700 rounded-xl p-6"><p className="text-dark-400">No report data available</p></div>

  const metrics = [
    { label: 'Total Return', value: `${report.return_pct.toFixed(2)}%`, icon: TrendingUp, color: report.return_pct >= 0 ? 'text-green-400' : 'text-red-400', bgColor: report.return_pct >= 0 ? 'bg-green-500/10' : 'bg-red-500/10', desc: 'Overall portfolio return' },
    { label: 'Win Rate', value: `${report.win_rate}%`, icon: Target, color: 'text-blue-400', bgColor: 'bg-blue-500/10', desc: 'Percentage of profitable trades' },
    { label: 'Max Drawdown', value: `${report.max_drawdown.toFixed(2)}%`, icon: TrendingDown, color: 'text-orange-400', bgColor: 'bg-orange-500/10', desc: 'Largest peak-to-trough decline' },
    { label: 'Total Trades', value: report.total_trades.toString(), icon: BarChart3, color: 'text-purple-400', bgColor: 'bg-purple-500/10', desc: 'Number of executed trades' },
    { label: 'Profit Factor', value: report.profit_factor.toFixed(2), icon: DollarSign, color: 'text-emerald-400', bgColor: 'bg-emerald-500/10', desc: 'Gross profit / Gross loss' },
    { label: 'Sharpe Ratio', value: report.sharpe_ratio.toFixed(2), icon: Award, color: 'text-cyan-400', bgColor: 'bg-cyan-500/10', desc: 'Risk-adjusted return measure' },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div><h3 className="text-xl font-bold text-white">Performance Report</h3><p className="text-sm text-dark-400 mt-1">Latest backtest performance metrics</p></div>
        <button onClick={fetchReport} className="flex items-center gap-2 px-4 py-2 bg-dark-800 hover:bg-dark-700 text-dark-400 hover:text-white rounded-lg text-sm transition-all"><RefreshCw className="w-4 h-4" />Refresh</button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {metrics.map(({ label, value, icon: Icon, color, bgColor, desc }) => (
          <div key={label} className="bg-dark-900 border border-dark-700 rounded-xl p-6 hover:border-dark-600 transition-all">
            <div className={`w-12 h-12 rounded-lg ${bgColor} flex items-center justify-center mb-4`}><Icon className={`w-6 h-6 ${color}`} /></div>
            <p className="text-sm text-dark-400 mb-1">{label}</p>
            <p className={`text-3xl font-bold ${color} mb-2`}>{value}</p>
            <p className="text-xs text-dark-500">{desc}</p>
          </div>
        ))}
      </div>
      <div className="bg-dark-900 border border-dark-700 rounded-xl p-6">
        <h4 className="text-lg font-semibold text-white mb-4">Summary</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div><p className="text-sm text-dark-400 mb-1">Risk Level</p><p className={clsx('text-lg font-medium', report.max_drawdown < 5 ? 'text-green-400' : report.max_drawdown < 10 ? 'text-yellow-400' : 'text-red-400')}>{report.max_drawdown < 5 ? 'Low' : report.max_drawdown < 10 ? 'Medium' : 'High'}</p></div>
          <div><p className="text-sm text-dark-400 mb-1">Strategy Quality</p><p className={clsx('text-lg font-medium', report.sharpe_ratio > 1.5 ? 'text-green-400' : report.sharpe_ratio > 1 ? 'text-yellow-400' : 'text-red-400')}>{report.sharpe_ratio > 1.5 ? 'Excellent' : report.sharpe_ratio > 1 ? 'Good' : 'Poor'}</p></div>
          <div><p className="text-sm text-dark-400 mb-1">Consistency</p><p className={clsx('text-lg font-medium', report.win_rate > 60 ? 'text-green-400' : report.win_rate > 50 ? 'text-yellow-400' : 'text-red-400')}>{report.win_rate > 60 ? 'High' : report.win_rate > 50 ? 'Medium' : 'Low'}</p></div>
        </div>
      </div>
    </div>
  )
}
