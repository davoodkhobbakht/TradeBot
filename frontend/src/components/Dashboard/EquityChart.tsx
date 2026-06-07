import { useEffect, useState } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { chartApi } from '../../api/client'
import type { EquityDataPoint } from '../../types'

export function EquityChart({ days = 30 }: { days?: number }) {
  const [data, setData] = useState<EquityDataPoint[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    chartApi.getEquity(days).then(setData).catch(console.error).finally(() => setLoading(false))
  }, [days])

  if (loading) return <div className="bg-dark-900 border border-dark-700 rounded-xl p-6"><h3 className="text-sm font-medium text-dark-400 uppercase tracking-wider mb-4">Equity Curve</h3><div className="flex items-center justify-center h-64 text-dark-500">Loading chart...</div></div>

  return (
    <div className="bg-dark-900 border border-dark-700 rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-dark-400 uppercase tracking-wider">Equity Curve</h3>
        <span className="text-xs text-dark-500">Last {days} days</span>
      </div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs><linearGradient id="eq" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} /><stop offset="95%" stopColor="#3b82f6" stopOpacity={0} /></linearGradient></defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="time" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} tickFormatter={(v) => { const d = new Date(v); return `${d.getMonth() + 1}/${d.getDate()}` }} />
            <YAxis stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} tickFormatter={(v) => `$${v.toFixed(0)}`} />
            <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#f8fafc' }} formatter={(v: number) => [`$${v.toFixed(2)}`, 'Equity']} />
            <Area type="monotone" dataKey="equity" stroke="#3b82f6" strokeWidth={2} fillOpacity={1} fill="url(#eq)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
