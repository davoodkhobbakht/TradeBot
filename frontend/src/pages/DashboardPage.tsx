import { StatusCard } from '../components/Dashboard/StatusCard'
import { MetricsCard } from '../components/Dashboard/MetricsCard'
import { EquityChart } from '../components/Dashboard/EquityChart'
import { useBotStatus } from '../hooks/useBotStatus'
import { reportsApi, botApi } from '../api/client'
import { useEffect, useState } from 'react'
import type { PerformanceReport } from '../types'

export function DashboardPage() {
  const { status, refetch } = useBotStatus()
  const [report, setReport] = useState<PerformanceReport | null>(null)
  useEffect(() => { reportsApi.getLatest().then(setReport).catch(console.error) }, [])
  return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-white">Dashboard</h1><p className="text-dark-400 mt-1">Overview of your trading bot</p></div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1"><StatusCard status={status} onStart={async () => { await botApi.start({ mode: 'live' }); refetch() }} onStop={async () => { await botApi.stop(); refetch() }} /></div>
        <div className="lg:col-span-2"><EquityChart days={30} /></div>
      </div>
      <MetricsCard report={report} />
    </div>
  )
}
