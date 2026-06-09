import { PerformanceReport } from '../components/Reports/PerformanceReport'
export function ReportsPage() {
  return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-white">Reports</h1><p className="text-dark-400 mt-1">View performance reports and analytics</p></div>
      <PerformanceReport />
    </div>
  )
}
