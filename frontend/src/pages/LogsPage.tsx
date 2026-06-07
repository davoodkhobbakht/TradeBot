import { LogViewer } from '../components/Logs/LogViewer'
export function LogsPage() {
  return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-white">Logs</h1><p className="text-dark-400 mt-1">Real-time bot activity logs</p></div>
      <LogViewer />
    </div>
  )
}
