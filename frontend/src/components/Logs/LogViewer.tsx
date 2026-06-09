import { useEffect, useState, useRef } from 'react'
import { ScrollText, Wifi, WifiOff, Trash2, Download } from 'lucide-react'
import { useWebSocket } from '../../hooks/useWebSocket'
import { logsApi } from '../../api/client'
import clsx from 'clsx'

export function LogViewer() {
  const [logs, setLogs] = useState<string[]>([])
  const [autoScroll, setAutoScroll] = useState(true)
  const [filter, setFilter] = useState('')
  const logsEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => { logsApi.get(200).then(setData).catch(console.error) }, [])

  useWebSocket({ url: '/ws/logs', onMessage: (data: unknown) => { const msg = data as { type: string; data: string[] }; if (msg.type === 'logs' && msg.data) setLogs(msg.data) }, enabled: true })

  useEffect(() => { if (autoScroll) logsEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [logs, autoScroll])

  const filteredLogs = filter ? logs.filter(log => log.toLowerCase().includes(filter.toLowerCase())) : logs

  const handleClear = () => setLogs([])
  const handleDownload = () => {
    const blob = new Blob([logs.join('\n')], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = `tradebot-logs-${new Date().toISOString().slice(0, 10)}.txt`; a.click()
    URL.revokeObjectURL(url)
  }

  const getLogColor = (log: string) => {
    if (log.includes('❌') || log.includes('ERROR') || log.includes('error')) return 'text-red-400'
    if (log.includes('✅') || log.includes('SUCCESS') || log.includes('success')) return 'text-green-400'
    if (log.includes('⚠️') || log.includes('WARNING') || log.includes('warning')) return 'text-yellow-400'
    if (log.includes('🚀') || log.includes('INFO')) return 'text-blue-400'
    return 'text-dark-300'
  }

  return (
    <div className="bg-dark-900 border border-dark-700 rounded-xl p-6 h-[calc(100vh-12rem)] flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3"><ScrollText className="w-5 h-5 text-primary-400" /><h3 className="text-lg font-semibold text-white">Live Logs</h3><span className="text-xs text-dark-500">({filteredLogs.length} lines)</span></div>
        <div className="flex items-center gap-3">
          <input type="text" placeholder="Filter logs..." value={filter} onChange={e => setFilter(e.target.value)} className="px-3 py-1.5 bg-dark-800 border border-dark-700 rounded-lg text-sm text-white placeholder-dark-500 focus:border-primary-500 focus:outline-none w-48" />
          <button onClick={() => setAutoScroll(!autoScroll)} className={clsx('flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs transition-all', autoScroll ? 'bg-primary-600/20 text-primary-400' : 'bg-dark-800 text-dark-400')}>{autoScroll ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}Auto-scroll</button>
          <button onClick={handleDownload} className="flex items-center gap-1 px-3 py-1.5 bg-dark-800 hover:bg-dark-700 text-dark-400 hover:text-white rounded-lg text-xs transition-all"><Download className="w-3 h-3" />Export</button>
          <button onClick={handleClear} className="flex items-center gap-1 px-3 py-1.5 bg-dark-800 hover:bg-red-600/20 text-dark-400 hover:text-red-400 rounded-lg text-xs transition-all"><Trash2 className="w-3 h-3" />Clear</button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto bg-dark-950 rounded-lg p-4 font-mono text-sm">
        {filteredLogs.length === 0 ? <div className="flex items-center justify-center h-full text-dark-500">No logs available</div> : (
          <div className="space-y-1">
            {filteredLogs.map((log, idx) => <div key={idx} className={clsx('py-0.5 hover:bg-dark-800/50 px-2 rounded', getLogColor(log))}>{log}</div>)}
            <div ref={logsEndRef} />
          </div>
        )}
      </div>
    </div>
  )
}
