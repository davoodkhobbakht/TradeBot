import { BotControls } from '../components/Bot/BotControls'
import { BotStatus } from '../components/Bot/BotStatus'
import { useBotStatus } from '../hooks/useBotStatus'

export function BotPage() {
  const { status, refetch } = useBotStatus()
  return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-white">Bot Control</h1><p className="text-dark-400 mt-1">Start, stop, and configure your trading bot</p></div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2"><BotControls status={status} onStatusChange={refetch} /></div>
        <div className="lg:col-span-1"><BotStatus status={status} /></div>
      </div>
    </div>
  )
}
