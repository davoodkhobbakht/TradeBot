import { PositionsTable } from '../components/Positions/PositionsTable'
export function PositionsPage() {
  return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-white">Positions</h1><p className="text-dark-400 mt-1">View current open trading positions</p></div>
      <PositionsTable />
    </div>
  )
}
