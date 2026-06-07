import { BacktestPanel } from '../components/Backtest/BacktestPanel'
export function BacktestPage() {
  return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-white">Backtesting</h1><p className="text-dark-400 mt-1">Run backtests to evaluate strategy performance</p></div>
      <BacktestPanel />
    </div>
  )
}
