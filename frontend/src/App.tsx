import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout/Layout'
import { DashboardPage } from './pages/DashboardPage'
import { BotPage } from './pages/BotPage'
import { ConfigPage } from './pages/ConfigPage'
import { PositionsPage } from './pages/PositionsPage'
import { BacktestPage } from './pages/BacktestPage'
import { ReportsPage } from './pages/ReportsPage'
import { LogsPage } from './pages/LogsPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<DashboardPage />} />
          <Route path="bot" element={<BotPage />} />
          <Route path="config" element={<ConfigPage />} />
          <Route path="positions" element={<PositionsPage />} />
          <Route path="backtest" element={<BacktestPage />} />
          <Route path="reports" element={<ReportsPage />} />
          <Route path="logs" element={<LogsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
export default App
