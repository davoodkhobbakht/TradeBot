import axios from 'axios'
import type { BotStatus, StartRequest, ConfigSnapshot, ConfigUpdate, Position, BacktestResult, PerformanceReport, EquityDataPoint } from '../types'

const api = axios.create({ baseURL: '/api', headers: { 'Content-Type': 'application/json' } })

export const botApi = {
  start: (req: StartRequest) => api.post<{ status: string; pid?: number }>('/bot/start', req).then(r => r.data),
  stop: () => api.post<{ status: string }>('/bot/stop').then(r => r.data),
  status: () => api.get<BotStatus>('/bot/status').then(r => r.data),
}
export const configApi = {
  get: () => api.get<ConfigSnapshot>('/config').then(r => r.data),
  update: (update: ConfigUpdate) => api.put<{ status: string; config: ConfigSnapshot }>('/config', update).then(r => r.data),
}
export const positionsApi = { get: () => api.get<Position[]>('/positions').then(r => r.data) }
export const backtestApi = {
  runSimple: () => api.post<{ job_id: string; status: string }>('/backtest/simple').then(r => r.data),
  runEnhanced: () => api.post<{ job_id: string; status: string }>('/backtest/enhanced').then(r => r.data),
  getResult: () => api.get<BacktestResult>('/backtest/result').then(r => r.data),
}
export const reportsApi = { getLatest: () => api.get<PerformanceReport>('/reports/latest').then(r => r.data) }
export const chartApi = { getEquity: (days: number = 30) => api.get<EquityDataPoint[]>('/chart/equity', { params: { days } }).then(r => r.data) }
export const logsApi = { get: (lines: number = 100) => api.get<string[]>('/logs', { params: { lines } }).then(r => r.data) }
export default api
