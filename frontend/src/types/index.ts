export interface BotStatus { status: 'running' | 'stopped'; mode: string | null; pid: number | null; uptime: string }
export interface StartRequest { mode: 'train' | 'simple' | 'enhanced' | 'validate' | 'live'; args?: string[] }
export interface TradeSettings { initial_capital: number; trade_fee: number; slippage: number; max_trades_per_symbol: number }
export interface MLSettings { target_lookahead: number; test_size: number; min_positive_samples: number }
export interface RLSettings { [key: string]: unknown }
export interface SymbolParams { stop_loss: number; take_profit: number; position_size: number }
export interface ConfigSnapshot { trade: TradeSettings; ml: MLSettings; rl: RLSettings; symbols: Record<string, SymbolParams> }
export interface ConfigUpdate { trade?: Partial<TradeSettings>; ml?: Partial<MLSettings>; rl?: Partial<RLSettings>; symbols?: Record<string, Partial<SymbolParams>> }
export interface Position { symbol: string; side: 'LONG' | 'SHORT'; entry: number; current: number; pnl_pct: number }
export interface BacktestResult { type: 'simple' | 'enhanced'; data?: unknown; error?: string }
export interface PerformanceReport { return_pct: number; win_rate: number; max_drawdown: number; total_trades: number; profit_factor: number; sharpe_ratio: number }
export interface EquityDataPoint { time: string; equity: number }
