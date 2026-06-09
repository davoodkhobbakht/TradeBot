import { useEffect, useState } from 'react'
import { Save, Loader2, AlertCircle, CheckCircle } from 'lucide-react'
import { configApi } from '../../api/client'
import type { ConfigSnapshot, ConfigUpdate } from '../../types'
import clsx from 'clsx'

export function ConfigEditor() {
  const [config, setConfig] = useState<ConfigSnapshot | null>(null)
  const [editConfig, setEditConfig] = useState<ConfigSnapshot | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  useEffect(() => { configApi.get().then(data => { setConfig(data); setEditConfig(data) }).catch(() => setMessage({ type: 'error', text: 'Failed to load configuration' })).finally(() => setLoading(false)) }, [])

  const handleSave = async () => {
    if (!editConfig) return
    setSaving(true); setMessage(null)
    try {
      const result = await configApi.update({ trade: editConfig.trade, ml: editConfig.ml, rl: editConfig.rl, symbols: editConfig.symbols })
      setConfig(result.config); setEditConfig(result.config)
      setMessage({ type: 'success', text: 'Configuration updated successfully' })
    } catch { setMessage({ type: 'error', text: 'Failed to save configuration' }) }
    finally { setSaving(false) }
  }

  const updateTrade = (k: string, v: number) => setEditConfig(prev => prev ? { ...prev, trade: { ...prev.trade, [k]: v } } : null)
  const updateML = (k: string, v: number) => setEditConfig(prev => prev ? { ...prev, ml: { ...prev.ml, [k]: v } } : null)
  const updateSymbol = (sym: string, k: string, v: number) => setEditConfig(prev => prev ? { ...prev, symbols: { ...prev.symbols, [sym]: { ...prev.symbols[sym], [k]: v } } } : null)

  if (loading) return <div className="bg-dark-900 border border-dark-700 rounded-xl p-6 flex items-center justify-center h-64"><Loader2 className="w-8 h-8 animate-spin text-primary-400" /></div>
  if (!editConfig) return <div className="bg-dark-900 border border-dark-700 rounded-xl p-6"><p className="text-dark-400">Failed to load configuration</p></div>

  return (
    <div className="space-y-6">
      {message && <div className={clsx('p-3 rounded-lg flex items-center gap-2 text-sm', message.type === 'success' ? 'bg-green-500/10 border border-green-500/30 text-green-400' : 'bg-red-500/10 border border-red-500/30 text-red-400')}>{message.type === 'success' ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}{message.text}</div>}
      
      <div className="bg-dark-900 border border-dark-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Trade Settings</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div><label className="block text-sm text-dark-400 mb-1">Initial Capital ($)</label><input type="number" value={editConfig.trade.initial_capital} onChange={e => updateTrade('initial_capital', parseFloat(e.target.value))} className="w-full px-3 py-2 bg-dark-800 border border-dark-700 rounded-lg text-white focus:border-primary-500 focus:outline-none" /></div>
          <div><label className="block text-sm text-dark-400 mb-1">Trade Fee (%)</label><input type="number" step="0.0001" value={editConfig.trade.trade_fee} onChange={e => updateTrade('trade_fee', parseFloat(e.target.value))} className="w-full px-3 py-2 bg-dark-800 border border-dark-700 rounded-lg text-white focus:border-primary-500 focus:outline-none" /></div>
          <div><label className="block text-sm text-dark-400 mb-1">Slippage (%)</label><input type="number" step="0.001" value={editConfig.trade.slippage} onChange={e => updateTrade('slippage', parseFloat(e.target.value))} className="w-full px-3 py-2 bg-dark-800 border border-dark-700 rounded-lg text-white focus:border-primary-500 focus:outline-none" /></div>
          <div><label className="block text-sm text-dark-400 mb-1">Max Trades per Symbol</label><input type="number" value={editConfig.trade.max_trades_per_symbol} onChange={e => updateTrade('max_trades_per_symbol', parseInt(e.target.value))} className="w-full px-3 py-2 bg-dark-800 border border-dark-700 rounded-lg text-white focus:border-primary-500 focus:outline-none" /></div>
        </div>
      </div>

      <div className="bg-dark-900 border border-dark-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Machine Learning Settings</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div><label className="block text-sm text-dark-400 mb-1">Target Lookahead (days)</label><input type="number" value={editConfig.ml.target_lookahead} onChange={e => updateML('target_lookahead', parseInt(e.target.value))} className="w-full px-3 py-2 bg-dark-800 border border-dark-700 rounded-lg text-white focus:border-primary-500 focus:outline-none" /></div>
          <div><label className="block text-sm text-dark-400 mb-1">Test Size</label><input type="number" step="0.05" value={editConfig.ml.test_size} onChange={e => updateML('test_size', parseFloat(e.target.value))} className="w-full px-3 py-2 bg-dark-800 border border-dark-700 rounded-lg text-white focus:border-primary-500 focus:outline-none" /></div>
          <div><label className="block text-sm text-dark-400 mb-1">Min Positive Samples</label><input type="number" step="0.01" value={editConfig.ml.min_positive_samples} onChange={e => updateML('min_positive_samples', parseFloat(e.target.value))} className="w-full px-3 py-2 bg-dark-800 border border-dark-700 rounded-lg text-white focus:border-primary-500 focus:outline-none" /></div>
        </div>
      </div>

      <div className="bg-dark-900 border border-dark-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Symbol Parameters</h3>
        <div className="space-y-4">
          {Object.entries(editConfig.symbols).map(([symbol, params]) => (
            <div key={symbol} className="p-4 bg-dark-800/50 rounded-lg">
              <h4 className="text-sm font-medium text-primary-400 mb-3">{symbol}</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div><label className="block text-xs text-dark-400 mb-1">Stop Loss (%)</label><input type="number" step="0.01" value={params.stop_loss} onChange={e => updateSymbol(symbol, 'stop_loss', parseFloat(e.target.value))} className="w-full px-3 py-2 bg-dark-800 border border-dark-700 rounded-lg text-white text-sm focus:border-primary-500 focus:outline-none" /></div>
                <div><label className="block text-xs text-dark-400 mb-1">Take Profit (%)</label><input type="number" step="0.01" value={params.take_profit} onChange={e => updateSymbol(symbol, 'take_profit', parseFloat(e.target.value))} className="w-full px-3 py-2 bg-dark-800 border border-dark-700 rounded-lg text-white text-sm focus:border-primary-500 focus:outline-none" /></div>
                <div><label className="block text-xs text-dark-400 mb-1">Position Size</label><input type="number" step="0.01" value={params.position_size} onChange={e => updateSymbol(symbol, 'position_size', parseFloat(e.target.value))} className="w-full px-3 py-2 bg-dark-800 border border-dark-700 rounded-lg text-white text-sm focus:border-primary-500 focus:outline-none" /></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="flex justify-end">
        <button onClick={handleSave} disabled={saving} className="flex items-center gap-2 px-6 py-3 bg-primary-600 hover:bg-primary-500 text-white rounded-lg text-sm font-medium transition-all disabled:opacity-50">
          {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}Save Configuration
        </button>
      </div>
    </div>
  )
}
