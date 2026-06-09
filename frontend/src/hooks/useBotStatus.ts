import { useState, useEffect, useCallback } from 'react'
import { botApi } from '../api/client'
import type { BotStatus } from '../types'

export function useBotStatus(pollInterval: number = 5000) {
  const [status, setStatus] = useState<BotStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const fetchStatus = useCallback(async () => {
    try { const data = await botApi.status(); setStatus(data) } catch (err) { console.error(err) }
    finally { setLoading(false) }
  }, [])
  useEffect(() => { fetchStatus(); const interval = setInterval(fetchStatus, pollInterval); return () => clearInterval(interval) }, [fetchStatus, pollInterval])
  return { status, loading, refetch: fetchStatus }
}
