import { useEffect, useRef, useState, useCallback } from 'react'

export function useWebSocket({ url, onMessage, reconnectInterval = 3000, enabled = true }: { url: string; onMessage: (data: unknown) => void; reconnectInterval?: number; enabled?: boolean }) {
  const [isConnected, setIsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>()
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  const connect = useCallback(() => {
    if (!enabled) return
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}${url}`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws
    ws.onopen = () => setIsConnected(true)
    ws.onmessage = (event) => { try { onMessageRef.current(JSON.parse(event.data)) } catch (err) { console.error('WS parse error:', err) } }
    ws.onclose = () => { setIsConnected(false); reconnectTimeoutRef.current = setTimeout(connect, reconnectInterval) }
    ws.onerror = () => ws.close()
  }, [url, reconnectInterval, enabled])

  useEffect(() => { connect(); return () => { if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current); if (wsRef.current) wsRef.current.close() } }, [connect])
  return { isConnected }
}
