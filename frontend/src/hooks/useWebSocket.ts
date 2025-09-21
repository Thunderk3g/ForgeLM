import { useState, useEffect, useRef, useCallback } from 'react';

export interface WebSocketMessage {
  type: string;
  timestamp: string;
  session_id?: string;
  [key: string]: any;
}

export interface UseWebSocketOptions {
  url: string;
  onMessage?: (message: WebSocketMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export interface UseWebSocketReturn {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  sendMessage: (message: any) => void;
  lastMessage: WebSocketMessage | null;
  connectionAttempts: number;
  disconnect: () => void;
  reconnect: () => void;
}

export const useWebSocket = (options: UseWebSocketOptions): UseWebSocketReturn => {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [connectionAttempts, setConnectionAttempts] = useState(0);

  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const mountedRef = useRef(true);

  const {
    url,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5
  } = options;

  const connect = useCallback(() => {
    if (!mountedRef.current || isConnecting || isConnected) return;

    setIsConnecting(true);
    setError(null);

    try {
      // Create WebSocket URL (convert http to ws)
      const wsUrl = url.replace(/^http/, 'ws');
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        if (!mountedRef.current) return;
        
        setIsConnected(true);
        setIsConnecting(false);
        setConnectionAttempts(0);
        setError(null);
        onConnect?.();
      };

      ws.current.onmessage = (event) => {
        if (!mountedRef.current) return;
        
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setLastMessage(message);
          onMessage?.(message);
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.current.onclose = (event) => {
        if (!mountedRef.current) return;
        
        setIsConnected(false);
        setIsConnecting(false);
        onDisconnect?.();

        // Only attempt reconnection if it wasn't a clean close and we haven't exceeded max attempts
        if (event.code !== 1000 && connectionAttempts < maxReconnectAttempts) {
          setConnectionAttempts(prev => prev + 1);
          reconnectTimeoutRef.current = setTimeout(() => {
            if (mountedRef.current) {
              connect();
            }
          }, reconnectInterval);
        }
      };

      ws.current.onerror = (event) => {
        if (!mountedRef.current) return;
        
        setError('WebSocket connection error');
        setIsConnecting(false);
        onError?.(event);
      };

    } catch (err) {
      setError('Failed to create WebSocket connection');
      setIsConnecting(false);
    }
  }, [url, isConnecting, isConnected, connectionAttempts, maxReconnectAttempts, reconnectInterval, onMessage, onConnect, onDisconnect, onError]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (ws.current) {
      ws.current.close(1000); // Clean close
      ws.current = null;
    }
    
    setIsConnected(false);
    setIsConnecting(false);
  }, []);

  const reconnect = useCallback(() => {
    disconnect();
    setConnectionAttempts(0);
    setTimeout(connect, 100);
  }, [disconnect, connect]);

  const sendMessage = useCallback((message: any) => {
    if (ws.current && isConnected) {
      try {
        ws.current.send(JSON.stringify(message));
      } catch (err) {
        console.error('Failed to send WebSocket message:', err);
        setError('Failed to send message');
      }
    } else {
      console.warn('WebSocket is not connected');
    }
  }, [isConnected]);

  // Initial connection
  useEffect(() => {
    connect();
    
    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, []);

  // Ping to keep connection alive
  useEffect(() => {
    if (!isConnected) return;

    const pingInterval = setInterval(() => {
      sendMessage({ type: 'ping' });
    }, 30000); // Ping every 30 seconds

    return () => clearInterval(pingInterval);
  }, [isConnected, sendMessage]);

  return {
    isConnected,
    isConnecting,
    error,
    sendMessage,
    lastMessage,
    connectionAttempts,
    disconnect,
    reconnect
  };
};