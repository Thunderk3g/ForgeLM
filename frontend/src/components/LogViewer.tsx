import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import { 
  Terminal, 
  Download, 
  Search, 
  Filter,
  Trash2,
  Play,
  Pause,
  Maximize2,
  Minimize2,
  Copy,
  Wifi,
  WifiOff,
  Circle,
  AlertTriangle,
  Info,
  Bug,
  Activity
} from 'lucide-react';

interface LogViewerProps {
  logs: string[];
  title?: string;
  maxHeight?: string;
  isConnected?: boolean;
  onReconnect?: () => void;
}

interface ParsedLog {
  id: string;
  timestamp: string;
  level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG' | 'TRAIN' | 'UNKNOWN';
  message: string;
  rawContent: string;
  lineNumber: number;
}

type LogLevel = ParsedLog['level'];

const LOG_LEVEL_CONFIG: Record<LogLevel, { 
  color: string; 
  bgColor: string; 
  icon: React.ComponentType<{ className?: string }>; 
  priority: number;
}> = {
  ERROR: { color: 'text-destructive', bgColor: 'bg-destructive/10', icon: AlertTriangle, priority: 1 },
  WARN: { color: 'text-yellow-500', bgColor: 'bg-yellow-50/10', icon: AlertTriangle, priority: 2 },
  INFO: { color: 'text-blue-500', bgColor: 'bg-blue-50/10', icon: Info, priority: 3 },
  TRAIN: { color: 'text-green-500', bgColor: 'bg-green-50/10', icon: Activity, priority: 4 },
  DEBUG: { color: 'text-purple-500', bgColor: 'bg-purple-50/10', icon: Bug, priority: 5 },
  UNKNOWN: { color: 'text-muted-foreground', bgColor: 'bg-muted/10', icon: Circle, priority: 6 }
};

export const LogViewer: React.FC<LogViewerProps> = ({ 
  logs, 
  title = "Training Logs", 
  maxHeight = "400px",
  isConnected = true,
  onReconnect
}) => {
  const logEndRef = useRef<HTMLDivElement>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);
  
  // Enhanced state management
  const [searchTerm, setSearchTerm] = useState('');
  const [activeFilters, setActiveFilters] = useState<Set<LogLevel>>(new Set());
  const [isAutoscrollEnabled, setIsAutoscrollEnabled] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [hoveredLine, setHoveredLine] = useState<number | null>(null);
  const [newLogAnimation, setNewLogAnimation] = useState<number | null>(null);

  // Parse logs with enhanced structure
  const parsedLogs = useMemo((): ParsedLog[] => {
    return logs.map((log, index) => {
      const id = `log-${index}-${Date.now()}`;
      
      // Enhanced log parsing with multiple patterns
      const patterns = [
        /^\[?(\d{4}-\d{2}-\d{2}[\s|T]\d{2}:\d{2}:\d{2}(?:\.\d{3})?)\]?\s*\[?(INFO|WARN|ERROR|DEBUG|TRAIN)\]?\s*:?\s*(.+)$/i,
        /^(\d{2}:\d{2}:\d{2})\s*\[?(INFO|WARN|ERROR|DEBUG|TRAIN)\]?\s*:?\s*(.+)$/i,
        /^\[?(INFO|WARN|ERROR|DEBUG|TRAIN)\]?\s*:?\s*(.+)$/i
      ];
      
      for (const pattern of patterns) {
        const match = log.match(pattern);
        if (match) {
          const [, timestamp, level, message] = match;
          return {
            id,
            timestamp: timestamp || new Date().toLocaleTimeString(),
            level: (level?.toUpperCase() as LogLevel) || 'UNKNOWN',
            message: message || log,
            rawContent: log,
            lineNumber: index + 1
          };
        }
      }
      
      // Fallback parsing
      return {
        id,
        timestamp: new Date().toLocaleTimeString(),
        level: 'UNKNOWN',
        message: log,
        rawContent: log,
        lineNumber: index + 1
      };
    });
  }, [logs]);

  // Filter and search logic
  const filteredLogs = useMemo(() => {
    let filtered = parsedLogs;
    
    // Apply level filters
    if (activeFilters.size > 0) {
      filtered = filtered.filter(log => activeFilters.has(log.level));
    }
    
    // Apply search term
    if (searchTerm) {
      const search = searchTerm.toLowerCase();
      filtered = filtered.filter(log => 
        log.message.toLowerCase().includes(search) ||
        log.level.toLowerCase().includes(search) ||
        log.timestamp.toLowerCase().includes(search)
      );
    }
    
    return filtered;
  }, [parsedLogs, activeFilters, searchTerm]);

  // Auto-scroll functionality
  useEffect(() => {
    if (isAutoscrollEnabled && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [filteredLogs, isAutoscrollEnabled]);

  // New log animation
  useEffect(() => {
    if (logs.length > 0) {
      const lastIndex = logs.length - 1;
      setNewLogAnimation(lastIndex);
      
      const timer = setTimeout(() => {
        setNewLogAnimation(null);
      }, 500);
      
      return () => clearTimeout(timer);
    }
  }, [logs.length]);

  // Enhanced download functionality
  const downloadLogs = useCallback(() => {
    const logContent = filteredLogs.map(log => 
      `[${log.timestamp}] [${log.level}] ${log.message}`
    ).join('\n');
    
    const blob = new Blob([logContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `logs-${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [filteredLogs]);

  // Copy log line functionality
  const copyLogLine = useCallback(async (log: ParsedLog) => {
    try {
      await navigator.clipboard.writeText(`[${log.timestamp}] [${log.level}] ${log.message}`);
    } catch (err) {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = `[${log.timestamp}] [${log.level}] ${log.message}`;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
    }
  }, []);

  // Clear logs functionality
  const clearLogs = useCallback(() => {
    // This would need to be implemented in the parent component
    // For now, we'll just clear the search and filters
    setSearchTerm('');
    setActiveFilters(new Set());
  }, []);

  // Toggle filter
  const toggleFilter = useCallback((level: LogLevel) => {
    setActiveFilters(prev => {
      const newSet = new Set(prev);
      if (newSet.has(level)) {
        newSet.delete(level);
      } else {
        newSet.add(level);
      }
      return newSet;
    });
  }, []);

  // Get log level counts
  const logCounts = useMemo(() => {
    const counts: Record<LogLevel, number> = {
      ERROR: 0, WARN: 0, INFO: 0, TRAIN: 0, DEBUG: 0, UNKNOWN: 0
    };
    parsedLogs.forEach(log => counts[log.level]++);
    return counts;
  }, [parsedLogs]);

  return (
    <div className={`rounded-lg border bg-card transition-all duration-300 ${
      isFullscreen ? 'fixed inset-4 z-50' : ''
    }`}>
      {/* Terminal Header */}
      <div className="flex items-center justify-between border-b p-4">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-3">
            <div className="rounded-md bg-primary/10 p-2">
              <Terminal className="h-4 w-4 text-primary" />
            </div>
            <div>
              <h3 className="font-semibold tracking-tight">{title}</h3>
              <p className="text-sm text-muted-foreground">Real-time training logs</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            {isConnected ? (
              <div className="flex items-center space-x-2 rounded-full bg-green-50/10 px-3 py-1">
                <div className="h-2 w-2 rounded-full bg-green-500" />
                <span className="text-xs font-medium text-green-500">Live</span>
              </div>
            ) : (
              <div className="flex items-center space-x-2">
                <div className="h-2 w-2 rounded-full bg-destructive" />
                <button 
                  onClick={onReconnect}
                  className="text-xs font-medium text-destructive hover:text-destructive/80 transition-colors"
                >
                  Reconnect
                </button>
              </div>
            )}
            
            <div className="flex items-center space-x-2 text-muted-foreground">
              <Activity className="h-3 w-3" />
              <span className="text-xs font-mono">{filteredLogs.length}/{logs.length}</span>
            </div>
          </div>
        </div>
        
        {/* Action Buttons */}
        <div className="flex items-center space-x-1">
          <div className="relative">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="btn-ghost p-2"
              title="Toggle filters"
            >
              <Filter className="h-4 w-4" />
            </button>
            
            {/* Filter Dropdown */}
            {showFilters && (
              <div className="absolute right-0 top-full mt-2 min-w-48 rounded-md border bg-popover p-2 shadow-lg z-10">
                <div className="space-y-1">
                  {Object.entries(LOG_LEVEL_CONFIG).map(([level, config]) => {
                    const Icon = config.icon;
                    const count = logCounts[level as LogLevel];
                    return (
                      <button
                        key={level}
                        onClick={() => toggleFilter(level as LogLevel)}
                        className={`w-full flex items-center justify-between px-3 py-2 rounded-sm text-sm transition-colors ${
                          activeFilters.has(level as LogLevel)
                            ? 'bg-accent text-accent-foreground'
                            : 'hover:bg-accent/50'
                        }`}
                      >
                        <div className="flex items-center space-x-2">
                          <Icon className={`h-3 w-3 ${config.color}`} />
                          <span>{level}</span>
                        </div>
                        <span className="text-xs text-muted-foreground">{count}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
          
          <button
            onClick={() => setIsAutoscrollEnabled(!isAutoscrollEnabled)}
            className={`btn-ghost p-2 ${
              isAutoscrollEnabled ? 'text-green-500 hover:text-green-600' : ''
            }`}
            title={isAutoscrollEnabled ? 'Disable autoscroll' : 'Enable autoscroll'}
          >
            {isAutoscrollEnabled ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </button>
          
          <button
            onClick={clearLogs}
            className="btn-ghost p-2"
            title="Clear filters"
          >
            <Trash2 className="h-4 w-4" />
          </button>
          
          {logs.length > 0 && (
            <button
              onClick={downloadLogs}
              className="btn-ghost p-2"
              title="Download logs"
            >
              <Download className="h-4 w-4" />
            </button>
          )}
          
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="btn-ghost p-2"
            title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
          >
            {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </button>
        </div>
      </div>
      
      {/* Search Bar */}
      <div className="border-b p-3">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search logs..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input pl-10"
          />
        </div>
      </div>
      
      {/* Log Content */}
      <div 
        ref={logContainerRef}
        className="bg-background font-mono text-sm overflow-y-auto relative"
        style={{ 
          maxHeight: isFullscreen ? 'calc(100vh - 200px)' : maxHeight,
          minHeight: '200px'
        }}
      >
        {filteredLogs.length === 0 ? (
          <div className="flex h-32 items-center justify-center text-muted-foreground">
            {logs.length === 0 ? (
              <div className="text-center space-y-2">
                <Terminal className="mx-auto h-8 w-8 opacity-50" />
                <div>No logs available yet...</div>
                <div className="text-xs">Waiting for training to start</div>
              </div>
            ) : (
              <div className="text-center space-y-2">
                <Search className="mx-auto h-8 w-8 opacity-50" />
                <div>No logs match your filters</div>
                <div className="text-xs">Try adjusting your search or filters</div>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-1 p-4">
            {filteredLogs.map((log, index) => {
              const config = LOG_LEVEL_CONFIG[log.level];
              const Icon = config.icon;
              const isNewLog = newLogAnimation === log.lineNumber - 1;
              
              return (
                <div
                  key={log.id}
                  className={`group flex items-start space-x-3 rounded-sm px-2 py-1 transition-all duration-200 ${
                    isNewLog ? 'animate-pulse bg-blue-50/10' : 'hover:bg-accent/50'
                  } ${config.bgColor}`}
                  onMouseEnter={() => setHoveredLine(index)}
                  onMouseLeave={() => setHoveredLine(null)}
                >
                  {/* Line number */}
                  <div className="w-12 flex-shrink-0 text-right text-xs text-muted-foreground leading-relaxed">
                    {log.lineNumber}
                  </div>
                  
                  {/* Timestamp */}
                  <div className="flex-shrink-0 text-xs text-muted-foreground leading-relaxed">
                    {log.timestamp}
                  </div>
                  
                  {/* Level badge */}
                  <div className={`flex flex-shrink-0 items-center space-x-1 ${config.color}`}>
                    <Icon className="h-3 w-3" />
                    <span className="min-w-12 text-xs font-medium">{log.level}</span>
                  </div>
                  
                  {/* Message */}
                  <div className="flex-1 whitespace-pre-wrap break-words leading-relaxed">
                    {log.message}
                  </div>
                  
                  {/* Copy button */}
                  {hoveredLine === index && (
                    <button
                      onClick={() => copyLogLine(log)}
                      className="flex-shrink-0 p-1 opacity-0 transition-opacity group-hover:opacity-100 btn-ghost"
                      title="Copy log line"
                    >
                      <Copy className="h-3 w-3" />
                    </button>
                  )}
                </div>
              );
            })}
            <div ref={logEndRef} />
          </div>
        )}
        
        {/* Streaming indicator */}
        {isConnected && logs.length > 0 && (
          <div className="absolute bottom-2 right-2 flex items-center space-x-2 rounded-md bg-background/95 border px-3 py-1 text-xs backdrop-blur-sm">
            <div className="h-2 w-2 animate-pulse rounded-full bg-green-500"></div>
            <span className="font-medium text-green-600">Live</span>
          </div>
        )}
      </div>
    </div>
  );
};