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
  ERROR: { color: 'text-red-400', bgColor: 'bg-red-950/30', icon: AlertTriangle, priority: 1 },
  WARN: { color: 'text-yellow-400', bgColor: 'bg-yellow-950/20', icon: AlertTriangle, priority: 2 },
  INFO: { color: 'text-blue-400', bgColor: 'bg-blue-950/20', icon: Info, priority: 3 },
  TRAIN: { color: 'text-green-400', bgColor: 'bg-green-950/20', icon: Activity, priority: 4 },
  DEBUG: { color: 'text-purple-400', bgColor: 'bg-purple-950/20', icon: Bug, priority: 5 },
  UNKNOWN: { color: 'text-gray-400', bgColor: 'bg-gray-950/10', icon: Circle, priority: 6 }
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
  const [isAutoscrollEnabled, setIsAutoscrollEnabled] = useState(true);
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
    <div className={`bg-slate-900 rounded-lg shadow-lg border border-slate-700 transition-all duration-300 ${
      isFullscreen ? 'fixed inset-4 z-50' : ''
    }`}>
      {/* Terminal Header */}
      <div className="flex items-center justify-between p-3 border-b border-slate-700 bg-slate-800 rounded-t-lg">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <Terminal className="w-5 h-5 text-slate-300" />
            <h3 className="font-semibold text-slate-100">{title}</h3>
          </div>
          
          <div className="flex items-center space-x-2">
            {isConnected ? (
              <div className="flex items-center space-x-1 text-green-400">
                <Wifi className="w-4 h-4" />
                <span className="text-xs">Connected</span>
              </div>
            ) : (
              <div className="flex items-center space-x-1 text-red-400">
                <WifiOff className="w-4 h-4" />
                <button 
                  onClick={onReconnect}
                  className="text-xs hover:text-red-300 transition-colors"
                >
                  Reconnect
                </button>
              </div>
            )}
            
            <div className="flex items-center space-x-1 text-slate-400">
              <Activity className="w-3 h-3" />
              <span className="text-xs">{filteredLogs.length}/{logs.length}</span>
            </div>
          </div>
        </div>
        
        {/* Action Buttons */}
        <div className="flex items-center space-x-1">
          <div className="relative">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded transition-colors"
              title="Toggle filters"
            >
              <Filter className="w-4 h-4" />
            </button>
            
            {/* Filter Dropdown */}
            {showFilters && (
              <div className="absolute right-0 top-full mt-2 bg-slate-800 border border-slate-600 rounded-md shadow-lg z-10 p-2 min-w-48">
                <div className="space-y-1">
                  {Object.entries(LOG_LEVEL_CONFIG).map(([level, config]) => {
                    const Icon = config.icon;
                    const count = logCounts[level as LogLevel];
                    return (
                      <button
                        key={level}
                        onClick={() => toggleFilter(level as LogLevel)}
                        className={`w-full flex items-center justify-between px-2 py-1 rounded text-sm transition-colors ${
                          activeFilters.has(level as LogLevel)
                            ? 'bg-slate-600 text-slate-100'
                            : 'text-slate-300 hover:bg-slate-700'
                        }`}
                      >
                        <div className="flex items-center space-x-2">
                          <Icon className={`w-3 h-3 ${config.color}`} />
                          <span>{level}</span>
                        </div>
                        <span className="text-xs opacity-60">{count}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
          
          <button
            onClick={() => setIsAutoscrollEnabled(!isAutoscrollEnabled)}
            className={`p-2 rounded transition-colors ${
              isAutoscrollEnabled 
                ? 'text-green-400 hover:text-green-300 hover:bg-slate-700' 
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
            }`}
            title={isAutoscrollEnabled ? 'Disable autoscroll' : 'Enable autoscroll'}
          >
            {isAutoscrollEnabled ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          
          <button
            onClick={clearLogs}
            className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded transition-colors"
            title="Clear filters"
          >
            <Trash2 className="w-4 h-4" />
          </button>
          
          {logs.length > 0 && (
            <button
              onClick={downloadLogs}
              className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded transition-colors"
              title="Download logs"
            >
              <Download className="w-4 h-4" />
            </button>
          )}
          
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded transition-colors"
            title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>
      
      {/* Search Bar */}
      <div className="p-3 border-b border-slate-700 bg-slate-800">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search logs..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-slate-700 border border-slate-600 rounded-md text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
          />
        </div>
      </div>
      
      {/* Log Content */}
      <div 
        ref={logContainerRef}
        className="bg-slate-950 text-slate-100 font-mono text-sm overflow-y-auto relative"
        style={{ 
          maxHeight: isFullscreen ? 'calc(100vh - 200px)' : maxHeight,
          minHeight: '200px'
        }}
      >
        {filteredLogs.length === 0 ? (
          <div className="flex items-center justify-center h-32 text-slate-500">
            {logs.length === 0 ? (
              <div className="text-center">
                <Terminal className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <div>No logs available yet...</div>
                <div className="text-xs mt-1">Waiting for training to start</div>
              </div>
            ) : (
              <div className="text-center">
                <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <div>No logs match your filters</div>
                <div className="text-xs mt-1">Try adjusting your search or filters</div>
              </div>
            )}
          </div>
        ) : (
          <div className="p-4 space-y-1">
            {filteredLogs.map((log, index) => {
              const config = LOG_LEVEL_CONFIG[log.level];
              const Icon = config.icon;
              const isNewLog = newLogAnimation === log.lineNumber - 1;
              
              return (
                <div
                  key={log.id}
                  className={`group flex items-start space-x-3 py-1 px-2 rounded transition-all duration-200 ${
                    isNewLog ? 'animate-pulse bg-blue-950/30' : 'hover:bg-slate-800/50'
                  } ${config.bgColor}`}
                  onMouseEnter={() => setHoveredLine(index)}
                  onMouseLeave={() => setHoveredLine(null)}
                >
                  {/* Line number */}
                  <div className="flex-shrink-0 w-12 text-right text-slate-500 text-xs leading-relaxed">
                    {log.lineNumber}
                  </div>
                  
                  {/* Timestamp */}
                  <div className="flex-shrink-0 text-slate-400 text-xs leading-relaxed">
                    {log.timestamp}
                  </div>
                  
                  {/* Level badge */}
                  <div className={`flex-shrink-0 flex items-center space-x-1 ${config.color}`}>
                    <Icon className="w-3 h-3" />
                    <span className="text-xs font-medium min-w-12">{log.level}</span>
                  </div>
                  
                  {/* Message */}
                  <div className="flex-1 whitespace-pre-wrap break-words leading-relaxed">
                    {log.message}
                  </div>
                  
                  {/* Copy button */}
                  {hoveredLine === index && (
                    <button
                      onClick={() => copyLogLine(log)}
                      className="flex-shrink-0 p-1 text-slate-400 hover:text-slate-200 opacity-0 group-hover:opacity-100 transition-opacity"
                      title="Copy log line"
                    >
                      <Copy className="w-3 h-3" />
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
          <div className="absolute bottom-2 right-2 flex items-center space-x-1 bg-slate-800/80 backdrop-blur-sm px-2 py-1 rounded text-xs text-slate-400">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span>Live</span>
          </div>
        )}
      </div>
    </div>
  );
};