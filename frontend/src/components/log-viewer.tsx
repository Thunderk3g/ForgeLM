"use client"

import type React from "react"
import { useRef, useEffect, useState, useMemo, useCallback } from "react"
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
  Circle,
  AlertTriangle,
  Info,
  Bug,
  Activity,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"

interface LogViewerProps {
  logs: string[]
  title?: string
  maxHeight?: string
  isConnected?: boolean
  onReconnect?: () => void
}

interface ParsedLog {
  id: string
  timestamp: string
  level: "INFO" | "WARN" | "ERROR" | "DEBUG" | "TRAIN" | "UNKNOWN"
  message: string
  rawContent: string
  lineNumber: number
}

type LogLevel = ParsedLog["level"]

const LOG_LEVEL_CONFIG: Record<
  LogLevel,
  {
    color: string
    bgColor: string
    icon: React.ComponentType<{ className?: string }>
    priority: number
  }
> = {
  ERROR: { color: "text-destructive", bgColor: "bg-destructive/10", icon: AlertTriangle, priority: 1 },
  WARN: { color: "text-secondary", bgColor: "bg-secondary/10", icon: AlertTriangle, priority: 2 },
  INFO: { color: "text-blue-500", bgColor: "bg-blue-50/10", icon: Info, priority: 3 },
  TRAIN: { color: "text-primary", bgColor: "bg-primary/10", icon: Activity, priority: 4 },
  DEBUG: { color: "text-purple-500", bgColor: "bg-purple-50/10", icon: Bug, priority: 5 },
  UNKNOWN: { color: "text-muted-foreground", bgColor: "bg-muted/10", icon: Circle, priority: 6 },
}

export const LogViewer: React.FC<LogViewerProps> = ({
  logs,
  title = "Training Logs",
  maxHeight = "400px",
  isConnected = true,
  onReconnect,
}) => {
  const logEndRef = useRef<HTMLDivElement>(null)
  const logContainerRef = useRef<HTMLDivElement>(null)

  const [searchTerm, setSearchTerm] = useState("")
  const [activeFilters, setActiveFilters] = useState<Set<LogLevel>>(new Set())
  const [isAutoscrollEnabled, setIsAutoscrollEnabled] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showFilters, setShowFilters] = useState(false)
  const [hoveredLine, setHoveredLine] = useState<number | null>(null)

  // Parse logs with enhanced structure
  const parsedLogs = useMemo((): ParsedLog[] => {
    return logs.map((log, index) => {
      const id = `log-${index}-${Date.now()}`

      // Enhanced log parsing with multiple patterns
      const patterns = [
        /^\[?(\d{4}-\d{2}-\d{2}[\s|T]\d{2}:\d{2}:\d{2}(?:\.\d{3})?)\]?\s*\[?(INFO|WARN|ERROR|DEBUG|TRAIN)\]?\s*:?\s*(.+)$/i,
        /^(\d{2}:\d{2}:\d{2})\s*\[?(INFO|WARN|ERROR|DEBUG|TRAIN)\]?\s*:?\s*(.+)$/i,
        /^\[?(INFO|WARN|ERROR|DEBUG|TRAIN)\]?\s*:?\s*(.+)$/i,
      ]

      for (const pattern of patterns) {
        const match = log.match(pattern)
        if (match) {
          const [, timestamp, level, message] = match
          return {
            id,
            timestamp: timestamp || new Date().toLocaleTimeString(),
            level: (level?.toUpperCase() as LogLevel) || "UNKNOWN",
            message: message || log,
            rawContent: log,
            lineNumber: index + 1,
          }
        }
      }

      // Fallback parsing
      return {
        id,
        timestamp: new Date().toLocaleTimeString(),
        level: "UNKNOWN",
        message: log,
        rawContent: log,
        lineNumber: index + 1,
      }
    })
  }, [logs])

  // Filter and search logic
  const filteredLogs = useMemo(() => {
    let filtered = parsedLogs

    // Apply level filters
    if (activeFilters.size > 0) {
      filtered = filtered.filter((log) => activeFilters.has(log.level))
    }

    // Apply search term
    if (searchTerm) {
      const search = searchTerm.toLowerCase()
      filtered = filtered.filter(
        (log) =>
          log.message.toLowerCase().includes(search) ||
          log.level.toLowerCase().includes(search) ||
          log.timestamp.toLowerCase().includes(search),
      )
    }

    return filtered
  }, [parsedLogs, activeFilters, searchTerm])

  // Auto-scroll functionality
  useEffect(() => {
    if (isAutoscrollEnabled && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [filteredLogs, isAutoscrollEnabled])

  // Enhanced download functionality
  const downloadLogs = useCallback(() => {
    const logContent = filteredLogs.map((log) => `[${log.timestamp}] [${log.level}] ${log.message}`).join("\n")

    const blob = new Blob([logContent], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `logs-${new Date().toISOString().slice(0, 19).replace(/[:.]/g, "-")}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [filteredLogs])

  // Copy log line functionality
  const copyLogLine = useCallback(async (log: ParsedLog) => {
    try {
      await navigator.clipboard.writeText(`[${log.timestamp}] [${log.level}] ${log.message}`)
    } catch (err) {
      // Fallback for older browsers
      const textArea = document.createElement("textarea")
      textArea.value = `[${log.timestamp}] [${log.level}] ${log.message}`
      document.body.appendChild(textArea)
      textArea.select()
      document.execCommand("copy")
      document.body.removeChild(textArea)
    }
  }, [])

  // Toggle filter
  const toggleFilter = useCallback((level: LogLevel) => {
    setActiveFilters((prev) => {
      const newSet = new Set(prev)
      if (newSet.has(level)) {
        newSet.delete(level)
      } else {
        newSet.add(level)
      }
      return newSet
    })
  }, [])

  // Get log level counts
  const logCounts = useMemo(() => {
    const counts: Record<LogLevel, number> = {
      ERROR: 0,
      WARN: 0,
      INFO: 0,
      TRAIN: 0,
      DEBUG: 0,
      UNKNOWN: 0,
    }
    parsedLogs.forEach((log) => counts[log.level]++)
    return counts
  }, [parsedLogs])

  return (
    <Card className={`transition-all duration-300 ${isFullscreen ? "fixed inset-4 z-50" : ""}`}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <div className="rounded-md bg-primary/10 p-2">
                <Terminal className="h-4 w-4 text-primary" />
              </div>
              <div>
                <CardTitle className="text-lg">{title}</CardTitle>
                <p className="text-sm text-muted-foreground">Real-time training logs</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {isConnected ? (
                <Badge variant="secondary" className="bg-primary/10 text-primary">
                  <div className="mr-2 h-2 w-2 rounded-full bg-primary" />
                  Live
                </Badge>
              ) : (
                <div className="flex items-center space-x-2">
                  <div className="h-2 w-2 rounded-full bg-destructive" />
                  <Button variant="ghost" size="sm" onClick={onReconnect} className="text-destructive">
                    Reconnect
                  </Button>
                </div>
              )}

              <div className="flex items-center space-x-2 text-muted-foreground">
                <Activity className="h-3 w-3" />
                <span className="text-xs font-mono">
                  {filteredLogs.length}/{logs.length}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-1">
            <Button variant="ghost" size="sm" onClick={() => setShowFilters(!showFilters)} className="relative">
              <Filter className="h-4 w-4" />
              {activeFilters.size > 0 && (
                <Badge variant="destructive" className="absolute -right-1 -top-1 h-4 w-4 p-0 text-xs">
                  {activeFilters.size}
                </Badge>
              )}
            </Button>

            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsAutoscrollEnabled(!isAutoscrollEnabled)}
              className={isAutoscrollEnabled ? "text-primary" : ""}
            >
              {isAutoscrollEnabled ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>

            <Button variant="ghost" size="sm" onClick={() => setSearchTerm("")}>
              <Trash2 className="h-4 w-4" />
            </Button>

            {logs.length > 0 && (
              <Button variant="ghost" size="sm" onClick={downloadLogs}>
                <Download className="h-4 w-4" />
              </Button>
            )}

            <Button variant="ghost" size="sm" onClick={() => setIsFullscreen(!isFullscreen)}>
              {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            </Button>
          </div>
        </div>

        {/* Search Bar */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search logs..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>

        {/* Filter Dropdown */}
        {showFilters && (
          <div className="space-y-2 rounded-md border bg-card p-3">
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(LOG_LEVEL_CONFIG).map(([level, config]) => {
                const Icon = config.icon
                const count = logCounts[level as LogLevel]
                return (
                  <Button
                    key={level}
                    variant={activeFilters.has(level as LogLevel) ? "default" : "outline"}
                    size="sm"
                    onClick={() => toggleFilter(level as LogLevel)}
                    className="justify-between"
                  >
                    <div className="flex items-center space-x-2">
                      <Icon className={`h-3 w-3 ${config.color}`} />
                      <span>{level}</span>
                    </div>
                    <span className="text-xs">{count}</span>
                  </Button>
                )
              })}
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent className="p-0">
        <div
          ref={logContainerRef}
          className="bg-background font-mono text-sm overflow-y-auto relative"
          style={{
            maxHeight: isFullscreen ? "calc(100vh - 300px)" : maxHeight,
            minHeight: "200px",
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
                const config = LOG_LEVEL_CONFIG[log.level]
                const Icon = config.icon

                return (
                  <div
                    key={log.id}
                    className={`group flex items-start space-x-3 rounded-sm px-2 py-1 transition-all duration-200 hover:bg-accent/50 ${config.bgColor}`}
                    onMouseEnter={() => setHoveredLine(index)}
                    onMouseLeave={() => setHoveredLine(null)}
                  >
                    {/* Line number */}
                    <div className="w-12 flex-shrink-0 text-right text-xs text-muted-foreground leading-relaxed">
                      {log.lineNumber}
                    </div>

                    {/* Timestamp */}
                    <div className="flex-shrink-0 text-xs text-muted-foreground leading-relaxed">{log.timestamp}</div>

                    {/* Level badge */}
                    <div className={`flex flex-shrink-0 items-center space-x-1 ${config.color}`}>
                      <Icon className="h-3 w-3" />
                      <span className="min-w-12 text-xs font-medium">{log.level}</span>
                    </div>

                    {/* Message */}
                    <div className="flex-1 whitespace-pre-wrap break-words leading-relaxed">{log.message}</div>

                    {/* Copy button */}
                    {hoveredLine === index && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => copyLogLine(log)}
                        className="flex-shrink-0 p-1 opacity-0 transition-opacity group-hover:opacity-100"
                      >
                        <Copy className="h-3 w-3" />
                      </Button>
                    )}
                  </div>
                )
              })}
              <div ref={logEndRef} />
            </div>
          )}

          {/* Streaming indicator */}
          {isConnected && logs.length > 0 && (
            <div className="absolute bottom-2 right-2 flex items-center space-x-2 rounded-md bg-background/95 border px-3 py-1 text-xs backdrop-blur-sm">
              <div className="h-2 w-2 animate-pulse rounded-full bg-primary"></div>
              <span className="font-medium text-primary">Live</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
