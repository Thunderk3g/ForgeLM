"use client"

import type React from "react"
import { useState, useEffect } from "react"
import {
  Activity,
  Clock,
  Zap,
  Play,
  Pause,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  XCircle,
  ArrowRight,
  FileText,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { getActiveTrainingSessions } from "../lib/api"

interface ActiveSession {
  task_id: string
  status: string
  started_at: string
  request_data: any
  last_activity: string
  current_status?: {
    task_id: string
    status: string
    progress: string
    current_epoch?: number
    total_epochs?: number
    current_step?: number
    total_steps?: number
    loss?: number
    learning_rate?: number
    progress_percentage?: number
    estimated_remaining_minutes?: number
  }
}

interface ActiveSessionsPanelProps {
  onConnectToSession: (taskId: string) => void
  onViewLogs?: (taskId: string) => void
  onCancelTraining?: (taskId: string) => void
  currentTaskId?: string | null
  refreshTrigger?: number
}

export const ActiveSessionsPanel: React.FC<ActiveSessionsPanelProps> = ({
  onConnectToSession,
  onViewLogs,
  onCancelTraining,
  currentTaskId,
  refreshTrigger = 0,
}) => {
  const [activeSessions, setActiveSessions] = useState<Record<string, ActiveSession>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<string>("")

  const loadActiveSessions = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await getActiveTrainingSessions()
      if (response.status === "success") {
        setActiveSessions(response.active_sessions)
        setLastUpdated(new Date().toLocaleTimeString())
      } else {
        setError("Failed to load active sessions")
      }
    } catch (err) {
      setError("Error connecting to server")
      console.error("Error loading active sessions:", err)
    } finally {
      setLoading(false)
    }
  }

  // Load sessions on mount and when refresh trigger changes
  useEffect(() => {
    loadActiveSessions()
  }, [refreshTrigger])

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(loadActiveSessions, 30000)
    return () => clearInterval(interval)
  }, [])

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case "running":
        return <Play className="w-4 h-4 text-primary" />
      case "completed":
        return <CheckCircle className="w-4 h-4 text-blue-400" />
      case "failed":
      case "error":
        return <XCircle className="w-4 h-4 text-destructive" />
      default:
        return <AlertCircle className="w-4 h-4 text-secondary" />
    }
  }

  const getProgressInfo = (session: ActiveSession) => {
    const current = session.current_status
    if (!current) return { progress: 0, text: "No status available" }

    if (
      current.progress_percentage !== undefined &&
      current.progress_percentage !== null &&
      typeof current.progress_percentage === "number"
    ) {
      return {
        progress: current.progress_percentage,
        text: `${current.progress_percentage.toFixed(1)}%`,
      }
    }

    if (current.current_step && current.total_steps) {
      const progress = (current.current_step / current.total_steps) * 100
      return {
        progress,
        text: `${current.current_step}/${current.total_steps} steps`,
      }
    }

    if (current.current_epoch && current.total_epochs) {
      const progress = (current.current_epoch / current.total_epochs) * 100
      return {
        progress,
        text: `${Number(current.current_epoch).toFixed(1)}/${current.total_epochs} epochs`,
      }
    }

    return { progress: 0, text: "Starting..." }
  }

  const formatTime = (timeString: string) => {
    try {
      return new Date(timeString).toLocaleString()
    } catch {
      return timeString
    }
  }

  const sessionCount = Object.keys(activeSessions).length

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="rounded-md bg-primary/10 p-2">
              <Activity className="h-5 w-5 text-primary" />
            </div>
            <div>
              <CardTitle className="text-lg">Active Training Sessions</CardTitle>
              <p className="text-sm text-muted-foreground">
                {sessionCount} active session{sessionCount !== 1 ? "s" : ""}
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {lastUpdated && <span className="text-xs text-muted-foreground">Updated: {lastUpdated}</span>}

            <Button variant="outline" size="sm" onClick={loadActiveSessions} disabled={loading}>
              <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {error && (
          <div className="mb-4 flex items-center space-x-2 rounded-md border border-destructive/20 bg-destructive/10 p-3">
            <AlertCircle className="h-4 w-4 text-destructive" />
            <span className="text-sm text-destructive">{error}</span>
          </div>
        )}

        {loading && sessionCount === 0 && (
          <div className="flex items-center justify-center py-12 text-muted-foreground">
            <RefreshCw className="mr-2 h-5 w-5 animate-spin" />
            <span>Loading active sessions...</span>
          </div>
        )}

        {!loading && sessionCount === 0 && !error && (
          <div className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground">
            <div className="rounded-full bg-muted p-3 mb-4">
              <Pause className="h-6 w-6 opacity-50" />
            </div>
            <h4 className="font-medium text-foreground mb-1">No active training sessions</h4>
            <p className="text-sm">Start a new training to see it here</p>
          </div>
        )}

        {sessionCount > 0 && (
          <div className="space-y-3">
            {Object.entries(activeSessions).map(([taskId, session]) => {
              const progressInfo = getProgressInfo(session)
              const isCurrentSession = currentTaskId === taskId
              const isRunning = session.current_status?.status === "running"

              return (
                <div
                  key={taskId}
                  className={`rounded-lg border p-4 transition-all ${
                    isCurrentSession ? "border-primary bg-primary/5" : "border-border hover:bg-accent/50"
                  }`}
                >
                  {/* Session Header */}
                  <div className="mb-4 flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="flex-shrink-0">
                        {getStatusIcon(session.current_status?.status || session.status)}
                      </div>

                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium truncate">Task {taskId.slice(0, 8)}...</p>
                        <p className="text-xs text-muted-foreground truncate">
                          {session.request_data?.base_model_dir?.split("/").pop() || "Unknown model"}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center space-x-2">
                      {isCurrentSession && <Badge variant="default">Connected</Badge>}

                      {/* View Logs Button - available for all sessions */}
                      {onViewLogs && (
                        <Button variant="outline" size="sm" onClick={() => onViewLogs(taskId)}>
                          <FileText className="mr-1 h-3 w-3" />
                          Logs
                        </Button>
                      )}

                      {!isCurrentSession && isRunning && (
                        <Button size="sm" onClick={() => onConnectToSession(taskId)}>
                          <ArrowRight className="mr-1 h-3 w-3" />
                          Connect
                        </Button>
                      )}

                      {/* Cancel Button - for running sessions only */}
                      {onCancelTraining &&
                        (session.current_status?.status === "running" || session.status === "running") && (
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick={() => {
                              if (
                                window.confirm(
                                  `Are you sure you want to cancel training session ${taskId.slice(
                                    0,
                                    8,
                                  )}...?\n\nThe training will stop at the next safe checkpoint to preserve data integrity.`,
                                )
                              ) {
                                onCancelTraining(taskId)
                              }
                            }}
                          >
                            <XCircle className="mr-1 h-3 w-3" />
                            Cancel
                          </Button>
                        )}
                    </div>
                  </div>

                  {/* Progress Bar */}
                  {progressInfo.progress > 0 && (
                    <div className="mb-4 space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">Progress</span>
                        <span className="font-medium">{progressInfo.text}</span>
                      </div>
                      <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
                        <div
                          className="h-full bg-primary transition-all duration-300 ease-out"
                          style={{ width: `${Math.min(progressInfo.progress, 100)}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Session Details */}
                  <div className="grid grid-cols-2 gap-4 text-xs">
                    <div className="space-y-1">
                      <div className="flex items-center space-x-1 text-muted-foreground">
                        <Clock className="h-3 w-3" />
                        <span>Started</span>
                      </div>
                      <p className="font-medium">{formatTime(session.started_at)}</p>
                    </div>

                    <div className="space-y-1">
                      <div className="flex items-center space-x-1 text-muted-foreground">
                        <Zap className="h-3 w-3" />
                        <span>Status</span>
                      </div>
                      <p className="font-medium">{session.current_status?.progress || "Running..."}</p>
                    </div>

                    {session.current_status?.loss && (
                      <div className="space-y-1">
                        <div className="flex items-center space-x-1 text-muted-foreground">
                          <Activity className="h-3 w-3" />
                          <span>Loss</span>
                        </div>
                        <p className="font-mono font-medium text-primary">{session.current_status.loss.toFixed(4)}</p>
                      </div>
                    )}

                    {session.current_status?.estimated_remaining_minutes && (
                      <div className="space-y-1">
                        <div className="flex items-center space-x-1 text-muted-foreground">
                          <Clock className="h-3 w-3" />
                          <span>ETA</span>
                        </div>
                        <p className="font-medium text-blue-600">
                          ~{Math.round(session.current_status.estimated_remaining_minutes)} min
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
