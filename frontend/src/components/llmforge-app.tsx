"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { Toaster, toast } from "react-hot-toast"
import { Brain, Activity, Zap, XCircle, MessageSquare, Settings, Home, Workflow, BookOpen } from "lucide-react"
import { Button } from "@/components/ui/button"

import { ModelSelector } from "./model-selector"
import { DataPreparation } from "./data-preparation"
import { TrainingForm } from "./training-form"
import { StatusCard } from "./status-card"
import { ProgressBar } from "./progress-bar"
import { LogViewer } from "./log-viewer"
import { MetricsChart } from "./metrics-chart"
import { ActiveSessionsPanel } from "./active-sessions-panel"
import { ComparisonChatInterface } from "./comparison-chat-interface"

import {
  healthCheck,
  selectModel,
  prepareData,
  startFineTuning,
  getTrainingStatus,
  getAvailableModels,
  streamTrainingLogs,
  getActiveTrainingSessions,
  cancelTraining,
} from "../lib/api"
import type { LogEntry } from "../lib/api"

import type { SelectModelRequest, PrepareDataRequest, FineTuneRequest, TrainingStatus, ModelInfo } from "../lib/types"

interface MetricData {
  epoch: number
  step: number
  loss?: number
  eval_loss?: number
  learning_rate?: number
}

interface LLMForgeAppProps {
  onNavigate: (page: "home" | "training" | "workflow" | "docs") => void
}

function LLMForgeApp({ onNavigate }: LLMForgeAppProps) {
  const [activeTab, setActiveTab] = useState<"training" | "chat">("training")
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking")
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null)
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [metrics, setMetrics] = useState<MetricData[]>([])
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [selectedModelInfo, setSelectedModelInfo] = useState<ModelInfo | null>(null)
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([])
  const [isStreamConnected, setIsStreamConnected] = useState<boolean>(false)
  const [streamConnection, setStreamConnection] = useState<{ close: () => void } | null>(null)
  const streamConnectionRef = useRef<{ close: () => void } | null>(null)
  // Removed unused state variables
  const [activeSessionsRefresh, setActiveSessionsRefresh] = useState(0)
  const [showActiveSessions, setShowActiveSessions] = useState(true)
  const [viewingLogsTaskId, setViewingLogsTaskId] = useState<string | null>(null)
  const [loading, setLoading] = useState({
    model: false,
    data: false,
    training: false,
  })

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await healthCheck()
        setApiStatus("online")
      } catch (error) {
        setApiStatus("offline")
      }
    }
    checkHealth()
  }, [])

  // Load available models and active sessions on mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        const response = await getAvailableModels()
        if (response.status === "success") {
          setAvailableModels(response.models)
        }
      } catch (error) {
        console.error("Failed to load available models:", error)
      }
    }

    const loadActiveSessions = async () => {
      try {
        const response = await getActiveTrainingSessions()
        if (response.status === "success") {
          // Active sessions loaded - trigger UI refresh if needed
          setActiveSessionsRefresh((prev) => prev + 1)
        }
      } catch (error) {
        console.error("Error loading active sessions:", error)
      }
    }

    if (apiStatus === "online") {
      loadModels()
      loadActiveSessions()
    }
  }, [apiStatus, currentTaskId])

  // Cleanup SSE connections on unmount
  useEffect(() => {
    return () => {
      if (streamConnection) {
        streamConnection.close()
      }
    }
  }, [])

  // Function to start log streaming for any task
  const startLogStreaming = useCallback((taskId: string, updateStatus = true) => {
    // Close existing connection if any
    if (streamConnectionRef.current) {
      streamConnectionRef.current.close()
    }

    const connection = streamTrainingLogs(
      taskId,
      // onLog
      (logEntry: LogEntry) => {
        setLogs((prev) => [...prev, logEntry])

        // Update metrics if log contains training data
        if (logEntry.type === "training" && logEntry.step && logEntry.loss !== undefined) {
          const newMetric: MetricData = {
            epoch: logEntry.epoch ?? 0,
            step: logEntry.step,
            loss: logEntry.loss ?? undefined,
            eval_loss: undefined, // Will be updated by status events
            learning_rate: logEntry.learning_rate ?? undefined,
          }

          setMetrics((prev) => {
            const exists = prev.find((m) => m.step === newMetric.step)
            if (exists) return prev
            return [...prev, newMetric]
          })
        }
      },
      // onStatus - only update status if this is the current task
      (status: TrainingStatus) => {
        if (updateStatus) {
          // Ensure status has the required fields with fallbacks
          const validatedStatus: TrainingStatus = {
            task_id: status.task_id || taskId || "",
            status: status.status || "running",
            progress: status.progress || "No progress information",
            current_step: status.current_step ?? undefined,
            total_steps: status.total_steps ?? undefined,
            current_epoch: status.current_epoch ?? undefined,
            total_epochs: status.total_epochs ?? undefined,
            loss: status.loss ?? undefined,
            learning_rate: status.learning_rate ?? undefined,
            message: status.message ?? undefined,
          }
          setTrainingStatus(validatedStatus)
        }
      },
      // onComplete
      () => {
        console.log("Training stream completed")
        setIsStreamConnected(false)
        setCurrentTaskId(null)

        // Refresh active sessions to remove completed session
        setActiveSessionsRefresh((prev) => prev + 1)
      },
      // onError
      (error: string) => {
        console.error("SSE Stream error:", error)
        toast.error(`Streaming error: ${error}`)
        setIsStreamConnected(false)
        // Fall back to polling on error
        fallbackToPolling(taskId)
      },
      // onConnect
      () => {
        console.log("SSE Stream connected")
        setIsStreamConnected(true)
      },
    )

    setStreamConnection(connection)
    streamConnectionRef.current = connection
  }, [])

  // SSE streaming for current training task
  useEffect(() => {
    if (currentTaskId && trainingStatus?.status === "running") {
      startLogStreaming(currentTaskId, true)
    } else {
      // Clean up connection when not training
      if (streamConnection) {
        streamConnection.close()
        setStreamConnection(null)
        setIsStreamConnected(false)
      }
    }
  }, [currentTaskId, trainingStatus?.status, startLogStreaming])

  // Fallback polling function in case SSE fails
  const fallbackToPolling = (taskId: string) => {
    const pollStatus = async () => {
      try {
        const status = await getTrainingStatus(taskId)
        setTrainingStatus(status)

        if (status.status === "completed" || status.status === "error") {
          setCurrentTaskId(null)
        } else {
          // Continue polling every 5 seconds
          setTimeout(pollStatus, 5000)
        }
      } catch (error) {
        console.error("Failed to fetch training status:", error)
      }
    }
    pollStatus()
  }

  const handleSelectModel = async (request: SelectModelRequest) => {
    setLoading((prev) => ({ ...prev, model: true }))
    try {
      const response = await selectModel(request)
      if (response.status === "success") {
        setSelectedModel(request.model_id)
        toast.success("Model downloaded successfully!")
      } else {
        toast.error(response.message)
      }
    } catch (error) {
      toast.error("Failed to download model")
    } finally {
      setLoading((prev) => ({ ...prev, model: false }))
    }
  }

  const handleModelSelected = (modelId: string) => {
    setSelectedModel(modelId)
    // Find the model info from available models
    const modelInfo = availableModels.find((model) => model.model_id === modelId)
    setSelectedModelInfo(modelInfo || null)
  }

  const handlePrepareData = async (request: PrepareDataRequest) => {
    setLoading((prev) => ({ ...prev, data: true }))
    try {
      const response = await prepareData(request)
      if (response.status === "success") {
        toast.success("Data prepared successfully!")
      } else {
        toast.error(response.message)
      }
    } catch (error) {
      toast.error("Failed to prepare data")
    } finally {
      setLoading((prev) => ({ ...prev, data: false }))
    }
  }

  const handleStartTraining = async (request: FineTuneRequest) => {
    setLoading((prev) => ({ ...prev, training: true }))
    try {
      const response = await startFineTuning(request)
      if (response.status === "started" && response.task_id) {
        setCurrentTaskId(response.task_id)
        setTrainingStatus({
          task_id: response.task_id,
          status: "running",
          progress: "Starting training...",
          message: response.message,
        })
        setLogs([])
        setMetrics([])

        // Trigger refresh of active sessions
        setActiveSessionsRefresh((prev) => prev + 1)

        toast.success("Training started!")
      } else {
        toast.error(response.message)
      }
    } catch (error) {
      toast.error("Failed to start training")
    } finally {
      setLoading((prev) => ({ ...prev, training: false }))
    }
  }

  const calculateProgress = (): number => {
    if (!trainingStatus) return 0

    // Use the computed progress_percentage if available
    if (trainingStatus.progress_percentage !== undefined) {
      return trainingStatus.progress_percentage
    }

    // Fallback to manual calculation
    if (trainingStatus.current_step && trainingStatus.total_steps) {
      return (trainingStatus.current_step / trainingStatus.total_steps) * 100
    }
    if (trainingStatus.current_epoch && trainingStatus.total_epochs) {
      return (trainingStatus.current_epoch / trainingStatus.total_epochs) * 100
    }
    return 0
  }

  const connectToActiveSession = async (taskId: string) => {
    setCurrentTaskId(taskId)
    setLogs([])
    setMetrics([])

    // Load current status
    try {
      const status = await getTrainingStatus(taskId)
      setTrainingStatus(status)
    } catch (error) {
      console.error("Error loading session status:", error)
    }

    // Start streaming logs
    startLogStreaming(taskId)

    // Trigger refresh of active sessions to update UI
    setActiveSessionsRefresh((prev) => prev + 1)

    toast.success(`Connected to training session ${taskId.slice(0, 8)}...`)
  }

  const handleViewLogs = (taskId: string) => {
    // Clear current logs and set new viewing task
    setLogs([])
    setMetrics([])
    setViewingLogsTaskId(taskId)

    // Start streaming logs for the selected task (don't update training status)
    startLogStreaming(taskId, false)

    toast.success(`Viewing logs for session ${taskId.slice(0, 8)}...`)
  }

  const handleCancelTraining = async (taskId: string) => {
    try {
      const response = await cancelTraining(taskId)
      if (response.status === "cancellation_requested") {
        toast.success(`Cancellation requested for ${taskId.slice(0, 8)}...`)

        // Refresh active sessions to show updated status
        setActiveSessionsRefresh((prev) => prev + 1)

        // If this is the current task, update status
        if (taskId === currentTaskId) {
          setTrainingStatus((prevStatus) =>
            prevStatus
              ? {
                  ...prevStatus,
                  status: "cancelling",
                  message: "Cancellation requested - stopping at next safe checkpoint",
                }
              : null,
          )
        }
      } else {
        toast.error(response.message || "Failed to cancel training")
      }
    } catch (error: any) {
      console.error("Cancellation error:", error)
      toast.error(`Failed to cancel training: ${error?.response?.data?.detail || error.message}`)
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "hsl(var(--card))",
            color: "hsl(var(--card-foreground))",
            border: "1px solid hsl(var(--border))",
          },
        }}
      />

      {/* Header - Following shadcn/ui navigation patterns */}
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto max-w-7xl">
          <div className="flex h-16 items-center justify-between px-4">
            <div className="flex items-center space-x-8">
              <div className="flex items-center space-x-3">
                <Brain className="h-8 w-8 text-primary" />
                <h1 className="text-2xl font-bold tracking-tight">LLMForge</h1>
              </div>

              {/* Navigation Tabs */}
              <nav className="flex items-center space-x-1">
                <Button variant="ghost" onClick={() => onNavigate("home")} className="text-sm">
                  <Home className="mr-2 h-4 w-4" />
                  Home
                </Button>

                <Button variant="default" className="text-sm">
                  <Settings className="mr-2 h-4 w-4" />
                  Fine-tuning
                </Button>

                <Button
                  onClick={() => setActiveTab("chat")}
                  className={`inline-flex items-center justify-center rounded-md px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                    activeTab === "chat" ? "bg-accent text-accent-foreground" : "text-muted-foreground"
                  }`}
                >
                  <MessageSquare className="mr-2 h-4 w-4" />
                  Model Comparison
                </Button>

                <Button variant="ghost" onClick={() => onNavigate("workflow")} className="text-sm">
                  <Workflow className="mr-2 h-4 w-4" />
                  Workflows
                </Button>

                <Button variant="ghost" onClick={() => onNavigate("docs")} className="text-sm">
                  <BookOpen className="mr-2 h-4 w-4" />
                  Documentation
                </Button>
              </nav>
            </div>

            <div className="flex items-center space-x-4">
              {activeTab === "training" && (
                <button
                  onClick={() => setShowActiveSessions(!showActiveSessions)}
                  className={`btn-outline ${showActiveSessions ? "bg-accent" : ""}`}
                  title="Toggle Active Sessions Panel"
                >
                  <Activity className="mr-2 h-4 w-4" />
                  Active Training
                </button>
              )}

              <div className="flex items-center space-x-2 rounded-md border px-3 py-1.5">
                <div
                  className={`h-2 w-2 rounded-full ${
                    apiStatus === "online" ? "bg-primary" : apiStatus === "offline" ? "bg-destructive" : "bg-secondary"
                  }`}
                />
                <span className="text-sm font-medium">
                  {apiStatus === "checking" ? "Checking..." : "API " + apiStatus}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      {activeTab === "chat" ? (
        <ComparisonChatInterface />
      ) : (
        <main className="container mx-auto max-w-7xl p-4 md:p-6 lg:p-8 space-y-8">
          {/* Training Progress Overview */}
          {trainingStatus && (
            <div className="space-y-6">
              <div className="card">
                <div className="card-header">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="rounded-md bg-primary/10 p-2">
                        <Zap className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <h2 className="text-lg font-semibold tracking-tight">Training Progress</h2>
                        <p className="text-sm text-muted-foreground">Monitor your model fine-tuning in real-time</p>
                      </div>
                    </div>

                    {/* Cancel Button for current training */}
                    {currentTaskId && (trainingStatus.status === "running" || trainingStatus.status === "started") && (
                      <button
                        onClick={() => {
                          if (
                            window.confirm(
                              `Are you sure you want to cancel the current training session?\n\nThe training will stop at the next safe checkpoint to preserve data integrity.`,
                            )
                          ) {
                            handleCancelTraining(currentTaskId)
                          }
                        }}
                        className="btn-destructive"
                        title="Cancel current training session"
                      >
                        <XCircle className="mr-2 h-4 w-4" />
                        Cancel Training
                      </button>
                    )}
                  </div>
                </div>

                <div className="card-content space-y-6">
                  <ProgressBar progress={calculateProgress()} label="Overall Progress" />

                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
                    <div className="space-y-2">
                      <p className="text-sm text-muted-foreground">Status</p>
                      <p
                        className={`text-sm font-medium ${
                          trainingStatus.status === "running"
                            ? "text-blue-400"
                            : trainingStatus.status === "completed"
                              ? "text-green-400"
                              : trainingStatus.status === "error"
                                ? "text-destructive"
                                : "text-foreground"
                        }`}
                      >
                        {trainingStatus.status}
                      </p>
                    </div>

                    {trainingStatus.current_epoch && trainingStatus.total_epochs && (
                      <div className="space-y-2">
                        <p className="text-sm text-muted-foreground">Epoch</p>
                        <p className="text-sm font-mono font-medium">
                          {trainingStatus.current_epoch}/{trainingStatus.total_epochs}
                        </p>
                      </div>
                    )}

                    {trainingStatus.current_step && trainingStatus.total_steps && (
                      <div className="space-y-2">
                        <p className="text-sm text-muted-foreground">Step</p>
                        <p className="text-sm font-mono font-medium">
                          {trainingStatus.current_step}/{trainingStatus.total_steps}
                        </p>
                      </div>
                    )}

                    {trainingStatus.loss !== undefined && trainingStatus.loss !== null && (
                      <div className="space-y-2">
                        <p className="text-sm text-muted-foreground">Loss</p>
                        <p className="text-sm font-mono font-medium text-green-400">{trainingStatus.loss.toFixed(4)}</p>
                      </div>
                    )}

                    {trainingStatus.learning_rate !== undefined && trainingStatus.learning_rate !== null && (
                      <div className="space-y-2">
                        <p className="text-sm text-muted-foreground">Learning Rate</p>
                        <p className="text-sm font-mono font-medium text-blue-400">
                          {trainingStatus.learning_rate.toExponential(2)}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="grid gap-8 lg:grid-cols-2">
            {/* Left Column - Setup & Configuration */}
            <div className="space-y-6">
              <ModelSelector
                onSelectModel={handleSelectModel}
                onModelSelected={handleModelSelected}
                isLoading={loading.model}
                selectedModel={selectedModel || undefined}
              />

              <DataPreparation onPrepareData={handlePrepareData} isLoading={loading.data} />

              <TrainingForm
                onStartTraining={handleStartTraining}
                isLoading={loading.training}
                selectedModel={selectedModel}
                selectedModelInfo={selectedModelInfo}
              />
            </div>

            {/* Right Column - Monitoring & Status */}
            <div className="space-y-6">
              {/* Status Overview */}
              <div className="space-y-4">
                <StatusCard title="Model Status" status="completed" message="Ready for fine-tuning" />
                <StatusCard title="Data Status" status="completed" message="Dataset prepared and ready" />
                {trainingStatus && (
                  <StatusCard
                    title="Training Status"
                    status={trainingStatus.status as any}
                    message={trainingStatus.message}
                    details={{
                      task_id: currentTaskId?.substring(0, 8) + "...",
                      progress: `${calculateProgress().toFixed(1)}%`,
                    }}
                  />
                )}
              </div>

              {/* Active Training Sessions Panel */}
              {showActiveSessions && (
                <ActiveSessionsPanel
                  onConnectToSession={connectToActiveSession}
                  onViewLogs={handleViewLogs}
                  onCancelTraining={handleCancelTraining}
                  currentTaskId={currentTaskId}
                  refreshTrigger={activeSessionsRefresh}
                />
              )}

              {/* Metrics Chart */}
              {metrics.length > 0 && <MetricsChart data={metrics} />}

              {/* Enhanced Terminal Logs - Always show for all LLM operations */}
              <LogViewer
                logs={logs.map((log) => `[${log.timestamp}] ${log.type.toUpperCase()}: ${log.message}`)}
                title={
                  viewingLogsTaskId
                    ? `Training Logs - Session ${viewingLogsTaskId.slice(0, 8)}...`
                    : currentTaskId
                      ? `Training Logs - Session ${currentTaskId.slice(0, 8)}...`
                      : "LLM Training Logs"
                }
                maxHeight="600px"
                isConnected={isStreamConnected}
                onReconnect={() => {
                  // Try to reconnect SSE stream if we have a current task
                  if (currentTaskId && trainingStatus?.status === "running") {
                    // Close existing connection
                    if (streamConnection) {
                      streamConnection.close()
                    }

                    // Trigger re-creation of SSE connection by updating state
                    setCurrentTaskId(currentTaskId)
                  } else if (currentTaskId) {
                    // Fallback to polling
                    fallbackToPolling(currentTaskId)
                  }
                }}
              />
            </div>
          </div>
        </main>
      )}
    </div>
  )
}

export default LLMForgeApp
