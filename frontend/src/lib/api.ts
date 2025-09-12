const API_BASE = "http://localhost:8000"

const createApiClient = () => {
  return {
    async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
      const url = `${API_BASE}${endpoint}`
      const config: RequestInit = {
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
        ...options,
      }

      try {
        const response = await fetch(url, config)

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        return await response.json()
      } catch (error) {
        console.error(`API request failed for ${endpoint}:`, error)
        throw error
      }
    },
  }
}

const api = createApiClient()

// SSE Log streaming types
export interface LogEntry {
  timestamp: string
  type: "info" | "training" | "success" | "error"
  message: string
  step?: number | null
  epoch?: number | null
  loss?: number | null
  learning_rate?: number | null
  progress_percentage?: number
}

export interface SSEEvent {
  type: "connected" | "log" | "status" | "heartbeat" | "complete" | "error"
  data: any
}

export const healthCheck = async (): Promise<{ status: string; service: string }> => {
  return api.request("/health")
}

export const getAvailableModels = async (): Promise<{
  status: "success" | "error"
  models: Array<{
    model_id: string
    local_name: string
    path: string
    status: "available" | "downloading" | "error"
    size_mb: number
    files: string[]
    has_tokenizer: boolean
    architecture?: string
    model_type?: string
  }>
  models_directory: string
  total_models: number
  message?: string
}> => {
  return api.request("/models/available")
}

export const selectModel = async (request: {
  model_id: string
  output_dir: string
  quantization?: string
}): Promise<{
  status: "success" | "error" | "started" | "running" | "completed"
  message: string
  model_path?: string
  task_id?: string
}> => {
  return api.request("/select-model", {
    method: "POST",
    body: JSON.stringify(request),
  })
}

export const prepareData = async (request: {
  input_file: string
  output_dir?: string
  task: "unsupervised" | "supervised" | "instruction"
  augment?: boolean
  balance?: boolean
  test_size?: number
  max_length?: number
}): Promise<{
  status: "success" | "error"
  message: string
  train_path?: string
  val_path?: string
}> => {
  return api.request("/prepare-data", {
    method: "POST",
    body: JSON.stringify(request),
  })
}

export const startFineTuning = async (request: {
  base_model_dir: string
  dataset_dir: string
  output_dir?: string
  ft_type: string
  peft_method: string
  alignment: string
  epochs: number
  batch_size: number
  learning_rate: number
  gradient_accumulation_steps: number
  max_seq_length: number
  reasoning_effort?: string
}): Promise<{
  status: "success" | "error" | "started" | "running" | "completed"
  message: string
  task_id?: string
  eval_results?: Record<string, any>
  optimization_info?: Record<string, any>
}> => {
  return api.request("/fine-tune", {
    method: "POST",
    body: JSON.stringify(request),
  })
}

export const getTrainingStatus = async (
  taskId: string,
): Promise<{
  task_id: string
  status: "running" | "completed" | "error" | "cancelling" | "cancelled" | "started"
  message?: string
  progress: string
  current_epoch?: number
  total_epochs?: number
  current_step?: number
  total_steps?: number
  loss?: number
  eval_loss?: number
  learning_rate?: number
  progress_percentage?: number
  started_at?: string
  updated_at?: string
  estimated_completion?: string
  estimated_remaining_minutes?: number
  logs?: string[]
  model_path?: string
  eval_results?: Record<string, any>
}> => {
  return api.request(`/fine-tune/status/${taskId}`)
}

export const cancelTraining = async (
  taskId: string,
): Promise<{
  status: "success" | "error" | "cancellation_requested"
  message: string
}> => {
  return api.request(`/fine-tune/cancel/${taskId}`, {
    method: "DELETE",
  })
}

export const streamTrainingLogs = (
  taskId: string,
  onLog: (log: LogEntry) => void,
  onStatus: (status: any) => void,
  onComplete: () => void,
  onError: (error: string) => void,
  onConnect: () => void,
): { close: () => void } => {
  const eventSource = new EventSource(`${API_BASE}/fine-tune/logs/${taskId}`)

  // Handle connection established
  eventSource.addEventListener("connected", (event) => {
    console.log("SSE Connected:", event.data)
    onConnect()
  })

  // Handle log entries
  eventSource.addEventListener("log", (event) => {
    try {
      const logEntry: LogEntry = JSON.parse(event.data)
      onLog(logEntry)
    } catch (error) {
      console.error("Error parsing log entry:", error)
    }
  })

  // Handle metrics events (training progress with structured data)
  eventSource.addEventListener("metrics", (event) => {
    try {
      const logEntry: LogEntry = JSON.parse(event.data)
      onLog(logEntry)

      // If there's status information in the metrics, update status too
      if (logEntry.step || logEntry.epoch || logEntry.loss) {
        const statusUpdate = {
          current_step: logEntry.step ?? undefined,
          current_epoch: logEntry.epoch ?? undefined,
          loss: logEntry.loss ?? undefined,
          learning_rate: logEntry.learning_rate ?? undefined,
          progress_percentage: logEntry.progress_percentage ?? undefined,
        }
        onStatus(statusUpdate)
      }
    } catch (error) {
      console.error("Error parsing metrics entry:", error)
    }
  })

  // Handle warning events
  eventSource.addEventListener("warning", (event) => {
    try {
      const logEntry: LogEntry = JSON.parse(event.data)
      onLog(logEntry)
    } catch (error) {
      console.error("Error parsing warning entry:", error)
    }
  })

  // Handle status updates
  eventSource.addEventListener("status", (event) => {
    try {
      const status = JSON.parse(event.data)
      onStatus(status)
    } catch (error) {
      console.error("Error parsing status:", error)
    }
  })

  // Handle heartbeat to keep connection alive
  eventSource.addEventListener("heartbeat", () => {
    console.log("SSE Heartbeat received")
  })

  // Handle completion
  eventSource.addEventListener("complete", (event) => {
    console.log("Training completed:", event.data)
    onComplete()
    eventSource.close()
  })

  // Handle errors
  eventSource.addEventListener("error", (event) => {
    try {
      const messageEvent = event as MessageEvent
      if (messageEvent.data) {
        const errorData = JSON.parse(messageEvent.data)
        onError(errorData.error || "Unknown SSE error")
      } else {
        onError("SSE connection error")
      }
    } catch (error) {
      onError("SSE connection error")
    }
    eventSource.close()
  })

  // Handle general EventSource errors
  eventSource.onerror = (event) => {
    console.error("SSE Connection error:", event)
    if (eventSource.readyState === EventSource.CLOSED) {
      onError("Connection closed")
    } else {
      onError("Connection error")
    }
  }

  return {
    close: () => {
      eventSource.close()
    },
  }
}

export const getActiveTrainingSessions = async (): Promise<{
  status: string
  active_sessions: Record<
    string,
    {
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
  >
  total_active: number
}> => {
  return api.request("/fine-tune/active")
}

export const getChatModelPairs = async (): Promise<{
  status: "success" | "error"
  model_pairs: Array<{
    pair_id: string
    base_model_name: string
    adapter_name: string
    base_model_size_mb: number
    adapter_size_mb: number
    estimated_memory_mb: number
    status: string
    load_time_ms?: number
  }>
}> => {
  return api.request("/chat/model-pairs")
}

export const getChatSessions = async (): Promise<{
  status: "success" | "error"
  sessions: Array<{
    session_id: string
    session_name: string
    pair_id: string
    model_info: {
      base_model: string
      adapter_name: string
    }
    created_at: string
    last_activity: string
    message_count: number
    status: string
  }>
}> => {
  return api.request("/chat/sessions")
}

export const createChatSession = async (request: {
  pair_id: string
  session_name?: string
}): Promise<{
  status: "success" | "error"
  session: {
    session_id: string
    session_name: string
    pair_id: string
    model_info: {
      base_model: string
      adapter_name: string
    }
    created_at: string
    last_activity: string
    message_count: number
    status: string
  }
  message?: string
}> => {
  return api.request("/chat/sessions", {
    method: "POST",
    body: JSON.stringify(request),
  })
}

export const getChatSessionHistory = async (
  sessionId: string,
): Promise<{
  status: "success" | "error"
  messages: Array<{
    role: string
    content: string
    timestamp: string
    model_type?: string
  }>
}> => {
  return api.request(`/chat/sessions/${sessionId}`)
}

export const deleteChatSession = async (
  sessionId: string,
): Promise<{
  status: "success" | "error"
  message?: string
}> => {
  return api.request(`/chat/sessions/${sessionId}`, {
    method: "DELETE",
  })
}

export const generateChatResponse = async (request: {
  session_id: string
  messages: Array<{
    role: string
    content: string
    timestamp: string
    model_type?: string
  }>
  generation_params: {
    max_new_tokens: number
    temperature: number
    do_sample: boolean
    top_p: number
  }
}): Promise<{
  status: "success" | "error"
  response: {
    base_response: string
    finetuned_response: string
    generation_time_ms: {
      base: number
      finetuned: number
      total: number
    }
    model_info: {
      base_model: string
      adapter_name: string
      pair_id: string
    }
    cache_metrics: {
      cache_hits: number
      cache_misses: number
    }
    session_id: string
    timestamp: string
  }
}> => {
  return api.request("/chat/generate", {
    method: "POST",
    body: JSON.stringify(request),
  })
}

export const getApiInfo = async (): Promise<{
  service: string
  version: string
  status: string
  endpoints: string[]
}> => {
  return api.request("/")
}

// ============================
// Configuration API Functions
// ============================

export const getSessionConfig = async (sessionId: string) => {
  return api.request(`/chat/sessions/${sessionId}/config`)
}

export const updateSessionConfig = async (sessionId: string, config: any) => {
  return api.request(`/chat/sessions/${sessionId}/config`, {
    method: "POST",
    body: JSON.stringify(config),
  })
}

export const validateSessionConfig = async (sessionId: string, config: any) => {
  return api.request(`/chat/sessions/${sessionId}/config/validate`, {
    method: "POST",
    body: JSON.stringify(config),
  })
}

export const getModelCapabilities = async (modelId: string) => {
  return api.request(`/chat/models/${modelId}/capabilities`)
}

export const listConfigPresets = async (category?: string) => {
  const params = category ? `?category=${encodeURIComponent(category)}` : ""
  return api.request(`/chat/config/presets${params}`)
}

export const getConfigPreset = async (presetId: string) => {
  return api.request(`/chat/config/presets/${presetId}`)
}

export const createConfigPreset = async (preset: any) => {
  return api.request("/chat/config/presets", {
    method: "POST",
    body: JSON.stringify(preset),
  })
}

export const deleteConfigPreset = async (presetId: string) => {
  return api.request(`/chat/config/presets/${presetId}`, {
    method: "DELETE",
  })
}

export const applyPresetToSession = async (sessionId: string, presetId: string) => {
  return api.request(`/chat/sessions/${sessionId}/config/apply-preset/${presetId}`, {
    method: "POST",
  })
}

// Enhanced comparison with configuration support
export const generateEnhancedComparison = async (request: any) => {
  return api.request("/chat/generate-enhanced", {
    method: "POST",
    body: JSON.stringify(request),
  })
}

// Stream enhanced comparison with configuration
export const streamEnhancedComparison = (
  request: any,
  onMessage: (data: any) => void,
  onError: (error: string) => void,
  onComplete: () => void
) => {
  const eventSource = new EventSource(
    `${API_BASE}/chat/stream-enhanced?` + 
    new URLSearchParams({
      session_id: request.session_id,
      config: JSON.stringify(request.config || {}),
      use_agent_orchestrator: request.use_agent_orchestrator?.toString() || "false"
    })
  )

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
    } catch (error) {
      console.error("Failed to parse SSE message:", error)
    }
  }

  eventSource.onerror = (event) => {
    console.error("SSE error:", event)
    onError("Connection error")
    eventSource.close()
  }

  // Handle completion
  eventSource.addEventListener("complete", () => {
    onComplete()
    eventSource.close()
  })

  return {
    close: () => eventSource.close()
  }
}

export default api
