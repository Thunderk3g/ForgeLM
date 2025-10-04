// Configuration types matching backend schemas

export interface GenerationConfig {
  temperature: number
  top_p: number
  top_k: number | null
  max_tokens: number
  presence_penalty: number
  frequency_penalty: number
  repetition_penalty: number
  do_sample: boolean
  early_stopping: boolean
  seed: number | null
  pad_token_id: number | null
  eos_token_id: number | null
}

export interface ModelSpecificConfig {
  system_prompt: string
  context_length: number
  generation_config: GenerationConfig
  model_id: string | null
  enabled: boolean
}

export interface DualModelConfig {
  session_id: string
  base_model: ModelSpecificConfig
  finetuned_model: ModelSpecificConfig
  sync_parameters: boolean
  config_name: string | null
  created_at: string
  updated_at: string
}

export interface ConfigPreset {
  preset_id: string
  name: string
  description: string
  category: string
  base_config: GenerationConfig
  finetuned_config: GenerationConfig
  system_prompts: Record<string, string>
  is_built_in: boolean
  created_by: string | null
  created_at: string
  usage_count: number
}

export interface ModelCapabilities {
  model_id: string
  model_type: string
  max_context_length: number
  supports_system_prompt: boolean
  default_parameters: GenerationConfig
  parameter_constraints: Record<string, Record<string, any>>
  recommended_presets: string[]
  memory_usage_mb: number | null
  estimated_tokens_per_second: number | null
}

export interface ConfigValidationResponse {
  is_valid: boolean
  warnings: string[]
  errors: string[]
  estimated_performance?: Record<string, any>
  memory_requirements?: Record<string, number>
}

export interface ConfigPresetListResponse {
  presets: ConfigPreset[]
  categories: string[]
  total_count: number
}

// Chat-related types
export interface ChatMessage {
  role: string
  content: string
  timestamp?: string
}

export interface ModelPair {
  pair_id: string
  base_model_name: string
  adapter_name: string
  base_model_size_mb: number
  adapter_size_mb: number
  estimated_memory_mb: number
  status: string
  load_time_ms?: number
}

export interface ChatSession {
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

// API Request types
export interface ConfigPresetRequest {
  name: string
  description?: string
  category: string
  base_config: GenerationConfig
  finetuned_config: GenerationConfig
  system_prompts?: Record<string, string>
}

export interface EnhancedComparisonRequest {
  session_id: string
  messages: ChatMessage[]
  config?: DualModelConfig
  use_agent_orchestrator?: boolean
  return_debug_info?: boolean
}

export interface ComparisonResponse {
  base_response: string
  finetuned_response: string
  generation_time_ms: Record<string, number>
  model_info: Record<string, string>
  cache_metrics: Record<string, number>
  session_id: string
  timestamp: string
  debug_info?: any
}