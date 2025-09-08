export interface SelectModelRequest {
  model_id: string;
  output_dir: string;
  quantization?: string;
}

export interface PrepareDataRequest {
  input_file: string;
  output_dir?: string;
  task: 'unsupervised' | 'supervised' | 'instruction';
  augment?: boolean;
  balance?: boolean;
  test_size?: number;
  max_length?: number;
}

export interface FineTuneRequest {
  base_model_dir: string;
  dataset_dir: string;
  output_dir?: string;
  ft_type: string;
  peft_method: string;
  alignment: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  gradient_accumulation_steps: number;
  max_seq_length: number;
  reasoning_effort?: string;
}

export interface ApiResponse {
  status: 'success' | 'error' | 'started' | 'running' | 'completed' | 'cancellation_requested';
  message: string;
  model_path?: string;
  train_path?: string;
  val_path?: string;
  task_id?: string;
  eval_results?: Record<string, any>;
  optimization_info?: Record<string, any>;
  optimization_summary?: Record<string, any>;
}

export interface TrainingStatus {
  task_id: string;
  status: 'running' | 'completed' | 'error' | 'cancelling' | 'cancelled' | 'started';
  message?: string;
  progress: string;
  current_epoch?: number;
  total_epochs?: number;
  current_step?: number;
  total_steps?: number;
  loss?: number;
  eval_loss?: number;
  learning_rate?: number;
  progress_percentage?: number;
  started_at?: string;
  updated_at?: string;
  estimated_completion?: string;
  estimated_remaining_minutes?: number;
  logs?: string[];
  model_path?: string;
  eval_results?: Record<string, any>;
}

export interface ActiveSession {
  task_id: string;
  status: string;
  started_at: string;
  request_data: any;
  last_activity: string;
  current_status?: TrainingStatus;
}

export interface ActiveSessionsResponse {
  status: string;
  active_sessions: Record<string, ActiveSession>;
  total_active: number;
}

export interface HealthResponse {
  status: string;
  service: string;
}

export interface ModelInfo {
  model_id: string;
  local_name: string;
  path: string;
  status: 'available' | 'downloading' | 'error';
  size_mb: number;
  files: string[];
  has_tokenizer: boolean;
  architecture?: string;
  model_type?: string;
}

export interface AvailableModelsResponse {
  status: 'success' | 'error';
  models: ModelInfo[];
  models_directory: string;
  total_models: number;
  message?: string;
}

// =====================================
// VISUALIZATION DATA INTERFACES
// =====================================

export interface TrainingMetric {
  step: number;
  loss: number;
  learning_rate: number;
  epoch: number;
  gradient_norm?: number;
  tokens_per_second?: number;
  memory_usage?: number;
  timestamp: string;
}

export interface ResourceMetric {
  timestamp: string;
  cpu_load: number;
  memory_pressure: number;
  gpu_utilization: number;
  resources: {
    gpu: {
      allocated_mb: number;
      total_mb: number;
    };
    memory: {
      used_mb: number;
      total_mb: number;
    };
  };
}

export interface TrainingAnalyticsResponse {
  status: string;
  task_id: string;
  training_metrics: TrainingMetric[];
  resource_metrics: ResourceMetric[];
  total_metrics: number;
}

export interface TokenDistribution {
  range: string;
  count: number;
  percentage: number;
}

export interface TemplateFormat {
  name: string;
  count: number;
  percentage: number;
  color: string;
}

export interface DatasetAnalyticsData {
  total_samples: number;
  token_distribution: TokenDistribution[];
  length_statistics: {
    mean: number;
    median: number;
    min: number;
    max: number;
    std_dev: number;
  };
  template_formats: TemplateFormat[];
  quality_metrics: {
    duplicate_ratio: number;
    avg_response_length: number;
    unique_instructions: number;
  };
  sample_examples: Array<{
    instruction: string;
    response: string;
    token_count: number;
    template_format: string;
  }>;
}

export interface DatasetAnalyticsResponse {
  status: string;
  dataset_path: string;
  analytics_data: DatasetAnalyticsData;
}

export interface ComparisonMetric {
  session_id: string;
  base_time_ms: number;
  finetuned_time_ms: number;
  base_response_length: number;
  finetuned_response_length: number;
  similarity_score: number;
  timestamp: string;
  user_message_length: number;
}

export interface ComparisonStats {
  total_comparisons: number;
  avg_similarity: number;
  avg_base_time: number;
  avg_finetuned_time: number;
  performance_improvement: number;
  model_pair_info: {
    base_model: string;
    finetuned_model: string;
  };
}

export interface ComparisonAnalyticsResponse {
  status: string;
  comparison_data: ComparisonMetric[];
  stats: ComparisonStats;
  time_range_hours: number;
}

export interface SystemHealthData {
  cpu: {
    usage_percent: number;
    load_average: number[];
  };
  memory: {
    total_gb: number;
    used_gb: number;
    available_gb: number;
    usage_percent: number;
  };
  gpu: {
    device_count: number;
    devices: Array<{
      name: string;
      memory_total_mb: number;
      memory_used_mb: number;
      utilization_percent: number;
      temperature_c: number;
    }>;
  };
  disk: {
    total_gb: number;
    used_gb: number;
    available_gb: number;
    usage_percent: number;
  };
  network: {
    bytes_sent: number;
    bytes_received: number;
  };
}

export interface SystemHealthResponse {
  status: string;
  timestamp: string;
  health_data: SystemHealthData;
}

export interface DashboardSummaryResponse {
  status: string;
  timestamp: string;
  summary: {
    active_trainings: number;
    completed_trainings: number;
    recent_comparisons: number;
    system_status: 'healthy' | 'warning' | 'critical';
    quick_stats: {
      avg_training_loss: number;
      avg_response_time: number;
      model_performance_score: number;
    };
  };
}