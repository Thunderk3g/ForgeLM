export interface SelectModelRequest {
  model_id: string;
  output_dir?: string;
  quantize?: 'none' | '4bit' | '8bit';
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
  model_name: string;
  output_dir?: string;
  ft_type: 'sft' | 'instruction' | 'unsupervised';
  peft_method?: 'qlora';
  alignment?: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  gradient_accumulation_steps: number;
  max_seq_length: number;
  reasoning_effort?: string;
}

export interface ApiResponse {
  status: 'success' | 'error' | 'started' | 'running' | 'completed';
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
  status: 'running' | 'completed' | 'error';
  message: string;
  progress?: number;
  current_epoch?: number;
  total_epochs?: number;
  current_step?: number;
  total_steps?: number;
  loss?: number;
  eval_loss?: number;
  learning_rate?: number;
  logs?: string[];
  model_path?: string;
  eval_results?: Record<string, any>;
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