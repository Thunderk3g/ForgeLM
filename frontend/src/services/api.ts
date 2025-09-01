import axios from 'axios';
import type {
  SelectModelRequest,
  PrepareDataRequest,
  FineTuneRequest,
  ApiResponse,
  TrainingStatus,
  HealthResponse,
  AvailableModelsResponse
} from '../types/api';

const API_BASE = 'http://localhost:80';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 300000, // 5 minutes for long operations
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health check
export const healthCheck = async (): Promise<HealthResponse> => {
  const response = await api.get('/health');
  return response.data;
};

// Get available models
export const getAvailableModels = async (): Promise<AvailableModelsResponse> => {
  const response = await api.get('/models/available');
  return response.data;
};

// Model selection
export const selectModel = async (request: SelectModelRequest): Promise<ApiResponse> => {
  const response = await api.post('/select-model', request);
  return response.data;
};

// Data preparation
export const prepareData = async (request: PrepareDataRequest): Promise<ApiResponse> => {
  const response = await api.post('/prepare-data', request);
  return response.data;
};

// Fine-tuning
export const startFineTuning = async (request: FineTuneRequest): Promise<ApiResponse> => {
  const response = await api.post('/fine-tune', request);
  return response.data;
};

// Training status
export const getTrainingStatus = async (taskId: string): Promise<TrainingStatus> => {
  const response = await api.get(`/fine-tune/status/${taskId}`);
  return response.data;
};

// API info
export const getApiInfo = async () => {
  const response = await api.get('/');
  return response.data;
};

export default api;