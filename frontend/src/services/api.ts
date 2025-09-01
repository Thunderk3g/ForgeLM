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

// SSE Log streaming types
export interface LogEntry {
  timestamp: string;
  type: 'info' | 'training' | 'success' | 'error';
  message: string;
  step?: number | null;
  epoch?: number | null;
  loss?: number | null;
  learning_rate?: number | null;
}

export interface SSEEvent {
  type: 'connected' | 'log' | 'status' | 'heartbeat' | 'complete' | 'error';
  data: any;
}

// Stream training logs using Server-Sent Events
export const streamTrainingLogs = (
  taskId: string,
  onLog: (log: LogEntry) => void,
  onStatus: (status: TrainingStatus) => void,
  onComplete: () => void,
  onError: (error: string) => void,
  onConnect: () => void
): { close: () => void } => {
  const eventSource = new EventSource(`${API_BASE}/fine-tune/logs/${taskId}`);
  
  // Handle different event types
  eventSource.addEventListener('connected', (event) => {
    console.log('SSE Connected:', event.data);
    onConnect();
  });
  
  eventSource.addEventListener('log', (event) => {
    try {
      const logEntry: LogEntry = JSON.parse(event.data);
      onLog(logEntry);
    } catch (error) {
      console.error('Error parsing log entry:', error);
    }
  });
  
  eventSource.addEventListener('status', (event) => {
    try {
      const status: TrainingStatus = JSON.parse(event.data);
      onStatus(status);
    } catch (error) {
      console.error('Error parsing status:', error);
    }
  });
  
  eventSource.addEventListener('heartbeat', () => {
    console.log('SSE Heartbeat received');
  });
  
  eventSource.addEventListener('complete', (event) => {
    console.log('Training completed:', event.data);
    onComplete();
    eventSource.close();
  });
  
  eventSource.addEventListener('error', (event) => {
    try {
      const messageEvent = event as MessageEvent;
      if (messageEvent.data) {
        const errorData = JSON.parse(messageEvent.data);
        onError(errorData.error || 'Unknown SSE error');
      } else {
        onError('SSE connection error');
      }
    } catch (error) {
      onError('SSE connection error');
    }
    eventSource.close();
  });
  
  // Handle general EventSource errors
  eventSource.onerror = (event) => {
    console.error('SSE Connection error:', event);
    if (eventSource.readyState === EventSource.CLOSED) {
      onError('Connection closed');
    } else {
      onError('Connection error');
    }
  };
  
  // Return close function for cleanup
  return {
    close: () => {
      eventSource.close();
    }
  };
};

// API info
export const getApiInfo = async () => {
  const response = await api.get('/');
  return response.data;
};

export default api;