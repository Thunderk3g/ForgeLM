import { useState, useEffect } from 'react';
import { Toaster, toast } from 'react-hot-toast';
import { Brain, Activity, Zap } from 'lucide-react';

// Components
import { ModelSelector } from './components/ModelSelector';
import { DataPreparation } from './components/DataPreparation';
import { TrainingForm } from './components/TrainingForm';
import { StatusCard } from './components/StatusCard';
import { ProgressBar } from './components/ProgressBar';
import { LogViewer } from './components/LogViewer';
import { MetricsChart } from './components/MetricsChart';

// API
import { 
  healthCheck, 
  selectModel, 
  prepareData, 
  startFineTuning, 
  getTrainingStatus,
  getAvailableModels,
  streamTrainingLogs
} from './services/api';
import type { LogEntry
} from './services/api';

// Types
import type { 
  SelectModelRequest, 
  PrepareDataRequest, 
  FineTuneRequest, 
  TrainingStatus,
  ModelInfo
} from './types/api';

interface MetricData {
  epoch: number;
  step: number;
  loss?: number;
  eval_loss?: number;
  learning_rate?: number;
}

function App() {
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [metrics, setMetrics] = useState<MetricData[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [selectedModelInfo, setSelectedModelInfo] = useState<ModelInfo | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [isStreamConnected, setIsStreamConnected] = useState<boolean>(false);
  const [streamConnection, setStreamConnection] = useState<{ close: () => void } | null>(null);
  const [loading, setLoading] = useState({
    model: false,
    data: false,
    training: false,
  });

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await healthCheck();
        setApiStatus('online');
      } catch (error) {
        setApiStatus('offline');
      }
    };
    checkHealth();
  }, []);

  // Load available models on mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        const response = await getAvailableModels();
        if (response.status === 'success') {
          setAvailableModels(response.models);
        }
      } catch (error) {
        console.error('Failed to load available models:', error);
      }
    };
    loadModels();
  }, []);

  // Cleanup SSE connections on unmount
  useEffect(() => {
    return () => {
      if (streamConnection) {
        streamConnection.close();
      }
    };
  }, []);

  // SSE streaming for training logs
  useEffect(() => {
    if (currentTaskId && trainingStatus?.status === 'running') {
      // Close existing connection if any
      if (streamConnection) {
        streamConnection.close();
      }
      
      const connection = streamTrainingLogs(
        currentTaskId,
        // onLog
        (logEntry: LogEntry) => {
          setLogs(prev => [...prev, logEntry]);
          
          // Update metrics if log contains training data
          if (logEntry.type === 'training' && logEntry.step && logEntry.loss !== undefined) {
            const newMetric: MetricData = {
              epoch: logEntry.epoch ?? 0,
              step: logEntry.step,
              loss: logEntry.loss ?? undefined,
              eval_loss: undefined, // Will be updated by status events
              learning_rate: logEntry.learning_rate ?? undefined,
            };
            
            setMetrics(prev => {
              const exists = prev.find(m => m.step === newMetric.step);
              if (exists) return prev;
              return [...prev, newMetric];
            });
          }
        },
        // onStatus
        (status: TrainingStatus) => {
          setTrainingStatus(status);
        },
        // onComplete
        () => {
          console.log('Training stream completed');
          setIsStreamConnected(false);
          setCurrentTaskId(null);
        },
        // onError
        (error: string) => {
          console.error('SSE Stream error:', error);
          toast.error(`Streaming error: ${error}`);
          setIsStreamConnected(false);
          // Fall back to polling on error
          fallbackToPolling(currentTaskId);
        },
        // onConnect
        () => {
          console.log('SSE Stream connected');
          setIsStreamConnected(true);
        }
      );
      
      setStreamConnection(connection);
      
      return () => {
        connection.close();
        setIsStreamConnected(false);
      };
    } else {
      // Clean up connection when not training
      if (streamConnection) {
        streamConnection.close();
        setStreamConnection(null);
        setIsStreamConnected(false);
      }
    }
  }, [currentTaskId, trainingStatus?.status]);
  
  // Fallback polling function in case SSE fails
  const fallbackToPolling = (taskId: string) => {
    const pollStatus = async () => {
      try {
        const status = await getTrainingStatus(taskId);
        setTrainingStatus(status);
        
        if (status.status === 'completed' || status.status === 'error') {
          setCurrentTaskId(null);
        } else {
          // Continue polling every 5 seconds
          setTimeout(pollStatus, 5000);
        }
      } catch (error) {
        console.error('Failed to fetch training status:', error);
      }
    };
    pollStatus();
  };

  const handleSelectModel = async (request: SelectModelRequest) => {
    setLoading(prev => ({ ...prev, model: true }));
    try {
      const response = await selectModel(request);
      if (response.status === 'success') {
        setSelectedModel(request.model_id);
        toast.success('Model downloaded successfully!');
      } else {
        toast.error(response.message);
      }
    } catch (error) {
      toast.error('Failed to download model');
    } finally {
      setLoading(prev => ({ ...prev, model: false }));
    }
  };

  const handleModelSelected = (modelId: string) => {
    setSelectedModel(modelId);
    // Find the model info from available models
    const modelInfo = availableModels.find(model => model.model_id === modelId);
    setSelectedModelInfo(modelInfo || null);
  };

  const handlePrepareData = async (request: PrepareDataRequest) => {
    setLoading(prev => ({ ...prev, data: true }));
    try {
      const response = await prepareData(request);
      if (response.status === 'success') {
        toast.success('Data prepared successfully!');
      } else {
        toast.error(response.message);
      }
    } catch (error) {
      toast.error('Failed to prepare data');
    } finally {
      setLoading(prev => ({ ...prev, data: false }));
    }
  };

  const handleStartTraining = async (request: FineTuneRequest) => {
    setLoading(prev => ({ ...prev, training: true }));
    try {
      const response = await startFineTuning(request);
      if (response.status === 'started' && response.task_id) {
        setCurrentTaskId(response.task_id);
        setTrainingStatus({ status: 'running', message: response.message });
        setLogs([]);
        setMetrics([]);
        toast.success('Training started!');
      } else {
        toast.error(response.message);
      }
    } catch (error) {
      toast.error('Failed to start training');
    } finally {
      setLoading(prev => ({ ...prev, training: false }));
    }
  };

  const calculateProgress = (): number => {
    if (!trainingStatus) return 0;
    if (trainingStatus.current_step && trainingStatus.total_steps) {
      return (trainingStatus.current_step / trainingStatus.total_steps) * 100;
    }
    if (trainingStatus.current_epoch && trainingStatus.total_epochs) {
      return (trainingStatus.current_epoch / trainingStatus.total_epochs) * 100;
    }
    return 0;
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Toaster 
        position="top-right"
        toastOptions={{
          style: {
            background: 'hsl(var(--card))',
            color: 'hsl(var(--card-foreground))',
            border: '1px solid hsl(var(--border))',
          },
        }}
      />
      
      {/* Header */}
      <header className="bg-card shadow-sm border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <Brain className="w-8 h-8 text-blue-400" />
              <h1 className="text-2xl font-bold text-gradient">LLMForge Dashboard</h1>
            </div>
            <div className="flex items-center space-x-2">
              <Activity className={`w-5 h-5 ${
                apiStatus === 'online' ? 'status-online' : 
                apiStatus === 'offline' ? 'status-error' : 
                'status-warning'
              }`} />
              <span className={`text-sm font-medium ${
                apiStatus === 'online' ? 'text-green-300' : 
                apiStatus === 'offline' ? 'text-red-300' : 
                'text-yellow-300'
              }`}>
                API {apiStatus === 'checking' ? 'Checking...' : apiStatus}
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Training Progress Overview */}
        {trainingStatus && (
          <div className="mb-8">
            <div className="card p-6 animate-fade-in">
              <div className="flex items-center space-x-2 mb-4">
                <Zap className="w-5 h-5 text-orange-400" />
                <h2 className="text-lg font-semibold text-card-foreground">Training Progress</h2>
              </div>
              
              <div className="space-y-4">
                <ProgressBar 
                  progress={calculateProgress()} 
                  label="Overall Progress" 
                />
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Status:</span>
                    <span className={`ml-2 font-medium capitalize ${
                      trainingStatus.status === 'running' ? 'text-blue-400' :
                      trainingStatus.status === 'completed' ? 'text-green-400' :
                      trainingStatus.status === 'error' ? 'text-red-400' :
                      'text-foreground'
                    }`}>{trainingStatus.status}</span>
                  </div>
                  {trainingStatus.current_epoch && trainingStatus.total_epochs && (
                    <div>
                      <span className="text-muted-foreground">Epoch:</span>
                      <span className="ml-2 font-medium text-foreground">{trainingStatus.current_epoch}/{trainingStatus.total_epochs}</span>
                    </div>
                  )}
                  {trainingStatus.current_step && trainingStatus.total_steps && (
                    <div>
                      <span className="text-muted-foreground">Step:</span>
                      <span className="ml-2 font-medium text-foreground">{trainingStatus.current_step}/{trainingStatus.total_steps}</span>
                    </div>
                  )}
                  {trainingStatus.loss !== undefined && (
                    <div>
                      <span className="text-muted-foreground">Loss:</span>
                      <span className="ml-2 font-medium font-mono text-green-400">{trainingStatus.loss.toFixed(4)}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Controls */}
          <div className="space-y-6">
            <ModelSelector 
              onSelectModel={handleSelectModel}
              onModelSelected={handleModelSelected}
              isLoading={loading.model}
              selectedModel={selectedModel || undefined}
            />
            
            <DataPreparation 
              onPrepareData={handlePrepareData}
              isLoading={loading.data}
            />
            
            <TrainingForm 
              onStartTraining={handleStartTraining}
              isLoading={loading.training}
              selectedModel={selectedModel}
              selectedModelInfo={selectedModelInfo}
            />
          </div>

          {/* Right Column - Monitoring */}
          <div className="space-y-6">
            {/* Status Overview */}
            <div className="grid grid-cols-1 gap-4">
              <StatusCard 
                title="Model Status" 
                status="completed" 
                message="Ready for fine-tuning"
              />
              <StatusCard 
                title="Data Status" 
                status="completed" 
                message="Dataset prepared and ready"
              />
              {trainingStatus && (
                <StatusCard 
                  title="Training Status" 
                  status={trainingStatus.status as any}
                  message={trainingStatus.message}
                  details={{
                    task_id: currentTaskId?.substring(0, 8) + '...',
                    progress: `${calculateProgress().toFixed(1)}%`
                  }}
                />
              )}
            </div>

            {/* Metrics Chart */}
            {metrics.length > 0 && (
              <MetricsChart data={metrics} />
            )}

            {/* Enhanced Terminal Logs - Always show for all LLM operations */}
            <LogViewer 
              logs={logs.map(log => `[${log.timestamp}] ${log.type.toUpperCase()}: ${log.message}`)} 
              title="LLM Training Logs"
              maxHeight="600px"
              isConnected={isStreamConnected}
              onReconnect={() => {
                // Try to reconnect SSE stream if we have a current task
                if (currentTaskId && trainingStatus?.status === 'running') {
                  // Close existing connection
                  if (streamConnection) {
                    streamConnection.close();
                  }
                  
                  // Trigger re-creation of SSE connection by updating state
                  setCurrentTaskId(currentTaskId);
                } else if (currentTaskId) {
                  // Fallback to polling
                  fallbackToPolling(currentTaskId);
                }
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App
