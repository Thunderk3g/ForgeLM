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
import { ActiveSessionsPanel } from './components/ActiveSessionsPanel';

// API
import { 
  healthCheck, 
  selectModel, 
  prepareData, 
  startFineTuning, 
  getTrainingStatus,
  getAvailableModels,
  streamTrainingLogs,
  getActiveTrainingSessions
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
  const [activeSessions, setActiveSessions] = useState<Record<string, any>>({});
  const [showActiveSessionsModal, setShowActiveSessionsModal] = useState<boolean>(false);
  const [activeSessionsRefresh, setActiveSessionsRefresh] = useState(0);
  const [showActiveSessions, setShowActiveSessions] = useState(true);
  const [viewingLogsTaskId, setViewingLogsTaskId] = useState<string | null>(null);
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

  // Load available models and active sessions on mount
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
    
    const loadActiveSessions = async () => {
      try {
        const response = await getActiveTrainingSessions();
        if (response.status === 'success') {
          setActiveSessions(response.active_sessions);
          
          // If there are active sessions and no current task, show modal
          if (Object.keys(response.active_sessions).length > 0 && !currentTaskId) {
            setShowActiveSessionsModal(true);
          }
        }
      } catch (error) {
        console.error('Error loading active sessions:', error);
      }
    };
    
    if (apiStatus === 'online') {
      loadModels();
      loadActiveSessions();
    }
  }, [apiStatus, currentTaskId]);

  // Cleanup SSE connections on unmount
  useEffect(() => {
    return () => {
      if (streamConnection) {
        streamConnection.close();
      }
    };
  }, []);

  // Function to start log streaming for any task
  const startLogStreaming = (taskId: string, updateStatus: boolean = true) => {
    // Close existing connection if any
    if (streamConnection) {
      streamConnection.close();
    }
      
    const connection = streamTrainingLogs(
      taskId,
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
      // onStatus - only update status if this is the current task
      (status: TrainingStatus) => {
        if (updateStatus) {
          // Ensure status has the required fields with fallbacks
          const validatedStatus: TrainingStatus = {
            task_id: status.task_id || taskId || '',
            status: status.status || 'pending',
            progress: status.progress || 'No progress information',
            current_step: status.current_step || null,
            total_steps: status.total_steps || null,
            current_epoch: status.current_epoch || null,
            total_epochs: status.total_epochs || null,
            loss: status.loss || null,
            learning_rate: status.learning_rate || null,
            message: status.message || null
          };
          setTrainingStatus(validatedStatus);
        }
      },
        // onComplete
        () => {
          console.log('Training stream completed');
          setIsStreamConnected(false);
          setCurrentTaskId(null);
          
          // Refresh active sessions to remove completed session
          setActiveSessionsRefresh(prev => prev + 1);
        },
        // onError
        (error: string) => {
          console.error('SSE Stream error:', error);
          toast.error(`Streaming error: ${error}`);
          setIsStreamConnected(false);
          // Fall back to polling on error
          fallbackToPolling(taskId);
        },
        // onConnect
        () => {
          console.log('SSE Stream connected');
          setIsStreamConnected(true);
        }
      );
      
    setStreamConnection(connection);
  };

  // SSE streaming for current training task
  useEffect(() => {
    if (currentTaskId && trainingStatus?.status === 'running') {
      startLogStreaming(currentTaskId, true);
    } else {
      // Clean up connection when not training
      if (streamConnection) {
        streamConnection.close();
        setStreamConnection(null);
        setIsStreamConnected(false);
      }
    }
    
    return () => {
      if (streamConnection) {
        streamConnection.close();
        setIsStreamConnected(false);
      }
    };
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
        
        // Trigger refresh of active sessions
        setActiveSessionsRefresh(prev => prev + 1);
        
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
    
    // Use the computed progress_percentage if available
    if (trainingStatus.progress_percentage !== undefined) {
      return trainingStatus.progress_percentage;
    }
    
    // Fallback to manual calculation
    if (trainingStatus.current_step && trainingStatus.total_steps) {
      return (trainingStatus.current_step / trainingStatus.total_steps) * 100;
    }
    if (trainingStatus.current_epoch && trainingStatus.total_epochs) {
      return (trainingStatus.current_epoch / trainingStatus.total_epochs) * 100;
    }
    return 0;
  };

  const connectToActiveSession = async (taskId: string) => {
    setCurrentTaskId(taskId);
    setLogs([]);
    setMetrics([]);
    setShowActiveSessionsModal(false);
    
    // Load current status
    try {
      const status = await getTrainingStatus(taskId);
      setTrainingStatus(status);
    } catch (error) {
      console.error('Error loading session status:', error);
    }
    
    // Start streaming logs
    startLogStreaming(taskId);
    
    // Trigger refresh of active sessions to update UI
    setActiveSessionsRefresh(prev => prev + 1);
    
    toast.success(`Connected to training session ${taskId.slice(0, 8)}...`);
  };

  const handleViewLogs = (taskId: string) => {
    // Clear current logs and set new viewing task
    setLogs([]);
    setMetrics([]);
    setViewingLogsTaskId(taskId);
    
    // Start streaming logs for the selected task (don't update training status)
    startLogStreaming(taskId, false);
    
    toast.success(`Viewing logs for session ${taskId.slice(0, 8)}...`);
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
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowActiveSessions(!showActiveSessions)}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  showActiveSessions 
                    ? 'bg-primary/20 text-primary border border-primary/30' 
                    : 'bg-accent text-accent-foreground border border-border hover:bg-accent/80'
                }`}
                title="Toggle Active Sessions Panel"
              >
                <Activity className="w-4 h-4 inline mr-1" />
                Active Training
              </button>
              
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
                
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 text-sm">
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
                  {trainingStatus.learning_rate !== undefined && (
                    <div>
                      <span className="text-muted-foreground">Learning Rate:</span>
                      <span className="ml-2 font-medium font-mono text-blue-400">{trainingStatus.learning_rate.toExponential(2)}</span>
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

            {/* Active Training Sessions Panel */}
            {showActiveSessions && (
              <ActiveSessionsPanel 
                onConnectToSession={connectToActiveSession}
                onViewLogs={handleViewLogs}
                currentTaskId={currentTaskId}
                refreshTrigger={activeSessionsRefresh}
              />
            )}

            {/* Metrics Chart */}
            {metrics.length > 0 && (
              <MetricsChart data={metrics} />
            )}

            {/* Enhanced Terminal Logs - Always show for all LLM operations */}
            <LogViewer 
              logs={logs.map(log => `[${log.timestamp}] ${log.type.toUpperCase()}: ${log.message}`)} 
              title={
                viewingLogsTaskId ? 
                  `Training Logs - Session ${viewingLogsTaskId.slice(0, 8)}...` : 
                  currentTaskId ? 
                    `Training Logs - Session ${currentTaskId.slice(0, 8)}...` :
                    "LLM Training Logs"
              }
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
