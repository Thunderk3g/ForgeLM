import React, { useState, useEffect, useCallback } from 'react';
import { 
  Play, 
  Pause, 
  Square, 
  RefreshCw, 
  AlertCircle, 
  CheckCircle, 
  Eye,
  Brain,
  BarChart3,
  Clock,
  FileText,
  Zap
} from 'lucide-react';
import type { PreprocessingConfig } from '../types/api';
import { useWebSocket, WebSocketMessage } from '../hooks/useWebSocket';

interface RealTimePreviewProps {
  inputFile: string;
  preprocessingConfig: PreprocessingConfig;
  onPreviewComplete?: (summary: any) => void;
  className?: string;
}

interface PreviewSample {
  index: number;
  original_text: string;
  processed_text: string;
  token_count: number;
  embedding_preview?: number[];
  quality_score: number;
}

interface PreviewSummary {
  total_samples: number;
  avg_quality_score: number;
  estimated_processing_time: number;
  estimated_output_size: string;
}

export const RealTimePreview: React.FC<RealTimePreviewProps> = ({
  inputFile,
  preprocessingConfig,
  onPreviewComplete,
  className = ''
}) => {
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [samples, setSamples] = useState<PreviewSample[]>([]);
  const [summary, setSummary] = useState<PreviewSummary | null>(null);
  const [progress, setProgress] = useState(0);
  const [currentSample, setCurrentSample] = useState<PreviewSample | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  // WebSocket connection for real-time updates
  const {
    isConnected,
    isConnecting,
    error: wsError,
    sendMessage,
    lastMessage,
    reconnect
  } = useWebSocket({
    url: sessionId ? `ws://localhost:8003/ws/${sessionId}` : '',
    onMessage: handleWebSocketMessage,
    onConnect: () => {
      console.log('WebSocket connected for preview');
    },
    onDisconnect: () => {
      console.log('WebSocket disconnected');
    }
  });

  function handleWebSocketMessage(message: WebSocketMessage) {
    switch (message.type) {
      case 'preview_started':
        setIsRunning(true);
        setError(null);
        setSamples([]);
        setProgress(0);
        break;

      case 'preview_sample':
        const sampleData = message.sample_data as PreviewSample;
        setSamples(prev => [...prev, sampleData]);
        setCurrentSample(sampleData);
        setProgress(message.progress || 0);
        break;

      case 'preview_completed':
        setIsRunning(false);
        setSummary(message.summary as PreviewSummary);
        setProgress(100);
        onPreviewComplete?.(message.summary);
        break;

      case 'preview_error':
        setError(message.error);
        setIsRunning(false);
        break;

      case 'pong':
        // Handle ping response
        break;

      default:
        console.log('Unknown message type:', message.type);
    }
  }

  const startPreview = useCallback(async () => {
    try {
      setError(null);
      
      // Start real-time preview session
      const response = await fetch('/api/data-generation/preprocessing/preview-realtime', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_file: inputFile,
          sample_size: preprocessingConfig.preview_samples,
          preprocessing_config: preprocessingConfig
        })
      });

      if (response.ok) {
        const data = await response.json();
        setSessionId(data.session_id);
      } else {
        throw new Error('Failed to start preview session');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start preview');
    }
  }, [inputFile, preprocessingConfig]);

  const stopPreview = useCallback(() => {
    setIsRunning(false);
    setIsPaused(false);
    if (sessionId && isConnected) {
      sendMessage({
        type: 'stop_preview',
        session_id: sessionId
      });
    }
  }, [sessionId, isConnected, sendMessage]);

  const pausePreview = useCallback(() => {
    setIsPaused(!isPaused);
    if (sessionId && isConnected) {
      sendMessage({
        type: isPaused ? 'resume_preview' : 'pause_preview',
        session_id: sessionId
      });
    }
  }, [isPaused, sessionId, isConnected, sendMessage]);

  const resetPreview = useCallback(() => {
    setIsRunning(false);
    setIsPaused(false);
    setSamples([]);
    setSummary(null);
    setProgress(0);
    setCurrentSample(null);
    setError(null);
    setSessionId(null);
  }, []);

  // Auto-start preview when session is connected
  useEffect(() => {
    if (sessionId && isConnected && !isRunning) {
      sendMessage({
        type: 'start_preview',
        config: {
          input_file: inputFile,
          sample_size: preprocessingConfig.preview_samples,
          enable_embeddings: preprocessingConfig.enable_embeddings,
          text_cleaning_level: preprocessingConfig.text_cleaning_level
        }
      });
    }
  }, [sessionId, isConnected, isRunning, sendMessage, inputFile, preprocessingConfig]);

  const renderCurrentSample = () => {
    if (!currentSample) return null;

    return (
      <div className="bg-card border rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-medium">Current Sample</h4>
          <div className="flex items-center space-x-2">
            <span className="text-xs text-muted-foreground">
              Sample {currentSample.index + 1}
            </span>
            <span className={`px-2 py-1 text-xs rounded ${
              currentSample.quality_score >= 0.8 ? 'bg-green-100 text-green-700' :
              currentSample.quality_score >= 0.6 ? 'bg-yellow-100 text-yellow-700' :
              'bg-red-100 text-red-700'
            }`}>
              Quality: {(currentSample.quality_score * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        <div className="space-y-3">
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-1">Original Text</div>
            <div className="text-sm bg-muted p-2 rounded text-muted-foreground">
              {currentSample.original_text}
            </div>
          </div>

          <div>
            <div className="text-xs font-medium text-muted-foreground mb-1">Processed Text</div>
            <div className="text-sm bg-background border p-2 rounded">
              {currentSample.processed_text}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Tokens:</span>
              <span className="ml-2 font-medium">{currentSample.token_count}</span>
            </div>
            {currentSample.embedding_preview && (
              <div>
                <span className="text-muted-foreground">Embedding:</span>
                <span className="ml-2 font-mono text-xs">
                  [{currentSample.embedding_preview.slice(0, 3).map(v => v.toFixed(3)).join(', ')}...]
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderSummary = () => {
    if (!summary) return null;

    return (
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <div className="flex items-center space-x-2 mb-3">
          <CheckCircle className="w-5 h-5 text-green-500" />
          <h4 className="font-medium text-green-800">Preview Complete</h4>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <div className="text-green-600">Total Samples</div>
            <div className="font-medium text-green-800">{summary.total_samples}</div>
          </div>
          <div>
            <div className="text-green-600">Avg Quality</div>
            <div className="font-medium text-green-800">
              {(summary.avg_quality_score * 100).toFixed(1)}%
            </div>
          </div>
          <div>
            <div className="text-green-600">Est. Time</div>
            <div className="font-medium text-green-800">
              {summary.estimated_processing_time.toFixed(1)}s
            </div>
          </div>
          <div>
            <div className="text-green-600">Est. Size</div>
            <div className="font-medium text-green-800">{summary.estimated_output_size}</div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Header with Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Eye className="w-5 h-5 text-blue-500" />
          <h3 className="font-medium">Real-time Preview</h3>
          <div className="flex items-center space-x-1">
            {isConnecting && <RefreshCw className="w-4 h-4 text-blue-500 animate-spin" />}
            {isConnected && <div className="w-2 h-2 bg-green-500 rounded-full" />}
            {wsError && <div className="w-2 h-2 bg-red-500 rounded-full" />}
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {!isRunning && !summary && (
            <button
              onClick={startPreview}
              disabled={isConnecting || !isConnected}
              className="flex items-center space-x-2 px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              <Play className="w-4 h-4" />
              <span>Start Preview</span>
            </button>
          )}

          {isRunning && (
            <>
              <button
                onClick={pausePreview}
                className="flex items-center space-x-2 px-3 py-2 bg-yellow-600 text-white rounded-md hover:bg-yellow-700 transition-colors"
              >
                {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                <span>{isPaused ? 'Resume' : 'Pause'}</span>
              </button>

              <button
                onClick={stopPreview}
                className="flex items-center space-x-2 px-3 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
              >
                <Square className="w-4 h-4" />
                <span>Stop</span>
              </button>
            </>
          )}

          <button
            onClick={resetPreview}
            className="flex items-center space-x-2 px-3 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      {(isRunning || progress > 0) && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Processing samples...</span>
            <span>{progress.toFixed(0)}%</span>
          </div>
          <div className="w-full bg-muted rounded-full h-2">
            <div 
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <span className="font-medium text-red-800">Preview Error</span>
          </div>
          <div className="text-sm text-red-700 mt-1">{error}</div>
        </div>
      )}

      {/* Current Sample Display */}
      {renderCurrentSample()}

      {/* Summary Display */}
      {renderSummary()}

      {/* Processed Samples List */}
      {samples.length > 0 && (
        <div className="bg-card border rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-medium">Processed Samples ({samples.length})</h4>
            <div className="flex items-center space-x-4 text-sm text-muted-foreground">
              <div className="flex items-center space-x-1">
                <FileText className="w-3 h-3" />
                <span>Text</span>
              </div>
              {preprocessingConfig.enable_embeddings && (
                <div className="flex items-center space-x-1">
                  <Brain className="w-3 h-3" />
                  <span>Embeddings</span>
                </div>
              )}
              <div className="flex items-center space-x-1">
                <BarChart3 className="w-3 h-3" />
                <span>Quality</span>
              </div>
            </div>
          </div>

          <div className="space-y-2 max-h-64 overflow-y-auto">
            {samples.map((sample) => (
              <div key={sample.index} className="border rounded p-2 text-sm">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium">Sample {sample.index + 1}</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-muted-foreground">
                      {sample.token_count} tokens
                    </span>
                    <span className={`px-1.5 py-0.5 text-xs rounded ${
                      sample.quality_score >= 0.8 ? 'bg-green-100 text-green-700' :
                      sample.quality_score >= 0.6 ? 'bg-yellow-100 text-yellow-700' :
                      'bg-red-100 text-red-700'
                    }`}>
                      {(sample.quality_score * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                <div className="text-muted-foreground truncate">
                  {sample.processed_text}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Connection Status */}
      {!isConnected && !isConnecting && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 text-yellow-500" />
              <span className="text-sm text-yellow-800">WebSocket disconnected</span>
            </div>
            <button
              onClick={reconnect}
              className="text-xs text-yellow-600 hover:text-yellow-800 underline"
            >
              Reconnect
            </button>
          </div>
        </div>
      )}
    </div>
  );
};