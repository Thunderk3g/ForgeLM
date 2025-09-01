import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  Clock, 
  Zap, 
  Play, 
  Pause,
  RefreshCw,
  Monitor,
  AlertCircle,
  CheckCircle,
  XCircle,
  ArrowRight,
  FileText,
  ExternalLink
} from 'lucide-react';
import { getActiveTrainingSessions } from '../services/api';
import type { ActiveSession, ActiveSessionsResponse } from '../types/api';

interface ActiveSessionsPanelProps {
  onConnectToSession: (taskId: string) => void;
  onViewLogs?: (taskId: string) => void;
  currentTaskId?: string | null;
  refreshTrigger?: number;
}

export const ActiveSessionsPanel: React.FC<ActiveSessionsPanelProps> = ({
  onConnectToSession,
  onViewLogs,
  currentTaskId,
  refreshTrigger = 0
}) => {
  const [activeSessions, setActiveSessions] = useState<Record<string, ActiveSession>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>('');

  const loadActiveSessions = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response: ActiveSessionsResponse = await getActiveTrainingSessions();
      if (response.status === 'success') {
        setActiveSessions(response.active_sessions);
        setLastUpdated(new Date().toLocaleTimeString());
      } else {
        setError('Failed to load active sessions');
      }
    } catch (err) {
      setError('Error connecting to server');
      console.error('Error loading active sessions:', err);
    } finally {
      setLoading(false);
    }
  };

  // Load sessions on mount and when refresh trigger changes
  useEffect(() => {
    loadActiveSessions();
  }, [refreshTrigger]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(loadActiveSessions, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'running':
        return <Play className="w-4 h-4 text-green-400" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-blue-400" />;
      case 'failed':
      case 'error':
        return <XCircle className="w-4 h-4 text-red-400" />;
      default:
        return <AlertCircle className="w-4 h-4 text-yellow-400" />;
    }
  };

  const getProgressInfo = (session: ActiveSession) => {
    const current = session.current_status;
    if (!current) return { progress: 0, text: 'No status available' };

    if (current.progress_percentage !== undefined) {
      return {
        progress: current.progress_percentage,
        text: `${current.progress_percentage.toFixed(1)}%`
      };
    }

    if (current.current_step && current.total_steps) {
      const progress = (current.current_step / current.total_steps) * 100;
      return {
        progress,
        text: `${current.current_step}/${current.total_steps} steps`
      };
    }

    if (current.current_epoch && current.total_epochs) {
      const progress = (current.current_epoch / current.total_epochs) * 100;
      return {
        progress,
        text: `${current.current_epoch.toFixed(1)}/${current.total_epochs} epochs`
      };
    }

    return { progress: 0, text: 'Starting...' };
  };

  const formatTime = (timeString: string) => {
    try {
      return new Date(timeString).toLocaleString();
    } catch {
      return timeString;
    }
  };

  const sessionCount = Object.keys(activeSessions).length;

  return (
    <div className="bg-card rounded-lg border border-border shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <Activity className="w-5 h-5 text-primary" />
            <h3 className="font-semibold text-card-foreground">Active Training Sessions</h3>
          </div>
          
          <div className="flex items-center space-x-2 text-sm text-muted-foreground">
            <Monitor className="w-4 h-4" />
            <span>{sessionCount} active</span>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {lastUpdated && (
            <span className="text-xs text-muted-foreground">
              Updated: {lastUpdated}
            </span>
          )}
          
          <button
            onClick={loadActiveSessions}
            disabled={loading}
            className="p-2 text-muted-foreground hover:text-primary hover:bg-accent rounded-md transition-colors disabled:opacity-50"
            title="Refresh sessions"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        {error && (
          <div className="flex items-center space-x-2 p-3 bg-destructive/10 border border-destructive/20 rounded-md mb-4">
            <AlertCircle className="w-4 h-4 text-destructive" />
            <span className="text-sm text-destructive">{error}</span>
          </div>
        )}

        {loading && sessionCount === 0 && (
          <div className="flex items-center justify-center p-8 text-muted-foreground">
            <RefreshCw className="w-5 h-5 animate-spin mr-2" />
            <span>Loading active sessions...</span>
          </div>
        )}

        {!loading && sessionCount === 0 && !error && (
          <div className="flex flex-col items-center justify-center p-8 text-muted-foreground">
            <Pause className="w-8 h-8 mb-2 opacity-50" />
            <p className="text-center">No active training sessions</p>
            <p className="text-sm text-center mt-1">Start a new training to see it here</p>
          </div>
        )}

        {sessionCount > 0 && (
          <div className="space-y-3">
            {Object.entries(activeSessions).map(([taskId, session]) => {
              const progressInfo = getProgressInfo(session);
              const isCurrentSession = currentTaskId === taskId;
              const isRunning = session.current_status?.status === 'running';

              return (
                <div
                  key={taskId}
                  className={`p-4 rounded-lg border transition-all ${
                    isCurrentSession 
                      ? 'border-primary bg-primary/5' 
                      : 'border-border bg-card hover:bg-accent/50'
                  }`}
                >
                  {/* Session Header */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(session.current_status?.status || session.status)}
                      
                      <div>
                        <p className="font-medium text-sm text-card-foreground">
                          Task {taskId.slice(0, 8)}...
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {session.request_data?.base_model_dir?.split('/').pop() || 'Unknown model'}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center space-x-2">
                      {isCurrentSession && (
                        <span className="px-2 py-1 text-xs bg-primary/20 text-primary rounded-full">
                          Connected
                        </span>
                      )}
                      
                      {/* View Logs Button - available for all sessions */}
                      {onViewLogs && (
                        <button
                          onClick={() => onViewLogs(taskId)}
                          className="flex items-center space-x-1 px-2 py-1 text-xs bg-accent hover:bg-accent/80 text-accent-foreground rounded-md transition-colors"
                          title="View logs for this session"
                        >
                          <FileText className="w-3 h-3" />
                          <span>Logs</span>
                        </button>
                      )}
                      
                      {!isCurrentSession && isRunning && (
                        <button
                          onClick={() => onConnectToSession(taskId)}
                          className="flex items-center space-x-1 px-3 py-1 text-xs bg-primary hover:bg-primary/90 text-primary-foreground rounded-md transition-colors"
                        >
                          <ArrowRight className="w-3 h-3" />
                          <span>Connect</span>
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Progress Bar */}
                  {progressInfo.progress > 0 && (
                    <div className="mb-3">
                      <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                        <span>Progress</span>
                        <span>{progressInfo.text}</span>
                      </div>
                      <div className="w-full bg-accent rounded-full h-1.5">
                        <div
                          className="bg-primary h-1.5 rounded-full transition-all duration-300"
                          style={{ width: `${Math.min(progressInfo.progress, 100)}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Session Details */}
                  <div className="grid grid-cols-2 gap-4 text-xs">
                    <div className="space-y-1">
                      <div className="flex items-center space-x-1 text-muted-foreground">
                        <Clock className="w-3 h-3" />
                        <span>Started:</span>
                      </div>
                      <p className="text-card-foreground">
                        {formatTime(session.started_at)}
                      </p>
                    </div>

                    <div className="space-y-1">
                      <div className="flex items-center space-x-1 text-muted-foreground">
                        <Zap className="w-3 h-3" />
                        <span>Status:</span>
                      </div>
                      <p className="text-card-foreground">
                        {session.current_status?.progress || 'Running...'}
                      </p>
                    </div>

                    {session.current_status?.loss && (
                      <div className="space-y-1">
                        <div className="flex items-center space-x-1 text-muted-foreground">
                          <Activity className="w-3 h-3" />
                          <span>Loss:</span>
                        </div>
                        <p className="text-card-foreground">
                          {session.current_status.loss.toFixed(4)}
                        </p>
                      </div>
                    )}

                    {session.current_status?.estimated_remaining_minutes && (
                      <div className="space-y-1">
                        <div className="flex items-center space-x-1 text-muted-foreground">
                          <Clock className="w-3 h-3" />
                          <span>ETA:</span>
                        </div>
                        <p className="text-card-foreground">
                          ~{Math.round(session.current_status.estimated_remaining_minutes)} min
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};