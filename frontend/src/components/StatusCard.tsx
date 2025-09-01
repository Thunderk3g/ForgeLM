import React from 'react';
import { CheckCircle, Clock, AlertCircle, Play } from 'lucide-react';

interface StatusCardProps {
  title: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  message?: string;
  details?: Record<string, any>;
}

const statusConfig = {
  pending: {
    icon: Clock,
    color: 'text-slate-400',
    bgColor: 'bg-slate-900/20',
    borderColor: 'border-slate-700/30'
  },
  running: {
    icon: Play,
    color: 'text-blue-400',
    bgColor: 'bg-blue-950/20',
    borderColor: 'border-blue-800/30'
  },
  completed: {
    icon: CheckCircle,
    color: 'text-green-400',
    bgColor: 'bg-green-950/20',
    borderColor: 'border-green-800/30'
  },
  error: {
    icon: AlertCircle,
    color: 'text-red-400',
    bgColor: 'bg-red-950/20',
    borderColor: 'border-red-800/30'
  }
};

export const StatusCard: React.FC<StatusCardProps> = ({ title, status, message, details }) => {
  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <div className={`p-4 rounded-lg border ${config.bgColor} ${config.borderColor} animate-fade-in`}>
      <div className="flex items-center space-x-3">
        <Icon className={`w-6 h-6 ${config.color}`} />
        <div className="flex-1">
          <h3 className="font-semibold text-foreground">{title}</h3>
          {message && <p className="text-sm text-muted-foreground mt-1">{message}</p>}
          {details && Object.keys(details).length > 0 && (
            <div className="mt-2 text-xs text-muted-foreground">
              {Object.entries(details).map(([key, value]) => (
                <div key={key} className="flex justify-between">
                  <span className="capitalize">{key.replace(/_/g, ' ')}:</span>
                  <span className="font-mono text-foreground">{String(value)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};