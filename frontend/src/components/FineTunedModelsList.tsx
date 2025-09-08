import React, { useState, useEffect } from 'react';
import { FolderOpen, Calendar, Settings, Cpu, FileText, CheckCircle, AlertCircle } from 'lucide-react';

interface FineTunedModel {
  name: string;
  path: string;
  size_mb: number;
  created_at: string;
  base_model: string;
  task_id?: string;
  training_config: {
    ft_type?: string;
    peft_method?: string;
    alignment?: string;
    epochs?: number;
    learning_rate?: number;
  };
  files: {
    has_adapter: boolean;
    has_merged: boolean;
    metadata_exists: boolean;
  };
  framework: string;
}

interface FineTunedModelsListProps {
  onSelectModel?: (model: FineTunedModel) => void;
}

export const FineTunedModelsList: React.FC<FineTunedModelsListProps> = ({ onSelectModel }) => {
  const [models, setModels] = useState<FineTunedModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchFineTunedModels();
  }, []);

  const fetchFineTunedModels = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/models/fine-tuned');
      const data = await response.json();
      
      if (data.status === 'success') {
        setModels(data.fine_tuned_models);
      } else {
        setError('Failed to fetch fine-tuned models');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    if (dateString === 'unknown') return 'Unknown';
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  const formatSize = (sizeInMB: number) => {
    if (sizeInMB < 1024) {
      return `${sizeInMB.toFixed(1)} MB`;
    } else {
      return `${(sizeInMB / 1024).toFixed(1)} GB`;
    }
  };

  if (loading) {
    return (
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold">Fine-Tuned Models</h3>
        </div>
        <div className="card-content">
          <div className="flex items-center justify-center p-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            <span className="ml-3 text-sm text-muted-foreground">Loading models...</span>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold">Fine-Tuned Models</h3>
        </div>
        <div className="card-content">
          <div className="flex items-center justify-center p-8">
            <AlertCircle className="h-8 w-8 text-red-500 mr-3" />
            <div>
              <p className="text-sm font-medium text-red-500">Error loading models</p>
              <p className="text-xs text-muted-foreground">{error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="rounded-md bg-primary/10 p-2">
              <FolderOpen className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-semibold tracking-tight">Fine-Tuned Models</h3>
              <p className="text-sm text-muted-foreground">
                {models.length} models organized in separate folders
              </p>
            </div>
          </div>
          <button
            onClick={fetchFineTunedModels}
            className="rounded-md border border-primary/20 bg-primary/5 p-2 hover:bg-primary/10 transition-colors"
          >
            <Settings className="h-4 w-4" />
          </button>
        </div>
      </div>
      
      <div className="card-content">
        {models.length === 0 ? (
          <div className="text-center p-8">
            <FolderOpen className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-sm font-medium text-muted-foreground">No fine-tuned models found</p>
            <p className="text-xs text-muted-foreground">Complete a fine-tuning session to see models here</p>
          </div>
        ) : (
          <div className="space-y-4">
            {models.map((model, index) => (
              <div
                key={index}
                className="border rounded-lg p-4 hover:bg-accent/50 transition-colors cursor-pointer"
                onClick={() => onSelectModel?.(model)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <h4 className="font-medium text-sm">{model.name}</h4>
                      <div className="flex space-x-1">
                        {model.files.has_adapter && (
                          <div className="rounded bg-green-100 dark:bg-green-900/20 px-2 py-1">
                            <CheckCircle className="h-3 w-3 text-green-600 dark:text-green-400 inline mr-1" />
                            <span className="text-xs text-green-600 dark:text-green-400">Adapter</span>
                          </div>
                        )}
                        {model.files.has_merged && (
                          <div className="rounded bg-blue-100 dark:bg-blue-900/20 px-2 py-1">
                            <FileText className="h-3 w-3 text-blue-600 dark:text-blue-400 inline mr-1" />
                            <span className="text-xs text-blue-600 dark:text-blue-400">Merged</span>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 text-xs text-muted-foreground mb-3">
                      <div className="space-y-1">
                        <div className="flex items-center">
                          <Calendar className="h-3 w-3 mr-1" />
                          <span>Created: {formatDate(model.created_at)}</span>
                        </div>
                        <div className="flex items-center">
                          <Cpu className="h-3 w-3 mr-1" />
                          <span>Size: {formatSize(model.size_mb)}</span>
                        </div>
                      </div>
                      <div className="space-y-1">
                        <div>Base: {model.base_model.split('/').pop() || 'Unknown'}</div>
                        <div>Framework: {model.framework}</div>
                      </div>
                    </div>
                    
                    {model.training_config && (
                      <div className="flex items-center space-x-4 text-xs">
                        {model.training_config.ft_type && (
                          <span className="bg-accent px-2 py-1 rounded">
                            {model.training_config.ft_type}
                          </span>
                        )}
                        {model.training_config.peft_method && (
                          <span className="bg-accent px-2 py-1 rounded">
                            {model.training_config.peft_method}
                          </span>
                        )}
                        {model.training_config.alignment && model.training_config.alignment !== 'none' && (
                          <span className="bg-accent px-2 py-1 rounded">
                            {model.training_config.alignment}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};