import React, { useState, useEffect } from 'react';
import { Brain, Zap, Shield, Database, CheckCircle, AlertCircle } from 'lucide-react';
import type { EmbeddingModelInfo, EmbeddingModelConfig, EmbeddingModelListResponse } from '../types/api';

interface EmbeddingModelSelectorProps {
  selectedModel?: EmbeddingModelConfig;
  onModelSelect: (config: EmbeddingModelConfig) => void;
  className?: string;
}

export const EmbeddingModelSelector: React.FC<EmbeddingModelSelectorProps> = ({
  selectedModel,
  onModelSelect,
  className = ''
}) => {
  const [availableModels, setAvailableModels] = useState<EmbeddingModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModelName, setSelectedModelName] = useState(
    selectedModel?.model_name || ''
  );

  useEffect(() => {
    fetchEmbeddingModels();
  }, []);

  const fetchEmbeddingModels = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/data-generation/embedding-models');
      const data: EmbeddingModelListResponse = await response.json();
      
      if (response.ok) {
        setAvailableModels(data.models);
      } else {
        setError('Failed to fetch embedding models');
      }
    } catch (err) {
      setError('Network error while fetching models');
    } finally {
      setLoading(false);
    }
  };

  const handleModelSelection = (model: EmbeddingModelInfo) => {
    const config: EmbeddingModelConfig = {
      model_name: model.model_name,
      model_type: model.model_type,
      max_sequence_length: model.max_sequence_length,
      normalize_embeddings: true,
      batch_size: 32,
      embedding_dimension: model.embedding_dimension
    };
    
    setSelectedModelName(model.model_name);
    onModelSelect(config);
  };

  const getModelIcon = (modelName: string) => {
    if (modelName.includes('bge')) return <Shield className="w-4 h-4" />;
    if (modelName.includes('jina')) return <Zap className="w-4 h-4" />;
    return <Brain className="w-4 h-4" />;
  };

  const getPerformanceBadge = (metrics?: { accuracy: number; speed: number }) => {
    if (!metrics) return null;
    
    if (metrics.accuracy >= 0.9) return 'High Accuracy';
    if (metrics.speed >= 0.9) return 'High Speed';
    return 'Balanced';
  };

  if (loading) {
    return (
      <div className={`bg-card rounded-lg border border-border p-4 ${className}`}>
        <div className="flex items-center space-x-2 mb-3">
          <Database className="w-5 h-5 text-blue-500 animate-spin" />
          <h4 className="font-medium text-foreground">Loading Embedding Models...</h4>
        </div>
        <div className="space-y-2">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-16 bg-muted rounded animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-card rounded-lg border border-destructive p-4 ${className}`}>
        <div className="flex items-center space-x-2 text-destructive">
          <AlertCircle className="w-5 h-5" />
          <span className="font-medium">Error: {error}</span>
        </div>
        <button
          onClick={fetchEmbeddingModels}
          className="mt-2 text-sm text-muted-foreground hover:text-foreground underline"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className={`bg-card rounded-lg border border-border p-4 ${className}`}>
      <div className="flex items-center space-x-2 mb-4">
        <Database className="w-5 h-5 text-blue-500" />
        <h4 className="font-medium text-foreground">Select Embedding Model</h4>
        <span className="text-xs text-muted-foreground">
          ({availableModels.length} models available)
        </span>
      </div>

      <div className="space-y-3">
        {availableModels.map((model) => {
          const isSelected = selectedModelName === model.model_name;
          const isAvailable = model.is_available;
          
          return (
            <div
              key={model.model_name}
              className={`
                relative p-3 rounded-lg border cursor-pointer transition-all
                ${isSelected 
                  ? 'border-primary bg-primary/5 ring-1 ring-primary' 
                  : 'border-border hover:border-primary/50 hover:bg-accent'
                }
                ${!isAvailable ? 'opacity-50 cursor-not-allowed' : ''}
              `}
              onClick={() => isAvailable && handleModelSelection(model)}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3">
                  <div className={`
                    flex items-center justify-center w-8 h-8 rounded-lg
                    ${isSelected ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}
                  `}>
                    {getModelIcon(model.model_name)}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2">
                      <h5 className="font-medium text-foreground truncate">
                        {model.model_name.split('/').pop()}
                      </h5>
                      {isSelected && (
                        <CheckCircle className="w-4 h-4 text-primary flex-shrink-0" />
                      )}
                      {!isAvailable && (
                        <AlertCircle className="w-4 h-4 text-destructive flex-shrink-0" />
                      )}
                    </div>
                    
                    <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                      {model.description}
                    </p>
                    
                    <div className="flex items-center space-x-4 mt-2 text-xs text-muted-foreground">
                      <span>Dim: {model.embedding_dimension}</span>
                      <span>Max: {model.max_sequence_length} tokens</span>
                      <span>{model.memory_requirements}</span>
                    </div>
                  </div>
                </div>
                
                <div className="flex flex-col items-end space-y-1">
                  {model.performance_metrics && (
                    <div className="flex space-x-1">
                      {getPerformanceBadge(model.performance_metrics) && (
                        <span className="px-2 py-1 text-xs bg-primary/10 text-primary rounded">
                          {getPerformanceBadge(model.performance_metrics)}
                        </span>
                      )}
                    </div>
                  )}
                  
                  <div className="flex space-x-1">
                    {model.supported_tasks.slice(0, 2).map(task => (
                      <span
                        key={task}
                        className="px-1.5 py-0.5 text-xs bg-muted text-muted-foreground rounded"
                      >
                        {task}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {selectedModel && (
        <div className="mt-4 p-3 bg-muted rounded-lg">
          <h5 className="font-medium text-foreground mb-2">Selected Configuration</h5>
          <div className="text-sm text-muted-foreground space-y-1">
            <div>Model: {selectedModel.model_name}</div>
            <div>Sequence Length: {selectedModel.max_sequence_length}</div>
            <div>Batch Size: {selectedModel.batch_size}</div>
            <div>Normalize: {selectedModel.normalize_embeddings ? 'Yes' : 'No'}</div>
          </div>
        </div>
      )}
    </div>
  );
};