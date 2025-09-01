import React, { useState, useEffect } from 'react';
import { Download, Cpu, Check, AlertCircle, RefreshCw } from 'lucide-react';
import { getAvailableModels } from '../services/api';
import type { SelectModelRequest, ModelInfo } from '../types/api';

interface ModelSelectorProps {
  onSelectModel: (request: SelectModelRequest) => void;
  onModelSelected: (modelId: string) => void;
  isLoading: boolean;
  selectedModel?: string;
}

const POPULAR_MODELS = [
  { id: 'unsloth/llama-3.1-8b-unsloth-bnb-4bit', name: 'Llama 3.1 8B (Unsloth)', size: '8B' },
  { id: 'unsloth/llama-3.1-70b-unsloth-bnb-4bit', name: 'Llama 3.1 70B (Unsloth)', size: '70B' },
  { id: 'unsloth/mistral-7b-unsloth-bnb-4bit', name: 'Mistral 7B (Unsloth)', size: '7B' },
  { id: 'unsloth/codellama-34b-unsloth-bnb-4bit', name: 'CodeLlama 34B (Unsloth)', size: '34B' },
];

export const ModelSelector: React.FC<ModelSelectorProps> = ({ 
  onSelectModel, 
  onModelSelected, 
  isLoading, 
  selectedModel 
}) => {
  const [localSelectedModel, setLocalSelectedModel] = useState(POPULAR_MODELS[0].id);
  const [customModel, setCustomModel] = useState('');
  const [useCustom, setUseCustom] = useState(false);
  const [quantize, setQuantize] = useState<'none' | '4bit' | '8bit'>('4bit');
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [viewMode, setViewMode] = useState<'popular' | 'available'>('available');

  // Load available models on component mount
  useEffect(() => {
    loadAvailableModels();
  }, []);

  const loadAvailableModels = async () => {
    setLoadingModels(true);
    try {
      const response = await getAvailableModels();
      if (response.status === 'success') {
        setAvailableModels(response.models);
      }
    } catch (error) {
      console.error('Failed to load available models:', error);
    } finally {
      setLoadingModels(false);
    }
  };

  const handleDownload = (e: React.FormEvent) => {
    e.preventDefault();
    const modelId = useCustom ? customModel : localSelectedModel;
    if (!modelId.trim()) return;

    onSelectModel({
      model_id: modelId,
      quantize,
    });
  };

  const handleSelectExistingModel = (modelId: string) => {
    onModelSelected(modelId);
  };

  const isModelAvailable = (modelId: string) => {
    return availableModels.some(model => model.model_id === modelId);
  };

  // const getModelInfo = (modelId: string) => {
  //   return availableModels.find(model => model.model_id === modelId);
  // };

  return (
    <div className="bg-card rounded-lg shadow-sm border-border border p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Cpu className="w-5 h-5 text-primary" />
          <h3 className="font-semibold text-foreground">Model Selection</h3>
        </div>
        <button
          onClick={loadAvailableModels}
          disabled={loadingModels}
          className="flex items-center space-x-1 text-sm text-primary hover:text-primary/80 transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loadingModels ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Selected Model Display */}
      {selectedModel && (
        <div className="mb-4 p-3 bg-green-500/10 border border-green-500/20 rounded-md">
          <div className="flex items-center space-x-2">
            <Check className="w-4 h-4 text-green-600" />
            <span className="text-sm font-medium text-green-400">Selected Model:</span>
            <span className="text-sm text-green-300">{selectedModel}</span>
          </div>
        </div>
      )}

      {/* View Mode Tabs */}
      <div className="flex space-x-4 mb-4 border-b border-border">
        <button
          onClick={() => setViewMode('available')}
          className={`pb-2 px-1 text-sm font-medium border-b-2 ${
            viewMode === 'available' 
              ? 'border-primary text-primary' 
              : 'border-transparent text-muted-foreground hover:text-foreground'
          }`}
        >
          Available Models ({availableModels.length})
        </button>
        <button
          onClick={() => setViewMode('popular')}
          className={`pb-2 px-1 text-sm font-medium border-b-2 ${
            viewMode === 'popular' 
              ? 'border-primary text-primary' 
              : 'border-transparent text-muted-foreground hover:text-foreground'
          }`}
        >
          Download New Model
        </button>
      </div>

      {/* Available Models View */}
      {viewMode === 'available' && (
        <div>
          {loadingModels ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="w-6 h-6 animate-spin text-primary" />
              <span className="ml-2 text-muted-foreground">Loading available models...</span>
            </div>
          ) : availableModels.length > 0 ? (
            <div className="space-y-3">
              {availableModels.map((model) => (
                <div
                  key={model.model_id}
                  className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                    selectedModel === model.model_id
                      ? 'border-green-500 bg-green-500/10'
                      : 'border-border hover:border-primary/50 hover:bg-primary/5'
                  }`}
                  onClick={() => handleSelectExistingModel(model.model_id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <h4 className="font-medium text-foreground">{model.model_id}</h4>
                        <span className="px-2 py-1 text-xs bg-green-500/20 text-green-400 rounded">
                          Available
                        </span>
                      </div>
                      <div className="mt-1 text-sm text-muted-foreground space-x-4">
                        <span>Size: {model.size_mb} MB</span>
                        {model.architecture && <span>Architecture: {model.architecture}</span>}
                        <span className={`${model.has_tokenizer ? 'text-green-400' : 'text-red-400'}`}>
                          Tokenizer: {model.has_tokenizer ? 'Available' : 'Missing'}
                        </span>
                      </div>
                    </div>
                    {selectedModel === model.model_id && (
                      <Check className="w-5 h-5 text-green-400" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <AlertCircle className="w-8 h-8 mx-auto mb-2" />
              <p>No models available locally</p>
              <p className="text-sm">Download a model to get started</p>
            </div>
          )}
        </div>
      )}

      {/* Download New Model View */}
      {viewMode === 'popular' && (
        <form onSubmit={handleDownload} className="space-y-4">
          <div>
            <label className="flex items-center space-x-2">
              <input
                type="radio"
                checked={!useCustom}
                onChange={() => setUseCustom(false)}
                className="text-primary accent-primary"
              />
              <span>Popular Models</span>
            </label>
            {!useCustom && (
              <div className="mt-2 ml-6">
                <select
                  value={localSelectedModel}
                  onChange={(e) => setLocalSelectedModel(e.target.value)}
                  className="w-full p-2 bg-background border border-input rounded-md text-foreground focus:ring-2 focus:ring-ring focus:border-ring transition-colors"
                >
                  {POPULAR_MODELS.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name} ({model.size}) 
                      {isModelAvailable(model.id) ? ' - Available' : ' - Need to download'}
                    </option>
                  ))}
                </select>
                {!useCustom && isModelAvailable(localSelectedModel) && (
                  <div className="mt-2 p-2 bg-yellow-500/10 border border-yellow-500/20 rounded text-sm text-yellow-400">
                    This model is already available locally. You can select it from the "Available Models" tab instead.
                  </div>
                )}
              </div>
            )}
          </div>

          <div>
            <label className="flex items-center space-x-2">
              <input
                type="radio"
                checked={useCustom}
                onChange={() => setUseCustom(true)}
                className="text-primary accent-primary"
              />
              <span>Custom Model ID</span>
            </label>
            {useCustom && (
              <div className="mt-2 ml-6">
                <input
                  type="text"
                  value={customModel}
                  onChange={(e) => setCustomModel(e.target.value)}
                  placeholder="e.g., microsoft/DialoGPT-medium"
                  className="w-full p-2 bg-background border border-input rounded-md text-foreground placeholder:text-muted-foreground focus:ring-2 focus:ring-ring focus:border-ring transition-colors"
                />
                {useCustom && isModelAvailable(customModel) && (
                  <div className="mt-2 p-2 bg-yellow-500/10 border border-yellow-500/20 rounded text-sm text-yellow-400">
                    This model is already available locally. You can select it from the "Available Models" tab instead.
                  </div>
                )}
              </div>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Quantization
            </label>
            <div className="flex space-x-4">
              {(['none', '4bit', '8bit'] as const).map((option) => (
                <label key={option} className="flex items-center space-x-2">
                  <input
                    type="radio"
                    value={option}
                    checked={quantize === option}
                    onChange={(e) => setQuantize(e.target.value as typeof quantize)}
                    className="text-primary accent-primary"
                  />
                  <span className="capitalize">{option}</span>
                </label>
              ))}
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading || (!localSelectedModel && !customModel.trim())}
            className="w-full flex items-center justify-center space-x-2 py-2 px-4 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Download className="w-4 h-4" />
            <span>{isLoading ? 'Downloading Model...' : 'Download Model'}</span>
          </button>
        </form>
      )}
    </div>
  );
};