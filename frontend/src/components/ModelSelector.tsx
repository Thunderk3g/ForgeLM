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
  const [quantization, setQuantization] = useState<string>('4bit');
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
      output_dir: '/app/models/base',
      quantization,
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
    <div className="card">
      <div className="card-header">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="rounded-md bg-primary/10 p-2">
              <Cpu className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-semibold tracking-tight">Model Selection</h3>
              <p className="text-sm text-muted-foreground">
                Choose or download a base model for fine-tuning
              </p>
            </div>
          </div>
          <button
            onClick={loadAvailableModels}
            disabled={loadingModels}
            className="btn-outline"
          >
            <RefreshCw className={`mr-2 h-4 w-4 ${loadingModels ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      <div className="card-content">
        {/* Selected Model Display */}
        {selectedModel && (
          <div className="mb-6 rounded-md border border-green-200/20 bg-green-50/5 p-4">
            <div className="flex items-center space-x-2">
              <Check className="h-4 w-4 text-green-500" />
              <span className="text-sm font-medium text-green-400">Selected Model:</span>
              <span className="text-sm font-mono text-green-300">{selectedModel}</span>
            </div>
          </div>
        )}

        {/* View Mode Tabs */}
        <div className="mb-6">
          <div className="flex space-x-1 rounded-md bg-muted p-1">
            <button
              onClick={() => setViewMode('available')}
              className={`flex-1 rounded-sm px-3 py-2 text-sm font-medium transition-colors ${
                viewMode === 'available' 
                  ? 'bg-background text-foreground shadow-sm' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              Available Models ({availableModels.length})
            </button>
            <button
              onClick={() => setViewMode('popular')}
              className={`flex-1 rounded-sm px-3 py-2 text-sm font-medium transition-colors ${
                viewMode === 'popular' 
                  ? 'bg-background text-foreground shadow-sm' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              Download New Model
            </button>
          </div>
        </div>

        {/* Available Models View */}
        {viewMode === 'available' && (
          <div className="space-y-4">
            {loadingModels ? (
              <div className="flex items-center justify-center py-12">
                <div className="flex items-center space-x-3">
                  <RefreshCw className="h-5 w-5 animate-spin text-primary" />
                  <span className="text-sm text-muted-foreground">Loading available models...</span>
                </div>
              </div>
            ) : availableModels.length > 0 ? (
              <div className="space-y-3">
                {availableModels.map((model) => (
                  <div
                    key={model.model_id}
                    className={`rounded-lg border p-4 transition-colors cursor-pointer ${
                      selectedModel === model.model_id
                        ? 'border-primary bg-accent'
                        : 'hover:border-accent-foreground/20 hover:bg-accent/50'
                    }`}
                    onClick={() => handleSelectExistingModel(model.model_id)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-3 mb-2">
                          <h4 className="font-medium text-foreground truncate">{model.model_id}</h4>
                          <div className="flex items-center space-x-2">
                            <div className="rounded-full bg-green-500/10 px-2 py-1">
                              <span className="text-xs font-medium text-green-600">Available</span>
                            </div>
                          </div>
                        </div>
                        <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
                          <span>Size: {Math.round(model.size_mb / 1024)} GB</span>
                          {model.architecture && <span>Architecture: {model.architecture}</span>}
                          <span className={`flex items-center space-x-1 ${model.has_tokenizer ? 'text-green-400' : 'text-destructive'}`}>
                            <span>Tokenizer: {model.has_tokenizer ? 'Available' : 'Missing'}</span>
                          </span>
                        </div>
                      </div>
                      {selectedModel === model.model_id && (
                        <div className="flex-shrink-0">
                          <div className="rounded-full bg-primary p-1">
                            <Check className="h-4 w-4 text-primary-foreground" />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="rounded-full bg-muted p-3 mb-4">
                  <AlertCircle className="h-6 w-6 text-muted-foreground" />
                </div>
                <h4 className="text-sm font-medium text-foreground mb-1">No models available locally</h4>
                <p className="text-sm text-muted-foreground">Download a model to get started</p>
              </div>
            )}
          </div>
        )}

        {/* Download New Model View */}
        {viewMode === 'popular' && (
          <form onSubmit={handleDownload} className="space-y-6">
            <div className="space-y-4">
              <div className="space-y-3">
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="radio"
                    checked={!useCustom}
                    onChange={() => setUseCustom(false)}
                    className="h-4 w-4 border border-primary text-primary focus:ring-2 focus:ring-primary focus:ring-offset-2"
                  />
                  <div>
                    <span className="text-sm font-medium">Popular Models</span>
                    <p className="text-xs text-muted-foreground">Pre-configured optimized models</p>
                  </div>
                </label>
                {!useCustom && (
                  <div className="ml-7 space-y-3">
                    <select
                      value={localSelectedModel}
                      onChange={(e) => setLocalSelectedModel(e.target.value)}
                      className="input"
                    >
                      {POPULAR_MODELS.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.name} ({model.size}) 
                          {isModelAvailable(model.id) ? ' - Available' : ' - Need to download'}
                        </option>
                      ))}
                    </select>
                    {!useCustom && isModelAvailable(localSelectedModel) && (
                      <div className="rounded-md border border-yellow-200/20 bg-yellow-50/5 p-3">
                        <p className="text-sm text-yellow-400">
                          This model is already available locally. You can select it from the "Available Models" tab instead.
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="space-y-3">
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="radio"
                    checked={useCustom}
                    onChange={() => setUseCustom(true)}
                    className="h-4 w-4 border border-primary text-primary focus:ring-2 focus:ring-primary focus:ring-offset-2"
                  />
                  <div>
                    <span className="text-sm font-medium">Custom Model ID</span>
                    <p className="text-xs text-muted-foreground">Enter any Hugging Face model ID</p>
                  </div>
                </label>
                {useCustom && (
                  <div className="ml-7 space-y-3">
                    <input
                      type="text"
                      value={customModel}
                      onChange={(e) => setCustomModel(e.target.value)}
                      placeholder="e.g., microsoft/DialoGPT-medium"
                      className="input"
                    />
                    {useCustom && isModelAvailable(customModel) && (
                      <div className="rounded-md border border-yellow-200/20 bg-yellow-50/5 p-3">
                        <p className="text-sm text-yellow-400">
                          This model is already available locally. You can select it from the "Available Models" tab instead.
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-3">
              <label className="text-sm font-medium leading-none">
                Quantization
              </label>
              <div className="flex space-x-6">
                {(['none', '4bit', '8bit'] as const).map((option) => (
                  <label key={option} className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="radio"
                      value={option}
                      checked={quantization === option}
                      onChange={(e) => setQuantization(e.target.value)}
                      className="h-4 w-4 border border-primary text-primary focus:ring-2 focus:ring-primary focus:ring-offset-2"
                    />
                    <span className="text-sm font-medium capitalize">{option}</span>
                  </label>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                Quantization reduces memory usage but may affect model quality
              </p>
            </div>

            <button
              type="submit"
              disabled={isLoading || (!localSelectedModel && !customModel.trim())}
              className="btn-primary w-full"
            >
              <Download className="mr-2 h-4 w-4" />
              {isLoading ? 'Downloading Model...' : 'Download Model'}
            </button>
          </form>
        )}
      </div>
    </div>
  );
};