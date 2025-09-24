import React, { useState } from 'react';
import { Play, Settings, Database, Eye, ChevronDown, ChevronRight } from 'lucide-react';
import type { FineTuneRequest, ModelInfo, PreprocessingConfig, PreprocessingAnalytics } from '../types/api';
import { PreprocessingConfigPanel } from './PreprocessingConfigPanel';
import { PreprocessingResultsViewer } from './PreprocessingResultsViewer';

interface TrainingFormProps {
  onStartTraining: (request: FineTuneRequest) => void;
  isLoading: boolean;
  selectedModel: string | null;
  selectedModelInfo?: ModelInfo | null;
  datasetPath?: string;
  preprocessingConfig?: PreprocessingConfig;
  analytics?: PreprocessingAnalytics;
  onUpdateConfig?: (config: PreprocessingConfig) => void;
}

export const TrainingForm: React.FC<TrainingFormProps> = ({ 
  onStartTraining, 
  isLoading, 
  selectedModel, 
  selectedModelInfo,
  datasetPath = '/app/data/prepared',
  preprocessingConfig = {
    enable_embeddings: false,
    text_cleaning_level: 'standard',
    cleaning_options: {
      deduplication_method: 'hash',
      normalize_whitespace: true,
      language_detection: false,
      utf8_sanitization: true
    },
    remove_duplicates_by_embedding: false,
    similarity_threshold: 0.95,
    generate_correlations: false,
    create_tensor_visualizations: false,
    quality_analysis: true,
    token_analysis: true,
    preview_samples: 10
  },
  analytics,
  onUpdateConfig
}) => {
  const [ftType, setFtType] = useState<string>('instruction');
  const [peftMethod, setPeftMethod] = useState<string>('qlora');
  const [alignment, setAlignment] = useState<string>('none');
  const [epochs, setEpochs] = useState(1);
  const [batchSize, setBatchSize] = useState(20);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [gradientAccumulation, setGradientAccumulation] = useState(4);
  const [maxSeqLength, setMaxSeqLength] = useState(1024);

  const [showDatasetPrep, setShowDatasetPrep] = useState(true);
  const [showVizPreview, setShowVizPreview] = useState(false);
  const [currentDatasetPath, setCurrentDatasetPath] = useState(datasetPath);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedModel) {
      alert('Please select a model first!');
      return;
    }
    
    onStartTraining({
      base_model_dir: selectedModelInfo?.path || '/app/models/base',
      dataset_dir: currentDatasetPath,
      output_dir: '/app/models/fine_tuned',
      ft_type: ftType,
      peft_method: peftMethod,
      alignment: alignment,
      epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
      gradient_accumulation_steps: gradientAccumulation,
      max_seq_length: maxSeqLength,
      preprocessing_config: preprocessingConfig,
    });
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-center space-x-3">
          <div className="rounded-md bg-primary/10 p-2">
            <Settings className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold tracking-tight">Fine-tuning Configuration</h3>
            <p className="text-sm text-muted-foreground">
              Configure training parameters for optimal results
            </p>
          </div>
        </div>
      </div>

      <div className="card-content">
        {selectedModel && (
          <div className="mb-6 rounded-md border border-green-200/20 bg-green-50/5 p-4">
            <div className="flex items-center space-x-2">
              <div className="h-2 w-2 rounded-full bg-green-500" />
              <p className="text-sm font-medium text-green-400">
                Selected Model: {selectedModel}
              </p>
            </div>
          </div>
        )}

        {!selectedModel && (
          <div className="mb-6 rounded-md border border-yellow-200/20 bg-yellow-50/5 p-4">
            <div className="flex items-center space-x-2">
              <div className="h-2 w-2 rounded-full bg-yellow-500" />
              <p className="text-sm font-medium text-yellow-400">
                Please select a model first before configuring fine-tuning
              </p>
            </div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-8">
          <div className="space-y-6">
            <div className="space-y-3">
              <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                Training Type
              </label>
              <div className="grid gap-3">
                {(['instruction', 'sft', 'unsupervised'] as const).map((option) => (
                  <label key={option} className="flex items-center space-x-3 cursor-pointer">
                    <input
                      type="radio"
                      value={option}
                      checked={ftType === option}
                      onChange={(e) => setFtType(e.target.value)}
                      className="h-4 w-4 border border-primary text-primary focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                    />
                    <div>
                      <p className="text-sm font-medium">
                        {option === 'sft' ? 'Supervised Fine-tuning' : `${option.charAt(0).toUpperCase() + option.slice(1)} Tuning`}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {option === 'instruction' ? 'Train on instruction-response pairs' : 
                         option === 'sft' ? 'Standard supervised fine-tuning' : 
                         'Train on unlabeled text data'}
                      </p>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Dataset Preparation Section */}
            <div className="space-y-4">
              <button
                type="button"
                onClick={() => setShowDatasetPrep(!showDatasetPrep)}
                className="flex items-center justify-between w-full p-3 border rounded-lg hover:bg-accent transition-colors"
              >
                <div className="flex items-center space-x-2">
                  <Database className="w-4 h-4 text-blue-500" />
                  <span className="font-medium">Dataset Preparation & Embeddings</span>
                </div>
                {showDatasetPrep ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
              </button>

              {showDatasetPrep && (
                <div className="space-y-4 ml-4 p-4 border rounded-lg bg-muted/50">
                  {/* Dataset Path */}
                  <div>
                    <label className="block text-sm font-medium mb-1">Dataset Path</label>
                    <input
                      type="text"
                      value={currentDatasetPath}
                      onChange={(e) => setCurrentDatasetPath(e.target.value)}
                      className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
                      placeholder="/app/data/prepared"
                    />
                  </div>

                  {/* Embedding Configuration */}
                  <div>
                    <PreprocessingConfigPanel
                      config={preprocessingConfig}
                      onConfigChange={(config) => onUpdateConfig?.(config)}
                      className="border rounded"
                    />
                  </div>

                  {/* Visualization Preview Toggle */}
                  {analytics && (
                    <div>
                      <button
                        type="button"
                        onClick={() => setShowVizPreview(!showVizPreview)}
                        className="flex items-center space-x-2 text-sm text-blue-600 hover:text-blue-800"
                      >
                        <Eye className="w-4 h-4" />
                        <span>{showVizPreview ? 'Hide' : 'Show'} Visualization Preview</span>
                      </button>
                      {showVizPreview && (
                        <div className="mt-3">
                          <PreprocessingResultsViewer
                            result={{ status: 'success', message: 'Preview', train_path: currentDatasetPath }}
                            className="border rounded p-4"
                          />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium leading-none">
                  PEFT Method
                </label>
                <select
                  value={peftMethod}
                  onChange={(e) => setPeftMethod(e.target.value)}
                  className="input"
                >
                  <option value="qlora">QLoRA - 4-bit quantized</option>
                  <option value="lora">LoRA - Standard precision</option>
                </select>
                <p className="text-xs text-muted-foreground">
                  Parameter-efficient fine-tuning method
                </p>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium leading-none">
                  Alignment Method
                </label>
                <select
                  value={alignment}
                  onChange={(e) => setAlignment(e.target.value)}
                  className="input"
                >
                  <option value="none">None</option>
                  <option value="dpo">DPO - Direct Preference</option>
                  <option value="ipo">IPO - Identity Preference</option>
                  <option value="kto">KTO - Kahneman-Tversky</option>
                </select>
                <p className="text-xs text-muted-foreground">
                  Human preference alignment technique
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium leading-none">
                  Epochs
                </label>
                <input
                  type="number"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  min="1"
                  max="20"
                  className="input"
                  placeholder="1"
                />
                <p className="text-xs text-muted-foreground">
                  Number of training passes through dataset
                </p>
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium leading-none">
                  Batch Size
                </label>
                <input
                  type="number"
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  min="1"
                  max="32"
                  className="input"
                  placeholder="20"
                />
                <p className="text-xs text-muted-foreground">
                  Samples processed per training step
                </p>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium leading-none">
                Learning Rate
              </label>
              <input
                type="number"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                min="0.00001"
                max="0.01"
                step="0.00001"
                className="input"
                placeholder="0.0001"
              />
              <p className="text-xs text-muted-foreground">
                Controls the step size during optimization (0.00001 - 0.01)
              </p>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium leading-none">
                  Gradient Accumulation Steps
                </label>
                <input
                  type="number"
                  value={gradientAccumulation}
                  onChange={(e) => setGradientAccumulation(parseInt(e.target.value))}
                  min="1"
                  max="32"
                  className="input"
                  placeholder="4"
                />
                <p className="text-xs text-muted-foreground">
                  Accumulate gradients for effective larger batch size
                </p>
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium leading-none">
                  Max Sequence Length
                </label>
                <input
                  type="number"
                  value={maxSeqLength}
                  onChange={(e) => setMaxSeqLength(parseInt(e.target.value))}
                  min="128"
                  max="4096"
                  step="128"
                  className="input"
                  placeholder="1024"
                />
                <p className="text-xs text-muted-foreground">
                  Maximum input token length (128 - 4096)
                </p>
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading || !selectedModel}
            className="btn-primary w-full"
          >
            <Play className="mr-2 h-4 w-4" />
            {isLoading ? 'Starting Training...' : 'Start Fine-tuning'}
          </button>
        </form>
      </div>
    </div>
  );
};