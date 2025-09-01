import React, { useState } from 'react';
import { Play, Settings } from 'lucide-react';
import type { FineTuneRequest, ModelInfo } from '../types/api';

interface TrainingFormProps {
  onStartTraining: (request: FineTuneRequest) => void;
  isLoading: boolean;
  selectedModel: string | null;
  selectedModelInfo?: ModelInfo | null;
}

export const TrainingForm: React.FC<TrainingFormProps> = ({ onStartTraining, isLoading, selectedModel, selectedModelInfo }) => {
  const [ftType, setFtType] = useState<'sft' | 'instruction' | 'unsupervised'>('instruction');
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(4);
  const [learningRate, setLearningRate] = useState(1e-4);
  const [gradientAccumulation, setGradientAccumulation] = useState(4);
  const [maxSeqLength, setMaxSeqLength] = useState(512);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedModel) {
      alert('Please select a model first!');
      return;
    }
    
    onStartTraining({
      base_model_dir: selectedModelInfo?.path || '/app/models/base',
      dataset_dir: '/app/data/prepared',
      model_name: selectedModel,
      ft_type: ftType,
      peft_method: 'qlora',
      epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
      gradient_accumulation_steps: gradientAccumulation,
      max_seq_length: maxSeqLength,
    });
  };

  return (
    <div className="bg-card rounded-lg shadow-sm border-border border p-6">
      <div className="flex items-center space-x-2 mb-4">
        <Settings className="w-5 h-5 text-purple-500" />
        <h3 className="font-semibold text-foreground">Fine-tuning Configuration</h3>
      </div>

      {selectedModel && (
        <div className="mb-4 p-3 bg-green-500/10 border border-green-500/20 rounded-md">
          <p className="text-sm text-green-400">
            <span className="font-medium">Selected Model:</span> {selectedModel}
          </p>
        </div>
      )}

      {!selectedModel && (
        <div className="mb-4 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-md">
          <p className="text-sm text-yellow-400">
            Please select a model first before configuring fine-tuning.
          </p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-foreground mb-2">
            Training Type
          </label>
          <div className="space-y-2">
            {(['instruction', 'sft', 'unsupervised'] as const).map((option) => (
              <label key={option} className="flex items-center space-x-2">
                <input
                  type="radio"
                  value={option}
                  checked={ftType === option}
                  onChange={(e) => setFtType(e.target.value as typeof ftType)}
                  className="text-purple-500 accent-purple-500"
                />
                <span className="capitalize">
                  {option === 'sft' ? 'Supervised Fine-tuning' : `${option} Tuning`}
                </span>
              </label>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-1">
              Epochs
            </label>
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              min="1"
              max="20"
              className="w-full p-2 bg-background border border-input rounded-md text-foreground focus:ring-2 focus:ring-ring focus:border-ring transition-colors"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1">
              Batch Size
            </label>
            <input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              min="1"
              max="32"
              className="w-full p-2 bg-background border border-input rounded-md text-foreground focus:ring-2 focus:ring-ring focus:border-ring transition-colors"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-foreground mb-1">
            Learning Rate
          </label>
          <input
            type="number"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            min="0.00001"
            max="0.01"
            step="0.00001"
            className="w-full p-2 border rounded-md focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-1">
              Gradient Accumulation
            </label>
            <input
              type="number"
              value={gradientAccumulation}
              onChange={(e) => setGradientAccumulation(parseInt(e.target.value))}
              min="1"
              max="32"
              className="w-full p-2 bg-background border border-input rounded-md text-foreground focus:ring-2 focus:ring-ring focus:border-ring transition-colors"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1">
              Max Sequence Length
            </label>
            <input
              type="number"
              value={maxSeqLength}
              onChange={(e) => setMaxSeqLength(parseInt(e.target.value))}
              min="128"
              max="4096"
              step="128"
              className="w-full p-2 bg-background border border-input rounded-md text-foreground focus:ring-2 focus:ring-ring focus:border-ring transition-colors"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={isLoading || !selectedModel}
          className="w-full flex items-center justify-center space-x-2 py-2 px-4 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Play className="w-4 h-4" />
          <span>{isLoading ? 'Starting Training...' : 'Start Fine-tuning'}</span>
        </button>
      </form>
    </div>
  );
};