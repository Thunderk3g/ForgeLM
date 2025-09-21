import React, { useState } from 'react';
import { Database, Upload, Settings, Eye, Sparkles, ChevronDown, ChevronRight } from 'lucide-react';
import type { PrepareDataRequest, PreprocessingConfig } from '../types/api';
import { PreprocessingConfigPanel } from './PreprocessingConfigPanel';

interface DataPreparationProps {
  onPrepareData: (request: PrepareDataRequest) => void;
  isLoading: boolean;
}

export const DataPreparation: React.FC<DataPreparationProps> = ({ onPrepareData, isLoading }) => {
  const [inputFile, setInputFile] = useState('/app/data/input/dataset.csv');
  const [task, setTask] = useState<'unsupervised' | 'supervised' | 'instruction'>('instruction');
  const [augment, setAugment] = useState(false);
  const [balance, setBalance] = useState(false);
  const [testSize, setTestSize] = useState(0.2);
  const [maxLength, setMaxLength] = useState(512);
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<string>('basic');
  
  // Advanced preprocessing configuration
  const [preprocessingConfig, setPreprocessingConfig] = useState<PreprocessingConfig>({
    enable_embeddings: false,
    text_cleaning_level: 'standard',
    remove_duplicates_by_embedding: false,
    similarity_threshold: 0.95,
    generate_correlations: false,
    create_tensor_visualizations: false,
    quality_analysis: true,
    token_analysis: true,
    preview_samples: 10
  });

  const predefinedTemplates = {
    basic: {
      name: 'Basic Preprocessing',
      description: 'Standard text cleaning with minimal processing',
      config: {
        enable_embeddings: false,
        text_cleaning_level: 'standard' as const,
        remove_duplicates_by_embedding: false,
        similarity_threshold: 0.95,
        generate_correlations: false,
        create_tensor_visualizations: false,
        quality_analysis: true,
        token_analysis: true,
        preview_samples: 10
      }
    },
    embedding_enhanced: {
      name: 'Embedding-Enhanced',
      description: 'Full preprocessing with embedding generation and analysis',
      config: {
        enable_embeddings: true,
        embedding_config: {
          model_name: 'sentence-transformers/all-mpnet-base-v2',
          model_type: 'sentence_transformer',
          max_sequence_length: 512,
          normalize_embeddings: true,
          batch_size: 32
        },
        text_cleaning_level: 'standard' as const,
        remove_duplicates_by_embedding: true,
        similarity_threshold: 0.95,
        generate_correlations: true,
        create_tensor_visualizations: true,
        quality_analysis: true,
        token_analysis: true,
        preview_samples: 10
      }
    },
    research_analysis: {
      name: 'Research & Analysis',
      description: 'Comprehensive analysis with visualizations',
      config: {
        enable_embeddings: true,
        embedding_config: {
          model_name: 'BAAI/bge-base-en-v1.5',
          model_type: 'sentence_transformer',
          max_sequence_length: 512,
          normalize_embeddings: true,
          batch_size: 16
        },
        text_cleaning_level: 'aggressive' as const,
        remove_duplicates_by_embedding: true,
        similarity_threshold: 0.9,
        generate_correlations: true,
        create_tensor_visualizations: true,
        quality_analysis: true,
        token_analysis: true,
        preview_samples: 20
      }
    }
  };

  const handleTemplateChange = (templateKey: string) => {
    setSelectedTemplate(templateKey);
    if (templateKey !== 'custom') {
      const template = predefinedTemplates[templateKey as keyof typeof predefinedTemplates];
      setPreprocessingConfig(template.config);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onPrepareData({
      input_file: inputFile,
      task,
      augment,
      balance,
      test_size: testSize,
      max_length: maxLength,
      preprocessing_config: showAdvancedConfig ? preprocessingConfig : undefined,
    });
  };

  return (
    <div className="space-y-6">
      <div className="bg-card rounded-lg shadow-sm border-border border p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <Database className="w-5 h-5 text-green-500" />
            <h3 className="font-semibold text-foreground">Prepare Training Data</h3>
          </div>
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-purple-500" />
            <span className="text-sm text-muted-foreground">Enhanced Preprocessing</span>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Basic Configuration */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-1">
                Input File Path
              </label>
              <input
                type="text"
                value={inputFile}
                onChange={(e) => setInputFile(e.target.value)}
                className="w-full p-2 bg-background border border-input rounded-md text-foreground placeholder:text-muted-foreground focus:ring-2 focus:ring-ring focus:border-ring transition-colors"
                placeholder="/app/data/input/dataset.csv"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Task Type
              </label>
              <div className="grid grid-cols-3 gap-3">
                {(['instruction', 'supervised', 'unsupervised'] as const).map((option) => (
                  <label
                    key={option}
                    className={`
                      flex items-center justify-center p-3 rounded-lg border cursor-pointer transition-all
                      ${task === option 
                        ? 'border-green-500 bg-green-50 text-green-700' 
                        : 'border-border hover:border-green-300 hover:bg-accent'
                      }
                    `}
                  >
                    <input
                      type="radio"
                      value={option}
                      checked={task === option}
                      onChange={(e) => setTask(e.target.value as typeof task)}
                      className="sr-only"
                    />
                    <span className="capitalize font-medium">{option}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-foreground mb-1">
                  Test Size: {testSize}
                </label>
                <input
                  type="range"
                  value={testSize}
                  onChange={(e) => setTestSize(parseFloat(e.target.value))}
                  min="0.1"
                  max="0.5"
                  step="0.05"
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>10%</span>
                  <span>50%</span>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-foreground mb-1">
                  Max Length: {maxLength}
                </label>
                <input
                  type="range"
                  value={maxLength}
                  onChange={(e) => setMaxLength(parseInt(e.target.value))}
                  min="128"
                  max="4096"
                  step="128"
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>128</span>
                  <span>4096</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <label className="flex items-center space-x-2 p-3 border rounded-lg hover:bg-accent cursor-pointer">
                <input
                  type="checkbox"
                  checked={augment}
                  onChange={(e) => setAugment(e.target.checked)}
                  className="accent-green-500"
                />
                <div>
                  <span className="font-medium">Data Augmentation</span>
                  <div className="text-xs text-muted-foreground">Enhance dataset diversity</div>
                </div>
              </label>
              <label className="flex items-center space-x-2 p-3 border rounded-lg hover:bg-accent cursor-pointer">
                <input
                  type="checkbox"
                  checked={balance}
                  onChange={(e) => setBalance(e.target.checked)}
                  className="accent-green-500"
                />
                <div>
                  <span className="font-medium">Balance Classes</span>
                  <div className="text-xs text-muted-foreground">Equalize class distribution</div>
                </div>
              </label>
            </div>
          </div>

          {/* Preprocessing Template Selection */}
          <div className="border-t pt-4">
            <label className="block text-sm font-medium text-foreground mb-3">
              Preprocessing Template
            </label>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {Object.entries(predefinedTemplates).map(([key, template]) => (
                <label
                  key={key}
                  className={`
                    p-3 rounded-lg border cursor-pointer transition-all
                    ${selectedTemplate === key 
                      ? 'border-purple-500 bg-purple-50 text-purple-700' 
                      : 'border-border hover:border-purple-300 hover:bg-accent'
                    }
                  `}
                >
                  <input
                    type="radio"
                    value={key}
                    checked={selectedTemplate === key}
                    onChange={(e) => handleTemplateChange(e.target.value)}
                    className="sr-only"
                  />
                  <div className="font-medium text-sm">{template.name}</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {template.description}
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Advanced Configuration Toggle */}
          <div className="border-t pt-4">
            <button
              type="button"
              onClick={() => setShowAdvancedConfig(!showAdvancedConfig)}
              className="flex items-center space-x-2 text-sm font-medium text-foreground hover:text-purple-600 transition-colors"
            >
              {showAdvancedConfig ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
              <Settings className="w-4 h-4" />
              <span>Advanced Preprocessing Configuration</span>
              {preprocessingConfig.enable_embeddings && (
                <span className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded">
                  Embeddings Enabled
                </span>
              )}
            </button>
          </div>

          {/* Submit Button */}
          <div className="space-y-2">
            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex items-center justify-center space-x-2 py-3 px-4 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Upload className="w-4 h-4" />
              <span>
                {isLoading ? 'Preparing Data...' : 
                 showAdvancedConfig && preprocessingConfig.enable_embeddings ? 
                 'Prepare Data with Embeddings' : 'Prepare Data'}
              </span>
            </button>
            
            {showAdvancedConfig && (
              <div className="flex space-x-2">
                <button
                  type="button"
                  className="flex-1 flex items-center justify-center space-x-2 py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                >
                  <Eye className="w-4 h-4" />
                  <span>Preview Changes</span>
                </button>
                <button
                  type="button"
                  className="flex-1 flex items-center justify-center space-x-2 py-2 px-4 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors"
                >
                  <Sparkles className="w-4 h-4" />
                  <span>Generate Visualizations</span>
                </button>
              </div>
            )}
          </div>
        </form>
      </div>

      {/* Advanced Configuration Panel */}
      {showAdvancedConfig && (
        <PreprocessingConfigPanel
          config={preprocessingConfig}
          onConfigChange={setPreprocessingConfig}
          className="animate-in slide-in-from-top-5 duration-200"
        />
      )}
    </div>
  );
};