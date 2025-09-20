import React, { useState } from 'react';
import { Settings, Sliders, Eye, BarChart3, Target, AlertTriangle, Info } from 'lucide-react';
import type { PreprocessingConfig, EmbeddingModelConfig } from '../types/api';
import { EmbeddingModelSelector } from './EmbeddingModelSelector';

interface PreprocessingConfigPanelProps {
  config: PreprocessingConfig;
  onConfigChange: (config: PreprocessingConfig) => void;
  className?: string;
}

export const PreprocessingConfigPanel: React.FC<PreprocessingConfigPanelProps> = ({
  config,
  onConfigChange,
  className = ''
}) => {
  const [expandedSections, setExpandedSections] = useState({
    embeddings: false,
    cleaning: false,
    analysis: false,
    deduplication: false
  });

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const updateConfig = (updates: Partial<PreprocessingConfig>) => {
    onConfigChange({ ...config, ...updates });
  };

  const handleEmbeddingModelSelect = (embeddingConfig: EmbeddingModelConfig) => {
    updateConfig({
      embedding_config: embeddingConfig,
      enable_embeddings: true
    });
  };

  const getCleaningLevelDescription = (level: string) => {
    switch (level) {
      case 'minimal':
        return 'Basic whitespace normalization only';
      case 'standard':
        return 'Remove special characters, normalize spacing';
      case 'aggressive':
        return 'Remove URLs, emails, short words, extensive cleaning';
      default:
        return '';
    }
  };

  return (
    <div className={`bg-card rounded-lg border border-border ${className}`}>
      <div className="p-4 border-b border-border">
        <div className="flex items-center space-x-2">
          <Settings className="w-5 h-5 text-purple-500" />
          <h3 className="font-semibold text-foreground">Advanced Preprocessing Configuration</h3>
        </div>
        <p className="text-sm text-muted-foreground mt-1">
          Configure embedding models, text cleaning, and analysis options
        </p>
      </div>

      <div className="p-4 space-y-4">
        {/* Embedding Configuration Section */}
        <div className="space-y-3">
          <div
            className="flex items-center justify-between cursor-pointer p-2 rounded hover:bg-accent"
            onClick={() => toggleSection('embeddings')}
          >
            <div className="flex items-center space-x-2">
              <Target className="w-4 h-4 text-blue-500" />
              <span className="font-medium">Embedding Generation</span>
              <div className={`
                px-2 py-1 text-xs rounded
                ${config.enable_embeddings 
                  ? 'bg-green-100 text-green-700' 
                  : 'bg-gray-100 text-gray-600'
                }
              `}>
                {config.enable_embeddings ? 'Enabled' : 'Disabled'}
              </div>
            </div>
            <div className="text-muted-foreground">
              {expandedSections.embeddings ? '−' : '+'}
            </div>
          </div>

          <div className="ml-6">
            <label className="flex items-center space-x-2 mb-3">
              <input
                type="checkbox"
                checked={config.enable_embeddings}
                onChange={(e) => updateConfig({ enable_embeddings: e.target.checked })}
                className="accent-blue-500"
              />
              <span className="text-sm">Enable embedding generation during preprocessing</span>
            </label>

            {expandedSections.embeddings && config.enable_embeddings && (
              <EmbeddingModelSelector
                selectedModel={config.embedding_config}
                onModelSelect={handleEmbeddingModelSelect}
                className="mt-3"
              />
            )}
          </div>
        </div>

        {/* Text Cleaning Section */}
        <div className="space-y-3">
          <div
            className="flex items-center justify-between cursor-pointer p-2 rounded hover:bg-accent"
            onClick={() => toggleSection('cleaning')}
          >
            <div className="flex items-center space-x-2">
              <Sliders className="w-4 h-4 text-green-500" />
              <span className="font-medium">Text Cleaning</span>
              <span className="px-2 py-1 text-xs bg-accent text-accent-foreground rounded capitalize">
                {config.text_cleaning_level}
              </span>
            </div>
            <div className="text-muted-foreground">
              {expandedSections.cleaning ? '−' : '+'}
            </div>
          </div>

          {expandedSections.cleaning && (
            <div className="ml-6 space-y-3">
              <div>
                <label className="block text-sm font-medium mb-2">Cleaning Level</label>
                <div className="space-y-2">
                  {(['minimal', 'standard', 'aggressive'] as const).map((level) => (
                    <label key={level} className="flex items-start space-x-3">
                      <input
                        type="radio"
                        value={level}
                        checked={config.text_cleaning_level === level}
                        onChange={(e) => updateConfig({ 
                          text_cleaning_level: e.target.value as typeof level 
                        })}
                        className="mt-1 accent-green-500"
                      />
                      <div>
                        <div className="capitalize font-medium">{level}</div>
                        <div className="text-xs text-muted-foreground">
                          {getCleaningLevelDescription(level)}
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Deduplication Section */}
        <div className="space-y-3">
          <div
            className="flex items-center justify-between cursor-pointer p-2 rounded hover:bg-accent"
            onClick={() => toggleSection('deduplication')}
          >
            <div className="flex items-center space-x-2">
              <Eye className="w-4 h-4 text-orange-500" />
              <span className="font-medium">Duplicate Removal</span>
              {config.remove_duplicates_by_embedding && config.enable_embeddings && (
                <span className="px-2 py-1 text-xs bg-orange-100 text-orange-700 rounded">
                  Embedding-based
                </span>
              )}
            </div>
            <div className="text-muted-foreground">
              {expandedSections.deduplication ? '−' : '+'}
            </div>
          </div>

          {expandedSections.deduplication && (
            <div className="ml-6 space-y-3">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.remove_duplicates_by_embedding}
                  onChange={(e) => updateConfig({ 
                    remove_duplicates_by_embedding: e.target.checked 
                  })}
                  disabled={!config.enable_embeddings}
                  className="accent-orange-500"
                />
                <span className="text-sm">Remove duplicates using embedding similarity</span>
              </label>

              {!config.enable_embeddings && config.remove_duplicates_by_embedding && (
                <div className="flex items-center space-x-2 text-amber-600 text-xs">
                  <AlertTriangle className="w-4 h-4" />
                  <span>Requires embedding generation to be enabled</span>
                </div>
              )}

              {config.remove_duplicates_by_embedding && config.enable_embeddings && (
                <div className="space-y-2">
                  <label className="block text-sm font-medium">
                    Similarity Threshold: {config.similarity_threshold}
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="1.0"
                    step="0.05"
                    value={config.similarity_threshold}
                    onChange={(e) => updateConfig({ 
                      similarity_threshold: parseFloat(e.target.value) 
                    })}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Less strict (0.5)</span>
                    <span>More strict (1.0)</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Analysis & Visualization Section */}
        <div className="space-y-3">
          <div
            className="flex items-center justify-between cursor-pointer p-2 rounded hover:bg-accent"
            onClick={() => toggleSection('analysis')}
          >
            <div className="flex items-center space-x-2">
              <BarChart3 className="w-4 h-4 text-purple-500" />
              <span className="font-medium">Analysis & Visualization</span>
              <span className="text-xs text-muted-foreground">
                ({[
                  config.generate_correlations,
                  config.create_tensor_visualizations, 
                  config.quality_analysis,
                  config.token_analysis
                ].filter(Boolean).length}/4 enabled)
              </span>
            </div>
            <div className="text-muted-foreground">
              {expandedSections.analysis ? '−' : '+'}
            </div>
          </div>

          {expandedSections.analysis && (
            <div className="ml-6 space-y-3">
              <div className="grid grid-cols-1 gap-3">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.generate_correlations}
                    onChange={(e) => updateConfig({ generate_correlations: e.target.checked })}
                    className="accent-purple-500"
                  />
                  <div>
                    <span className="text-sm font-medium">Generate Correlation Analysis</span>
                    <div className="text-xs text-muted-foreground">
                      Analyze feature relationships and data correlations
                    </div>
                  </div>
                </label>

                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.create_tensor_visualizations}
                    onChange={(e) => updateConfig({ create_tensor_visualizations: e.target.checked })}
                    className="accent-purple-500"
                  />
                  <div>
                    <span className="text-sm font-medium">Create Tensor Visualizations</span>
                    <div className="text-xs text-muted-foreground">
                      Generate PCA, t-SNE and embedding distribution plots
                    </div>
                  </div>
                </label>

                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.quality_analysis}
                    onChange={(e) => updateConfig({ quality_analysis: e.target.checked })}
                    className="accent-purple-500"
                  />
                  <div>
                    <span className="text-sm font-medium">Quality Analysis</span>
                    <div className="text-xs text-muted-foreground">
                      Analyze dataset quality, completeness, and richness
                    </div>
                  </div>
                </label>

                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.token_analysis}
                    onChange={(e) => updateConfig({ token_analysis: e.target.checked })}
                    className="accent-purple-500"
                  />
                  <div>
                    <span className="text-sm font-medium">Token Distribution Analysis</span>
                    <div className="text-xs text-muted-foreground">
                      Analyze token counts, vocabulary, and distribution patterns
                    </div>
                  </div>
                </label>
              </div>

              <div className="mt-4">
                <label className="block text-sm font-medium mb-2">
                  Preview Sample Size: {config.preview_samples}
                </label>
                <input
                  type="range"
                  min="5"
                  max="50"
                  step="5"
                  value={config.preview_samples}
                  onChange={(e) => updateConfig({ 
                    preview_samples: parseInt(e.target.value) 
                  })}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>5 samples</span>
                  <span>50 samples</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Configuration Summary */}
        <div className="bg-muted rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-2">
            <Info className="w-4 h-4 text-blue-500" />
            <span className="font-medium text-sm">Configuration Summary</span>
          </div>
          <div className="text-xs text-muted-foreground space-y-1">
            <div>
              Embedding Model: {config.embedding_config?.model_name || 'None selected'}
            </div>
            <div>
              Text Cleaning: {config.text_cleaning_level} level
            </div>
            <div>
              Analysis Features: {[
                config.generate_correlations && 'Correlations',
                config.create_tensor_visualizations && 'Tensors',
                config.quality_analysis && 'Quality',
                config.token_analysis && 'Tokens'
              ].filter(Boolean).join(', ') || 'None'}
            </div>
            {config.remove_duplicates_by_embedding && (
              <div>
                Deduplication: Similarity threshold {config.similarity_threshold}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};