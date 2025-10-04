import React, { useState } from 'react';
import { Settings, Sliders, Eye, BarChart3, Target, AlertTriangle, Info, Zap, Plus, Scale } from 'lucide-react';
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
    raw_cleaning: false,
    tokenization: false,
    data_augmentation: false,
    class_balancing: false,
    analysis: false,
    deduplication: false
  });

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const defaultCleaningOptions = {
    deduplication_method: 'hash' as const,
    normalize_whitespace: true,
    language_detection: false,
    utf8_sanitization: true
  };

  const defaultTokenizationOptions = {
    method: 'bpe' as const,
    chunking_strategy: 'fixed' as const,
    chunk_size: 512,
    overlap: 0,
    max_length_handling: 'truncate' as const
  };

  const defaultDataAugmentationOptions = {
    enabled: false,
    techniques: [] as ('back_translation' | 'synonym_replacement' | 'paraphrasing' | 'noise_injection')[],
    ratio: 0.2,
    preview: false
  };

  const defaultClassBalancingOptions = {
    enabled: false,
    method: 'none' as const,
    target_ratio: 1.0,
    stratified_split: true,
    k_neighbors: 5
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

          <div className="ml-6 space-y-4">
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
              <div className="space-y-4">
                {/* Model Cards with Trade-offs */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* MiniLM Card */}
                  <div className={`
                    p-4 border rounded-lg cursor-pointer transition-all
                    ${config.embedding_config?.model_name === 'all-MiniLM-L6-v2'
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-border hover:border-blue-300'
                    }
                  `} onClick={() => handleEmbeddingModelSelect({
                    model_name: 'all-MiniLM-L6-v2',
                    model_type: 'sentence-transformers',
                    max_sequence_length: 512,
                    normalize_embeddings: true,
                    batch_size: 32,
                    device: 'cpu',
                    embedding_dimension: 384
                  })}>
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                        <h4 className="font-medium">all-MiniLM-L6-v2</h4>
                      </div>
                      {config.embedding_config?.model_name === 'all-MiniLM-L6-v2' && (
                        <div className="text-blue-500">✓</div>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">Lightweight sentence embeddings</p>
                    <div className="mt-3 space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span>Pros:</span>
                        <span className="font-medium">Fast & Light</span>
                      </div>
                      <div className="text-xs text-muted-foreground">• 384-dim, 22MB model size</div>
                      <div className="text-xs text-muted-foreground">• Excellent for semantic search</div>
                      <div className="text-xs text-muted-foreground">• CPU-friendly inference</div>
                      <div className="flex items-center justify-between text-xs mt-2">
                        <span>Cons:</span>
                        <span className="font-medium">General Purpose</span>
                      </div>
                      <div className="text-xs text-muted-foreground">• Less specialized for domains</div>
                    </div>
                  </div>

                  {/* BGE Small Card */}
                  <div className={`
                    p-4 border rounded-lg cursor-pointer transition-all
                    ${config.embedding_config?.model_name === 'bge-small-en'
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-border hover:border-blue-300'
                    }
                  `} onClick={() => handleEmbeddingModelSelect({
                    model_name: 'bge-small-en',
                    model_type: 'bge',
                    max_sequence_length: 512,
                    normalize_embeddings: true,
                    batch_size: 16,
                    device: 'cpu',
                    embedding_dimension: 384
                  })}>
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <h4 className="font-medium">bge-small-en</h4>
                      </div>
                      {config.embedding_config?.model_name === 'bge-small-en' && (
                        <div className="text-blue-500">✓</div>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">BAAI General Embedding</p>
                    <div className="mt-3 space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span>Pros:</span>
                        <span className="font-medium">High Accuracy</span>
                      </div>
                      <div className="text-xs text-muted-foreground">• State-of-the-art retrieval</div>
                      <div className="text-xs text-muted-foreground">• Strong multilingual support</div>
                      <div className="text-xs text-muted-foreground">• Optimized for RAG systems</div>
                      <div className="flex items-center justify-between text-xs mt-2">
                        <span>Cons:</span>
                        <span className="font-medium">Slightly Slower</span>
                      </div>
                      <div className="text-xs text-muted-foreground">• 33MB model size</div>
                    </div>
                  </div>

                  {/* Add more model cards as needed */}
                  <div className="md:col-span-2 text-xs text-muted-foreground text-center py-2 border-t">
                    More models available: paraphrase-MiniLM, multi-qa-MiniLM, etc.
                  </div>
                </div>

                {/* Batch Size Configuration */}
                <div>
                  <label className="block text-sm font-medium mb-2">Batch Size</label>
                  <input
                    type="number"
                    min="1"
                    max="128"
                    step="1"
                    value={config.embedding_config?.batch_size ?? 32}
                    onChange={(e) => {
                      if (config.embedding_config) {
                        handleEmbeddingModelSelect({
                          ...config.embedding_config,
                          batch_size: parseInt(e.target.value)
                        });
                      }
                    }}
                    className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                  <div className="text-xs text-muted-foreground mt-1">
                    Number of samples processed simultaneously (higher = faster but more memory)
                  </div>
                </div>
              </div>
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

        {/* Raw Data Cleaning Section */}
        <div className="space-y-3">
          <div
            className="flex items-center justify-between cursor-pointer p-2 rounded hover:bg-accent"
            onClick={() => toggleSection('raw_cleaning')}
          >
            <div className="flex items-center space-x-2">
              <Zap className="w-4 h-4 text-indigo-500" />
              <span className="font-medium">Raw Data Cleaning</span>
              <span className="text-xs text-muted-foreground">
                Advanced cleaning pipeline
              </span>
            </div>
            <div className="text-muted-foreground">
              {expandedSections.raw_cleaning ? '−' : '+'}
            </div>
          </div>

          {expandedSections.raw_cleaning && (
            <div className="ml-6 space-y-4">
              {/* Deduplication Method */}
              <div>
                <label className="block text-sm font-medium mb-2">Deduplication Method</label>
                <div className="space-y-2">
                  {(['none', 'hash', 'similarity'] as const).map((method) => (
                    <label key={method} className="flex items-start space-x-3">
                      <input
                        type="radio"
                        value={method}
                        checked={(config.cleaning_options?.deduplication_method || 'hash') === method}
                        onChange={(e) => {
                          const current = config.cleaning_options ?? defaultCleaningOptions;
                          updateConfig({
                            cleaning_options: {
                              ...current,
                              deduplication_method: e.target.value as typeof method
                            }
                          });
                        }}
                        className="mt-1 accent-indigo-500"
                      />
                      <div>
                        <div className="capitalize font-medium">{method}</div>
                        <div className="text-xs text-muted-foreground">
                          {method === 'none' ? 'No deduplication' :
                           method === 'hash' ? 'Fast hashing for exact duplicates' :
                           'Semantic similarity using embeddings'}
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Individual Cleaning Options */}
              <div className="space-y-3">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.cleaning_options?.normalize_whitespace ?? true}
                    onChange={(e) => {
                      const current = config.cleaning_options ?? defaultCleaningOptions;
                      updateConfig({
                        cleaning_options: {
                          ...current,
                          normalize_whitespace: e.target.checked
                        }
                      });
                    }}
                    className="accent-indigo-500"
                  />
                  <span className="text-sm">Normalize Whitespace</span>
                  <div className="text-xs text-muted-foreground ml-6">
                    Standardize multiple spaces and line breaks
                  </div>
                </label>

                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.cleaning_options?.language_detection ?? false}
                    onChange={(e) => {
                      const current = config.cleaning_options ?? defaultCleaningOptions;
                      updateConfig({
                        cleaning_options: {
                          ...current,
                          language_detection: e.target.checked
                        }
                      });
                    }}
                    className="accent-indigo-500"
                  />
                  <span className="text-sm">Language Detection</span>
                  <div className="text-xs text-muted-foreground ml-6">
                    Detect and filter by language using langdetect
                  </div>
                </label>

                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.cleaning_options?.utf8_sanitization ?? true}
                    onChange={(e) => {
                      const current = config.cleaning_options ?? defaultCleaningOptions;
                      updateConfig({
                        cleaning_options: {
                          ...current,
                          utf8_sanitization: e.target.checked
                        }
                      });
                    }}
                    className="accent-indigo-500"
                  />
                  <span className="text-sm">UTF-8 Sanitization</span>
                  <div className="text-xs text-muted-foreground ml-6">
                    Remove non-UTF-8 characters with regex
                  </div>
                </label>
              </div>
            </div>
          )}
        </div>

        {/* Tokenization & Chunking Section */}
        <div className="space-y-3">
          <div
            className="flex items-center justify-between cursor-pointer p-2 rounded hover:bg-accent"
            onClick={() => toggleSection('tokenization')}
          >
            <div className="flex items-center space-x-2">
              <Zap className="w-4 h-4 text-indigo-500" />
              <span className="font-medium">Tokenization & Chunking</span>
              <span className="text-xs text-muted-foreground">
                Configure tokenization and chunking strategies
              </span>
            </div>
            <div className="text-muted-foreground">
              {expandedSections.tokenization ? '−' : '+'}
            </div>
          </div>

          {expandedSections.tokenization && (
            <div className="ml-6 space-y-4">
              {/* Tokenization Method */}
              <div>
                <label className="block text-sm font-medium mb-2">Tokenization Method</label>
                <div className="space-y-2">
                  {(['bpe', 'sentencepiece', 'custom'] as const).map((method) => (
                    <label key={method} className="flex items-start space-x-3">
                      <input
                        type="radio"
                        value={method}
                        checked={(config.tokenization_and_chunking?.method || 'bpe') === method}
                        onChange={(e) => {
                          const current = config.tokenization_and_chunking ?? defaultTokenizationOptions;
                          updateConfig({
                            tokenization_and_chunking: {
                              ...current,
                              method: e.target.value as typeof method
                            }
                          });
                        }}
                        className="mt-1 accent-indigo-500"
                      />
                      <div>
                        <div className="capitalize font-medium">{method}</div>
                        <div className="text-xs text-muted-foreground">
                          {method === 'bpe' ? 'Byte-Pair Encoding (default for GPT models)' :
                           method === 'sentencepiece' ? 'SentencePiece (recommended for multilingual)' :
                           'Custom tokenizer implementation'}
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Chunking Strategy */}
              <div>
                <label className="block text-sm font-medium mb-2">Chunking Strategy</label>
                <div className="space-y-2">
                  {(['fixed', 'sliding_window', 'semantic'] as const).map((strategy) => (
                    <label key={strategy} className="flex items-start space-x-3">
                      <input
                        type="radio"
                        value={strategy}
                        checked={(config.tokenization_and_chunking?.chunking_strategy || 'fixed') === strategy}
                        onChange={(e) => {
                          const current = config.tokenization_and_chunking ?? defaultTokenizationOptions;
                          updateConfig({
                            tokenization_and_chunking: {
                              ...current,
                              chunking_strategy: e.target.value as typeof strategy
                            }
                          });
                        }}
                        className="mt-1 accent-indigo-500"
                      />
                      <div>
                        <div className="capitalize font-medium">{strategy.replace('_', ' ')}</div>
                        <div className="text-xs text-muted-foreground">
                          {strategy === 'fixed' ? 'Fixed-length chunks without overlap' :
                           strategy === 'sliding_window' ? 'Overlapping sliding window approach' :
                           'Semantic-aware chunking using sentence boundaries'}
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Chunk Size Slider */}
              <div>
                <label className="block text-sm font-medium mb-2">Chunk Size (tokens)</label>
                <input
                  type="range"
                  min="64"
                  max="4096"
                  step="64"
                  value={config.tokenization_and_chunking?.chunk_size || defaultTokenizationOptions.chunk_size}
                  onChange={(e) => {
                    const current = config.tokenization_and_chunking ?? defaultTokenizationOptions;
                    updateConfig({
                      tokenization_and_chunking: {
                        ...current,
                        chunk_size: parseInt(e.target.value)
                      }
                    });
                  }}
                  className="w-full accent-indigo-500"
                />
                <div className="text-xs text-muted-foreground mt-1">
                  {config.tokenization_and_chunking?.chunk_size || defaultTokenizationOptions.chunk_size} tokens
                </div>
              </div>

              {/* Overlap Slider */}
              <div>
                <label className="block text-sm font-medium mb-2">Overlap (tokens)</label>
                <input
                  type="range"
                  min="0"
                  max="512"
                  step="32"
                  value={config.tokenization_and_chunking?.overlap || defaultTokenizationOptions.overlap}
                  onChange={(e) => {
                    const current = config.tokenization_and_chunking ?? defaultTokenizationOptions;
                    updateConfig({
                      tokenization_and_chunking: {
                        ...current,
                        overlap: parseInt(e.target.value)
                      }
                    });
                  }}
                  className="w-full accent-indigo-500"
                />
                <div className="text-xs text-muted-foreground mt-1">
                  {config.tokenization_and_chunking?.overlap || defaultTokenizationOptions.overlap} tokens overlap
                </div>
              </div>

              {/* Max Length Handling */}
              <div>
                <label className="block text-sm font-medium mb-2">Max Length Handling</label>
                <div className="space-y-2">
                  {(['truncate', 'pad', 'split'] as const).map((handling) => (
                    <label key={handling} className="flex items-start space-x-3">
                      <input
                        type="radio"
                        value={handling}
                        checked={(config.tokenization_and_chunking?.max_length_handling || 'truncate') === handling}
                        onChange={(e) => {
                          const current = config.tokenization_and_chunking ?? defaultTokenizationOptions;
                          updateConfig({
                            tokenization_and_chunking: {
                              ...current,
                              max_length_handling: e.target.value as typeof handling
                            }
                          });
                        }}
                        className="mt-1 accent-indigo-500"
                      />
                      <div>
                        <div className="capitalize font-medium">{handling}</div>
                        <div className="text-xs text-muted-foreground">
                          {handling === 'truncate' ? 'Truncate exceeding tokens' :
                           handling === 'pad' ? 'Pad short sequences with padding token' :
                           'Split long sequences into multiple chunks'}
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Data Augmentation Section */}
        <div className="space-y-3">
          <div
            className="flex items-center justify-between cursor-pointer p-2 rounded hover:bg-accent"
            onClick={() => toggleSection('data_augmentation')}
          >
            <div className="flex items-center space-x-2">
              <Plus className="w-4 h-4 text-green-500" />
              <span className="font-medium">Data Augmentation</span>
              <span className="text-xs text-muted-foreground">
                Enhance dataset diversity and size
              </span>
            </div>
            <div className="text-muted-foreground">
              {expandedSections.data_augmentation ? '−' : '+'}
            </div>
          </div>

          {expandedSections.data_augmentation && (
            <div className="ml-6 space-y-4">
              {/* Enable Augmentation */}
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.data_augmentation?.enabled ?? false}
                  onChange={(e) => {
                    const current = config.data_augmentation ?? defaultDataAugmentationOptions;
                    updateConfig({
                      data_augmentation: {
                        ...current,
                        enabled: e.target.checked
                      }
                    });
                  }}
                  className="accent-green-500"
                />
                <span className="text-sm font-medium">Enable Data Augmentation</span>
              </label>

              {/* Techniques Selection */}
              <div>
                <label className="block text-sm font-medium mb-2">Augmentation Techniques</label>
                <div className="space-y-2">
                  {(['back_translation', 'synonym_replacement', 'paraphrasing', 'noise_injection'] as const).map((technique) => (
                    <label key={technique} className="flex items-start space-x-3">
                      <input
                        type="checkbox"
                        checked={(config.data_augmentation?.techniques ?? []).includes(technique)}
                        onChange={(e) => {
                          const current = config.data_augmentation ?? defaultDataAugmentationOptions;
                          const techniques = e.target.checked
                            ? [...(current.techniques ?? []), technique]
                            : (current.techniques ?? []).filter(t => t !== technique);
                          updateConfig({
                            data_augmentation: {
                              ...current,
                              techniques
                            }
                          });
                        }}
                        className="mt-1 accent-green-500"
                        disabled={!config.data_augmentation?.enabled}
                      />
                      <div>
                        <div className="capitalize font-medium">{technique.replace('_', ' ')}</div>
                        <div className="text-xs text-muted-foreground">
                          {technique === 'back_translation' ? 'Translate to another language and back' :
                           technique === 'synonym_replacement' ? 'Replace words with synonyms' :
                           technique === 'paraphrasing' ? 'Rewrite sentences while preserving meaning' :
                           'Add controlled noise to text'}
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Augmentation Ratio Slider */}
              <div>
                <label className="block text-sm font-medium mb-2">Augmentation Ratio</label>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={config.data_augmentation?.ratio ?? defaultDataAugmentationOptions.ratio}
                  onChange={(e) => {
                    const current = config.data_augmentation ?? defaultDataAugmentationOptions;
                    updateConfig({
                      data_augmentation: {
                        ...current,
                        ratio: parseFloat(e.target.value)
                      }
                    });
                  }}
                  className="w-full accent-green-500"
                  disabled={!config.data_augmentation?.enabled}
                />
                <div className="text-xs text-muted-foreground mt-1">
                  {((config.data_augmentation?.ratio ?? defaultDataAugmentationOptions.ratio) * 100).toFixed(0)}% of original data
                </div>
              </div>

              {/* Preview Toggle */}
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.data_augmentation?.preview ?? false}
                  onChange={(e) => {
                    const current = config.data_augmentation ?? defaultDataAugmentationOptions;
                    updateConfig({
                      data_augmentation: {
                        ...current,
                        preview: e.target.checked
                      }
                    });
                  }}
                  className="accent-green-500"
                  disabled={!config.data_augmentation?.enabled}
                />
                <span className="text-sm">Show augmentation preview in results</span>
              </label>

              {/* Selected Techniques Count */}
              {config.data_augmentation?.enabled && (
                <div className="text-xs text-muted-foreground">
                  {config.data_augmentation.techniques?.length ?? 0} techniques selected
                </div>
              )}
            </div>
          )}
        </div>

        {/* Class Balancing Section */}
        <div className="space-y-3">
          <div
            className="flex items-center justify-between cursor-pointer p-2 rounded hover:bg-accent"
            onClick={() => toggleSection('class_balancing')}
          >
            <div className="flex items-center space-x-2">
              <Scale className="w-4 h-4 text-blue-500" />
              <span className="font-medium">Class Balancing</span>
              <span className="text-xs text-muted-foreground">
                Balance class distribution for supervised learning
              </span>
            </div>
            <div className="text-muted-foreground">
              {expandedSections.class_balancing ? '−' : '+'}
            </div>
          </div>

          {expandedSections.class_balancing && (
            <div className="ml-6 space-y-4">
              {/* Enable Class Balancing */}
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.class_balancing?.enabled ?? false}
                  onChange={(e) => {
                    const current = config.class_balancing ?? defaultClassBalancingOptions;
                    updateConfig({
                      class_balancing: {
                        ...current,
                        enabled: e.target.checked
                      }
                    });
                  }}
                  className="accent-blue-500"
                />
                <span className="text-sm font-medium">Enable Class Balancing</span>
              </label>

              {/* Balancing Method */}
              <div>
                <label className="block text-sm font-medium mb-2">Balancing Method</label>
                <div className="space-y-2">
                  {(['none', 'upsample', 'downsample', 'smote'] as const).map((method) => (
                    <label key={method} className="flex items-start space-x-3">
                      <input
                        type="radio"
                        value={method}
                        checked={(config.class_balancing?.method || 'none') === method}
                        onChange={(e) => {
                          const current = config.class_balancing ?? defaultClassBalancingOptions;
                          updateConfig({
                            class_balancing: {
                              ...current,
                              method: e.target.value as typeof method
                            }
                          });
                        }}
                        className="mt-1 accent-blue-500"
                        disabled={!config.class_balancing?.enabled}
                      />
                      <div>
                        <div className="capitalize font-medium">{method}</div>
                        <div className="text-xs text-muted-foreground">
                          {method === 'none' ? 'No balancing (original distribution)' :
                           method === 'upsample' ? 'Duplicate minority class samples' :
                           method === 'downsample' ? 'Remove majority class samples' :
                           'Synthetic Minority Over-sampling Technique (SMOTE)'}
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Target Ratio Slider */}
              <div>
                <label className="block text-sm font-medium mb-2">Target Class Ratio</label>
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  value={config.class_balancing?.target_ratio ?? defaultClassBalancingOptions.target_ratio}
                  onChange={(e) => {
                    const current = config.class_balancing ?? defaultClassBalancingOptions;
                    updateConfig({
                      class_balancing: {
                        ...current,
                        target_ratio: parseFloat(e.target.value)
                      }
                    });
                  }}
                  className="w-full accent-blue-500"
                  disabled={!config.class_balancing?.enabled}
                />
                <div className="text-xs text-muted-foreground mt-1">
                  {config.class_balancing?.target_ratio ?? defaultClassBalancingOptions.target_ratio}:1 minority to majority ratio
                </div>
              </div>

              {/* Stratified Split Toggle */}
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.class_balancing?.stratified_split ?? true}
                  onChange={(e) => {
                    const current = config.class_balancing ?? defaultClassBalancingOptions;
                    updateConfig({
                      class_balancing: {
                        ...current,
                        stratified_split: e.target.checked
                      }
                    });
                  }}
                  className="accent-blue-500"
                  disabled={!config.class_balancing?.enabled}
                />
                <span className="text-sm">Use stratified train/test split</span>
              </label>

              {/* K Neighbors for SMOTE */}
              {config.class_balancing?.method === 'smote' && config.class_balancing?.enabled && (
                <div>
                  <label className="block text-sm font-medium mb-2">SMOTE K-Neighbors</label>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    step="1"
                    value={config.class_balancing?.k_neighbors ?? defaultClassBalancingOptions.k_neighbors}
                    onChange={(e) => {
                      const current = config.class_balancing ?? defaultClassBalancingOptions;
                      updateConfig({
                        class_balancing: {
                          ...current,
                          k_neighbors: parseInt(e.target.value)
                        }
                      });
                    }}
                    className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                  <div className="text-xs text-muted-foreground mt-1">
                    Number of nearest neighbors for synthetic sample generation
                  </div>
                </div>
              )}
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
            {config.cleaning_options && (
              <div>
                Raw Cleaning: Dedup {config.cleaning_options.deduplication_method},
                Whitespace: {config.cleaning_options.normalize_whitespace ? 'On' : 'Off'},
                Lang Detect: {config.cleaning_options.language_detection ? 'On' : 'Off'},
                UTF-8: {config.cleaning_options.utf8_sanitization ? 'On' : 'Off'}
              </div>
            )}
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