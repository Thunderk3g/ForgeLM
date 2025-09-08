import React, { useState, useEffect } from 'react';
import { Play, Square, Eye, Download, RefreshCw, Settings, Database, Zap, CheckCircle, XCircle, Clock } from 'lucide-react';

interface GenerationConfig {
  task_type: string;
  template_format: string;
  num_samples: number;
  model_pair_id?: string;
  domain: string;
  difficulty_level: string;
  generation_strategy: string;
  quality_threshold: number;
  max_tokens: number;
  temperature: number;
  diversity_factor: number;
  seed_prompts?: string[];
}

interface GenerationTask {
  task_id: string;
  status: string;
  progress: number;
  samples_generated: number;
  samples_validated: number;
  started_at: string;
  completed_at?: string;
  dataset_path?: string;
  config: {
    task_type: string;
    template_format: string;
    num_samples: number;
  };
  errors?: string[];
  error?: string;
}

interface GeneratedDataset {
  filename: string;
  path: string;
  created_at: string;
  size_mb: number;
  sample_count: number;
  template_format: string;
  task_type: string;
  quality_score: number;
}

interface DataGenerationProps {
  onDatasetGenerated?: (dataset: GeneratedDataset) => void;
}

const DataGeneration: React.FC<DataGenerationProps> = ({ onDatasetGenerated }) => {
  // State for form configuration
  const [config, setConfig] = useState<GenerationConfig>({
    task_type: 'instruction_tuning',
    template_format: 'alpaca',
    num_samples: 100,
    domain: 'general',
    difficulty_level: 'mixed',
    generation_strategy: 'self_supervised',
    quality_threshold: 0.7,
    max_tokens: 512,
    temperature: 0.8,
    diversity_factor: 0.5
  });

  // State for generation management
  const [currentTask, setCurrentTask] = useState<GenerationTask | null>(null);
  const [generatedDatasets, setGeneratedDatasets] = useState<GeneratedDataset[]>([]);
  const [activeTab, setActiveTab] = useState<'configure' | 'monitor' | 'datasets'>('configure');
  const [isGenerating, setIsGenerating] = useState(false);
  const [validationResult, setValidationResult] = useState<any>(null);
  const [previewData, setPreviewData] = useState<any>(null);
  const [showPreview, setShowPreview] = useState(false);
  
  // Available options (would be fetched from API)
  const templates = {
    task_types: [
      { id: 'instruction_tuning', name: 'Instruction Tuning', description: 'Generate instruction-response pairs for fine-tuning' },
      { id: 'conversation', name: 'Conversation', description: 'Generate conversational exchanges' },
      { id: 'domain_specific', name: 'Domain Specific', description: 'Generate domain-specific training data' }
    ],
    template_formats: [
      { id: 'alpaca', name: 'Alpaca', description: 'Stanford Alpaca format with instruction/input/output' },
      { id: 'chatml', name: 'ChatML', description: 'OpenAI ChatML format with role-based messages' },
      { id: 'sharegpt', name: 'ShareGPT', description: 'ShareGPT conversation format' },
      { id: 'qwen', name: 'Qwen', description: 'Qwen chat format with system/user/assistant roles' }
    ],
    domains: [
      { id: 'general', name: 'General Knowledge', description: 'General purpose knowledge and reasoning' },
      { id: 'code', name: 'Programming', description: 'Code generation and programming assistance' },
      { id: 'math', name: 'Mathematics', description: 'Mathematical problems and explanations' },
      { id: 'science', name: 'Science', description: 'Scientific concepts and explanations' }
    ]
  };

  // Load generated datasets on component mount
  useEffect(() => {
    loadGeneratedDatasets();
  }, []);

  // Poll for task status when generating
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isGenerating && currentTask) {
      interval = setInterval(async () => {
        try {
          const response = await fetch(`/api/data-generation/status/${currentTask.task_id}`);
          if (response.ok) {
            const updatedTask = await response.json();
            setCurrentTask(updatedTask);
            
            if (updatedTask.status === 'completed' || updatedTask.status === 'error' || updatedTask.status === 'cancelled') {
              setIsGenerating(false);
              if (updatedTask.status === 'completed') {
                loadGeneratedDatasets(); // Refresh datasets list
                if (onDatasetGenerated) {
                  onDatasetGenerated({
                    filename: updatedTask.dataset_path?.split('/').pop() || '',
                    path: updatedTask.dataset_path || '',
                    created_at: updatedTask.completed_at || '',
                    size_mb: 0, // Will be updated when datasets list refreshes
                    sample_count: updatedTask.samples_validated,
                    template_format: updatedTask.config.template_format,
                    task_type: updatedTask.config.task_type,
                    quality_score: 0 // Will be updated when datasets list refreshes
                  });
                }
              }
            }
          }
        } catch (error) {
          console.error('Error polling task status:', error);
        }
      }, 2000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isGenerating, currentTask, onDatasetGenerated]);

  const loadGeneratedDatasets = async () => {
    try {
      const response = await fetch('/api/data-generation/datasets');
      if (response.ok) {
        const data = await response.json();
        setGeneratedDatasets(data.datasets || []);
      }
    } catch (error) {
      console.error('Error loading generated datasets:', error);
    }
  };

  const validateConfig = async () => {
    try {
      const response = await fetch('/api/data-generation/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      
      if (response.ok) {
        const result = await response.json();
        setValidationResult(result);
      }
    } catch (error) {
      console.error('Error validating config:', error);
    }
  };

  const startGeneration = async () => {
    try {
      setIsGenerating(true);
      const response = await fetch('/api/data-generation/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      
      if (response.ok) {
        const result = await response.json();
        setCurrentTask({
          task_id: result.task_id,
          status: 'initializing',
          progress: 0,
          samples_generated: 0,
          samples_validated: 0,
          started_at: new Date().toISOString(),
          config: result.config
        });
        setActiveTab('monitor');
      } else {
        setIsGenerating(false);
        const error = await response.json();
        alert(`Generation failed: ${error.detail?.message || error.detail}`);
      }
    } catch (error) {
      setIsGenerating(false);
      console.error('Error starting generation:', error);
      alert('Failed to start generation');
    }
  };

  const stopGeneration = async () => {
    if (currentTask) {
      try {
        const response = await fetch(`/api/data-generation/cancel/${currentTask.task_id}`, {
          method: 'POST'
        });
        
        if (response.ok) {
          setIsGenerating(false);
          setCurrentTask(prev => prev ? { ...prev, status: 'cancelled' } : null);
        }
      } catch (error) {
        console.error('Error stopping generation:', error);
      }
    }
  };

  const previewDataset = async (filename: string) => {
    try {
      const response = await fetch(`/api/data-generation/preview/${filename}?limit=3`);
      if (response.ok) {
        const data = await response.json();
        setPreviewData(data);
        setShowPreview(true);
      }
    } catch (error) {
      console.error('Error previewing dataset:', error);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'error':
      case 'cancelled':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'running':
        return <RefreshCw className="w-5 h-5 text-blue-500 animate-spin" />;
      default:
        return <Clock className="w-5 h-5 text-yellow-500" />;
    }
  };

  const formatFileSize = (sizeInMB: number) => {
    if (sizeInMB < 1) {
      return `${Math.round(sizeInMB * 1024)} KB`;
    }
    return `${sizeInMB.toFixed(1)} MB`;
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-2">
          <Database className="w-8 h-8 text-blue-600" />
          Automated Data Generation
        </h1>
        <p className="text-gray-600 mt-2">
          Generate high-quality training datasets using your available models
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b mb-6">
        <button
          className={`px-4 py-2 font-medium ${activeTab === 'configure' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-500'}`}
          onClick={() => setActiveTab('configure')}
        >
          <Settings className="w-4 h-4 inline mr-2" />
          Configure
        </button>
        <button
          className={`px-4 py-2 font-medium ${activeTab === 'monitor' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-500'}`}
          onClick={() => setActiveTab('monitor')}
        >
          <Zap className="w-4 h-4 inline mr-2" />
          Monitor
        </button>
        <button
          className={`px-4 py-2 font-medium ${activeTab === 'datasets' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-500'}`}
          onClick={() => setActiveTab('datasets')}
        >
          <Database className="w-4 h-4 inline mr-2" />
          Generated Datasets
        </button>
      </div>

      {/* Configuration Tab */}
      {activeTab === 'configure' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Basic Configuration */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Basic Configuration</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Task Type</label>
                  <select
                    value={config.task_type}
                    onChange={(e) => setConfig(prev => ({ ...prev, task_type: e.target.value }))}
                    className="w-full p-2 border rounded-md"
                  >
                    {templates.task_types.map(type => (
                      <option key={type.id} value={type.id}>{type.name}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Template Format</label>
                  <select
                    value={config.template_format}
                    onChange={(e) => setConfig(prev => ({ ...prev, template_format: e.target.value }))}
                    className="w-full p-2 border rounded-md"
                  >
                    {templates.template_formats.map(format => (
                      <option key={format.id} value={format.id}>{format.name}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Domain</label>
                  <select
                    value={config.domain}
                    onChange={(e) => setConfig(prev => ({ ...prev, domain: e.target.value }))}
                    className="w-full p-2 border rounded-md"
                  >
                    {templates.domains.map(domain => (
                      <option key={domain.id} value={domain.id}>{domain.name}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Number of Samples</label>
                  <input
                    type="number"
                    value={config.num_samples}
                    onChange={(e) => setConfig(prev => ({ ...prev, num_samples: parseInt(e.target.value) || 100 }))}
                    min={1}
                    max={10000}
                    className="w-full p-2 border rounded-md"
                  />
                </div>
              </div>
            </div>

            {/* Advanced Configuration */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Advanced Settings</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Quality Threshold ({config.quality_threshold})
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.1"
                    value={config.quality_threshold}
                    onChange={(e) => setConfig(prev => ({ ...prev, quality_threshold: parseFloat(e.target.value) }))}
                    className="w-full"
                  />
                  <div className="text-xs text-gray-500">Higher values = stricter quality filtering</div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Temperature ({config.temperature})
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="1.5"
                    step="0.1"
                    value={config.temperature}
                    onChange={(e) => setConfig(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                    className="w-full"
                  />
                  <div className="text-xs text-gray-500">Higher values = more creative responses</div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Max Tokens</label>
                  <select
                    value={config.max_tokens}
                    onChange={(e) => setConfig(prev => ({ ...prev, max_tokens: parseInt(e.target.value) }))}
                    className="w-full p-2 border rounded-md"
                  >
                    <option value={256}>256 (Short)</option>
                    <option value={512}>512 (Medium)</option>
                    <option value={1024}>1024 (Long)</option>
                    <option value={2048}>2048 (Very Long)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Generation Strategy</label>
                  <select
                    value={config.generation_strategy}
                    onChange={(e) => setConfig(prev => ({ ...prev, generation_strategy: e.target.value }))}
                    className="w-full p-2 border rounded-md"
                  >
                    <option value="self_supervised">Self-Supervised</option>
                    <option value="iterative_refinement">Iterative Refinement</option>
                    <option value="multi_model_consensus">Multi-Model Consensus</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* Validation Results */}
          {validationResult && (
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Configuration Validation</h3>
              {validationResult.valid ? (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-green-600">
                    <CheckCircle className="w-5 h-5" />
                    Configuration is valid and ready for generation
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div>
                      <strong>Estimated Time:</strong> ~{validationResult.estimates?.estimated_time_minutes} minutes
                    </div>
                    <div>
                      <strong>Expected Dataset Size:</strong> {validationResult.estimates?.estimated_dataset_size_mb} MB
                    </div>
                    <div>
                      <strong>Expected Valid Samples:</strong> {validationResult.estimates?.expected_valid_samples}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-red-600">
                  <XCircle className="w-5 h-5" />
                  {validationResult.message}
                </div>
              )}
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-4">
            <button
              onClick={validateConfig}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
            >
              <CheckCircle className="w-4 h-4" />
              Validate Configuration
            </button>
            
            <button
              onClick={startGeneration}
              disabled={isGenerating || !validationResult?.valid}
              className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 flex items-center gap-2"
            >
              <Play className="w-4 h-4" />
              Start Generation
            </button>
          </div>
        </div>
      )}

      {/* Monitor Tab */}
      {activeTab === 'monitor' && (
        <div className="space-y-6">
          {currentTask ? (
            <div className="bg-white p-6 rounded-lg shadow">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-lg font-semibold flex items-center gap-2">
                    {getStatusIcon(currentTask.status)}
                    Generation Task: {currentTask.task_id.substring(0, 8)}...
                  </h3>
                  <p className="text-gray-600">
                    {currentTask.config.task_type} • {currentTask.config.template_format} • {currentTask.config.num_samples} samples
                  </p>
                </div>
                
                {isGenerating && (
                  <button
                    onClick={stopGeneration}
                    className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center gap-2"
                  >
                    <Square className="w-4 h-4" />
                    Stop
                  </button>
                )}
              </div>

              {/* Progress Bar */}
              <div className="mb-4">
                <div className="flex justify-between text-sm text-gray-600 mb-1">
                  <span>Progress</span>
                  <span>{currentTask.progress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${currentTask.progress}%` }}
                  />
                </div>
              </div>

              {/* Statistics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <div className="font-medium text-gray-900">{currentTask.samples_generated}</div>
                  <div className="text-gray-600">Generated</div>
                </div>
                <div>
                  <div className="font-medium text-gray-900">{currentTask.samples_validated}</div>
                  <div className="text-gray-600">Validated</div>
                </div>
                <div>
                  <div className="font-medium text-gray-900">{currentTask.status}</div>
                  <div className="text-gray-600">Status</div>
                </div>
                <div>
                  <div className="font-medium text-gray-900">
                    {new Date(currentTask.started_at).toLocaleTimeString()}
                  </div>
                  <div className="text-gray-600">Started</div>
                </div>
              </div>

              {/* Errors */}
              {currentTask.error && (
                <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
                  <div className="text-red-800 font-medium">Error:</div>
                  <div className="text-red-700">{currentTask.error}</div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              No active generation task. Start a new generation from the Configure tab.
            </div>
          )}
        </div>
      )}

      {/* Datasets Tab */}
      {activeTab === 'datasets' && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Generated Datasets ({generatedDatasets.length})</h3>
            <button
              onClick={loadGeneratedDatasets}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
          </div>

          {generatedDatasets.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {generatedDatasets.map((dataset, index) => (
                <div key={index} className="bg-white p-4 rounded-lg shadow hover:shadow-md transition-shadow">
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex-1 min-w-0">
                      <h4 className="font-medium text-gray-900 truncate">{dataset.filename}</h4>
                      <p className="text-sm text-gray-600">{dataset.task_type} • {dataset.template_format}</p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-2 text-xs text-gray-600 mb-3">
                    <div>{dataset.sample_count} samples</div>
                    <div>{formatFileSize(dataset.size_mb)}</div>
                    <div>Quality: {(dataset.quality_score * 100).toFixed(0)}%</div>
                    <div>{new Date(dataset.created_at).toLocaleDateString()}</div>
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() => previewDataset(dataset.filename)}
                      className="flex-1 px-3 py-1 bg-blue-100 text-blue-700 rounded text-xs hover:bg-blue-200 flex items-center justify-center gap-1"
                    >
                      <Eye className="w-3 h-3" />
                      Preview
                    </button>
                    <button
                      onClick={() => window.open(`/api/data-generation/download/${dataset.filename}`, '_blank')}
                      className="flex-1 px-3 py-1 bg-green-100 text-green-700 rounded text-xs hover:bg-green-200 flex items-center justify-center gap-1"
                    >
                      <Download className="w-3 h-3" />
                      Download
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              No datasets generated yet. Start your first generation from the Configure tab.
            </div>
          )}
        </div>
      )}

      {/* Preview Modal */}
      {showPreview && previewData && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl max-h-[80vh] overflow-auto">
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-lg font-semibold">{previewData.dataset_filename}</h3>
                  <p className="text-gray-600">
                    {previewData.total_samples} total samples • Showing first {previewData.preview_count}
                  </p>
                </div>
                <button
                  onClick={() => setShowPreview(false)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ×
                </button>
              </div>

              <div className="space-y-4">
                {previewData.preview_samples?.map((sample: any, index: number) => (
                  <div key={index} className="border rounded-lg p-4">
                    <div className="space-y-2">
                      <div>
                        <div className="text-sm font-medium text-gray-700">Instruction:</div>
                        <div className="text-sm text-gray-900 bg-gray-50 p-2 rounded">
                          {sample.instruction}
                        </div>
                      </div>
                      <div>
                        <div className="text-sm font-medium text-gray-700">Response:</div>
                        <div className="text-sm text-gray-900 bg-gray-50 p-2 rounded">
                          {sample.response}
                        </div>
                      </div>
                      {sample.metadata?.quality_score && (
                        <div className="text-xs text-gray-500">
                          Quality Score: {(sample.metadata.quality_score * 100).toFixed(0)}%
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataGeneration;