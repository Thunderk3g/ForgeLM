import React, { useState, useEffect } from 'react';
import { 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  Download, 
  Eye, 
  BarChart3,
  FileText,
  Brain,
  Sparkles,
  ExternalLink
} from 'lucide-react';
import type { ApiResponse, PreprocessingAnalytics } from '../types/api';
import { PreprocessingVisualizationDashboard } from './PreprocessingVisualizationDashboard';

interface PreprocessingResultsViewerProps {
  result: ApiResponse;
  className?: string;
}

export const PreprocessingResultsViewer: React.FC<PreprocessingResultsViewerProps> = ({
  result,
  className = ''
}) => {
  const isPreview = result.message.includes('Preview');
  const [showVisualizations, setShowVisualizations] = useState(isPreview);
  const [analytics, setAnalytics] = useState<PreprocessingAnalytics | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (result.status === 'success' && result.train_path) {
      if (isPreview) {
        // Mock analytics for preview mode to show visualizations
        setAnalytics({
          quality_metrics: {
            overall_quality_score: 0.85,
            avg_text_length: 245,
            vocabulary_richness: 0.78,
            data_completeness: 0.98
          },
          embedding_stats: {
            model_name: 'sentence-transformers/all-MiniLM-L6-v2',
            embedding_dimension: 384,
            embedding_shape: [1000, 384],
            mean_norm: 1.0
          },
          token_distribution: {
            token_statistics: {
              mean_tokens_per_sample: 128,
              median_tokens_per_sample: 120,
              total_estimated_tokens: 128000
            },
            vocabulary_stats: {
              total_unique_words: 4500,
              most_frequent_words: [['the', 250], ['is', 180], ['and', 150], ['to', 120], ['of', 100]]
            }
          },
          preprocessing_time_ms: 2500,
          duplicate_removal_stats: {
            original_count: 1000,
            duplicates_removed: 50,
            final_count: 950
          },
          tensor_visualizations: {
            tensor_shape: [1000, 384],
            tensor_stats: {
              mean: 0.001,
              std: 0.15,
              min: -0.5,
              max: 0.5,
              norm: 1.0
            },
            visualization_data: {
              pca_components: [[0.1, 0.2], [0.3, 0.4]],
              tsne_coordinates: [[0.5, 0.6], [0.7, 0.8]],
              umap_embeddings: [[0.9, 1.0], [1.1, 1.2]]
            }
          },
          correlation_analysis: {
            correlation_matrix: [[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]],
            key_correlations: { 'text_length_tokens': 0.65, 'quality_score': 'medium' }
          },
          embedding_evaluation: {
            intrinsic_metrics: {
              cosine_similarity: 0.75,
              alignment: 0.82,
              uniformity: 0.78
            },
            downstream_evals: {
              task: 'classification',
              classification_accuracy: 0.88,
              f1_score: 0.85
            },
            visualization_available: true
          }
        });
        setLoading(false);
      } else {
        loadAnalytics();
      }
    }
  }, [result]);

  const loadAnalytics = async () => {
    if (!result.train_path) return;
    
    try {
      setLoading(true);
      const response = await fetch(`/api/data-generation/preprocessing/status/${encodeURIComponent(result.train_path)}`);
      if (response.ok) {
        const data = await response.json();
        if (data.analytics) {
          setAnalytics(data.analytics);
        }
      }
    } catch (error) {
      console.error('Failed to load analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = () => {
    switch (result.status) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      case 'running':
        return <Clock className="w-5 h-5 text-blue-500 animate-spin" />;
      default:
        return <Clock className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusColor = () => {
    switch (result.status) {
      case 'success':
        return 'bg-green-50 border-green-200';
      case 'error':
        return 'bg-red-50 border-red-200';
      case 'running':
        return 'bg-blue-50 border-blue-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  const hasAdvancedFeatures = analytics && (
    analytics.embedding_stats ||
    analytics.tensor_visualizations ||
    analytics.correlation_analysis ||
    analytics.token_distribution ||
    analytics.embedding_evaluation
  );

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Main Results Card */}
      <div className={`rounded-lg border p-6 ${getStatusColor()}`}>
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-3">
            {getStatusIcon()}
            <div>
              <h3 className="font-semibold text-lg">
                Data Preparation {result.status === 'success' ? 'Complete' : 'Status'}
              </h3>
              <p className="text-muted-foreground mt-1">{result.message}</p>
            </div>
          </div>
          
          {result.status === 'success' && (
            <div className="flex space-x-2">
              <button
                onClick={() => setShowVisualizations(!showVisualizations)}
                className="flex items-center space-x-2 px-3 py-2 bg-white border rounded-md hover:bg-gray-50 transition-colors"
              >
                <BarChart3 className="w-4 h-4" />
                <span className="text-sm">
                  {showVisualizations ? 'Hide' : 'Show'} Analytics
                </span>
              </button>
            </div>
          )}
        </div>

        {/* File Paths */}
        {result.status === 'success' && result.train_path && (
          <div className="mt-4 space-y-3">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white rounded-lg border p-3">
                <div className="flex items-center space-x-2 mb-2">
                  <FileText className="w-4 h-4 text-blue-500" />
                  <span className="font-medium text-sm">Training Data</span>
                </div>
                <div className="font-mono text-xs text-muted-foreground break-all">
                  {result.train_path}
                </div>
                <button className="mt-2 text-xs text-blue-600 hover:text-blue-800 flex items-center space-x-1">
                  <Download className="w-3 h-3" />
                  <span>Download</span>
                </button>
              </div>
              
              {result.val_path && (
                <div className="bg-white rounded-lg border p-3">
                  <div className="flex items-center space-x-2 mb-2">
                    <FileText className="w-4 h-4 text-green-500" />
                    <span className="font-medium text-sm">Validation Data</span>
                  </div>
                  <div className="font-mono text-xs text-muted-foreground break-all">
                    {result.val_path}
                  </div>
                  <button className="mt-2 text-xs text-blue-600 hover:text-blue-800 flex items-center space-x-1">
                    <Download className="w-3 h-3" />
                    <span>Download</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Optimization Info */}
        {result.optimization_info && (
          <div className="mt-4">
            <h4 className="font-medium mb-2">Optimization Information</h4>
            <div className="bg-white rounded-lg border p-3">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <div className="text-muted-foreground">Format</div>
                  <div className="font-medium">{result.optimization_info.format_applied}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Total Samples</div>
                  <div className="font-medium">{result.optimization_info.total_samples}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Unsloth Optimized</div>
                  <div className="font-medium">
                    {result.optimization_info.unsloth_optimized ? 'Yes' : 'No'}
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground">Packing</div>
                  <div className="font-medium">
                    {result.optimization_info.packing_recommended ? 'Recommended' : 'Standard'}
                  </div>
                </div>
                {result.optimization_info.has_embeddings && (
                  <>
                    <div>
                      <div className="text-muted-foreground">Embeddings</div>
                      <div className="font-medium flex items-center space-x-1">
                        <Brain className="w-3 h-3 text-blue-500" />
                        <span>Generated</span>
                      </div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Embedding Dim</div>
                      <div className="font-medium">{result.optimization_info.embedding_dimension}</div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Quick Analytics Summary */}
        {analytics && !loading && (
          <div className="mt-4">
            <h4 className="font-medium mb-2 flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-purple-500" />
              <span>Processing Summary</span>
            </h4>
            <div className="bg-white rounded-lg border p-3">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                {analytics.quality_metrics && (
                  <div>
                    <div className="text-muted-foreground">Quality Score</div>
                    <div className={`font-medium ${
                      analytics.quality_metrics.overall_quality_score >= 0.8 ? 'text-green-600' :
                      analytics.quality_metrics.overall_quality_score >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {(analytics.quality_metrics.overall_quality_score * 100).toFixed(0)}%
                    </div>
                  </div>
                )}
                
                {analytics.preprocessing_time_ms && (
                  <div>
                    <div className="text-muted-foreground">Processing Time</div>
                    <div className="font-medium">
                      {(analytics.preprocessing_time_ms / 1000).toFixed(1)}s
                    </div>
                  </div>
                )}
                
                {analytics.duplicate_removal_stats && (
                  <div>
                    <div className="text-muted-foreground">Duplicates Removed</div>
                    <div className="font-medium">
                      {analytics.duplicate_removal_stats.duplicates_removed}
                    </div>
                  </div>
                )}
                
                {analytics.token_distribution && (
                  <div>
                    <div className="text-muted-foreground">Avg Tokens/Sample</div>
                    <div className="font-medium">
                      {analytics.token_distribution.token_statistics.mean_tokens_per_sample.toFixed(0)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Embedding Evaluation Section */}
        {analytics?.embedding_evaluation && (
          <div className="mt-6">
            <h4 className="font-medium mb-3 flex items-center space-x-2">
              <Brain className="w-4 h-4 text-purple-500" />
              <span>Embedding Evaluation</span>
            </h4>
            <div className="bg-white rounded-lg border p-4 space-y-4">
              {/* Intrinsic Metrics */}
              <div>
                <h5 className="font-medium text-sm mb-2">Intrinsic Metrics</h5>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground">Cosine Similarity</div>
                    <div className="font-medium text-green-600">
                      {analytics.embedding_evaluation.intrinsic_metrics.cosine_similarity.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Alignment</div>
                    <div className="font-medium text-blue-600">
                      {analytics.embedding_evaluation.intrinsic_metrics.alignment.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Uniformity</div>
                    <div className="font-medium text-purple-600">
                      {analytics.embedding_evaluation.intrinsic_metrics.uniformity.toFixed(3)}
                    </div>
                  </div>
                </div>
                <div className="text-xs text-muted-foreground mt-2">
                  Higher values indicate better embedding quality (cosine: 0-1, alignment/uniformity: 0-1)
                </div>
              </div>

              {/* Downstream Evaluation */}
              {analytics.embedding_evaluation.downstream_evals && (
                <div>
                  <h5 className="font-medium text-sm mb-2">Downstream Task Performance</h5>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <div className="text-muted-foreground">Task</div>
                      <div className="font-medium capitalize">
                        {analytics.embedding_evaluation.downstream_evals.task}
                      </div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Accuracy</div>
                      <div className="font-medium text-green-600">
                        {(analytics.embedding_evaluation.downstream_evals.classification_accuracy * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">F1 Score</div>
                      <div className="font-medium text-blue-600">
                        {analytics.embedding_evaluation.downstream_evals.f1_score.toFixed(3)}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Visualization Toggle */}
              {analytics.embedding_evaluation.visualization_available && (
                <div className="pt-3 border-t">
                  <button
                    onClick={() => setShowVisualizations(!showVisualizations)}
                    className="flex items-center space-x-2 px-3 py-2 bg-blue-50 border border-blue-200 rounded-md hover:bg-blue-100 transition-colors text-sm"
                  >
                    <Eye className="w-4 h-4" />
                    <span>
                      {showVisualizations ? 'Hide' : 'Show'} Embedding Visualizations
                    </span>
                  </button>
                  {showVisualizations && (
                    <div className="mt-3 p-3 bg-gray-50 rounded">
                      <p className="text-xs text-muted-foreground">
                        PCA and t-SNE projections of embedding space available in the visualization dashboard below.
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Advanced Features Available */}
        {hasAdvancedFeatures && (
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <Brain className="w-4 h-4 text-blue-500" />
              <span className="font-medium text-sm text-blue-800">
                Advanced Analytics Available
              </span>
            </div>
            <div className="text-xs text-blue-600 mt-1">
              Embedding analysis, tensor visualizations, and detailed metrics are ready to view.
            </div>
            <div className="flex space-x-2 mt-2">
              {analytics.embedding_stats && (
                <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                  Embeddings
                </span>
              )}
              {analytics.tensor_visualizations && (
                <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                  Tensors
                </span>
              )}
              {analytics.correlation_analysis && (
                <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                  Correlations
                </span>
              )}
              {analytics.token_distribution && (
                <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                  Token Analysis
                </span>
              )}
            </div>
          </div>
        )}

        {/* Visualization URLs */}
        {result.status === 'success' && result.optimization_info?.visualization_urls && (
          <div className="mt-4">
            <h4 className="font-medium mb-2">Available Visualizations</h4>
            <div className="flex flex-wrap gap-2">
              {Object.entries(result.optimization_info.visualization_urls).map(([type, url]) => (
                <a
                  key={type}
                  href={url as string}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center space-x-2 px-3 py-2 bg-white border rounded-md hover:bg-gray-50 transition-colors text-sm"
                >
                  <ExternalLink className="w-3 h-3" />
                  <span className="capitalize">{type.replace('_', ' ')}</span>
                </a>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Visualization Dashboard */}
      {showVisualizations && result.status === 'success' && result.train_path && (
        <PreprocessingVisualizationDashboard
          datasetPath={result.train_path}
          analytics={analytics || undefined}
          className="animate-in slide-in-from-top-5 duration-200"
        />
      )}

      {/* Error Details */}
      {result.status === 'error' && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <AlertCircle className="w-4 h-4 text-red-500" />
            <span className="font-medium text-red-800">Error Details</span>
          </div>
          <pre className="text-sm text-red-700 bg-red-100 p-2 rounded overflow-x-auto">
            {result.message}
          </pre>
        </div>
      )}
    </div>
  );
};