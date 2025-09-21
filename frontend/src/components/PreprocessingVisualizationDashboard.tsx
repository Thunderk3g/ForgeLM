import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  Eye, 
  Brain, 
  Target, 
  Download,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  Info
} from 'lucide-react';
import type { PreprocessingAnalytics, TensorVisualizationData } from '../types/api';

interface PreprocessingVisualizationDashboardProps {
  datasetPath: string;
  analytics?: PreprocessingAnalytics;
  className?: string;
}

export const PreprocessingVisualizationDashboard: React.FC<PreprocessingVisualizationDashboardProps> = ({
  datasetPath,
  analytics,
  className = ''
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'embeddings' | 'quality' | 'tokens'>('overview');
  const [visualizations, setVisualizations] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (datasetPath) {
      loadVisualizationData();
    }
  }, [datasetPath]);

  const loadVisualizationData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('/api/visualization/preprocessing/visualize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_path: datasetPath,
          include_embeddings: true,
          include_correlations: true,
          include_token_analysis: true,
          include_quality_metrics: true
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setVisualizations(data.visualizations);
      } else {
        setError('Failed to load visualization data');
      }
    } catch (err) {
      setError('Network error while loading visualizations');
    } finally {
      setLoading(false);
    }
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'embeddings', label: 'Embeddings', icon: Brain },
    { id: 'quality', label: 'Quality', icon: Target },
    { id: 'tokens', label: 'Tokens', icon: TrendingUp }
  ];

  const renderOverviewTab = () => (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Quality Metrics Card */}
      {analytics?.quality_metrics && (
        <div className="bg-card border rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-3">
            <Target className="w-5 h-5 text-green-500" />
            <h4 className="font-medium">Dataset Quality</h4>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Overall Score</span>
              <div className="flex items-center space-x-2">
                <div className={`w-16 h-2 rounded-full ${
                  analytics.quality_metrics.overall_quality_score >= 0.8 ? 'bg-green-500' :
                  analytics.quality_metrics.overall_quality_score >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                }`} />
                <span className="font-medium">
                  {(analytics.quality_metrics.overall_quality_score * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-muted-foreground">Avg Text Length</div>
                <div className="font-medium">
                  {analytics.quality_metrics.avg_text_length.toFixed(0)} chars
                </div>
              </div>
              <div>
                <div className="text-muted-foreground">Vocabulary Richness</div>
                <div className="font-medium">
                  {(analytics.quality_metrics.vocabulary_richness * 100).toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-muted-foreground">Data Completeness</div>
                <div className="font-medium">
                  {(analytics.quality_metrics.data_completeness * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Embedding Info Card */}
      {analytics?.embedding_stats && (
        <div className="bg-card border rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-3">
            <Brain className="w-5 h-5 text-blue-500" />
            <h4 className="font-medium">Embedding Statistics</h4>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Model</span>
              <span className="font-mono text-xs">
                {analytics.embedding_stats.model_name.split('/').pop()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Dimension</span>
              <span className="font-medium">{analytics.embedding_stats.embedding_dimension}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Shape</span>
              <span className="font-mono text-xs">
                [{analytics.embedding_stats.embedding_shape.join(', ')}]
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Mean Norm</span>
              <span className="font-medium">{analytics.embedding_stats.mean_norm.toFixed(3)}</span>
            </div>
          </div>
        </div>
      )}

      {/* Token Statistics Card */}
      {analytics?.token_distribution && (
        <div className="bg-card border rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-3">
            <TrendingUp className="w-5 h-5 text-purple-500" />
            <h4 className="font-medium">Token Distribution</h4>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Mean Tokens/Sample</span>
              <span className="font-medium">
                {analytics.token_distribution.token_statistics.mean_tokens_per_sample.toFixed(0)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Total Tokens</span>
              <span className="font-medium">
                {analytics.token_distribution.token_statistics.total_estimated_tokens.toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Vocabulary Size</span>
              <span className="font-medium">
                {analytics.token_distribution.vocabulary_stats.total_unique_words.toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Processing Time Card */}
      {analytics?.preprocessing_time_ms && (
        <div className="bg-card border rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-3">
            <RefreshCw className="w-5 h-5 text-orange-500" />
            <h4 className="font-medium">Processing Statistics</h4>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Processing Time</span>
              <span className="font-medium">
                {(analytics.preprocessing_time_ms / 1000).toFixed(1)}s
              </span>
            </div>
            {analytics.duplicate_removal_stats && (
              <>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Duplicates Removed</span>
                  <span className="font-medium">
                    {analytics.duplicate_removal_stats.duplicates_removed}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Retention Rate</span>
                  <span className="font-medium">
                    {(
                      (analytics.duplicate_removal_stats.final_count / 
                       analytics.duplicate_removal_stats.original_count) * 100
                    ).toFixed(1)}%
                  </span>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );

  const renderEmbeddingsTab = () => (
    <div className="space-y-6">
      {analytics?.tensor_visualizations ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Tensor Stats */}
          <div className="bg-card border rounded-lg p-4">
            <h4 className="font-medium mb-3">Tensor Statistics</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Shape</span>
                <span className="font-mono text-xs">
                  [{analytics.tensor_visualizations.tensor_shape.join(', ')}]
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Mean</span>
                <span className="font-medium">
                  {analytics.tensor_visualizations.tensor_stats.mean.toFixed(6)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Std</span>
                <span className="font-medium">
                  {analytics.tensor_visualizations.tensor_stats.std.toFixed(6)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Norm</span>
                <span className="font-medium">
                  {analytics.tensor_visualizations.tensor_stats.norm.toFixed(3)}
                </span>
              </div>
            </div>
          </div>

          {/* Visualization Placeholder */}
          <div className="bg-card border rounded-lg p-4">
            <h4 className="font-medium mb-3">PCA Visualization</h4>
            <div className="aspect-square bg-muted rounded flex items-center justify-center">
              <div className="text-center text-muted-foreground">
                <Eye className="w-8 h-8 mx-auto mb-2" />
                <div className="text-sm">Interactive PCA plot</div>
                <div className="text-xs">would be rendered here</div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-card border rounded-lg p-8 text-center">
          <Brain className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
          <h4 className="font-medium mb-2">No Embedding Data</h4>
          <p className="text-sm text-muted-foreground">
            Enable embedding generation in preprocessing configuration to see tensor visualizations.
          </p>
        </div>
      )}
    </div>
  );

  const renderQualityTab = () => (
    <div className="space-y-6">
      {analytics?.quality_metrics ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Quality Score */}
          <div className="bg-card border rounded-lg p-4">
            <h4 className="font-medium mb-3">Quality Score</h4>
            <div className="text-center">
              <div className={`text-3xl font-bold mb-2 ${
                analytics.quality_metrics.overall_quality_score >= 0.8 ? 'text-green-500' :
                analytics.quality_metrics.overall_quality_score >= 0.6 ? 'text-yellow-500' : 'text-red-500'
              }`}>
                {(analytics.quality_metrics.overall_quality_score * 100).toFixed(0)}%
              </div>
              <div className="text-sm text-muted-foreground">Overall Quality</div>
            </div>
          </div>

          {/* Text Length Distribution */}
          <div className="bg-card border rounded-lg p-4">
            <h4 className="font-medium mb-3">Text Length</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Average</span>
                <span className="font-medium">
                  {analytics.quality_metrics.avg_text_length.toFixed(0)}
                </span>
              </div>
              <div className="w-full bg-muted rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full" 
                  style={{ 
                    width: `${Math.min(analytics.quality_metrics.avg_text_length / 1000 * 100, 100)}%` 
                  }} 
                />
              </div>
              <div className="text-xs text-muted-foreground">Characters per sample</div>
            </div>
          </div>

          {/* Data Completeness */}
          <div className="bg-card border rounded-lg p-4">
            <h4 className="font-medium mb-3">Completeness</h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm">Data Complete</span>
                <CheckCircle2 className={`w-4 h-4 ${
                  analytics.quality_metrics.data_completeness >= 0.95 ? 'text-green-500' : 'text-yellow-500'
                }`} />
              </div>
              <div className="text-2xl font-bold">
                {(analytics.quality_metrics.data_completeness * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground">
                Non-empty samples
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-card border rounded-lg p-8 text-center">
          <Target className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
          <h4 className="font-medium mb-2">No Quality Analysis</h4>
          <p className="text-sm text-muted-foreground">
            Enable quality analysis in preprocessing configuration to see quality metrics.
          </p>
        </div>
      )}
    </div>
  );

  const renderTokensTab = () => (
    <div className="space-y-6">
      {analytics?.token_distribution ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Token Statistics */}
          <div className="bg-card border rounded-lg p-4">
            <h4 className="font-medium mb-3">Token Statistics</h4>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Mean per Sample</span>
                <span className="font-medium">
                  {analytics.token_distribution.token_statistics.mean_tokens_per_sample.toFixed(0)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Median per Sample</span>
                <span className="font-medium">
                  {analytics.token_distribution.token_statistics.median_tokens_per_sample.toFixed(0)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Total Estimated</span>
                <span className="font-medium">
                  {analytics.token_distribution.token_statistics.total_estimated_tokens.toLocaleString()}
                </span>
              </div>
            </div>
          </div>

          {/* Vocabulary Statistics */}
          <div className="bg-card border rounded-lg p-4">
            <h4 className="font-medium mb-3">Vocabulary</h4>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Unique Words</span>
                <span className="font-medium">
                  {analytics.token_distribution.vocabulary_stats.total_unique_words.toLocaleString()}
                </span>
              </div>
              
              {analytics.token_distribution.vocabulary_stats.most_frequent_words && (
                <div>
                  <div className="text-sm text-muted-foreground mb-2">Most Frequent</div>
                  <div className="space-y-1">
                    {analytics.token_distribution.vocabulary_stats.most_frequent_words.slice(0, 5).map(([word, count], index) => (
                      <div key={index} className="flex justify-between text-xs">
                        <span className="font-mono">{word}</span>
                        <span>{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-card border rounded-lg p-8 text-center">
          <TrendingUp className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
          <h4 className="font-medium mb-2">No Token Analysis</h4>
          <p className="text-sm text-muted-foreground">
            Enable token analysis in preprocessing configuration to see token distribution.
          </p>
        </div>
      )}
    </div>
  );

  if (loading) {
    return (
      <div className={`bg-card rounded-lg border p-8 ${className}`}>
        <div className="text-center">
          <RefreshCw className="w-8 h-8 mx-auto mb-4 text-muted-foreground animate-spin" />
          <div className="font-medium mb-2">Loading Visualizations</div>
          <div className="text-sm text-muted-foreground">
            Generating analytics and visualizations...
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-card rounded-lg border border-destructive p-8 ${className}`}>
        <div className="text-center">
          <AlertCircle className="w-8 h-8 mx-auto mb-4 text-destructive" />
          <div className="font-medium mb-2 text-destructive">Error Loading Visualizations</div>
          <div className="text-sm text-muted-foreground mb-4">{error}</div>
          <button
            onClick={loadVisualizationData}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-card rounded-lg border ${className}`}>
      <div className="p-4 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-blue-500" />
            <h3 className="font-semibold">Preprocessing Analytics</h3>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={loadVisualizationData}
              className="p-2 hover:bg-accent rounded transition-colors"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              className="p-2 hover:bg-accent rounded transition-colors"
              title="Export Data"
            >
              <Download className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex space-x-1 mt-4">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`
                  flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors
                  ${activeTab === tab.id 
                    ? 'bg-primary text-primary-foreground' 
                    : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                  }
                `}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="p-4">
        {activeTab === 'overview' && renderOverviewTab()}
        {activeTab === 'embeddings' && renderEmbeddingsTab()}
        {activeTab === 'quality' && renderQualityTab()}
        {activeTab === 'tokens' && renderTokensTab()}
      </div>
    </div>
  );
};