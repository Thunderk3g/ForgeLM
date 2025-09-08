import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, ScatterPlot, Scatter } from 'recharts';
import { Database, FileText, TrendingUp, Filter } from 'lucide-react';

interface TokenDistribution {
  range: string;
  count: number;
  percentage: number;
}

interface TemplateFormat {
  name: string;
  count: number;
  percentage: number;
  color: string;
}

interface DatasetAnalyticsData {
  total_samples: number;
  token_distribution: TokenDistribution[];
  length_statistics: {
    mean: number;
    median: number;
    min: number;
    max: number;
    std_dev: number;
  };
  template_formats: TemplateFormat[];
  quality_metrics: {
    duplicate_ratio: number;
    avg_response_length: number;
    unique_instructions: number;
  };
  sample_examples: Array<{
    instruction: string;
    response: string;
    token_count: number;
    template_format: string;
  }>;
}

interface DatasetAnalyticsProps {
  datasetPath: string;
  analyticsData: DatasetAnalyticsData | null;
  isLoading: boolean;
  onRefresh: () => void;
}

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1'];

export const DatasetAnalytics: React.FC<DatasetAnalyticsProps> = ({
  datasetPath,
  analyticsData,
  isLoading,
  onRefresh
}) => {
  const [selectedView, setSelectedView] = useState<'overview' | 'distribution' | 'quality' | 'samples'>('overview');
  const [filterBy, setFilterBy] = useState<string>('all');

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow border p-8">
        <div className="flex items-center justify-center space-x-2">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
          <span className="text-gray-600">Analyzing dataset...</span>
        </div>
      </div>
    );
  }

  if (!analyticsData) {
    return (
      <div className="bg-white rounded-lg shadow border p-8">
        <div className="text-center">
          <Database className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Dataset Analytics Available</h3>
          <p className="text-gray-500 mb-4">Load a dataset to see detailed analytics and insights.</p>
          <button
            onClick={onRefresh}
            className="btn-primary"
          >
            Load Dataset Analytics
          </button>
        </div>
      </div>
    );
  }

  const { total_samples, token_distribution, length_statistics, template_formats, quality_metrics, sample_examples } = analyticsData;

  return (
    <div className="space-y-6">
      {/* Header with dataset info */}
      <div className="bg-white rounded-lg shadow border p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Database className="h-8 w-8 text-blue-500" />
            <div>
              <h2 className="text-xl font-bold text-gray-900">Dataset Analytics</h2>
              <p className="text-sm text-gray-600 mt-1">Path: {datasetPath}</p>
            </div>
          </div>
          <button
            onClick={onRefresh}
            className="btn-secondary flex items-center space-x-2"
          >
            <TrendingUp className="h-4 w-4" />
            <span>Refresh</span>
          </button>
        </div>

        {/* Quick stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm font-medium text-gray-600">Total Samples</p>
            <p className="text-2xl font-bold text-gray-900">{total_samples.toLocaleString()}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm font-medium text-gray-600">Avg Length</p>
            <p className="text-2xl font-bold text-gray-900">{length_statistics.mean.toFixed(0)}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm font-medium text-gray-600">Unique Instructions</p>
            <p className="text-2xl font-bold text-gray-900">{quality_metrics.unique_instructions.toLocaleString()}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm font-medium text-gray-600">Duplicate Ratio</p>
            <p className="text-2xl font-bold text-gray-900">{(quality_metrics.duplicate_ratio * 100).toFixed(1)}%</p>
          </div>
        </div>
      </div>

      {/* Navigation tabs */}
      <div className="bg-white rounded-lg shadow border">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6" aria-label="Tabs">
            {[
              { id: 'overview', name: 'Overview', icon: Database },
              { id: 'distribution', name: 'Token Distribution', icon: BarChart },
              { id: 'quality', name: 'Quality Analysis', icon: TrendingUp },
              { id: 'samples', name: 'Sample Explorer', icon: FileText }
            ].map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setSelectedView(tab.id as any)}
                  className={`${
                    selectedView === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{tab.name}</span>
                </button>
              );
            })}
          </nav>
        </div>

        <div className="p-6">
          {selectedView === 'overview' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold mb-4">Template Format Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={template_formats}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percentage }) => `${name} (${percentage.toFixed(1)}%)`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="count"
                    >
                      {template_formats.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color || COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold mb-4">Length Statistics</h3>
                <div className="space-y-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-600">Mean Length</span>
                      <span className="text-lg font-bold text-gray-900">{length_statistics.mean.toFixed(0)} tokens</span>
                    </div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-600">Median Length</span>
                      <span className="text-lg font-bold text-gray-900">{length_statistics.median.toFixed(0)} tokens</span>
                    </div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-600">Min Length</span>
                      <span className="text-lg font-bold text-gray-900">{length_statistics.min} tokens</span>
                    </div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-600">Max Length</span>
                      <span className="text-lg font-bold text-gray-900">{length_statistics.max} tokens</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium text-gray-600">Std Deviation</span>
                      <span className="text-lg font-bold text-gray-900">{length_statistics.std_dev.toFixed(0)}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedView === 'distribution' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Token Length Distribution</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={token_distribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="range" 
                    label={{ value: 'Token Length Range', position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis />
                  <Tooltip 
                    formatter={(value, name) => [
                      `${value} samples (${token_distribution.find(d => d.count === value)?.percentage?.toFixed(1) || 0}%)`,
                      'Count'
                    ]}
                  />
                  <Bar dataKey="count" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {selectedView === 'quality' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg p-6">
                  <h4 className="text-lg font-semibold text-blue-900 mb-2">Data Quality Score</h4>
                  <p className="text-3xl font-bold text-blue-700">
                    {((1 - quality_metrics.duplicate_ratio) * 100).toFixed(0)}%
                  </p>
                  <p className="text-sm text-blue-600 mt-2">Based on uniqueness and completeness</p>
                </div>
                
                <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-lg p-6">
                  <h4 className="text-lg font-semibold text-green-900 mb-2">Response Quality</h4>
                  <p className="text-3xl font-bold text-green-700">
                    {quality_metrics.avg_response_length.toFixed(0)}
                  </p>
                  <p className="text-sm text-green-600 mt-2">Average response length (tokens)</p>
                </div>
                
                <div className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg p-6">
                  <h4 className="text-lg font-semibold text-purple-900 mb-2">Diversity Score</h4>
                  <p className="text-3xl font-bold text-purple-700">
                    {((quality_metrics.unique_instructions / total_samples) * 100).toFixed(0)}%
                  </p>
                  <p className="text-sm text-purple-600 mt-2">Unique instruction ratio</p>
                </div>
              </div>
            </div>
          )}

          {selectedView === 'samples' && (
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Sample Explorer</h3>
                <div className="flex items-center space-x-2">
                  <Filter className="h-4 w-4 text-gray-500" />
                  <select
                    value={filterBy}
                    onChange={(e) => setFilterBy(e.target.value)}
                    className="input text-sm"
                  >
                    <option value="all">All Templates</option>
                    {template_formats.map(format => (
                      <option key={format.name} value={format.name}>{format.name}</option>
                    ))}
                  </select>
                </div>
              </div>
              
              <div className="space-y-4">
                {sample_examples
                  .filter(sample => filterBy === 'all' || sample.template_format === filterBy)
                  .slice(0, 5)
                  .map((sample, index) => (
                    <div key={index} className="bg-gray-50 rounded-lg p-4 border">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-medium bg-blue-100 text-blue-800 px-2 py-1 rounded">
                          {sample.template_format}
                        </span>
                        <span className="text-xs text-gray-500">{sample.token_count} tokens</span>
                      </div>
                      <div className="space-y-2">
                        <div>
                          <p className="text-sm font-medium text-gray-700">Instruction:</p>
                          <p className="text-sm text-gray-600 bg-white p-2 rounded border">
                            {sample.instruction.slice(0, 200)}
                            {sample.instruction.length > 200 && '...'}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-gray-700">Response:</p>
                          <p className="text-sm text-gray-600 bg-white p-2 rounded border">
                            {sample.response.slice(0, 200)}
                            {sample.response.length > 200 && '...'}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};