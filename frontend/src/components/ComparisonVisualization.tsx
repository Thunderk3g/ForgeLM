import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, LineChart, Line } from 'recharts';
import { Zap, Clock, TrendingUp, BarChart3, Eye, Filter } from 'lucide-react';

interface ComparisonMetric {
  session_id: string;
  base_time_ms: number;
  finetuned_time_ms: number;
  base_response_length: number;
  finetuned_response_length: number;
  similarity_score: number;
  timestamp: string;
  user_message_length: number;
}

interface ComparisonStats {
  total_comparisons: number;
  avg_similarity: number;
  avg_base_time: number;
  avg_finetuned_time: number;
  performance_improvement: number;
  model_pair_info: {
    base_model: string;
    finetuned_model: string;
  };
}

interface ComparisonVisualizationProps {
  comparisonData: ComparisonMetric[];
  stats: ComparisonStats;
  isLoading: boolean;
  onRefresh: () => void;
}

export const ComparisonVisualization: React.FC<ComparisonVisualizationProps> = ({
  comparisonData,
  stats,
  isLoading,
  onRefresh
}) => {
  const [selectedView, setSelectedView] = useState<'performance' | 'similarity' | 'trends' | 'heatmap'>('performance');
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h' | 'all'>('6h');

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow border p-8">
        <div className="flex items-center justify-center space-x-2">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
          <span className="text-gray-600">Loading comparison data...</span>
        </div>
      </div>
    );
  }

  if (!comparisonData.length) {
    return (
      <div className="bg-white rounded-lg shadow border p-8">
        <div className="text-center">
          <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Comparison Data Available</h3>
          <p className="text-gray-500 mb-4">Start generating model comparisons to see analytics.</p>
          <button
            onClick={onRefresh}
            className="btn-primary"
          >
            Refresh Data
          </button>
        </div>
      </div>
    );
  }

  // Filter data by time range
  const filterDataByTime = (data: ComparisonMetric[]) => {
    if (timeRange === 'all') return data;
    
    const now = new Date();
    const hoursBack = timeRange === '1h' ? 1 : timeRange === '6h' ? 6 : 24;
    const cutoff = new Date(now.getTime() - (hoursBack * 60 * 60 * 1000));
    
    return data.filter(item => new Date(item.timestamp) >= cutoff);
  };

  const filteredData = filterDataByTime(comparisonData);

  // Performance comparison data
  const performanceData = filteredData.map((item, index) => ({
    comparison: index + 1,
    base_time: item.base_time_ms,
    finetuned_time: item.finetuned_time_ms,
    speedup: ((item.base_time_ms - item.finetuned_time_ms) / item.base_time_ms) * 100
  }));

  // Similarity vs Response Length scatter data
  const similarityScatterData = filteredData.map((item) => ({
    similarity: item.similarity_score * 100,
    length_ratio: item.finetuned_response_length / Math.max(item.base_response_length, 1),
    base_time: item.base_time_ms,
    finetuned_time: item.finetuned_time_ms
  }));

  // Trend data over time
  const trendData = filteredData.map((item, index) => ({
    comparison: index + 1,
    timestamp: new Date(item.timestamp).toLocaleTimeString(),
    similarity: item.similarity_score * 100,
    base_time: item.base_time_ms,
    finetuned_time: item.finetuned_time_ms
  }));

  // Response length distribution
  const lengthComparisonData = filteredData.reduce((acc: any[], item) => {
    const baseRange = Math.floor(item.base_response_length / 50) * 50;
    const finetunedRange = Math.floor(item.finetuned_response_length / 50) * 50;
    
    const existingBase = acc.find(d => d.range === `${baseRange}-${baseRange + 49}` && d.type === 'Base');
    const existingFinetuned = acc.find(d => d.range === `${finetunedRange}-${finetunedRange + 49}` && d.type === 'Fine-tuned');
    
    if (existingBase) existingBase.count++;
    else acc.push({ range: `${baseRange}-${baseRange + 49}`, count: 1, type: 'Base' });
    
    if (existingFinetuned) existingFinetuned.count++;
    else acc.push({ range: `${finetunedRange}-${finetunedRange + 49}`, count: 1, type: 'Fine-tuned' });
    
    return acc;
  }, []);

  return (
    <div className="space-y-6">
      {/* Header with model info and stats */}
      <div className="bg-white rounded-lg shadow border p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <BarChart3 className="h-8 w-8 text-blue-500" />
            <div>
              <h2 className="text-xl font-bold text-gray-900">Model Comparison Analytics</h2>
              <p className="text-sm text-gray-600">
                {stats.model_pair_info.base_model} vs {stats.model_pair_info.finetuned_model}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Filter className="h-4 w-4 text-gray-500" />
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value as any)}
                className="input text-sm"
              >
                <option value="1h">Last 1 hour</option>
                <option value="6h">Last 6 hours</option>
                <option value="24h">Last 24 hours</option>
                <option value="all">All time</option>
              </select>
            </div>
            <button
              onClick={onRefresh}
              className="btn-secondary flex items-center space-x-2"
            >
              <TrendingUp className="h-4 w-4" />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Quick stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <Zap className="h-5 w-5 text-blue-600" />
              <div>
                <p className="text-sm font-medium text-blue-700">Performance Boost</p>
                <p className="text-2xl font-bold text-blue-800">
                  {stats.performance_improvement > 0 ? '+' : ''}{stats.performance_improvement.toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <Clock className="h-5 w-5 text-green-600" />
              <div>
                <p className="text-sm font-medium text-green-700">Avg Response Time</p>
                <p className="text-2xl font-bold text-green-800">
                  {stats.avg_finetuned_time.toFixed(0)}ms
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <Eye className="h-5 w-5 text-purple-600" />
              <div>
                <p className="text-sm font-medium text-purple-700">Avg Similarity</p>
                <p className="text-2xl font-bold text-purple-800">
                  {(stats.avg_similarity * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-orange-50 to-orange-100 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5 text-orange-600" />
              <div>
                <p className="text-sm font-medium text-orange-700">Total Comparisons</p>
                <p className="text-2xl font-bold text-orange-800">
                  {filteredData.length}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Chart navigation */}
      <div className="bg-white rounded-lg shadow border">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6" aria-label="Tabs">
            {[
              { id: 'performance', name: 'Performance', icon: Zap },
              { id: 'similarity', name: 'Similarity Analysis', icon: Eye },
              { id: 'trends', name: 'Trends', icon: TrendingUp },
              { id: 'heatmap', name: 'Distribution', icon: BarChart3 }
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
          {selectedView === 'performance' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Response Time Comparison</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={performanceData.slice(-20)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="comparison" />
                  <YAxis />
                  <Tooltip 
                    formatter={(value, name) => [
                      `${Number(value).toFixed(0)}ms`,
                      name === 'base_time' ? 'Base Model' : 'Fine-tuned Model'
                    ]}
                  />
                  <Legend />
                  <Bar dataKey="base_time" fill="#8884d8" name="Base Model" />
                  <Bar dataKey="finetuned_time" fill="#82ca9d" name="Fine-tuned Model" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {selectedView === 'similarity' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Response Similarity vs Length Ratio</h3>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart data={similarityScatterData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="similarity" 
                    domain={[0, 100]}
                    label={{ value: 'Response Similarity (%)', position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis 
                    dataKey="length_ratio"
                    label={{ value: 'Length Ratio (Fine-tuned/Base)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    formatter={(value, name) => [
                      name === 'length_ratio' ? Number(value).toFixed(2) : `${Number(value).toFixed(1)}%`,
                      name === 'length_ratio' ? 'Length Ratio' : 'Similarity'
                    ]}
                  />
                  <Scatter dataKey="length_ratio" fill="#8884d8" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}

          {selectedView === 'trends' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Performance Trends Over Time</h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={trendData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis yAxisId="time" orientation="left" />
                  <YAxis yAxisId="similarity" orientation="right" domain={[0, 100]} />
                  <Tooltip />
                  <Legend />
                  <Line 
                    yAxisId="time"
                    type="monotone" 
                    dataKey="base_time" 
                    stroke="#8884d8" 
                    strokeWidth={2}
                    name="Base Model Time (ms)"
                  />
                  <Line 
                    yAxisId="time"
                    type="monotone" 
                    dataKey="finetuned_time" 
                    stroke="#82ca9d" 
                    strokeWidth={2}
                    name="Fine-tuned Model Time (ms)"
                  />
                  <Line 
                    yAxisId="similarity"
                    type="monotone" 
                    dataKey="similarity" 
                    stroke="#ffc658" 
                    strokeWidth={2}
                    name="Similarity (%)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {selectedView === 'heatmap' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Response Length Distribution</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={lengthComparisonData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="range" 
                    label={{ value: 'Response Length Range (tokens)', position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" fill="#8884d8" name="Count" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};