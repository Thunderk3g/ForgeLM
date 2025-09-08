import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Activity, Cpu, Memory, Gpu } from 'lucide-react';

interface TrainingMetric {
  step: number;
  loss: number;
  learning_rate: number;
  epoch: number;
  gradient_norm?: number;
  tokens_per_second?: number;
  memory_usage?: number;
  timestamp: string;
}

interface ResourceMetric {
  timestamp: string;
  cpu_load: number;
  memory_pressure: number;
  gpu_utilization: number;
  resources: {
    gpu: {
      allocated_mb: number;
      total_mb: number;
    };
    memory: {
      used_mb: number;
      total_mb: number;
    };
  };
}

interface TrainingVisualizationProps {
  taskId: string;
  trainingMetrics: TrainingMetric[];
  resourceMetrics: ResourceMetric[];
  isRealTime?: boolean;
}

export const TrainingVisualization: React.FC<TrainingVisualizationProps> = ({ 
  taskId, 
  trainingMetrics, 
  resourceMetrics, 
  isRealTime = false 
}) => {
  const [selectedChart, setSelectedChart] = useState<'loss' | 'resources' | 'performance'>('loss');
  
  // Format data for loss chart
  const lossData = trainingMetrics.map(metric => ({
    step: metric.step,
    loss: metric.loss,
    learning_rate: metric.learning_rate * 10000, // Scale for visibility
    epoch: metric.epoch
  }));
  
  // Format data for resource chart
  const resourceData = resourceMetrics.map(metric => ({
    time: new Date(metric.timestamp).toLocaleTimeString(),
    cpu: metric.cpu_load,
    memory: metric.memory_pressure,
    gpu: metric.gpu_utilization
  })).slice(-20); // Show last 20 data points
  
  // Performance metrics
  const performanceData = trainingMetrics.filter(m => m.tokens_per_second).map(metric => ({
    step: metric.step,
    tokens_per_second: metric.tokens_per_second,
    gradient_norm: metric.gradient_norm
  }));
  
  // Current statistics
  const latestMetric = trainingMetrics[trainingMetrics.length - 1];
  const latestResource = resourceMetrics[resourceMetrics.length - 1];

  return (
    <div className="space-y-6">
      {/* Header with current stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg shadow border">
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-blue-500" />
            <div>
              <p className="text-sm font-medium text-gray-600">Current Loss</p>
              <p className="text-xl font-bold text-gray-900">
                {latestMetric?.loss?.toFixed(4) || 'N/A'}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow border">
          <div className="flex items-center space-x-2">
            <Cpu className="h-5 w-5 text-green-500" />
            <div>
              <p className="text-sm font-medium text-gray-600">CPU Usage</p>
              <p className="text-xl font-bold text-gray-900">
                {latestResource?.cpu_load?.toFixed(1) || 'N/A'}%
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow border">
          <div className="flex items-center space-x-2">
            <Memory className="h-5 w-5 text-purple-500" />
            <div>
              <p className="text-sm font-medium text-gray-600">Memory Usage</p>
              <p className="text-xl font-bold text-gray-900">
                {latestResource?.memory_pressure?.toFixed(1) || 'N/A'}%
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow border">
          <div className="flex items-center space-x-2">
            <Gpu className="h-5 w-5 text-orange-500" />
            <div>
              <p className="text-sm font-medium text-gray-600">GPU Usage</p>
              <p className="text-xl font-bold text-gray-900">
                {latestResource?.gpu_utilization?.toFixed(1) || 'N/A'}%
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Chart selection tabs */}
      <div className="bg-white rounded-lg shadow border">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6" aria-label="Tabs">
            {[
              { id: 'loss', name: 'Training Loss', icon: Activity },
              { id: 'resources', name: 'System Resources', icon: Cpu },
              { id: 'performance', name: 'Performance', icon: Gpu }
            ].map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setSelectedChart(tab.id as any)}
                  className={`${
                    selectedChart === tab.id
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
          {selectedChart === 'loss' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Training Loss & Learning Rate</h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={lossData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="step" 
                    label={{ value: 'Training Steps', position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis yAxisId="loss" orientation="left" />
                  <YAxis yAxisId="lr" orientation="right" />
                  <Tooltip 
                    labelFormatter={(value) => `Step: ${value}`}
                    formatter={(value, name) => [
                      name === 'loss' ? Number(value).toFixed(4) : Number(value).toFixed(8),
                      name === 'loss' ? 'Loss' : 'Learning Rate (×10⁻⁴)'
                    ]}
                  />
                  <Legend />
                  <Line 
                    yAxisId="loss"
                    type="monotone" 
                    dataKey="loss" 
                    stroke="#8884d8" 
                    strokeWidth={2}
                    dot={false}
                    name="Loss"
                  />
                  <Line 
                    yAxisId="lr"
                    type="monotone" 
                    dataKey="learning_rate" 
                    stroke="#82ca9d" 
                    strokeWidth={2}
                    dot={false}
                    name="Learning Rate (×10⁻⁴)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {selectedChart === 'resources' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">System Resource Usage (Real-time)</h3>
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={resourceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip 
                    formatter={(value, name) => [`${Number(value).toFixed(1)}%`, name]}
                  />
                  <Legend />
                  <Area 
                    type="monotone" 
                    dataKey="cpu" 
                    stackId="1"
                    stroke="#8884d8" 
                    fill="#8884d8" 
                    fillOpacity={0.3}
                    name="CPU Usage"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="memory" 
                    stackId="2"
                    stroke="#82ca9d" 
                    fill="#82ca9d" 
                    fillOpacity={0.3}
                    name="Memory Usage"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="gpu" 
                    stackId="3"
                    stroke="#ffc658" 
                    fill="#ffc658" 
                    fillOpacity={0.3}
                    name="GPU Usage"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          {selectedChart === 'performance' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Training Performance Metrics</h3>
              {performanceData.length > 0 ? (
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="step" />
                    <YAxis yAxisId="tokens" orientation="left" />
                    <YAxis yAxisId="gradient" orientation="right" />
                    <Tooltip />
                    <Legend />
                    <Line 
                      yAxisId="tokens"
                      type="monotone" 
                      dataKey="tokens_per_second" 
                      stroke="#8884d8" 
                      strokeWidth={2}
                      name="Tokens/Second"
                    />
                    <Line 
                      yAxisId="gradient"
                      type="monotone" 
                      dataKey="gradient_norm" 
                      stroke="#82ca9d" 
                      strokeWidth={2}
                      name="Gradient Norm"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-64 text-gray-500">
                  <p>Performance metrics will appear when training data is available</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Real-time indicator */}
      {isRealTime && (
        <div className="flex items-center justify-center">
          <div className="flex items-center space-x-2 text-green-600">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium">Real-time updates active</span>
          </div>
        </div>
      )}
    </div>
  );
};