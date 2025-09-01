import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface MetricData {
  epoch: number;
  step: number;
  loss?: number;
  eval_loss?: number;
  learning_rate?: number;
}

interface MetricsChartProps {
  data: MetricData[];
  title?: string;
}

export const MetricsChart: React.FC<MetricsChartProps> = ({ data, title = "Training Metrics" }) => {
  if (data.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="font-semibold text-gray-900 mb-4">{title}</h3>
        <div className="h-64 flex items-center justify-center text-gray-500">
          No metrics data available yet...
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <h3 className="font-semibold text-gray-900 mb-4">{title}</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="step" 
              label={{ value: 'Training Steps', position: 'insideBottom', offset: -5 }}
            />
            <YAxis />
            <Tooltip 
              formatter={(value, name) => [
                typeof value === 'number' ? value.toFixed(4) : value, 
                name
              ]}
              labelFormatter={(step) => `Step: ${step}`}
            />
            <Legend />
            {data.some(d => d.loss !== undefined) && (
              <Line 
                type="monotone" 
                dataKey="loss" 
                stroke="#ef4444" 
                strokeWidth={2}
                name="Training Loss"
                dot={false}
              />
            )}
            {data.some(d => d.eval_loss !== undefined) && (
              <Line 
                type="monotone" 
                dataKey="eval_loss" 
                stroke="#22c55e" 
                strokeWidth={2}
                name="Validation Loss"
                dot={false}
              />
            )}
            {data.some(d => d.learning_rate !== undefined) && (
              <Line 
                type="monotone" 
                dataKey="learning_rate" 
                stroke="#3b82f6" 
                strokeWidth={2}
                name="Learning Rate"
                dot={false}
                yAxisId="lr"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};