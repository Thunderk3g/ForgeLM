import React, { useState } from 'react';
import { Database, Upload } from 'lucide-react';
import type { PrepareDataRequest } from '../types/api';

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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onPrepareData({
      input_file: inputFile,
      task,
      augment,
      balance,
      test_size: testSize,
      max_length: maxLength,
    });
  };

  return (
    <div className="bg-card rounded-lg shadow-sm border-border border p-6">
      <div className="flex items-center space-x-2 mb-4">
        <Database className="w-5 h-5 text-green-500" />
        <h3 className="font-semibold text-foreground">Prepare Training Data</h3>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
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
          <div className="space-y-2">
            {(['instruction', 'supervised', 'unsupervised'] as const).map((option) => (
              <label key={option} className="flex items-center space-x-2">
                <input
                  type="radio"
                  value={option}
                  checked={task === option}
                  onChange={(e) => setTask(e.target.value as typeof task)}
                  className="text-green-500 accent-green-500"
                />
                <span className="capitalize">{option} Learning</span>
              </label>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-1">
              Test Size
            </label>
            <input
              type="number"
              value={testSize}
              onChange={(e) => setTestSize(parseFloat(e.target.value))}
              min="0.1"
              max="0.5"
              step="0.05"
              className="w-full p-2 bg-background border border-input rounded-md text-foreground focus:ring-2 focus:ring-ring focus:border-ring transition-colors"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1">
              Max Length
            </label>
            <input
              type="number"
              value={maxLength}
              onChange={(e) => setMaxLength(parseInt(e.target.value))}
              min="128"
              max="4096"
              step="128"
              className="w-full p-2 bg-background border border-input rounded-md text-foreground focus:ring-2 focus:ring-ring focus:border-ring transition-colors"
            />
          </div>
        </div>

        <div className="flex space-x-4">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={augment}
              onChange={(e) => setAugment(e.target.checked)}
              className="text-green-500 accent-green-500"
            />
            <span>Data Augmentation</span>
          </label>
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={balance}
              onChange={(e) => setBalance(e.target.checked)}
              className="text-green-500 accent-green-500"
            />
            <span>Balance Classes</span>
          </label>
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="w-full flex items-center justify-center space-x-2 py-2 px-4 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Upload className="w-4 h-4" />
          <span>{isLoading ? 'Preparing Data...' : 'Prepare Data'}</span>
        </button>
      </form>
    </div>
  );
};