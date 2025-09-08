import React, { useState } from 'react';
import { BarChart3, Activity, Database, Eye, RefreshCw, TrendingUp } from 'lucide-react';
import { TrainingVisualization } from './TrainingVisualization';
import { DatasetAnalytics } from './DatasetAnalytics';
import { useVisualizationDashboard } from '../hooks/useVisualizationData';
import type { TrainingMetric, ResourceMetric } from '../types/api';

interface VisualizationDashboardProps {
  currentTaskId?: string | null;
  selectedDatasetPath?: string | null;
}

export const VisualizationDashboard: React.FC<VisualizationDashboardProps> = ({
  currentTaskId = null,
  selectedDatasetPath = null
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'training' | 'dataset'>('overview');
  const [timeRangeHours, setTimeRangeHours] = useState(24);

  const {
    trainingAnalytics,
    datasetAnalytics,
    systemHealth,
    dashboardSummary,
    isLoading,
    hasError,
    errors,
    refetchAll
  } = useVisualizationDashboard(currentTaskId, selectedDatasetPath, timeRangeHours);

  // Transform data for TrainingVisualization component
  const trainingMetrics: TrainingMetric[] = trainingAnalytics?.training_metrics || [];
  const resourceMetrics: ResourceMetric[] = trainingAnalytics?.resource_metrics || [];

  const tabs = [
    { id: 'overview', name: 'Dashboard Overview', icon: BarChart3, disabled: false },
    { id: 'training', name: 'Training Analytics', icon: Activity, disabled: !currentTaskId },
    { id: 'dataset', name: 'Dataset Analytics', icon: Database, disabled: !selectedDatasetPath }
  ];

  return (
    <div className="space-y-6">
      {/* Header with refresh button */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <BarChart3 className="h-8 w-8 text-primary" />
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Analytics Dashboard</h1>
            <p className="text-muted-foreground">
              Comprehensive visualization and analysis of your LLM training pipeline
            </p>
          </div>
        </div>
        <button
          onClick={refetchAll}
          disabled={isLoading}
          className="inline-flex items-center justify-center rounded-md border border-input bg-background hover:bg-accent hover:text-accent-foreground px-4 py-2 text-sm font-medium shadow-sm transition-colors disabled:pointer-events-none disabled:opacity-50"
        >
          <RefreshCw className={`mr-2 h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh All Data
        </button>
      </div>

      {/* Dashboard Summary Cards (Overview) */}
      {dashboardSummary && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="rounded-lg border bg-card text-card-foreground shadow-sm">
            <div className="p-6 flex flex-row items-center justify-between space-y-0 pb-2">
              <div className="tracking-tight text-sm font-medium">Active Trainings</div>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="p-6 pt-0">
              <div className="text-2xl font-bold">{dashboardSummary?.summary?.active_trainings ?? 0}</div>
            </div>
          </div>

          <div className="rounded-lg border bg-card text-card-foreground shadow-sm">
            <div className="p-6 flex flex-row items-center justify-between space-y-0 pb-2">
              <div className="tracking-tight text-sm font-medium">Completed Trainings</div>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="p-6 pt-0">
              <div className="text-2xl font-bold">{dashboardSummary?.summary?.completed_trainings ?? 0}</div>
            </div>
          </div>

          <div className="rounded-lg border bg-card text-card-foreground shadow-sm">
            <div className="p-6 flex flex-row items-center justify-between space-y-0 pb-2">
              <div className="tracking-tight text-sm font-medium">Recent Comparisons</div>
              <Eye className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="p-6 pt-0">
              <div className="text-2xl font-bold">{dashboardSummary?.summary?.recent_comparisons ?? 0}</div>
            </div>
          </div>

          <div className="rounded-lg border bg-card text-card-foreground shadow-sm">
            <div className="p-6 flex flex-row items-center justify-between space-y-0 pb-2">
              <div className="tracking-tight text-sm font-medium">System Status</div>
              <div className={`h-2 w-2 rounded-full ${
                dashboardSummary?.summary?.system_status === 'healthy' ? 'bg-green-500' :
                dashboardSummary?.summary?.system_status === 'warning' ? 'bg-yellow-500' :
                'bg-red-500'
              }`} />
            </div>
            <div className="p-6 pt-0">
              <div className="text-lg font-semibold capitalize">{dashboardSummary?.summary?.system_status ?? 'unknown'}</div>
            </div>
          </div>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="border-b border-border">
        <nav className="flex space-x-8" aria-label="Tabs">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => !tab.disabled && setActiveTab(tab.id as any)}
                disabled={tab.disabled}
                className={`${
                  activeTab === tab.id
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground hover:border-border'
                } ${
                  tab.disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2`}
              >
                <Icon className="h-4 w-4" />
                <span>{tab.name}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Error Display */}
      {hasError && (
        <div className="rounded-md bg-destructive/15 p-4">
          <div className="flex">
            <div className="ml-3">
              <h3 className="text-sm font-medium text-destructive">
                Error loading visualization data
              </h3>
              <div className="mt-2 text-sm text-destructive">
                <ul className="list-disc pl-5 space-y-1">
                  {Object.entries(errors).map(([key, error]) => 
                    error && (
                      <li key={key}>
                        <strong>{key}:</strong> {error}
                      </li>
                    )
                  )}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab Content */}
      <div className="space-y-6">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-6">
              <h2 className="text-xl font-semibold mb-4">Quick Stats Overview</h2>
              {dashboardSummary ? (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-background rounded-lg">
                    <div className="text-2xl font-bold text-primary">
                      {dashboardSummary?.summary?.quick_stats?.avg_training_loss?.toFixed(4) ?? '0.0000'}
                    </div>
                    <p className="text-sm text-muted-foreground">Average Training Loss</p>
                  </div>
                  <div className="text-center p-4 bg-background rounded-lg">
                    <div className="text-2xl font-bold text-green-600">
                      {Math.round(dashboardSummary?.summary?.quick_stats?.avg_response_time ?? 0)}ms
                    </div>
                    <p className="text-sm text-muted-foreground">Average Response Time</p>
                  </div>
                  <div className="text-center p-4 bg-background rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">
                      {Math.round((dashboardSummary?.summary?.quick_stats?.model_performance_score ?? 0) * 100)}%
                    </div>
                    <p className="text-sm text-muted-foreground">Performance Score</p>
                  </div>
                </div>
              ) : (
                <div className="text-center text-muted-foreground">
                  {isLoading ? 'Loading dashboard data...' : 'No dashboard data available'}
                </div>
              )}
            </div>

            {/* System Health Overview */}
            {systemHealth && (
              <div className="rounded-lg border bg-card text-card-foreground shadow-sm p-6">
                <h2 className="text-xl font-semibold mb-4">System Health</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-4 bg-background rounded-lg">
                    <div className="text-lg font-bold">{systemHealth?.health_data?.cpu?.usage_percent?.toFixed(1) ?? '0.0'}%</div>
                    <p className="text-sm text-muted-foreground">CPU Usage</p>
                  </div>
                  <div className="text-center p-4 bg-background rounded-lg">
                    <div className="text-lg font-bold">{systemHealth?.health_data?.memory?.usage_percent?.toFixed(1) ?? '0.0'}%</div>
                    <p className="text-sm text-muted-foreground">Memory Usage</p>
                  </div>
                  <div className="text-center p-4 bg-background rounded-lg">
                    <div className="text-lg font-bold">{systemHealth?.health_data?.gpu?.device_count ?? 0}</div>
                    <p className="text-sm text-muted-foreground">GPU Devices</p>
                  </div>
                  <div className="text-center p-4 bg-background rounded-lg">
                    <div className="text-lg font-bold">{systemHealth?.health_data?.disk?.usage_percent?.toFixed(1) ?? '0.0'}%</div>
                    <p className="text-sm text-muted-foreground">Disk Usage</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'training' && currentTaskId && (
          <TrainingVisualization
            taskId={currentTaskId}
            trainingMetrics={trainingMetrics}
            resourceMetrics={resourceMetrics}
            isRealTime={true}
          />
        )}

        {activeTab === 'dataset' && selectedDatasetPath && datasetAnalytics && (
          <DatasetAnalytics
            datasetPath={selectedDatasetPath}
            analyticsData={datasetAnalytics.analytics_data}
            isLoading={isLoading}
            onRefresh={refetchAll}
          />
        )}


        {/* Empty states for disabled tabs */}
        {activeTab === 'training' && !currentTaskId && (
          <div className="text-center py-12">
            <Activity className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium text-foreground mb-2">No Active Training Session</h3>
            <p className="text-muted-foreground">Start a training session to view analytics and metrics.</p>
          </div>
        )}

        {activeTab === 'dataset' && !selectedDatasetPath && (
          <div className="text-center py-12">
            <Database className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium text-foreground mb-2">No Dataset Selected</h3>
            <p className="text-muted-foreground">Select a dataset to view analytics and insights.</p>
          </div>
        )}
      </div>
    </div>
  );
};