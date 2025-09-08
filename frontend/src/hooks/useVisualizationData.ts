import { useState, useEffect, useCallback } from 'react';
import {
  getTrainingAnalytics,
  getTrainingVisualizationData,
  getDatasetAnalytics,
  getComparisonAnalytics,
  getSystemHealth,
  getDashboardSummary
} from '../services/api';
import type {
  TrainingAnalyticsResponse,
  DatasetAnalyticsResponse,
  ComparisonAnalyticsResponse,
  SystemHealthResponse,
  DashboardSummaryResponse
} from '../types/api';

// Hook for training analytics
export const useTrainingAnalytics = (taskId: string | null, limit: number = 100) => {
  const [data, setData] = useState<TrainingAnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    if (!taskId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await getTrainingAnalytics(taskId, limit);
      setData(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to fetch training analytics');
    } finally {
      setLoading(false);
    }
  }, [taskId, limit]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

// Hook for enhanced training visualization data
export const useTrainingVisualization = (taskId: string | null, limit: number = 100) => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    if (!taskId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await getTrainingVisualizationData(taskId, limit);
      setData(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to fetch training visualization data');
    } finally {
      setLoading(false);
    }
  }, [taskId, limit]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

// Hook for dataset analytics
export const useDatasetAnalytics = (datasetPath: string | null) => {
  const [data, setData] = useState<DatasetAnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    if (!datasetPath) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await getDatasetAnalytics(datasetPath);
      setData(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to fetch dataset analytics');
    } finally {
      setLoading(false);
    }
  }, [datasetPath]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

// Hook for comparison analytics
export const useComparisonAnalytics = (timeRangeHours: number = 24) => {
  const [data, setData] = useState<ComparisonAnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await getComparisonAnalytics(timeRangeHours);
      setData(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to fetch comparison analytics');
    } finally {
      setLoading(false);
    }
  }, [timeRangeHours]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

// Hook for system health
export const useSystemHealth = (refreshInterval?: number) => {
  const [data, setData] = useState<SystemHealthResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await getSystemHealth();
      setData(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to fetch system health');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    
    // Set up auto-refresh if interval is provided
    if (refreshInterval) {
      const interval = setInterval(fetchData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchData, refreshInterval]);

  return { data, loading, error, refetch: fetchData };
};

// Hook for dashboard summary
export const useDashboardSummary = (
  includeTraining: boolean = true,
  includeComparisons: boolean = true,
  includeSystem: boolean = true,
  refreshInterval?: number
) => {
  const [data, setData] = useState<DashboardSummaryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await getDashboardSummary(includeTraining, includeComparisons, includeSystem);
      setData(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to fetch dashboard summary');
    } finally {
      setLoading(false);
    }
  }, [includeTraining, includeComparisons, includeSystem]);

  useEffect(() => {
    fetchData();
    
    // Set up auto-refresh if interval is provided
    if (refreshInterval) {
      const interval = setInterval(fetchData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchData, refreshInterval]);

  return { data, loading, error, refetch: fetchData };
};

// Combined hook for all visualization data
export const useVisualizationDashboard = (
  taskId: string | null = null,
  datasetPath: string | null = null,
  timeRangeHours: number = 24
) => {
  const trainingAnalytics = useTrainingAnalytics(taskId);
  const trainingVisualization = useTrainingVisualization(taskId);
  const datasetAnalytics = useDatasetAnalytics(datasetPath);
  const comparisonAnalytics = useComparisonAnalytics(timeRangeHours);
  const systemHealth = useSystemHealth(30000); // Refresh every 30 seconds
  const dashboardSummary = useDashboardSummary(true, true, true, 60000); // Refresh every minute

  const isLoading = trainingAnalytics.loading || 
                   trainingVisualization.loading || 
                   datasetAnalytics.loading || 
                   comparisonAnalytics.loading || 
                   systemHealth.loading || 
                   dashboardSummary.loading;

  const hasError = trainingAnalytics.error || 
                   trainingVisualization.error || 
                   datasetAnalytics.error || 
                   comparisonAnalytics.error || 
                   systemHealth.error || 
                   dashboardSummary.error;

  const refetchAll = useCallback(() => {
    trainingAnalytics.refetch();
    trainingVisualization.refetch();
    datasetAnalytics.refetch();
    comparisonAnalytics.refetch();
    systemHealth.refetch();
    dashboardSummary.refetch();
  }, [
    trainingAnalytics.refetch,
    trainingVisualization.refetch,
    datasetAnalytics.refetch,
    comparisonAnalytics.refetch,
    systemHealth.refetch,
    dashboardSummary.refetch
  ]);

  return {
    trainingAnalytics: trainingAnalytics.data,
    trainingVisualization: trainingVisualization.data,
    datasetAnalytics: datasetAnalytics.data,
    comparisonAnalytics: comparisonAnalytics.data,
    systemHealth: systemHealth.data,
    dashboardSummary: dashboardSummary.data,
    isLoading,
    hasError,
    errors: {
      trainingAnalytics: trainingAnalytics.error,
      trainingVisualization: trainingVisualization.error,
      datasetAnalytics: datasetAnalytics.error,
      comparisonAnalytics: comparisonAnalytics.error,
      systemHealth: systemHealth.error,
      dashboardSummary: dashboardSummary.error
    },
    refetchAll
  };
};