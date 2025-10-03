"""
Pydantic schemas for Visualization Service
"""

from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime


class AnalyticsQuery(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    metric_type: str
    aggregation: Optional[str] = "daily"
    filters: Optional[Dict[str, Any]] = None


class DashboardConfig(BaseModel):
    dashboard_name: str
    widgets: List[Dict[str, Any]]
    layout: Dict[str, Any]
    refresh_interval: int = 30


class MetricsResponse(BaseModel):
    metrics: Dict[str, Any]
    timestamp: str
    query_time_ms: float


class TrainingAnalytics(BaseModel):
    training_id: str
    model_name: str
    metrics: Dict[str, List[float]]
    timestamps: List[str]
    duration_minutes: float
    status: str


class ComparisonAnalytics(BaseModel):
    session_id: str
    comparison_count: int
    avg_response_time_ms: float
    model_performance: Dict[str, Any]
    user_preferences: Dict[str, float]


class ChartData(BaseModel):
    chart_type: str
    title: str
    data: List[Dict[str, Any]]
    labels: List[str]
    options: Optional[Dict[str, Any]] = None


class TensorVisualizationRequest(BaseModel):
    """Request for tensor visualization generation"""
    dataset_path: str
    embeddings_path: Optional[str] = None
    visualization_types: List[str] = ["pca", "tsne", "correlation", "distribution"]
    output_format: str = "json"


class TensorVisualizationResponse(BaseModel):
    """Response containing tensor visualization data"""
    status: str
    dataset_path: str
    has_embeddings: bool
    embedding_shape: Optional[List[int]] = None
    visualization_data: Dict[str, Any]
    timestamp: str


class CorrelationMatrixRequest(BaseModel):
    """Request for correlation matrix visualization"""
    dataset_path: str
    feature_subset: Optional[List[str]] = None
    correlation_threshold: Optional[float] = None


class CorrelationMatrixResponse(BaseModel):
    """Response containing correlation matrix visualization"""
    status: str
    correlation_matrix: Dict[str, Any]
    feature_names: List[str]
    plot: Optional[str] = None
    timestamp: str


class TokenDistributionRequest(BaseModel):
    """Request for token distribution visualization"""
    dataset_path: str
    analysis_type: str = "histogram"  # histogram, statistics, vocabulary
    sample_size: Optional[int] = None


class TokenDistributionResponse(BaseModel):
    """Response containing token distribution visualization"""
    status: str
    visualization_data: Dict[str, Any]
    timestamp: str


class PreprocessingVisualizationRequest(BaseModel):
    """Request for comprehensive preprocessing visualization"""
    dataset_path: str
    include_embeddings: bool = True
    include_correlations: bool = True
    include_token_analysis: bool = True
    include_quality_metrics: bool = True


class PreprocessingVisualizationResponse(BaseModel):
    """Response containing comprehensive preprocessing visualizations"""
    status: str
    dataset_path: str
    visualizations: Dict[str, Any]
    summary_statistics: Dict[str, Any]
    recommendations: List[str]
    timestamp: str