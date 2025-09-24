"""
LLMForge Visualization Microservice

Independent service for analytics, monitoring dashboards,
and data visualization for training and comparison metrics.
"""

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import uvicorn

# Import core visualization components
from services.visualization_service import VisualizationService
from models.schemas import (
    AnalyticsQuery, DashboardConfig, MetricsResponse,
    TrainingAnalytics, ComparisonAnalytics, ChartData,
    TensorVisualizationRequest, TensorVisualizationResponse,
    CorrelationMatrixRequest, CorrelationMatrixResponse,
    TokenDistributionRequest, TokenDistributionResponse,
    PreprocessingVisualizationRequest, PreprocessingVisualizationResponse
)
from services.service_orchestrator import register_with_orchestrator
from utils.logging import get_logger
from utils.performance import performance_monitor
from config.settings import settings

# Prometheus metrics
visualization_requests_total = Counter('visualization_requests_total', 'Total visualization requests')
dashboard_renders = Counter('dashboard_renders_total', 'Dashboard render requests', ['dashboard_type'])
analytics_queries = Counter('analytics_queries_total', 'Analytics query requests', ['query_type'])
chart_generations = Counter('chart_generations_total', 'Chart generation requests', ['chart_type'])
data_export_requests = Counter('data_export_requests_total', 'Data export requests', ['format'])

# Global service instance
visualization_service: Optional[VisualizationService] = None
logger = get_logger(__name__)


async def register_service():
    """Register this service with the orchestrator"""
    orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8080')
    service_id = f"visualization-service-{uuid.uuid4().hex[:8]}"
    
    try:
        await register_with_orchestrator(
            orchestrator_url=orchestrator_url,
            service_name="llmforge-visualization",
            service_id=service_id,
            host=os.getenv('SERVICE_ADVERTISED_HOST', 'llmforge-visualization'),
            port=int(os.getenv('SERVICE_PORT', '8004')),
            health_endpoint="/health",
            metadata={
                "type": "visualization",
                "dashboard_types": ["training", "comparison", "system"],
                "export_formats": ["json", "csv", "pdf"],
                "real_time_updates": True
            }
        )
        logger.info(f"Registered with orchestrator as {service_id}")
    except Exception as e:
        logger.warning(f"Failed to register with orchestrator: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the visualization service"""
    global visualization_service
    
    # Startup
    logger.info("Starting Visualization Microservice...")
    
    # Initialize service
    visualization_service = VisualizationService()
    
    # Register with orchestrator
    await register_service()
    
    # Start performance monitoring
    performance_monitor.start_monitoring()
    
    logger.info("Visualization Microservice started successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down Visualization Microservice...")
    if visualization_service:
        await visualization_service.cleanup()
    performance_monitor.stop_monitoring()
    logger.info("Visualization Microservice shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LLMForge Visualization Service",
    description="Microservice for analytics and monitoring dashboards",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint with detailed service status"""
    try:
        dashboard_count = 0
        cached_charts = 0
        
        if visualization_service:
            dashboard_count = len(visualization_service.list_dashboards())
            cached_charts = visualization_service.get_cache_size()
        
        return {
            "status": "healthy",
            "service": "visualization",
            "version": "1.0.0",
            "active_dashboards": dashboard_count,
            "cached_charts": cached_charts,
            "system_resources": {
                "memory_usage_mb": 0,  # Could add actual memory monitoring
                "cpu_usage_percent": 0
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return Response(
            content=f'{{"status": "unhealthy", "error": "{str(e)}"}}',
            status_code=503,
            media_type="application/json"
        )


@app.get("/analytics/training/{job_id}", response_model=TrainingAnalytics)
async def get_training_analytics(job_id: str):
    """Get training analytics for a specific job"""
    analytics_queries.labels(query_type="training").inc()
    
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        analytics = await visualization_service.get_training_analytics(job_id)
        return analytics
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    except Exception as e:
        logger.error(f"Training analytics failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/comparison/{session_id}", response_model=ComparisonAnalytics)
async def get_comparison_analytics(session_id: str):
    """Get comparison analytics for a specific session"""
    analytics_queries.labels(query_type="comparison").inc()
    
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        analytics = await visualization_service.get_comparison_analytics(session_id)
        return analytics
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Comparison session {session_id} not found")
    except Exception as e:
        logger.error(f"Comparison analytics failed for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/query", response_model=MetricsResponse)
async def query_analytics(query: AnalyticsQuery):
    """Execute custom analytics query"""
    analytics_queries.labels(query_type="custom").inc()
    
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        result = await visualization_service.execute_query(query)
        return result
        
    except Exception as e:
        logger.error(f"Analytics query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/{dashboard_type}")
async def get_dashboard(dashboard_type: str):
    """Get dashboard configuration and data"""
    dashboard_renders.labels(dashboard_type=dashboard_type).inc()
    
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        dashboard = await visualization_service.get_dashboard(dashboard_type)
        return dashboard
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dashboard {dashboard_type} not found")
    except Exception as e:
        logger.error(f"Dashboard retrieval failed for {dashboard_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dashboard/{dashboard_type}")
async def create_dashboard(dashboard_type: str, config: DashboardConfig):
    """Create or update dashboard configuration"""
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        success = await visualization_service.create_dashboard(dashboard_type, config)
        if success:
            return {"dashboard_type": dashboard_type, "status": "created"}
        else:
            raise HTTPException(status_code=400, detail="Failed to create dashboard")
    except Exception as e:
        logger.error(f"Dashboard creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/charts/{chart_type}")
async def generate_chart(
    chart_type: str,
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    filters: Optional[str] = Query(None)
):
    """Generate chart data"""
    chart_generations.labels(chart_type=chart_type).inc()
    
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        chart_data = await visualization_service.generate_chart(
            chart_type=chart_type,
            start_time=start_time,
            end_time=end_time,
            filters=filters or {}
        )
        return chart_data
        
    except Exception as e:
        logger.error(f"Chart generation failed for {chart_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/system")
async def get_system_metrics():
    """Get system-wide metrics"""
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        metrics = await visualization_service.get_system_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"System metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/training")
async def get_training_metrics(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None)
):
    """Get training-related metrics"""
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        metrics = await visualization_service.get_training_metrics(
            start_time=start_time,
            end_time=end_time
        )
        return metrics
        
    except Exception as e:
        logger.error(f"Training metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/comparison")
async def get_comparison_metrics(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None)
):
    """Get comparison-related metrics"""
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        metrics = await visualization_service.get_comparison_metrics(
            start_time=start_time,
            end_time=end_time
        )
        return metrics
        
    except Exception as e:
        logger.error(f"Comparison metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/export/{export_type}")
async def export_data(
    export_type: str,
    format: str = Query("json", pattern="^(json|csv|pdf)$"),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None)
):
    """Export data in various formats"""
    data_export_requests.labels(format=format).inc()
    
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        # Generate export data
        export_data = await visualization_service.export_data(
            export_type=export_type,
            format=format,
            start_time=start_time,
            end_time=end_time
        )
        
        # Set appropriate content type
        content_types = {
            "json": "application/json",
            "csv": "text/csv",
            "pdf": "application/pdf"
        }
        
        return Response(
            content=export_data,
            media_type=content_types.get(format, "application/octet-stream"),
            headers={
                "Content-Disposition": f"attachment; filename=export_{export_type}_{datetime.now().isoformat()}.{format}"
            }
        )
        
    except Exception as e:
        logger.error(f"Data export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/live/training/{job_id}")
async def stream_training_metrics(job_id: str):
    """Stream real-time training metrics"""
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        async def metrics_generator():
            async for metrics in visualization_service.stream_training_metrics(job_id):
                yield f"data: {metrics}\n\n"
        
        return StreamingResponse(
            metrics_generator(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    except Exception as e:
        logger.error(f"Training metrics streaming failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/live/comparison/{session_id}")
async def stream_comparison_metrics(session_id: str):
    """Stream real-time comparison metrics"""
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        async def metrics_generator():
            async for metrics in visualization_service.stream_comparison_metrics(session_id):
                yield f"data: {metrics}\n\n"
        
        return StreamingResponse(
            metrics_generator(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Comparison session {session_id} not found")
    except Exception as e:
        logger.error(f"Comparison metrics streaming failed for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports/training/{job_id}")
async def generate_training_report(job_id: str):
    """Generate comprehensive training report"""
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        report = await visualization_service.generate_training_report(job_id)
        return report
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    except Exception as e:
        logger.error(f"Training report generation failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports/comparison/{session_id}")
async def generate_comparison_report(session_id: str):
    """Generate comprehensive comparison report"""
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        report = await visualization_service.generate_comparison_report(session_id)
        return report
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Comparison session {session_id} not found")
    except Exception as e:
        logger.error(f"Comparison report generation failed for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboards")
async def list_dashboards():
    """List available dashboards"""
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        dashboards = visualization_service.list_dashboards()
        return {"dashboards": dashboards}
    except Exception as e:
        logger.error(f"List dashboards failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chart_types")
async def list_chart_types():
    """List supported chart types"""
    return {
        "chart_types": [
            "line",
            "bar",
            "pie",
            "scatter",
            "heatmap",
            "histogram",
            "box",
            "radar",
            "area",
            "gauge"
        ]
    }


@app.post("/cache/clear")
async def clear_cache():
    """Clear visualization cache"""
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        await visualization_service.clear_cache()
        return {"status": "cleared", "message": "Visualization cache cleared successfully"}
    except Exception as e:
        logger.error(f"Clear cache failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tensor/visualize", response_model=TensorVisualizationResponse)
async def generate_tensor_visualization(request: TensorVisualizationRequest):
    """Generate comprehensive tensor visualizations"""
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        result = await visualization_service.generate_tensor_visualization(
            dataset_path=request.dataset_path,
            embeddings_path=request.embeddings_path
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "Tensor visualization failed"))
        
        return TensorVisualizationResponse(
            status=result["status"],
            dataset_path=result["dataset_path"],
            has_embeddings=result["has_embeddings"],
            embedding_shape=result.get("embedding_shape"),
            visualization_data=result["visualization_data"],
            timestamp=result["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Tensor visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/correlation/matrix", response_model=CorrelationMatrixResponse)
async def generate_correlation_matrix(request: CorrelationMatrixRequest):
    """Generate correlation matrix visualization"""
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        result = await visualization_service.generate_correlation_matrix_visualization(
            dataset_path=request.dataset_path
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "Correlation matrix generation failed"))
        
        return CorrelationMatrixResponse(
            status=result["status"],
            correlation_matrix=result["correlation_matrix"],
            feature_names=result["feature_names"],
            plot=result.get("plot"),
            timestamp=result["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Correlation matrix generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/token/distribution", response_model=TokenDistributionResponse)
async def generate_token_distribution(request: TokenDistributionRequest):
    """Generate token distribution visualization"""
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        result = await visualization_service.generate_token_distribution_visualization(
            dataset_path=request.dataset_path
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "Token distribution generation failed"))
        
        return TokenDistributionResponse(
            status=result["status"],
            visualization_data=result["visualization_data"],
            timestamp=result["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Token distribution generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocessing/visualize", response_model=PreprocessingVisualizationResponse)
async def generate_preprocessing_visualizations(request: PreprocessingVisualizationRequest):
    """Generate comprehensive preprocessing visualizations"""
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        visualizations = {}
        summary_statistics = {}
        recommendations = []
        preprocessing_analytics = {}
        
        try:
            import json
            from pathlib import Path
            analytics_path = Path(request.dataset_path) / "preprocessing_analytics.json"
            if analytics_path.exists():
                with open(analytics_path, 'r') as f:
                    preprocessing_analytics = json.load(f)
                logger.info(f"Loaded preprocessing analytics from {analytics_path}")
        except Exception as e:
            logger.warning(f"Could not load preprocessing analytics: {e}")
        
        if request.include_embeddings:
            embeddings_path = f"{request.dataset_path}/embeddings.npy"
            from pathlib import Path
            if Path(embeddings_path).exists():
                tensor_result = await visualization_service.generate_tensor_visualization(
                    dataset_path=request.dataset_path,
                    embeddings_path=embeddings_path
                )
                if tensor_result.get("status") == "success":
                    vis_data = tensor_result.get("visualization_data", {})
                    preprocessing_summary = vis_data.get("preprocessing_summary", {})
                    
                    if preprocessing_summary.get("tensor_visualizations"):
                        visualizations["tensor_visualizations"] = preprocessing_summary["tensor_visualizations"]
                    
                    if preprocessing_summary.get("embedding_stats"):
                        visualizations["embedding_stats"] = preprocessing_summary["embedding_stats"]
                    elif preprocessing_analytics.get("embedding_stats"):
                        visualizations["embedding_stats"] = preprocessing_analytics["embedding_stats"]
                    
                    recommendations.append("Embeddings generated successfully - consider using for similarity analysis")
            else:
                logger.warning(f"Embeddings file not found at {embeddings_path}")
                if preprocessing_analytics.get("embedding_stats"):
                    visualizations["embedding_stats"] = preprocessing_analytics["embedding_stats"]
        
        if request.include_correlations:
            correlation_result = await visualization_service.generate_correlation_matrix_visualization(
                dataset_path=request.dataset_path
            )
            if correlation_result.get("status") == "success":
                visualizations["correlation_analysis"] = correlation_result
                recommendations.append("Correlation matrix shows feature relationships")
        
        if request.include_token_analysis:
            if preprocessing_analytics.get("token_distribution"):
                visualizations["token_distribution"] = preprocessing_analytics["token_distribution"]
                token_stats = preprocessing_analytics["token_distribution"].get("token_statistics", {})
                if token_stats.get("mean_tokens_per_sample", 0) > 512:
                    recommendations.append("Average token length is high - consider sequence length adjustments")
            else:
                token_result = await visualization_service.generate_token_distribution_visualization(
                    dataset_path=request.dataset_path
                )
                if token_result.get("status") == "success"):
                    token_vis_data = token_result.get("visualization_data", {})
                    visualizations["token_distribution"] = {
                        "token_statistics": token_vis_data.get("token_statistics", {}),
                        "vocabulary_stats": token_vis_data.get("vocabulary_statistics", {})
                    }
        
        if request.include_quality_metrics:
            if preprocessing_analytics.get("quality_metrics"):
                visualizations["quality_metrics"] = preprocessing_analytics["quality_metrics"]
                summary_statistics = preprocessing_analytics["quality_metrics"]
                if summary_statistics.get("overall_quality_score", 0) < 0.7:
                    recommendations.append("Dataset quality score is below recommended threshold")
        
        if preprocessing_analytics.get("preprocessing_time_ms"):
            visualizations["preprocessing_time_ms"] = preprocessing_analytics["preprocessing_time_ms"]
        
        if preprocessing_analytics.get("duplicate_removal_stats"):
            visualizations["duplicate_removal_stats"] = preprocessing_analytics["duplicate_removal_stats"]
        
        return PreprocessingVisualizationResponse(
            status="success",
            dataset_path=request.dataset_path,
            visualizations=visualizations,
            summary_statistics=summary_statistics,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Preprocessing visualization generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset/{dataset_id}/analytics")
async def get_dataset_analytics(dataset_id: str):
    """Get comprehensive analytics for a specific dataset"""
    if not visualization_service:
        raise HTTPException(status_code=503, detail="Visualization service not initialized")
    
    try:
        analytics = await visualization_service.get_dataset_analytics(dataset_id)
        return analytics
        
    except Exception as e:
        logger.error(f"Dataset analytics failed for {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/embedding/models/performance")
async def get_embedding_model_performance():
    """Get performance comparison of different embedding models"""
    try:
        # Mock performance data - would be real metrics in production
        performance_data = {
            "models": [
                {
                    "name": "sentence-transformers/all-mpnet-base-v2",
                    "accuracy_score": 0.89,
                    "speed_tokens_per_second": 1200,
                    "memory_usage_mb": 1100,
                    "use_cases": ["general", "similarity", "classification"]
                },
                {
                    "name": "BAAI/bge-base-en-v1.5", 
                    "accuracy_score": 0.92,
                    "speed_tokens_per_second": 1100,
                    "memory_usage_mb": 1050,
                    "use_cases": ["retrieval", "similarity", "classification"]
                },
                {
                    "name": "jinaai/jina-embeddings-v2-small-en",
                    "accuracy_score": 0.85,
                    "speed_tokens_per_second": 2400,
                    "memory_usage_mb": 400,
                    "use_cases": ["fast_processing", "similarity"]
                }
            ],
            "benchmark_date": datetime.now().isoformat(),
            "recommendations": {
                "best_accuracy": "BAAI/bge-base-en-v1.5",
                "best_speed": "jinaai/jina-embeddings-v2-small-en",
                "best_balanced": "sentence-transformers/all-mpnet-base-v2"
            }
        }
        return performance_data
        
    except Exception as e:
        logger.error(f"Embedding model performance query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualization/templates")
async def get_visualization_templates():
    """Get predefined visualization templates"""
    templates = {
        "preprocessing_overview": {
            "name": "Preprocessing Overview",
            "description": "Complete overview of data preprocessing results",
            "components": ["tensor_analysis", "correlation_matrix", "token_distribution", "quality_metrics"]
        },
        "embedding_analysis": {
            "name": "Embedding Analysis",
            "description": "Focus on embedding quality and characteristics",
            "components": ["pca_visualization", "tsne_plot", "embedding_distribution", "similarity_analysis"]
        },
        "quality_report": {
            "name": "Data Quality Report",
            "description": "Comprehensive data quality assessment",
            "components": ["quality_scores", "completeness_analysis", "vocabulary_richness", "duplicate_detection"]
        }
    }
    return {"templates": templates}


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


# Background task for updating metrics
async def update_metrics():
    """Background task to update system metrics"""
    while True:
        try:
            if visualization_service:
                # Update visualization-specific metrics here
                pass
            
            await asyncio.sleep(10)  # Update every 10 seconds
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
            await asyncio.sleep(30)  # Wait longer on error


async def main():
    """Main entry point for the visualization service"""
    # Start metrics update task
    asyncio.create_task(update_metrics())
    
    # Run the service
    port = int(os.getenv('SERVICE_PORT', '8004'))
    host = os.getenv('SERVICE_HOST', '0.0.0.0')
    
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())