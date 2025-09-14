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
from .services.visualization_service import VisualizationService
from .models.schemas import (
    AnalyticsQuery, DashboardConfig, MetricsResponse,
    TrainingAnalytics, ComparisonAnalytics, ChartData
)
from .services.service_orchestrator import register_with_orchestrator
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
            host=os.getenv('SERVICE_HOST', 'localhost'),
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
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    filters: Optional[Dict[str, Any]] = Query(None)
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
    format: str = Query("json", regex="^(json|csv|pdf)$"),
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


if __name__ == "__main__":
    # Start metrics update task
    asyncio.create_task(update_metrics())
    
    # Run the service
    port = int(os.getenv('SERVICE_PORT', '8004'))
    host = os.getenv('SERVICE_HOST', '0.0.0.0')
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )