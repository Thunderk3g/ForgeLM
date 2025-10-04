"""
LLMForge Data Generation Microservice

Independent service for automated dataset creation, validation,
and template management for training data generation.
"""

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Response, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import uvicorn

# Import core data generation components (local to this microservice)
from services.data_generation_service import DataGenerationService
from services.template_service import TemplateService
from services.data_preparation_service import DataPreparationService
from services.service_orchestrator import register_with_orchestrator
from services.websocket_service import get_websocket_service

# Import shared utilities and schemas from parent directories
from models.schemas import (
    DataGenerationRequest, DataGenerationResponse, TemplateConfig,
    DatasetValidationRequest, ValidationResult, PrepareDataRequest, PrepareDataResponse,
    EmbeddingModelListResponse, PreprocessingPreviewRequest, PreprocessingPreviewResponse,
    VisualizationGenerationRequest, VisualizationGenerationResponse
)
from utils.logging import get_logger
from utils.performance import performance_monitor
from config.settings import settings

# Prometheus metrics
generation_requests_total = Counter('generation_requests_total', 'Total data generation requests')
generation_duration_seconds = Histogram('generation_duration_seconds', 'Data generation duration')
active_generation_jobs = Gauge('active_generation_jobs', 'Number of active generation jobs')
templates_used = Counter('templates_used_total', 'Templates used for generation', ['template_name'])
validation_errors = Counter('validation_errors_total', 'Data validation errors')
generated_samples = Counter('generated_samples_total', 'Total generated data samples')

# Global service instances
data_generation_service: Optional[DataGenerationService] = None
template_service: Optional[TemplateService] = None
data_preparation_service: Optional[DataPreparationService] = None
logger = get_logger(__name__)


async def register_service():
    """Register this service with the orchestrator"""
    orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8080')
    service_id = f"data-generation-service-{uuid.uuid4().hex[:8]}"
    
    try:
        await register_with_orchestrator(
            orchestrator_url=orchestrator_url,
            service_name="llmforge-data-generation",
            service_id=service_id,
            host=os.getenv('SERVICE_ADVERTISED_HOST', 'llmforge-data-generation'),
            port=int(os.getenv('SERVICE_PORT', '8003')),
            health_endpoint="/health",
            metadata={
                "type": "data_generation",
                "templates_supported": True,
                "validation_enabled": True,
                "max_concurrent_jobs": 5
            }
        )
        logger.info(f"Registered with orchestrator as {service_id}")
    except Exception as e:
        logger.warning(f"Failed to register with orchestrator: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the data generation service"""
    global data_generation_service, template_service, data_preparation_service
    
    # Startup
    logger.info("Starting Data Generation Microservice...")
    
    # Initialize services
    template_service = TemplateService()
    data_generation_service = DataGenerationService(template_service)
    data_preparation_service = DataPreparationService()
    
    # Register with orchestrator
    await register_service()
    
    # Start performance monitoring
    performance_monitor.start_monitoring()
    
    # Start metrics update task
    metrics_task = asyncio.create_task(update_metrics())
    
    logger.info("Data Generation Microservice started successfully")
    yield
    
    # Cancel metrics task on shutdown
    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass
    
    # Shutdown
    logger.info("Shutting down Data Generation Microservice...")
    if data_generation_service:
        await data_generation_service.cleanup()
    performance_monitor.stop_monitoring()
    logger.info("Data Generation Microservice shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LLMForge Data Generation Service",
    description="Microservice for automated dataset creation and validation",
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
        template_count = 0
        active_jobs = 0
        
        if template_service:
            template_count = len(template_service.list_templates())
        if data_generation_service:
            active_jobs = data_generation_service.get_active_job_count()
        
        return {
            "status": "healthy",
            "service": "data_generation",
            "version": "1.0.0",
            "available_templates": template_count,
            "active_generation_jobs": active_jobs,
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


@app.post("/prepare-data", response_model=PrepareDataResponse)
async def prepare_data(request: PrepareDataRequest):
    """Clean and prepare data for fine-tuning"""
    if not data_preparation_service:
        raise HTTPException(status_code=503, detail="Data preparation service not initialized")
    
    try:
        logger.info(f"Starting data preparation for {request.input_file}")
        response = await data_preparation_service.prepare_data(request)
        
        if response.status == "success":
            logger.info(f"Data preparation completed successfully: {response.message}")
        else:
            logger.error(f"Data preparation failed: {response.message}")
        
        return response
        
    except Exception as e:
        logger.error(f"Data preparation request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=DataGenerationResponse)
async def generate_data(request: DataGenerationRequest, background_tasks: BackgroundTasks):
    """Generate training data using specified template"""
    generation_requests_total.inc()
    
    if not data_generation_service:
        raise HTTPException(status_code=503, detail="Data generation service not initialized")
    
    try:
        # Start data generation job
        job_id = await data_generation_service.start_generation(request)
        active_generation_jobs.inc()
        
        # Track template usage
        templates_used.labels(template_name=request.template_name).inc()
        
        logger.info(f"Started data generation job {job_id}")
        
        return DataGenerationResponse(
            job_id=job_id,
            status="started",
            message="Data generation job started successfully",
            estimated_completion_time=data_generation_service.estimate_completion_time(request)
        )
        
    except Exception as e:
        logger.error(f"Data generation request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate/{job_id}/status")
async def get_generation_status(job_id: str):
    """Get status of a data generation job"""
    if not data_generation_service:
        raise HTTPException(status_code=503, detail="Data generation service not initialized")
    
    try:
        status = await data_generation_service.get_job_status(job_id)
        return status
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except Exception as e:
        logger.error(f"Status check failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate/{job_id}/progress")
async def stream_generation_progress(job_id: str):
    """Stream generation progress"""
    if not data_generation_service:
        raise HTTPException(status_code=503, detail="Data generation service not initialized")
    
    try:
        async def progress_generator():
            async for progress_data in data_generation_service.stream_progress(job_id):
                yield f"data: {progress_data}\n\n"
        
        return StreamingResponse(
            progress_generator(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except Exception as e:
        logger.error(f"Progress streaming failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/{job_id}/stop")
async def stop_generation(job_id: str):
    """Stop a data generation job"""
    if not data_generation_service:
        raise HTTPException(status_code=503, detail="Data generation service not initialized")
    
    try:
        success = await data_generation_service.stop_job(job_id)
        if success:
            active_generation_jobs.dec()
            return {"job_id": job_id, "status": "stopped"}
        else:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or already completed")
    except Exception as e:
        logger.error(f"Stop generation failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs")
async def list_generation_jobs():
    """List all data generation jobs"""
    if not data_generation_service:
        raise HTTPException(status_code=503, detail="Data generation service not initialized")
    
    try:
        jobs = await data_generation_service.list_jobs()
        return {"jobs": jobs}
    except Exception as e:
        logger.error(f"List jobs failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/templates")
async def list_templates():
    """List available data generation templates"""
    if not template_service:
        raise HTTPException(status_code=503, detail="Template service not initialized")
    
    try:
        templates = template_service.list_templates()
        return {"templates": templates}
    except Exception as e:
        logger.error(f"List templates failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/templates/{template_name}")
async def get_template(template_name: str):
    """Get template configuration"""
    if not template_service:
        raise HTTPException(status_code=503, detail="Template service not initialized")
    
    try:
        template = template_service.get_template(template_name)
        if template:
            return template
        else:
            raise HTTPException(status_code=404, detail=f"Template {template_name} not found")
    except Exception as e:
        logger.error(f"Get template failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/templates/{template_name}")
async def create_template(template_name: str, config: TemplateConfig):
    """Create or update a template"""
    if not template_service:
        raise HTTPException(status_code=503, detail="Template service not initialized")
    
    try:
        success = template_service.create_template(template_name, config)
        if success:
            return {"template_name": template_name, "status": "created"}
        else:
            raise HTTPException(status_code=400, detail="Failed to create template")
    except Exception as e:
        logger.error(f"Create template failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/templates/{template_name}")
async def delete_template(template_name: str):
    """Delete a template"""
    if not template_service:
        raise HTTPException(status_code=503, detail="Template service not initialized")
    
    try:
        success = template_service.delete_template(template_name)
        if success:
            return {"template_name": template_name, "status": "deleted"}
        else:
            raise HTTPException(status_code=404, detail=f"Template {template_name} not found")
    except Exception as e:
        logger.error(f"Delete template failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate", response_model=ValidationResult)
async def validate_dataset(request: DatasetValidationRequest):
    """Validate a generated dataset"""
    if not data_generation_service:
        raise HTTPException(status_code=503, detail="Data generation service not initialized")
    
    try:
        result = await data_generation_service.validate_dataset(request)
        
        if not result.is_valid:
            validation_errors.inc()
        
        return result
        
    except Exception as e:
        validation_errors.inc()
        logger.error(f"Dataset validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/convert")
async def convert_dataset_format(
    source_path: str,
    target_path: str,
    source_format: str,
    target_format: str
):
    """Convert dataset between different formats"""
    if not data_generation_service:
        raise HTTPException(status_code=503, detail="Data generation service not initialized")
    
    try:
        success = await data_generation_service.convert_format(
            source_path, target_path, source_format, target_format
        )
        
        if success:
            return {
                "status": "converted",
                "source": source_path,
                "target": target_path,
                "source_format": source_format,
                "target_format": target_format
            }
        else:
            raise HTTPException(status_code=400, detail="Conversion failed")
            
    except Exception as e:
        logger.error(f"Dataset conversion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/formats")
async def list_supported_formats():
    """List supported dataset formats"""
    return {
        "supported_formats": [
            "jsonl",
            "json",
            "csv",
            "parquet",
            "huggingface",
            "alpaca",
            "chatml",
            "sharegpt"
        ]
    }


@app.get("/stats/{job_id}")
async def get_generation_stats(job_id: str):
    """Get detailed statistics for a generation job"""
    if not data_generation_service:
        raise HTTPException(status_code=503, detail="Data generation service not initialized")
    
    try:
        stats = await data_generation_service.get_job_stats(job_id)
        
        # Update metrics
        if 'samples_generated' in stats:
            generated_samples.inc(stats['samples_generated'])
        
        return stats
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except Exception as e:
        logger.error(f"Get stats failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch/generate")
async def batch_generate(requests: List[DataGenerationRequest]):
    """Start multiple data generation jobs"""
    if not data_generation_service:
        raise HTTPException(status_code=503, detail="Data generation service not initialized")
    
    try:
        job_ids = []
        for request in requests:
            job_id = await data_generation_service.start_generation(request)
            job_ids.append(job_id)
            active_generation_jobs.inc()
            templates_used.labels(template_name=request.template_name).inc()
        
        return {
            "job_ids": job_ids,
            "status": "started",
            "message": f"Started {len(job_ids)} data generation jobs"
        }
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


@app.get("/quality/analysis/{job_id}")
async def analyze_data_quality(job_id: str):
    """Analyze quality of generated data"""
    if not data_generation_service:
        raise HTTPException(status_code=503, detail="Data generation service not initialized")
    
    try:
        analysis = await data_generation_service.analyze_quality(job_id)
        return analysis
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except Exception as e:
        logger.error(f"Quality analysis failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/embedding-models", response_model=EmbeddingModelListResponse)
async def get_embedding_models():
    """Get available embedding models for preprocessing"""
    if not data_preparation_service:
        raise HTTPException(status_code=503, detail="Data preparation service not initialized")
    
    try:
        models = data_preparation_service.get_embedding_models()
        return models
    except Exception as e:
        logger.error(f"Failed to get embedding models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocessing/preview", response_model=PreprocessingPreviewResponse)
async def preview_preprocessing(request: PreprocessingPreviewRequest):
    """Preview preprocessing results on a sample of data"""
    if not data_preparation_service:
        raise HTTPException(status_code=503, detail="Data preparation service not initialized")
    
    try:
        logger.info(f"Starting preprocessing preview for {request.input_file}")
        
        # Create a mock response for now - would implement actual preview logic
        preview_response = PreprocessingPreviewResponse(
            status="success",
            message=f"Preview generated for {request.sample_size} samples",
            preview_samples=[],
            estimated_processing_time=30.0,
            estimated_output_size="~500MB",
            validation_warnings=[],
            preprocessing_summary={
                "sample_size": request.sample_size,
                "embedding_enabled": request.preprocessing_config.enable_embeddings,
                "cleaning_level": request.preprocessing_config.text_cleaning_level
            }
        )
        
        return preview_response
        
    except Exception as e:
        logger.error(f"Preprocessing preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-visualizations", response_model=VisualizationGenerationResponse)
async def generate_preprocessing_visualizations(request: VisualizationGenerationRequest):
    """Generate visualizations for preprocessed data"""
    try:
        logger.info(f"Generating visualizations for dataset: {request.dataset_path}")
        
        # This would integrate with the visualization service
        visualization_response = VisualizationGenerationResponse(
            status="success",
            message="Visualizations generated successfully",
            visualization_data={
                "correlation_matrix": {"generated": True},
                "tensor_analysis": {"generated": True},
                "token_distribution": {"generated": True}
            },
            file_paths={
                "correlation_heatmap": "/visualizations/correlation.html",
                "tensor_plot": "/visualizations/tensor.html",
                "distribution_chart": "/visualizations/distribution.html"
            },
            generation_time_ms=1500.0,
            metadata={
                "dataset_path": request.dataset_path,
                "visualization_types": request.visualization_types,
                "output_format": request.output_format
            }
        )
        
        return visualization_response
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/preprocessing/config-templates")
async def get_preprocessing_config_templates():
    """Get predefined preprocessing configuration templates"""
    templates = {
        "basic": {
            "name": "Basic Preprocessing",
            "description": "Standard text cleaning with minimal processing",
            "config": {
                "enable_embeddings": False,
                "text_cleaning_level": "standard",
                "remove_duplicates_by_embedding": False,
                "generate_correlations": False,
                "create_tensor_visualizations": False,
                "quality_analysis": True,
                "token_analysis": True
            }
        },
        "embedding_enhanced": {
            "name": "Embedding-Enhanced Preprocessing", 
            "description": "Full preprocessing with embedding generation and analysis",
            "config": {
                "enable_embeddings": True,
                "embedding_config": {
                    "model_name": "sentence-transformers/all-mpnet-base-v2",
                    "model_type": "sentence_transformer",
                    "max_sequence_length": 512,
                    "normalize_embeddings": True,
                    "batch_size": 32
                },
                "text_cleaning_level": "standard",
                "remove_duplicates_by_embedding": True,
                "similarity_threshold": 0.95,
                "generate_correlations": True,
                "create_tensor_visualizations": True,
                "quality_analysis": True,
                "token_analysis": True
            }
        },
        "research_analysis": {
            "name": "Research & Analysis",
            "description": "Comprehensive analysis with visualizations for research purposes",
            "config": {
                "enable_embeddings": True,
                "embedding_config": {
                    "model_name": "BAAI/bge-base-en-v1.5",
                    "model_type": "sentence_transformer",
                    "max_sequence_length": 512,
                    "normalize_embeddings": True,
                    "batch_size": 16
                },
                "text_cleaning_level": "aggressive",
                "remove_duplicates_by_embedding": True,
                "similarity_threshold": 0.9,
                "generate_correlations": True,
                "create_tensor_visualizations": True,
                "quality_analysis": True,
                "token_analysis": True,
                "preview_samples": 20
            }
        }
    }
    
    return {"templates": templates}


@app.get("/preprocessing/status/{dataset_path}")
async def get_preprocessing_status(dataset_path: str):
    """Get preprocessing status and analytics for a dataset"""
    try:
        from pathlib import Path
        import json
        
        dataset_path_obj = Path(dataset_path)
        if not dataset_path_obj.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Check for preprocessing analytics
        analytics_path = dataset_path_obj / "preprocessing_analytics.json"
        if analytics_path.exists():
            with open(analytics_path, 'r') as f:
                analytics = json.load(f)
            
            return {
                "status": "completed",
                "dataset_path": dataset_path,
                "analytics": analytics,
                "has_embeddings": analytics.get("embedding_stats") is not None,
                "has_visualizations": analytics.get("tensor_visualizations") is not None
            }
        else:
            return {
                "status": "not_processed",
                "dataset_path": dataset_path,
                "message": "No preprocessing analytics found"
            }
            
    except Exception as e:
        logger.error(f"Failed to get preprocessing status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time preprocessing updates"""
    await websocket.accept()
    websocket_service = get_websocket_service()
    
    try:
        # Register connection
        await websocket_service.register_connection(websocket, session_id)
        
        while True:
            # Listen for messages
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await handle_websocket_message(websocket_service, session_id, message)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error for session {session_id}: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        pass
    finally:
        await websocket_service.unregister_connection(session_id)


async def handle_websocket_message(ws_service, session_id: str, message: dict):
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")
    
    if message_type == "ping":
        await ws_service.send_message(session_id, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })
    
    elif message_type == "start_preview":
        config = message.get("config", {})
        await ws_service.start_preview_session(session_id, config)
        
        # Start actual preview processing (mock implementation)
        asyncio.create_task(process_preview_session(ws_service, session_id, config))
    
    elif message_type == "stop_preview":
        await ws_service.send_message(session_id, {
            "type": "preview_stopped",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
    
    elif message_type == "pause_preview":
        await ws_service.send_message(session_id, {
            "type": "preview_paused",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
    
    elif message_type == "resume_preview":
        await ws_service.send_message(session_id, {
            "type": "preview_resumed",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })


async def process_preview_session(ws_service, session_id: str, config: dict):
    """Process a preview session with real-time updates"""
    try:
        sample_size = config.get("sample_size", 10)
        
        for i in range(sample_size):
            # Simulate sample processing
            await asyncio.sleep(0.5)  # Simulate processing time
            
            sample_data = {
                "index": i,
                "original_text": f"Sample {i+1} original text...",
                "processed_text": f"Sample {i+1} processed text...",
                "token_count": 50 + i * 5,
                "quality_score": 0.7 + (i % 3) * 0.1,
                "embedding_preview": [0.1, 0.2, 0.3] if config.get("enable_embeddings") else None
            }
            
            await ws_service.send_preview_sample(session_id, sample_data, i)
        
        # Send completion summary
        summary = {
            "total_samples": sample_size,
            "avg_quality_score": 0.8,
            "estimated_processing_time": sample_size * 2.0,
            "estimated_output_size": f"~{sample_size * 50}MB"
        }
        
        await ws_service.complete_preview_session(session_id, summary)
        
    except Exception as e:
        await ws_service.send_message(session_id, {
            "type": "preview_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })


@app.post("/preprocessing/preview-realtime")
async def start_realtime_preview(request: dict):
    """Start a real-time preprocessing preview session"""
    session_id = str(uuid.uuid4())
    
    return {
        "session_id": session_id,
        "websocket_url": f"/ws/{session_id}",
        "status": "session_created",
        "message": "Connect to WebSocket to receive real-time updates"
    }


# Background task for updating metrics
async def update_metrics():
    """Background task to update system metrics"""
    while True:
        try:
            if data_generation_service:
                active_jobs = data_generation_service.get_active_job_count()
                active_generation_jobs.set(active_jobs)
            
            await asyncio.sleep(10)  # Update every 10 seconds
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
            await asyncio.sleep(30)  # Wait longer on error


if __name__ == "__main__":
    # Run the service
    port = int(os.getenv('SERVICE_PORT', '8003'))
    host = os.getenv('SERVICE_HOST', '0.0.0.0')
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )