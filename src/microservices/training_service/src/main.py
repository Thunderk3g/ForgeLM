"""
LLMForge Training Microservice

Independent training service with GPU optimization, Unsloth integration,
and high-performance fine-tuning capabilities.
"""

import asyncio
import os
import signal
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import uvicorn

# Import core training components (now local to this microservice)
from services.training_service import TrainingService
from services.training_worker import process_manager
from models.schemas import FineTuneRequest, TrainingStatus
from services.service_orchestrator import register_with_orchestrator

# Import local utilities
from utils.logging import get_logger
from utils.performance import performance_monitor
from config.settings import settings

# Prometheus metrics
training_requests_total = Counter('training_requests_total', 'Total training requests')
training_duration_seconds = Histogram('training_duration_seconds', 'Training duration in seconds')
active_training_jobs = Gauge('active_training_jobs', 'Number of active training jobs')
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization percentage')
training_errors_total = Counter('training_errors_total', 'Total training errors')

# Global service instance
training_service: Optional[TrainingService] = None
metrics_task: Optional[asyncio.Task] = None
logger = get_logger(__name__)


def signal_handler(signum, frame):
    """Enhanced signal handler for graceful container shutdown"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    
    # Cancel metrics task if running
    global metrics_task
    if metrics_task and not metrics_task.done():
        metrics_task.cancel()
        logger.info("Cancelled metrics task")
    
    # Stop any active training jobs
    if training_service:
        try:
            # This would need to be implemented in TrainingService
            logger.info("Stopping all active training jobs...")
        except Exception as e:
            logger.error(f"Error stopping training jobs: {e}")
    
    # Force exit after timeout to prevent hanging
    def force_exit():
        time.sleep(10)
        logger.warning("Force exit after 10s timeout")
        os._exit(0)
    
    threading.Thread(target=force_exit, daemon=True).start()
    logger.info("Signal handler completed")


# Register signal handlers for container environment
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


async def register_service():
    """Register this service with the orchestrator"""
    orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8080')
    service_id = f"training-service-{uuid.uuid4().hex[:8]}"
    
    try:
        await register_with_orchestrator(
            orchestrator_url=orchestrator_url,
            service_name="llmforge-training",
            service_id=service_id,
            host=os.getenv('SERVICE_ADVERTISED_HOST', 'llmforge-training'),
            port=int(os.getenv('SERVICE_PORT', '8001')),
            health_endpoint="/health",
            metadata={
                "type": "training",
                "gpu_enabled": True,
                "unsloth_available": True,
                "max_concurrent_jobs": 2
            }
        )
        logger.info(f"Registered with orchestrator as {service_id}")
    except Exception as e:
        logger.warning(f"Failed to register with orchestrator: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the training service"""
    global training_service, metrics_task
    
    # Startup
    logger.info("Starting Training Microservice...")
    training_service = TrainingService()
    
    # Register with orchestrator
    await register_service()
    
    # Start performance monitoring
    performance_monitor.start_monitoring()
    
    # FIXED: Properly managed background task
    metrics_task = asyncio.create_task(update_metrics())
    logger.info("Training Microservice started successfully")
    yield
    
    # FIXED: Proper cleanup
    logger.info("Shutting down Training Microservice...")
    if metrics_task and not metrics_task.done():
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass
    
    if training_service:
        await training_service.cleanup()
    performance_monitor.stop_monitoring()
    logger.info("Training Microservice shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LLMForge Training Service",
    description="Microservice for ML model fine-tuning with GPU optimization",
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
        gpu_info = {}
        if training_service:
            gpu_info = training_service.get_gpu_status()
        
        return {
            "status": "healthy",
            "service": "training",
            "version": "1.0.0",
            "gpu_available": gpu_info.get('available', False),
            "active_jobs": process_manager.get_active_job_count() if process_manager else 0,
            "system_resources": {
                "gpu_memory_used": gpu_info.get('memory_used_mb', 0),
                "gpu_memory_total": gpu_info.get('memory_total_mb', 0),
                "gpu_utilization": gpu_info.get('utilization_percent', 0)
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return Response(
            content=f'{{"status": "unhealthy", "error": "{str(e)}"}}',
            status_code=503,
            media_type="application/json"
        )


@app.post("/fine-tune")
async def fine_tune(request: FineTuneRequest, background_tasks: BackgroundTasks):
    """Start a fine-tuning job"""
    training_requests_total.inc()
    
    if not training_service:
        training_errors_total.inc()
        raise HTTPException(status_code=503, detail="Training service not initialized")
    
    try:
        # Fix model path if it's pointing to base directory instead of specific model
        if request.base_model_dir == "/app/models/base":
            # Look for the actual model directory
            base_models_dir = "/app/models/base"
            if os.path.exists(base_models_dir):
                model_subdirs = [d for d in os.listdir(base_models_dir) 
                               if os.path.isdir(os.path.join(base_models_dir, d)) and 
                               os.path.exists(os.path.join(base_models_dir, d, "config.json"))]
                if model_subdirs:
                    # Use the first available model (or could implement selection logic)
                    request.base_model_dir = os.path.join(base_models_dir, model_subdirs[0])
                    logger.info(f"Corrected model path from /app/models/base to {request.base_model_dir}")
        
        # Validate request
        if not os.path.exists(request.base_model_dir):
            training_errors_total.inc()
            raise HTTPException(status_code=400, detail=f"Base model directory not found: {request.base_model_dir}")
        
        if not os.path.exists(request.dataset_dir):
            training_errors_total.inc()
            raise HTTPException(status_code=400, detail=f"Dataset directory not found: {request.dataset_dir}")
        
        # Start training job
        task_id = await training_service.start_fine_tuning(request)
        active_training_jobs.inc()
        
        logger.info(f"Started training job {task_id}")
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": "Fine-tuning job started successfully",
            "estimated_duration_minutes": training_service.estimate_training_time(request)
        }
        
    except Exception as e:
        training_errors_total.inc()
        logger.error(f"Fine-tuning request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{task_id}")
async def get_training_status(task_id: str):
    """Get status of a training job"""
    if not training_service:
        raise HTTPException(status_code=503, detail="Training service not initialized")
    
    try:
        status = await training_service.get_training_status(task_id)
        return status
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    except Exception as e:
        logger.error(f"Status check failed for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/{task_id}")
async def get_training_logs(task_id: str):
    """Stream training logs for a job"""
    if not training_service:
        raise HTTPException(status_code=503, detail="Training service not initialized")
    
    try:
        async def log_generator():
            async for log_entry in training_service.stream_logs(task_id):
                yield f"data: {log_entry}\n\n"
        
        return StreamingResponse(
            log_generator(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    except Exception as e:
        logger.error(f"Log streaming failed for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop/{task_id}")
async def stop_training(task_id: str):
    """Stop a training job"""
    if not training_service:
        raise HTTPException(status_code=503, detail="Training service not initialized")
    
    try:
        success = await training_service.stop_training(task_id)
        if success:
            active_training_jobs.dec()
            return {"task_id": task_id, "status": "stopped"}
        else:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found or already completed")
    except Exception as e:
        logger.error(f"Stop training failed for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs")
async def list_training_jobs():
    """List all training jobs"""
    if not training_service:
        raise HTTPException(status_code=503, detail="Training service not initialized")
    
    try:
        jobs = await training_service.list_training_jobs()
        return {"jobs": jobs}
    except Exception as e:
        logger.error(f"List jobs failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fine-tune/active")
async def list_active_training_jobs():
    """List active training jobs (backward compatibility endpoint)"""
    if not training_service:
        raise HTTPException(status_code=503, detail="Training service not initialized")
    
    try:
        # Get all jobs and filter for active ones
        all_jobs = await training_service.list_training_jobs()
        active_jobs = [job for job in all_jobs if job.get('status') in ['running', 'starting', 'in_progress']]
        
        return {
            "active_jobs": active_jobs,
            "total_active": len(active_jobs),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"List active jobs failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/available")
async def list_available_models():
    """List available base models"""
    if not training_service:
        raise HTTPException(status_code=503, detail="Training service not initialized")
    
    try:
        models = await training_service.list_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"List models failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets/available")
async def list_available_datasets():
    """List available datasets"""
    if not training_service:
        raise HTTPException(status_code=503, detail="Training service not initialized")
    
    try:
        datasets = await training_service.list_available_datasets()
        return {"datasets": datasets}
    except Exception as e:
        logger.error(f"List datasets failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/gpu")
async def get_gpu_info():
    """Get GPU system information"""
    if not training_service:
        raise HTTPException(status_code=503, detail="Training service not initialized")
    
    try:
        gpu_info = training_service.get_gpu_status()
        
        # Update Prometheus metrics
        gpu_utilization.set(gpu_info.get('utilization_percent', 0))
        
        return gpu_info
    except Exception as e:
        logger.error(f"GPU info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


@app.post("/optimize/memory")
async def optimize_memory():
    """Trigger memory optimization"""
    if not training_service:
        raise HTTPException(status_code=503, detail="Training service not initialized")
    
    try:
        await training_service.optimize_memory()
        return {"status": "optimized", "message": "Memory optimization completed"}
    except Exception as e:
        logger.error(f"Memory optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/templates")
async def list_training_templates():
    """List available training configuration templates"""
    if not training_service:
        raise HTTPException(status_code=503, detail="Training service not initialized")
    
    try:
        templates = await training_service.list_training_templates()
        return {"templates": templates}
    except Exception as e:
        logger.error(f"List templates failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/hardware")
async def get_hardware_profile():
    """Get detailed hardware profile and adaptive configuration"""
    if not training_service:
        raise HTTPException(status_code=503, detail="Training service not initialized")
    
    try:
        profile = await training_service.get_hardware_profile()
        return profile
    except Exception as e:
        logger.error(f"Hardware profile failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Frontend compatibility endpoints with /fine-tune prefix
@app.get("/fine-tune/status/{task_id}")
async def get_training_status_alt(task_id: str):
    """Get status of a training job (frontend compatibility)"""
    return await get_training_status(task_id)


@app.get("/fine-tune/logs/{task_id}")
async def get_training_logs_alt(task_id: str):
    """Stream training logs for a job (frontend compatibility)"""
    return await get_training_logs(task_id)


# Background task for updating metrics
async def update_metrics():
    """Background task to update system metrics"""
    while True:
        try:
            if training_service:
                active_jobs = process_manager.get_active_job_count() if process_manager else 0
                active_training_jobs.set(active_jobs)
                
                gpu_info = training_service.get_gpu_status()
                gpu_utilization.set(gpu_info.get('utilization_percent', 0))
            
            await asyncio.sleep(10)  # Update every 10 seconds
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
            await asyncio.sleep(30)  # Wait longer on error


if __name__ == "__main__":
    # Start metrics update task
    
    # Run the service
    port = int(os.getenv('SERVICE_PORT', '8001'))
    host = os.getenv('SERVICE_HOST', '0.0.0.0')
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )