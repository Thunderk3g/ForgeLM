"""
LLMForge Model Management Microservice

Independent service for model lifecycle management, caching,
storage optimization, and version control.
"""

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Response, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import uvicorn

# Import core model management components (local to this microservice)
from services.model_service import ModelService
from models.schemas import (
    ModelLoadRequest, ModelInfo, ModelConfig, CacheStats,
    ModelUploadRequest, ModelVersionInfo
)
from services.service_orchestrator import register_with_orchestrator
from utils.logging import get_logger
from utils.performance import performance_monitor
from utils.memory import get_gpu_memory_info
from config.settings import settings

# Prometheus metrics
model_load_requests_total = Counter('model_load_requests_total', 'Total model load requests')
model_load_duration_seconds = Histogram('model_load_duration_seconds', 'Model load duration')
models_loaded = Gauge('models_loaded', 'Number of currently loaded models')
cache_hits = Counter('model_cache_hits_total', 'Model cache hits')
cache_misses = Counter('model_cache_misses_total', 'Model cache misses')
gpu_memory_usage_bytes = Gauge('gpu_memory_usage_bytes', 'GPU memory usage in bytes')
model_errors_total = Counter('model_errors_total', 'Total model-related errors', ['error_type'])

# Global service instance
model_service: Optional[ModelService] = None
logger = get_logger(__name__)


async def register_service():
    """Register this service with the orchestrator"""
    orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8080')
    service_id = f"model-management-service-{uuid.uuid4().hex[:8]}"
    
    try:
        await register_with_orchestrator(
            orchestrator_url=orchestrator_url,
            service_name="llmforge-model-management",
            service_id=service_id,
            host=os.getenv('SERVICE_ADVERTISED_HOST', 'llmforge-model-management'),
            port=int(os.getenv('SERVICE_PORT', '8005')),
            health_endpoint="/health",
            metadata={
                "type": "model_management",
                "gpu_enabled": True,
                "cache_enabled": True,
                "version_control": True,
                "max_loaded_models": 5
            }
        )
        logger.info(f"Registered with orchestrator as {service_id}")
    except Exception as e:
        logger.warning(f"Failed to register with orchestrator: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the model management service"""
    global model_service
    
    # Startup
    logger.info("Starting Model Management Microservice...")
    
    # Initialize service
    model_service = ModelService()
    
    # Register with orchestrator
    await register_service()
    
    # Start performance monitoring
    performance_monitor.start_monitoring()
    
    # Start metrics update task
    metrics_task = asyncio.create_task(update_metrics())
    
    logger.info("Model Management Microservice started successfully")
    yield
    
    # Cancel metrics task on shutdown
    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass
    
    # Shutdown
    logger.info("Shutting down Model Management Microservice...")
    if model_service:
        await model_service.cleanup()
    performance_monitor.stop_monitoring()
    logger.info("Model Management Microservice shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LLMForge Model Management Service",
    description="Microservice for model lifecycle management and caching",
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
        loaded_models = 0
        cache_size = 0
        gpu_info = {}
        
        if model_service:
            loaded_models = len(model_service.get_loaded_models())
            cache_size = model_service.get_cache_size()
            gpu_info = get_gpu_memory_info()
        
        return {
            "status": "healthy",
            "service": "model_management",
            "version": "1.0.0",
            "loaded_models": loaded_models,
            "cache_size": cache_size,
            "gpu_info": gpu_info,
            "system_resources": {
                "gpu_memory_used_mb": gpu_info.get('used_mb', 0),
                "gpu_memory_total_mb": gpu_info.get('total_mb', 0)
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return Response(
            content=f'{{"status": "unhealthy", "error": "{str(e)}"}}',
            status_code=503,
            media_type="application/json"
        )


@app.post("/models/load")
async def load_model(request: ModelLoadRequest):
    """Load a model into memory"""
    model_load_requests_total.inc()
    
    if not model_service:
        model_errors_total.labels(error_type="service_unavailable").inc()
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        # Load the model
        model_info = await model_service.load_model(
            model_path=request.model_path,
            model_config=request.config,
            force_reload=request.force_reload
        )
        
        models_loaded.inc()
        
        logger.info(f"Loaded model: {request.model_path}")
        
        return {
            "status": "loaded",
            "model_info": model_info,
            "message": f"Model {request.model_path} loaded successfully"
        }
        
    except FileNotFoundError:
        model_errors_total.labels(error_type="model_not_found").inc()
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")
    except Exception as e:
        model_errors_total.labels(error_type="load_error").inc()
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/unload")
async def unload_model(model_id: str):
    """Unload a model from memory"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        success = await model_service.unload_model(model_id)
        if success:
            models_loaded.dec()
            return {"status": "unloaded", "model_id": model_id}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        logger.error(f"Model unloading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/loaded")
async def list_loaded_models():
    """List currently loaded models"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        models = model_service.get_loaded_models()
        return {"loaded_models": models}
    except Exception as e:
        logger.error(f"List loaded models failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/available")
async def list_available_models():
    """List available models for loading"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        models = await model_service.list_available_models()
        return {"available_models": models}
    except Exception as e:
        logger.error(f"List available models failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/select-model")
async def select_model(request: dict):
    """Select and prepare a model for use (backward compatibility endpoint)"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        model_name = request.get('model_name')
        model_path = request.get('model_path')
        
        if not model_name and not model_path:
            raise HTTPException(status_code=400, detail="Either model_name or model_path must be provided")
        
        # If model_name is provided, try to find the model path
        if model_name and not model_path:
            available_models = await model_service.list_available_models()
            matching_models = [m for m in available_models if model_name.lower() in m.get('name', '').lower()]
            if matching_models:
                model_path = matching_models[0].get('path')
            else:
                # Try to download the model if it's not available locally
                try:
                    download_info = await model_service.download_model(model_name=model_name)
                    model_path = download_info.get('path')
                except Exception as e:
                    logger.warning(f"Failed to download model {model_name}: {e}")
                    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # Load the model if it's not already loaded
        try:
            model_info = await model_service.load_model(model_path=model_path)
            
            return {
                "status": "success",
                "message": f"Model {model_name or model_path} selected successfully",
                "model_info": model_info,
                "selected_model": {
                    "name": model_name or model_path,
                    "path": model_path,
                    "loaded": True
                }
            }
        except Exception as load_error:
            logger.error(f"Failed to load selected model: {load_error}")
            return {
                "status": "partial_success",
                "message": f"Model {model_name or model_path} found but failed to load",
                "model_path": model_path,
                "error": str(load_error),
                "selected_model": {
                    "name": model_name or model_path,
                    "path": model_path,
                    "loaded": False
                }
            }
            
    except Exception as e:
        logger.error(f"Model selection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_id}/info")
async def get_model_info(model_id: str):
    """Get detailed information about a model"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        model_info = await model_service.get_model_info(model_id)
        if model_info:
            return model_info
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        logger.error(f"Get model info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    model_name: str = None,
    description: str = None
):
    """Upload a new model"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        # Save uploaded file
        upload_path = await model_service.save_uploaded_model(
            file=file,
            model_name=model_name,
            description=description
        )
        
        return {
            "status": "uploaded",
            "model_path": upload_path,
            "message": f"Model uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Model upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        success = await model_service.delete_model(model_id)
        if success:
            return {"status": "deleted", "model_id": model_id}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        logger.error(f"Model deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        stats = model_service.get_cache_stats()
        
        # Update Prometheus metrics
        cache_hits.inc(stats.get('hits', 0))
        cache_misses.inc(stats.get('misses', 0))
        
        return stats
    except Exception as e:
        logger.error(f"Get cache stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear")
async def clear_cache():
    """Clear model cache"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        await model_service.clear_cache()
        return {"status": "cleared", "message": "Model cache cleared successfully"}
    except Exception as e:
        logger.error(f"Clear cache failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/optimize")
async def optimize_memory():
    """Optimize memory usage"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        stats = await model_service.optimize_memory()
        return {
            "status": "optimized",
            "freed_memory_mb": stats.get('freed_memory_mb', 0),
            "message": "Memory optimization completed"
        }
    except Exception as e:
        logger.error(f"Memory optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/storage/stats")
async def get_storage_stats():
    """Get storage statistics"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        stats = await model_service.get_storage_stats()
        return stats
    except Exception as e:
        logger.error(f"Get storage stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_id}/version")
async def create_model_version(model_id: str, version_info: ModelVersionInfo):
    """Create a new version of a model"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        version = await model_service.create_model_version(model_id, version_info)
        return {
            "status": "created",
            "model_id": model_id,
            "version": version,
            "message": "Model version created successfully"
        }
    except Exception as e:
        logger.error(f"Create model version failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_id}/versions")
async def list_model_versions(model_id: str):
    """List versions of a model"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        versions = await model_service.list_model_versions(model_id)
        return {"versions": versions}
    except Exception as e:
        logger.error(f"List model versions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/download")
async def download_model(
    model_name: str,
    source: str = "huggingface",
    revision: str = "main"
):
    """Download a model from external source"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        download_info = await model_service.download_model(
            model_name=model_name,
            source=source,
            revision=revision
        )
        
        return {
            "status": "downloaded",
            "model_path": download_info.get('path'),
            "size_mb": download_info.get('size_mb'),
            "message": f"Model {model_name} downloaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/search")
async def search_models(query: str, source: str = "all"):
    """Search for models"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        results = await model_service.search_models(query, source)
        return {"results": results}
    except Exception as e:
        logger.error(f"Model search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_id}/benchmark")
async def benchmark_model(model_id: str, benchmark_config: Dict[str, Any]):
    """Run benchmark on a model"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        benchmark_results = await model_service.benchmark_model(model_id, benchmark_config)
        return {
            "status": "completed",
            "results": benchmark_results,
            "message": f"Benchmark completed for model {model_id}"
        }
    except Exception as e:
        logger.error(f"Model benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


@app.get("/gpu/info")
async def get_gpu_info():
    """Get GPU information"""
    try:
        gpu_info = get_gpu_memory_info()
        
        # Update Prometheus metrics
        if 'used_bytes' in gpu_info:
            gpu_memory_usage_bytes.set(gpu_info['used_bytes'])
        
        return gpu_info
    except Exception as e:
        logger.error(f"GPU info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/pairs/create")
async def create_model_pair(base_model_id: str, finetuned_model_id: str):
    """Create a model pair for comparison"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        pair_id = await model_service.create_model_pair(base_model_id, finetuned_model_id)
        return {
            "status": "created",
            "pair_id": pair_id,
            "base_model": base_model_id,
            "finetuned_model": finetuned_model_id
        }
    except Exception as e:
        logger.error(f"Create model pair failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/pairs")
async def list_model_pairs():
    """List all model pairs"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    try:
        pairs = await model_service.list_model_pairs()
        return {"pairs": pairs}
    except Exception as e:
        logger.error(f"List model pairs failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task for updating metrics
async def update_metrics():
    """Background task to update system metrics"""
    while True:
        try:
            if model_service:
                # Update model count
                loaded_models_count = len(model_service.get_loaded_models())
                models_loaded.set(loaded_models_count)
                
                # Update GPU memory usage
                gpu_info = get_gpu_memory_info()
                if 'used_bytes' in gpu_info:
                    gpu_memory_usage_bytes.set(gpu_info['used_bytes'])
                
                # Update cache metrics
                cache_stats = model_service.get_cache_stats()
                cache_hits.inc(cache_stats.get('hits_since_last_update', 0))
                cache_misses.inc(cache_stats.get('misses_since_last_update', 0))
            
            await asyncio.sleep(10)  # Update every 10 seconds
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
            await asyncio.sleep(30)  # Wait longer on error


if __name__ == "__main__":
    # Run the service
    port = int(os.getenv('SERVICE_PORT', '8005'))
    host = os.getenv('SERVICE_HOST', '0.0.0.0')
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )