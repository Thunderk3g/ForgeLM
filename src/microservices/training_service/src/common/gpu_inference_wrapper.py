"""
GPU Inference Wrapper

Provides GPU-orchestrated inference capabilities for chat comparison and 
model management services. Handles model loading, caching, and inference
with automatic GPU lock management.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, Union
from contextlib import asynccontextmanager
import torch

from .gpu_client import GPUClient, GPUOperationType, GPULockPriority, get_service_id


class GPUInferenceManager:
    """Manages GPU inference operations with orchestration"""
    
    def __init__(self, service_id: str = None):
        self.service_id = service_id or get_service_id()
        self.gpu_client = GPUClient(self.service_id)
        self.logger = logging.getLogger(f"{__name__}.{self.service_id}")
        
        # Model cache for loaded models
        self._loaded_models: Dict[str, Any] = {}
        self._model_load_times: Dict[str, float] = {}
        
    async def load_model_with_gpu_lock(
        self,
        model_name: str,
        model_loader: Callable,
        cache_key: str = None,
        estimated_load_time: int = 300,
        force_reload: bool = False
    ) -> Any:
        """
        Load model with GPU orchestration
        
        Args:
            model_name: Name/path of the model
            model_loader: Function to load the model
            cache_key: Cache key for the model (defaults to model_name)
            estimated_load_time: Estimated loading time in seconds
            force_reload: Force reload even if cached
            
        Returns:
            Loaded model
        """
        cache_key = cache_key or model_name
        
        # Check cache first
        if not force_reload and cache_key in self._loaded_models:
            self.logger.info(f"Using cached model: {cache_key}")
            return self._loaded_models[cache_key]
        
        self.logger.info(f"Loading model {model_name} with GPU lock")
        
        async with self.gpu_client.gpu_lock_context(
            GPUOperationType.MODEL_LOADING,
            estimated_duration=estimated_load_time,
            priority=GPULockPriority.HIGH,
            metadata={"model_name": model_name, "cache_key": cache_key}
        ) as lock:
            if not lock:
                raise RuntimeError(f"Could not acquire GPU lock for model loading: {model_name}")
            
            start_time = time.time()
            try:
                # Execute model loader
                if asyncio.iscoroutinefunction(model_loader):
                    model = await model_loader()
                else:
                    model = model_loader()
                
                # Cache the model
                self._loaded_models[cache_key] = model
                self._model_load_times[cache_key] = time.time() - start_time
                
                self.logger.info(
                    f"Model {model_name} loaded successfully in "
                    f"{self._model_load_times[cache_key]:.2f}s"
                )
                return model
                
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
                # Clear any partial cache entry
                self._loaded_models.pop(cache_key, None)
                self._model_load_times.pop(cache_key, None)
                raise
    
    async def run_inference_with_gpu_lock(
        self,
        inference_func: Callable,
        estimated_time: int = 60,
        priority: GPULockPriority = GPULockPriority.NORMAL,
        metadata: Dict[str, Any] = None
    ) -> Any:
        """
        Run inference with GPU orchestration
        
        Args:
            inference_func: Function to run inference
            estimated_time: Estimated inference time in seconds
            priority: GPU lock priority
            metadata: Additional metadata for the operation
            
        Returns:
            Inference result
        """
        self.logger.debug("Running inference with GPU lock")
        
        async with self.gpu_client.gpu_lock_context(
            GPUOperationType.INFERENCE,
            estimated_duration=estimated_time,
            priority=priority,
            metadata=metadata or {}
        ) as lock:
            if not lock:
                raise RuntimeError("Could not acquire GPU lock for inference")
            
            start_time = time.time()
            try:
                # Execute inference
                if asyncio.iscoroutinefunction(inference_func):
                    result = await inference_func()
                else:
                    result = inference_func()
                
                inference_time = time.time() - start_time
                self.logger.debug(f"Inference completed in {inference_time:.2f}s")
                return result
                
            except Exception as e:
                self.logger.error(f"Inference failed: {e}")
                raise
    
    async def batch_inference_with_gpu_lock(
        self,
        inference_batches: list,
        batch_processor: Callable,
        estimated_time_per_batch: int = 30,
        max_batch_time: int = 300
    ) -> list:
        """
        Run batch inference with a single GPU lock
        
        Args:
            inference_batches: List of inference batches
            batch_processor: Function to process each batch
            estimated_time_per_batch: Estimated time per batch
            max_batch_time: Maximum time for entire batch operation
            
        Returns:
            List of batch results
        """
        total_estimated_time = min(
            len(inference_batches) * estimated_time_per_batch,
            max_batch_time
        )
        
        self.logger.info(f"Running batch inference with {len(inference_batches)} batches")
        
        async with self.gpu_client.gpu_lock_context(
            GPUOperationType.INFERENCE,
            estimated_duration=total_estimated_time,
            priority=GPULockPriority.NORMAL,
            metadata={"batch_count": len(inference_batches)}
        ) as lock:
            if not lock:
                raise RuntimeError("Could not acquire GPU lock for batch inference")
            
            results = []
            start_time = time.time()
            
            try:
                for i, batch in enumerate(inference_batches):
                    batch_start = time.time()
                    
                    if asyncio.iscoroutinefunction(batch_processor):
                        result = await batch_processor(batch)
                    else:
                        result = batch_processor(batch)
                    
                    results.append(result)
                    batch_time = time.time() - batch_start
                    
                    self.logger.debug(
                        f"Batch {i+1}/{len(inference_batches)} completed in {batch_time:.2f}s"
                    )
                
                total_time = time.time() - start_time
                self.logger.info(f"Batch inference completed in {total_time:.2f}s")
                return results
                
            except Exception as e:
                self.logger.error(f"Batch inference failed: {e}")
                raise
    
    def get_cached_model(self, cache_key: str) -> Optional[Any]:
        """Get cached model without GPU operations"""
        return self._loaded_models.get(cache_key)
    
    def clear_model_cache(self, cache_key: str = None):
        """Clear model cache (optionally for specific key)"""
        if cache_key:
            model = self._loaded_models.pop(cache_key, None)
            self._model_load_times.pop(cache_key, None)
            if model:
                self._cleanup_model(model)
                self.logger.info(f"Cleared cached model: {cache_key}")
        else:
            for cache_key, model in self._loaded_models.items():
                self._cleanup_model(model)
            self._loaded_models.clear()
            self._model_load_times.clear()
            self.logger.info("Cleared all cached models")
    
    def _cleanup_model(self, model):
        """Cleanup model resources"""
        try:
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.warning(f"Error during model cleanup: {e}")
    
    async def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status including model cache info"""
        status = await self.gpu_client.get_gpu_status()
        
        status["model_cache"] = {
            "cached_models": list(self._loaded_models.keys()),
            "cache_count": len(self._loaded_models),
            "load_times": self._model_load_times.copy()
        }
        
        return status
    
    async def cleanup(self):
        """Cleanup on service shutdown"""
        self.clear_model_cache()
        await self.gpu_client.cleanup_all_locks()
        self.logger.info("GPU inference manager cleanup completed")


@asynccontextmanager
async def gpu_inference_context(
    service_id: str = None,
    estimated_time: int = 60,
    priority: GPULockPriority = GPULockPriority.NORMAL
):
    """Context manager for simple GPU inference operations"""
    manager = GPUInferenceManager(service_id)
    
    async with manager.gpu_client.gpu_lock_context(
        GPUOperationType.INFERENCE,
        estimated_duration=estimated_time,
        priority=priority
    ) as lock:
        if not lock:
            raise RuntimeError("Could not acquire GPU lock for inference")
        yield manager


# Decorator for automatic GPU orchestration
def gpu_orchestrated(
    operation_type: GPUOperationType = GPUOperationType.INFERENCE,
    estimated_duration: int = 60,
    priority: GPULockPriority = GPULockPriority.NORMAL
):
    """Decorator to automatically wrap functions with GPU orchestration"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            service_id = kwargs.pop('_gpu_service_id', None) or get_service_id()
            gpu_client = GPUClient(service_id)
            
            async with gpu_client.gpu_lock_context(
                operation_type,
                estimated_duration=estimated_duration,
                priority=priority
            ) as lock:
                if not lock:
                    raise RuntimeError(f"Could not acquire GPU lock for {operation_type.value}")
                
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator