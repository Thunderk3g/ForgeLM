"""
GPU Inference Wrapper

Runs inference for the chat comparison and model management services under a
GPU lock held through the orchestrator.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Callable

# Each service used to keep its own copy of this module, and __name__ there gave
# the logger name below. It is fixed now that the one copy lives in llmforge_common,
# so log lines, and anything that searches or filters them, read as they did.
_LOGGER_NAME = "common.gpu_inference_wrapper"

from .gpu_client import GPUClient, GPUOperationType, GPULockPriority, get_service_id


class GPUInferenceManager:
    """Manages GPU inference operations with orchestration"""
    
    def __init__(self, service_id: str = None):
        self.service_id = service_id or get_service_id()
        self.gpu_client = GPUClient(self.service_id)
        self.logger = logging.getLogger(f"{_LOGGER_NAME}.{self.service_id}")
    
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
