"""
GPU Client Library

Provides a high-level interface for services to acquire and manage GPU locks
through the GPU Orchestrator. Includes automatic retry logic, health monitoring,
and context manager support for clean resource management.
"""

import asyncio
import time
import os
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
import aiohttp

from .gpu_orchestrator import (
    GPULockRequest, 
    GPULock, 
    GPUOperationType, 
    GPULockPriority
)


class GPUClient:
    """High-level GPU client for services to interact with GPU orchestrator"""
    
    def __init__(
        self,
        service_id: str,
        orchestrator_url: str = None,
        max_retries: int = 5,
        retry_delay: float = 30.0,
        timeout: int = 300
    ):
        self.service_id = service_id
        self.orchestrator_url = orchestrator_url or os.getenv(
            'ORCHESTRATOR_URL', 
            'http://orchestrator:8080'
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.logger = logging.getLogger(f"{__name__}.{service_id}")
        
        # Active locks held by this client
        self.active_locks: Dict[str, GPULock] = {}
        
    async def acquire_gpu_lock(
        self,
        operation: GPUOperationType,
        estimated_duration: int = 3600,
        priority: GPULockPriority = GPULockPriority.NORMAL,
        metadata: Dict[str, Any] = None,
        wait_for_availability: bool = True
    ) -> Optional[GPULock]:
        """
        Acquire GPU lock with automatic retry and queuing
        
        Args:
            operation: Type of GPU operation
            estimated_duration: Estimated duration in seconds
            priority: Lock priority level
            metadata: Additional metadata
            wait_for_availability: Whether to wait and retry if not immediately available
            
        Returns:
            GPULock if successful, None if failed
        """
        self.logger.info(f"Requesting GPU lock for {operation.value} (estimated {estimated_duration}s)")
        
        request_data = {
            "operation": operation.value,
            "ttl": estimated_duration,
            "priority": priority.value,
            "service_id": self.service_id,
            "metadata": metadata or {}
        }
        
        retries = 0
        start_time = time.time()
        
        while retries < self.max_retries:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                    async with session.post(
                        f"{self.orchestrator_url}/gpu/acquire",
                        json=request_data
                    ) as response:
                        result = await response.json()
                        
                        if response.status == 200 and result.get("acquired"):
                            # Lock acquired successfully
                            lock = GPULock(
                                token=result["token"],
                                key=result["key"],
                                holder_service=self.service_id,
                                operation=operation,
                                acquired_at=time.time(),
                                expires_at=time.time() + result["ttl"],
                                priority=priority,
                                metadata=metadata or {}
                            )
                            
                            self.active_locks[lock.key] = lock
                            elapsed = time.time() - start_time
                            self.logger.info(
                                f"GPU lock acquired for {operation.value} after {elapsed:.1f}s "
                                f"(attempts: {retries + 1})"
                            )
                            return lock
                        
                        elif response.status == 200 and not result.get("acquired"):
                            # Lock busy, check if we should wait
                            if not wait_for_availability:
                                self.logger.info(f"GPU lock busy for {operation.value}, not waiting")
                                return None
                            
                            retry_after = result.get("retry_after", self.retry_delay)
                            elapsed = time.time() - start_time
                            
                            if elapsed > self.timeout:
                                self.logger.error(
                                    f"Timeout waiting for GPU lock ({self.timeout}s), giving up"
                                )
                                return None
                            
                            self.logger.info(
                                f"GPU lock busy for {operation.value}, retrying in {retry_after}s "
                                f"(attempt {retries + 1}/{self.max_retries})"
                            )
                            await asyncio.sleep(retry_after)
                        
                        else:
                            self.logger.error(f"Failed to acquire GPU lock: {result}")
                            return None
                            
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout acquiring GPU lock (attempt {retries + 1})")
            except Exception as e:
                self.logger.error(f"Error acquiring GPU lock: {e}")
            
            retries += 1
            if retries < self.max_retries and wait_for_availability:
                await asyncio.sleep(min(self.retry_delay, 60))  # Cap at 1 minute
        
        self.logger.error(f"Failed to acquire GPU lock after {retries} attempts")
        return None
    
    async def release_gpu_lock(self, lock: GPULock) -> bool:
        """
        Release GPU lock
        
        Args:
            lock: GPU lock to release
            
        Returns:
            True if successfully released
        """
        if not lock or lock.key not in self.active_locks:
            self.logger.warning("Attempting to release unknown or already released lock")
            return False
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(
                    f"{self.orchestrator_url}/gpu/release",
                    json={
                        "token": lock.token,
                        "key": lock.key
                    }
                ) as response:
                    result = await response.json()
                    
                    if response.status == 200 and result.get("released"):
                        self.active_locks.pop(lock.key, None)
                        self.logger.info(f"GPU lock released successfully for {lock.operation.value}")
                        return True
                    else:
                        self.logger.error(f"Failed to release GPU lock: {result}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Error releasing GPU lock: {e}")
            return False
    
    async def extend_gpu_lock(self, lock: GPULock, additional_seconds: int) -> bool:
        """
        Extend GPU lock duration
        
        Args:
            lock: GPU lock to extend
            additional_seconds: Additional duration in seconds
            
        Returns:
            True if successfully extended
        """
        if not lock or lock.key not in self.active_locks:
            self.logger.warning("Attempting to extend unknown lock")
            return False
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(
                    f"{self.orchestrator_url}/gpu/extend",
                    json={
                        "token": lock.token,
                        "key": lock.key,
                        "additional_seconds": additional_seconds
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("extended"):
                            lock.expires_at += additional_seconds
                            self.logger.info(f"GPU lock extended by {additional_seconds}s")
                            return True
                    
                    self.logger.error(f"Failed to extend GPU lock: {await response.text()}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error extending GPU lock: {e}")
            return False
    
    async def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU orchestration status"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{self.orchestrator_url}/metrics") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"Failed to get status: {response.status}"}
        except Exception as e:
            self.logger.error(f"Error getting GPU status: {e}")
            return {"error": str(e)}
    
    async def heartbeat_lock(self, lock: GPULock) -> bool:
        """Send heartbeat to keep lock alive (extend by 60s)"""
        if not lock or lock.key not in self.active_locks:
            return False
        
        # Only extend if lock is expiring soon (within 5 minutes)
        time_until_expiry = lock.expires_at - time.time()
        if time_until_expiry > 300:  # 5 minutes
            return True
        
        return await self.extend_gpu_lock(lock, 300)  # Extend by 5 minutes
    
    async def cleanup_all_locks(self):
        """Release all active locks held by this client"""
        locks_to_release = list(self.active_locks.values())
        for lock in locks_to_release:
            try:
                await self.release_gpu_lock(lock)
            except Exception as e:
                self.logger.error(f"Error releasing lock during cleanup: {e}")
    
    @asynccontextmanager
    async def gpu_lock_context(
        self,
        operation: GPUOperationType,
        estimated_duration: int = 3600,
        priority: GPULockPriority = GPULockPriority.NORMAL,
        metadata: Dict[str, Any] = None,
        wait_for_availability: bool = True,
        heartbeat_enabled: bool = True
    ) -> AsyncGenerator[Optional[GPULock], None]:
        """
        Context manager for GPU lock with automatic cleanup and heartbeat
        
        Usage:
            async with gpu_client.gpu_lock_context(GPUOperationType.TRAINING) as lock:
                if lock:
                    # Perform GPU operations
                    pass
        """
        lock = None
        heartbeat_task = None
        
        try:
            # Acquire lock
            lock = await self.acquire_gpu_lock(
                operation=operation,
                estimated_duration=estimated_duration,
                priority=priority,
                metadata=metadata,
                wait_for_availability=wait_for_availability
            )
            
            if lock and heartbeat_enabled:
                # Start heartbeat task
                heartbeat_task = asyncio.create_task(self._heartbeat_loop(lock))
            
            yield lock
            
        finally:
            # Cleanup
            if heartbeat_task:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            if lock:
                await self.release_gpu_lock(lock)
    
    async def _heartbeat_loop(self, lock: GPULock):
        """Background heartbeat loop to keep lock alive"""
        try:
            while True:
                await asyncio.sleep(240)  # Heartbeat every 4 minutes
                success = await self.heartbeat_lock(lock)
                if not success:
                    self.logger.warning(f"Heartbeat failed for lock {lock.key}")
                    break
        except asyncio.CancelledError:
            pass


class GPULockManager:
    """Simplified GPU lock manager for common use cases"""
    
    def __init__(self, service_id: str):
        self.client = GPUClient(service_id)
        self.logger = logging.getLogger(f"{__name__}.{service_id}")
    
    async def with_gpu_for_training(
        self,
        func,
        estimated_duration: int = 3600,
        priority: GPULockPriority = GPULockPriority.NORMAL
    ):
        """Execute function with GPU lock for training operations"""
        async with self.client.gpu_lock_context(
            GPUOperationType.TRAINING,
            estimated_duration=estimated_duration,
            priority=priority
        ) as lock:
            if lock:
                self.logger.info("Starting GPU training operation")
                try:
                    result = await func() if asyncio.iscoroutinefunction(func) else func()
                    self.logger.info("GPU training operation completed")
                    return result
                except Exception as e:
                    self.logger.error(f"GPU training operation failed: {e}")
                    raise
            else:
                raise RuntimeError("Could not acquire GPU lock for training")
    
    async def with_gpu_for_inference(
        self,
        func,
        estimated_duration: int = 300,
        priority: GPULockPriority = GPULockPriority.HIGH
    ):
        """Execute function with GPU lock for inference operations"""
        async with self.client.gpu_lock_context(
            GPUOperationType.INFERENCE,
            estimated_duration=estimated_duration,
            priority=priority
        ) as lock:
            if lock:
                self.logger.info("Starting GPU inference operation")
                try:
                    result = await func() if asyncio.iscoroutinefunction(func) else func()
                    self.logger.info("GPU inference operation completed")
                    return result
                except Exception as e:
                    self.logger.error(f"GPU inference operation failed: {e}")
                    raise
            else:
                raise RuntimeError("Could not acquire GPU lock for inference")


# Convenience functions for quick integration
async def create_gpu_client(service_id: str) -> GPUClient:
    """Create and initialize GPU client"""
    return GPUClient(service_id)


def get_service_id() -> str:
    """Get service ID from environment or generate one"""
    service_id = os.getenv('SERVICE_ADVERTISED_HOST')
    if not service_id:
        service_id = os.getenv('HOSTNAME', f'service_{int(time.time())}')
    return service_id