"""
GPU Orchestrator Service

Provides centralized GPU resource management with distributed locking,
priority-based scheduling, and comprehensive monitoring for LLMForge services.
"""

import asyncio
import time
import json
import logging
import uuid
import os
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class GPUOperationType(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    MODEL_LOADING = "model_loading"
    GENERAL = "general"


class GPULockPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class GPULockRequest:
    service_id: str
    operation: GPUOperationType
    estimated_duration: int  # seconds
    priority: GPULockPriority = GPULockPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if isinstance(self.operation, str):
            self.operation = GPUOperationType(self.operation)
        if isinstance(self.priority, int):
            self.priority = GPULockPriority(self.priority)


@dataclass
class GPULock:
    token: str
    key: str
    holder_service: str
    operation: GPUOperationType
    acquired_at: datetime
    expires_at: datetime
    priority: GPULockPriority
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at
    
    @property
    def ttl_seconds(self) -> int:
        return max(0, int((self.expires_at - datetime.now()).total_seconds()))


class GPUOrchestrator:
    """Enhanced GPU orchestrator with distributed locking and priority scheduling"""
    
    def __init__(self, redis_host: str = "redis", redis_port: int = 6379):
        self.logger = logging.getLogger(__name__)
        
        # Redis setup
        if REDIS_AVAILABLE:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            self.logger.info("GPU Orchestrator initialized with Redis backend")
        else:
            self.redis_client = None
            self.logger.warning("Redis not available - GPU orchestration disabled")
        
        # Lock management
        self.active_locks: Dict[str, GPULock] = {}
        self.pending_requests: List[GPULockRequest] = []
        
        # Configuration
        self.max_queue_size = 100
        self.default_ttl = 3600  # 1 hour
        self.heartbeat_interval = 30  # seconds
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._queue_processor_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the GPU orchestrator background tasks"""
        if not self.redis_client:
            return
            
        try:
            # Test Redis connection
            await self.redis_client.ping()
            self.logger.info("Redis connection established")
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_locks())
            self._queue_processor_task = asyncio.create_task(self._process_pending_requests())
            
            self.logger.info("GPU Orchestrator started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start GPU Orchestrator: {e}")
            raise
    
    async def stop(self):
        """Stop the GPU orchestrator and cleanup"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._queue_processor_task:
            self._queue_processor_task.cancel()
        
        if self.redis_client:
            await self.redis_client.close()
        
        self.logger.info("GPU Orchestrator stopped")
    
    async def acquire_gpu_lock(self, request: GPULockRequest) -> Optional[GPULock]:
        """
        Acquire GPU lock with priority-based scheduling
        
        Args:
            request: GPU lock request with service details
            
        Returns:
            GPULock if successful, None if should queue/retry
        """
        if not self.redis_client:
            raise RuntimeError("Redis not available for GPU locking")
        
        operation_key = f"gpu_lock:{request.operation.value}"
        
        # Check if operation is already locked
        current_lock = await self._get_current_lock(operation_key)
        if current_lock and not current_lock.is_expired:
            # Add to pending queue if high priority or queue not full
            if request.priority.value >= GPULockPriority.HIGH.value or len(self.pending_requests) < self.max_queue_size:
                self.pending_requests.append(request)
                self.pending_requests.sort(key=lambda x: (-x.priority.value, x.created_at))
                self.logger.info(f"GPU request queued for {request.service_id}, position: {len(self.pending_requests)}")
            
            return None
        
        # Attempt to acquire lock
        lock = await self._create_lock(request, operation_key)
        if lock:
            await self._store_lock_metadata(lock)
            self.active_locks[lock.key] = lock
            self.logger.info(f"GPU lock acquired by {request.service_id} for {request.operation.value}")
            return lock
        
        return None
    
    async def release_gpu_lock(self, token: str, key: str) -> bool:
        """
        Release GPU lock using token verification
        
        Args:
            token: Lock token for verification
            key: Lock key to release
            
        Returns:
            True if successfully released, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            # Verify token matches current lock
            current_value = await self.redis_client.get(key)
            if not current_value:
                self.logger.warning(f"Lock {key} not found for release")
                return False
            
            lock_data = json.loads(current_value)
            if lock_data.get("token") != token:
                self.logger.error(f"Invalid token for lock {key}")
                return False
            
            # Delete the lock
            deleted = await self.redis_client.delete(key)
            if deleted:
                # Remove from active locks
                self.active_locks.pop(key, None)
                
                # Clean up metadata
                await self.redis_client.delete(f"{key}_metadata")
                
                self.logger.info(f"GPU lock {key} released successfully")
                
                # Trigger queue processing
                asyncio.create_task(self._process_next_in_queue(key.split(":")[-1]))
                
                return True
            else:
                self.logger.warning(f"Lock {key} was already expired")
                return False
                
        except Exception as e:
            self.logger.error(f"Error releasing GPU lock {key}: {e}")
            return False
    
    async def extend_gpu_lock(self, token: str, key: str, additional_seconds: int) -> bool:
        """Extend an existing GPU lock"""
        if not self.redis_client:
            return False
        
        try:
            current_value = await self.redis_client.get(key)
            if not current_value:
                return False
            
            lock_data = json.loads(current_value)
            if lock_data.get("token") != token:
                return False
            
            # Extend TTL
            extended = await self.redis_client.expire(key, additional_seconds)
            if extended:
                # Update metadata
                lock_data["expires_at"] = (datetime.now() + timedelta(seconds=additional_seconds)).isoformat()
                await self.redis_client.set(key, json.dumps(lock_data), ex=additional_seconds)
                
                self.logger.info(f"GPU lock {key} extended by {additional_seconds}s")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error extending GPU lock {key}: {e}")
            return False
    
    async def get_gpu_status(self) -> Dict[str, Any]:
        """Get comprehensive GPU orchestration status"""
        if not self.redis_client:
            return {"error": "Redis not available"}
        
        try:
            status = {
                "gpu_locks": {},
                "pending_requests": len(self.pending_requests),
                "queue": [],
                "redis_connected": True
            }
            
            # Get all GPU locks
            for operation in GPUOperationType:
                key = f"gpu_lock:{operation.value}"
                lock_data = await self.redis_client.get(key)
                ttl = await self.redis_client.ttl(key)
                
                if lock_data:
                    lock_info = json.loads(lock_data)
                    status["gpu_locks"][operation.value] = {
                        "locked": True,
                        "holder": lock_info.get("service_id"),
                        "operation": operation.value,
                        "ttl_seconds": ttl,
                        "acquired_at": lock_info.get("acquired_at"),
                        "expires_at": lock_info.get("expires_at")
                    }
                else:
                    status["gpu_locks"][operation.value] = {
                        "locked": False,
                        "available": True
                    }
            
            # Get pending queue info
            for req in self.pending_requests[:10]:  # Show first 10
                status["queue"].append({
                    "service_id": req.service_id,
                    "operation": req.operation.value,
                    "priority": req.priority.value,
                    "waiting_since": req.created_at.isoformat()
                })
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting GPU status: {e}")
            return {"error": str(e)}
    
    async def _create_lock(self, request: GPULockRequest, key: str) -> Optional[GPULock]:
        """Create a new GPU lock"""
        token = f"{request.service_id}_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        ttl = min(request.estimated_duration, self.default_ttl)
        
        lock_data = {
            "token": token,
            "service_id": request.service_id,
            "operation": request.operation.value,
            "acquired_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=ttl)).isoformat(),
            "priority": request.priority.value,
            "metadata": request.metadata
        }
        
        # Atomic SET if not exists with TTL
        success = await self.redis_client.set(
            key,
            json.dumps(lock_data),
            nx=True,  # Only set if not exists
            ex=ttl
        )
        
        if success:
            return GPULock(
                token=token,
                key=key,
                holder_service=request.service_id,
                operation=request.operation,
                acquired_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=ttl),
                priority=request.priority,
                metadata=request.metadata
            )
        
        return None
    
    async def _get_current_lock(self, key: str) -> Optional[GPULock]:
        """Get current lock information"""
        try:
            lock_data = await self.redis_client.get(key)
            if not lock_data:
                return None
            
            data = json.loads(lock_data)
            return GPULock(
                token=data["token"],
                key=key,
                holder_service=data["service_id"],
                operation=GPUOperationType(data["operation"]),
                acquired_at=datetime.fromisoformat(data["acquired_at"]),
                expires_at=datetime.fromisoformat(data["expires_at"]),
                priority=GPULockPriority(data["priority"]),
                metadata=data.get("metadata", {})
            )
        except Exception as e:
            self.logger.error(f"Error getting current lock for {key}: {e}")
            return None
    
    async def _store_lock_metadata(self, lock: GPULock):
        """Store additional lock metadata"""
        try:
            metadata_key = f"{lock.key}_metadata"
            metadata = {
                "service_id": lock.holder_service,
                "operation": lock.operation.value,
                "acquired_at": lock.acquired_at.isoformat(),
                "estimated_duration": lock.ttl_seconds,
                "metadata": lock.metadata
            }
            
            await self.redis_client.set(
                metadata_key,
                json.dumps(metadata),
                ex=lock.ttl_seconds + 300  # Keep metadata slightly longer
            )
        except Exception as e:
            self.logger.error(f"Error storing lock metadata: {e}")
    
    async def _cleanup_expired_locks(self):
        """Background task to cleanup expired locks"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Clean up expired locks from memory
                expired_keys = [key for key, lock in self.active_locks.items() if lock.is_expired]
                for key in expired_keys:
                    del self.active_locks[key]
                    self.logger.debug(f"Cleaned up expired lock: {key}")
                
                # Clean up stale pending requests (older than 1 hour)
                now = datetime.now()
                self.pending_requests = [
                    req for req in self.pending_requests
                    if now - req.created_at < timedelta(hours=1)
                ]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
    
    async def _process_pending_requests(self):
        """Background task to process pending GPU requests"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                if not self.pending_requests:
                    continue
                
                # Process highest priority requests first
                self.pending_requests.sort(key=lambda x: (-x.priority.value, x.created_at))
                
                processed = []
                for request in self.pending_requests[:5]:  # Process up to 5 at a time
                    operation_key = f"gpu_lock:{request.operation.value}"
                    current_lock = await self._get_current_lock(operation_key)
                    
                    if not current_lock or current_lock.is_expired:
                        # Try to acquire lock
                        lock = await self._create_lock(request, operation_key)
                        if lock:
                            await self._store_lock_metadata(lock)
                            self.active_locks[lock.key] = lock
                            processed.append(request)
                            self.logger.info(f"Processed queued request for {request.service_id}")
                
                # Remove processed requests
                for request in processed:
                    self.pending_requests.remove(request)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in queue processor: {e}")
    
    async def _process_next_in_queue(self, operation_type: str):
        """Process next request in queue for specific operation"""
        try:
            # Find next request for this operation type
            next_request = None
            for i, request in enumerate(self.pending_requests):
                if request.operation.value == operation_type:
                    next_request = self.pending_requests.pop(i)
                    break
            
            if next_request:
                operation_key = f"gpu_lock:{operation_type}"
                lock = await self._create_lock(next_request, operation_key)
                if lock:
                    await self._store_lock_metadata(lock)
                    self.active_locks[lock.key] = lock
                    self.logger.info(f"Processed next queued request for {next_request.service_id}")
        except Exception as e:
            self.logger.error(f"Error processing next in queue: {e}")


# Global orchestrator instance
_gpu_orchestrator: Optional[GPUOrchestrator] = None

async def get_gpu_orchestrator() -> GPUOrchestrator:
    """Get or create global GPU orchestrator instance"""
    global _gpu_orchestrator
    if _gpu_orchestrator is None:
        _gpu_orchestrator = GPUOrchestrator()
        await _gpu_orchestrator.start()
    return _gpu_orchestrator