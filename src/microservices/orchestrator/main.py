"""
LLMForge Service Orchestrator Main

FastAPI application that runs the service orchestrator
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from service_orchestrator import get_orchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management"""
    global orchestrator_instance
    
    # Startup
    logger.info("Starting LLMForge Service Orchestrator...")
    orchestrator_instance = await get_orchestrator()
    logger.info("Service Orchestrator started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Service Orchestrator...")
    if orchestrator_instance:
        # Stop via service orchestrator stop if available
        try:
            await orchestrator_instance.stop()
        except AttributeError:
            # Fallback: stop registry directly
            try:
                await orchestrator_instance.registry.stop()
            except Exception:
                pass
    logger.info("Service Orchestrator stopped")

# Create FastAPI app
app = FastAPI(
    title="LLMForge Service Orchestrator",
    description="Python-based agent orchestrator for LLMForge microservices",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "llmforge-orchestrator",
        "timestamp": "2025-01-11T12:00:00Z",
        "version": "1.0.0"
    }

@app.post("/register")
async def register_service(service_data: dict):
    """Register a new service with the orchestrator"""
    try:
        orchestrator = await get_orchestrator()
        # Accept the JSON payload as sent by clients and adapt to internal registry
        # Expected keys from clients: service_name, service_id, host, port, health_endpoint, metadata
        try:
            instance_data = {
                "service_name": service_data.get("service_name") or service_data.get("name"),
                "service_id": service_data.get("service_id") or service_data.get("id") or service_data.get("serviceId"),
                "host": service_data.get("host"),
                "port": int(service_data.get("port")),
                "health_endpoint": service_data.get("health_endpoint", "/health"),
                "metadata": service_data.get("metadata", {}),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid registration payload: {e}")

        if not instance_data["service_name"] or not instance_data["service_id"]:
            raise HTTPException(status_code=400, detail="Missing service_name or service_id")

        # Create and register the instance
        from service_orchestrator import ServiceInstance
        instance = ServiceInstance(
            id=instance_data["service_id"],
            name=instance_data["service_name"],
            host=instance_data["host"],
            port=instance_data["port"],
            health_endpoint=instance_data["health_endpoint"],
            metadata=instance_data["metadata"],
        )

        orchestrator.registry.register_service(instance)
        await orchestrator._broadcast_service_update()
        result = {"status": "registered", "service": instance.name, "id": instance.id}
        return result
    except Exception as e:
        logger.error(f"Service registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/deregister/{service_id}")
async def deregister_service(service_id: str):
    """Deregister a service from the orchestrator"""
    try:
        orchestrator = await get_orchestrator()
        result = await orchestrator.deregister_service(service_id)
        return result
    except Exception as e:
        logger.error(f"Service deregistration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/services")
async def list_services():
    """List all registered services"""
    try:
        orchestrator = await get_orchestrator()
        services = {}
        for service_name, instances in orchestrator.registry.services.items():
            services[service_name] = [
                {
                    "id": instance.id,
                    "url": instance.url,
                    "state": instance.state.value,
                    "response_time": instance.response_time_ms,
                    "active_connections": instance.active_connections,
                    "success_rate": instance.success_rate
                }
                for instance in instances
            ]
        return {"services": services}
    except Exception as e:
        logger.error(f"Failed to list services: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/discover/{service_name}")
async def discover_service(service_name: str):
    """Discover a healthy instance and return its URL and metadata"""
    try:
        orchestrator = await get_orchestrator()
        instances = orchestrator.registry.get_healthy_instances(service_name)
        selected = orchestrator.load_balancer.select_instance(service_name, instances)
        if selected:
            selected.active_connections += 1
            return {
                "service": service_name,
                "instance": {
                    "id": selected.id,
                    "url": selected.url,
                    "metadata": selected.metadata
                }
            }
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found or unavailable")
    except Exception as e:
        logger.error(f"Service discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/release/{service_id}")
async def release_connection(service_id: str):
    """Release a connection from a service"""
    try:
        orchestrator = await get_orchestrator()
        await orchestrator.release_connection(service_id)
        return {"status": "connection_released", "service_id": service_id}
    except Exception as e:
        logger.error(f"Connection release failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time service updates"""
    try:
        orchestrator = await get_orchestrator()
        await orchestrator.add_websocket_connection(websocket)
        
        # Keep connection alive
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if orchestrator_instance:
            await orchestrator_instance.remove_websocket_connection(websocket)

@app.get("/metrics")
async def get_metrics():
    """Get service metrics and performance data including GPU orchestration"""
    try:
        orchestrator = await get_orchestrator()
        metrics = {}
        
        # Service metrics
        for service_name, instances in orchestrator.registry.services.items():
            service_metrics = {
                "instance_count": len(instances),
                "healthy_instances": len([i for i in instances if i.state.value == "healthy"]),
                "avg_response_time": sum(i.response_time_ms for i in instances) / len(instances) if instances else 0,
                "total_requests": sum(i.success_count + i.error_count for i in instances),
                "success_rate": sum(i.success_rate for i in instances) / len(instances) if instances else 0
            }
            circuit_breaker = orchestrator.registry.circuit_breakers.get(service_name)
            if circuit_breaker:
                service_metrics["circuit_breaker_state"] = circuit_breaker.state
            metrics[service_name] = service_metrics
        
        # Add GPU lock status if Redis available
        if orchestrator.redis_client:
            try:
                operations = ["training", "inference", "model_loading", "general"]
                gpu_locks = {}
                
                for operation in operations:
                    key = f"gpu_lock:{operation}"
                    lock_value = await orchestrator.redis_client.get(key)
                    ttl = await orchestrator.redis_client.ttl(key)
                    
                    if lock_value:
                        import json as json_lib
                        lock_data = json_lib.loads(lock_value)
                        gpu_locks[operation] = {
                            "locked": True,
                            "holder": lock_data.get("service_id"),
                            "ttl_seconds": ttl,
                            "acquired_at": lock_data.get("acquired_at"),
                            "expires_at": lock_data.get("expires_at")
                        }
                    else:
                        gpu_locks[operation] = {
                            "locked": False,
                            "available": True
                        }
                
                metrics["gpu_lock"] = gpu_locks
                
            except Exception as e:
                logger.error(f"Failed to get GPU lock metrics: {e}")
                metrics["gpu_lock"] = {"error": str(e)}
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gpu/acquire")
async def acquire_gpu_lock(request: dict):
    """Acquire GPU lock with TTL and priority support"""
    try:
        orchestrator = await get_orchestrator()
        if not orchestrator.redis_client:
            raise HTTPException(status_code=503, detail="Redis not available for locking")
        
        operation = request.get("operation", "general")
        estimated_ttl = request.get("ttl", 3600)  # default 1 hour
        priority = request.get("priority", 1)  # lower number = higher priority
        service_id = request.get("service_id")
        
        if not service_id:
            raise HTTPException(status_code=400, detail="service_id required")
        
        lock_key = f"gpu_lock:{operation}"
        from datetime import datetime
        lock_value = f"{service_id}:{datetime.now().isoformat()}"
        
        # Create lock data
        import json as json_lib
        lock_data = {
            "token": lock_value,
            "service_id": service_id,
            "operation": operation,
            "acquired_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=estimated_ttl)).isoformat(),
            "priority": priority,
            "metadata": request.get("metadata", {})
        }
        
        # Simple lock: SET if not exists, with EX ttl
        acquired = await orchestrator.redis_client.set(
            lock_key,
            json_lib.dumps(lock_data),
            nx=True,  # only set if not exists
            ex=estimated_ttl
        )
        
        if acquired:
            logger.info(f"GPU lock acquired by {service_id} for {operation}, TTL: {estimated_ttl}s")
            return {
                "acquired": True,
                "token": lock_value,
                "key": lock_key,
                "ttl": estimated_ttl,
                "message": "GPU lock acquired successfully"
            }
        else:
            # Check TTL for wait suggestion
            ttl = await orchestrator.redis_client.ttl(lock_key)
            wait_time = max(30, ttl // 2) if ttl > 0 else 30
            logger.info(f"GPU lock busy for {operation}, suggesting wait {wait_time}s")
            return {
                "acquired": False,
                "key": lock_key,
                "ttl_remaining": ttl,
                "retry_after": wait_time,
                "message": "GPU lock busy, retry after suggested time"
            }
    except Exception as e:
        logger.error(f"GPU lock acquisition failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gpu/release")
async def release_gpu_lock(request: dict):
    """Release GPU lock using token"""
    try:
        orchestrator = await get_orchestrator()
        if not orchestrator.redis_client:
            raise HTTPException(status_code=503, detail="Redis not available for locking")
        
        token = request.get("token")
        key = request.get("key", "gpu_lock:general")
        
        if not token:
            raise HTTPException(status_code=400, detail="token required")
        
        # Verify and delete if matches
        current_value = await orchestrator.redis_client.get(key)
        if current_value:
            import json as json_lib
            lock_data = json_lib.loads(current_value)
            if lock_data.get("token") == token:
                deleted = await orchestrator.redis_client.delete(key)
                if deleted:
                    logger.info(f"GPU lock released for key {key}")
                    return {"released": True, "message": "GPU lock released successfully"}
                else:
                    return {"released": False, "message": "Lock already expired"}
            else:
                raise HTTPException(status_code=400, detail="Invalid token or lock not held")
        else:
            raise HTTPException(status_code=400, detail="Lock not found or already expired")
    except Exception as e:
        logger.error(f"GPU lock release failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("SERVICE_PORT", 8080))
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )