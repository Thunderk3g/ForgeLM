"""
LLMForge Service Orchestrator Main

FastAPI application that runs the service orchestrator
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager
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
        return orchestrator.get_service_overview()
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
    """Get service metrics and performance data"""
    try:
        orchestrator = await get_orchestrator()
        return orchestrator.get_service_overview()
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
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