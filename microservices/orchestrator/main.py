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

from service_orchestrator import ServiceOrchestrator, get_orchestrator

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
        await orchestrator_instance.stop()
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
        result = await orchestrator.register_service(service_data)
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

@app.get("/route/{service_name}")
async def route_to_service(service_name: str, capabilities: str = None):
    """Route a request to the best available service instance"""
    try:
        orchestrator = await get_orchestrator()
        capabilities_list = capabilities.split(",") if capabilities else None
        
        service_instance = await orchestrator.route_request(service_name, capabilities_list)
        
        if service_instance:
            return {
                "service_id": service_instance.service_id,
                "service_name": service_instance.service_name,
                "url": service_instance.url,
                "state": service_instance.state.value,
                "response_time_ms": service_instance.response_time_ms
            }
        else:
            raise HTTPException(
                status_code=503, 
                detail=f"No healthy instances available for service '{service_name}'"
            )
    except Exception as e:
        logger.error(f"Service routing failed: {e}")
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