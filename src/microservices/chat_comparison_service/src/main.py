"""
LLMForge Chat Comparison Microservice

Independent service for dual model inference and comparison with
FastLibra KV cache optimization and real-time streaming.
"""

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import uvicorn

# Import core comparison components (local to this microservice)
from services.comparison_service import ComparisonService
from services.model_service import ModelService
from services.chat_service import ChatService
from models.schemas import (
    ComparisonRequest, ComparisonResponse, ChatMessage, 
    SessionCreateRequest, SessionInfo, DualModelConfig
)
from services.service_orchestrator import register_with_orchestrator
from utils.logging import get_logger
from utils.performance import performance_monitor
from config.settings import settings

# Prometheus metrics
comparison_requests_total = Counter('comparison_requests_total', 'Total comparison requests')
comparison_duration_seconds = Histogram('comparison_duration_seconds', 'Comparison duration in seconds')
active_sessions = Gauge('active_sessions', 'Number of active chat sessions')
model_cache_hits = Counter('model_cache_hits_total', 'Model cache hits')
model_cache_misses = Counter('model_cache_misses_total', 'Model cache misses')
websocket_connections = Gauge('websocket_connections', 'Number of active WebSocket connections')
comparison_errors_total = Counter('comparison_errors_total', 'Total comparison errors')

# Global service instances
comparison_service: Optional[ComparisonService] = None
model_service: Optional[ModelService] = None
chat_service: Optional[ChatService] = None
logger = get_logger(__name__)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        websocket_connections.inc()

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        websocket_connections.dec()

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.active_connections.remove(connection)
            websocket_connections.dec()

connection_manager = ConnectionManager()


async def register_service():
    """Register this service with the orchestrator"""
    orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8080')
    service_id = f"chat-comparison-service-{uuid.uuid4().hex[:8]}"
    
    try:
        await register_with_orchestrator(
            orchestrator_url=orchestrator_url,
            service_name="llmforge-chat-comparison",
            service_id=service_id,
            host=os.getenv('SERVICE_ADVERTISED_HOST', 'llmforge-chat-comparison'),
            port=int(os.getenv('SERVICE_PORT', '8002')),
            health_endpoint="/health",
            metadata={
                "type": "chat_comparison",
                "gpu_enabled": True,
                "supports_streaming": True,
                "max_concurrent_sessions": 10
            }
        )
        logger.info(f"Registered with orchestrator as {service_id}")
    except Exception as e:
        logger.warning(f"Failed to register with orchestrator: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the chat comparison service"""
    global comparison_service, model_service, chat_service
    
    # Startup
    logger.info("Starting Chat Comparison Microservice...")
    
    # Initialize services
    model_service = ModelService()
    chat_service = ChatService(model_service)
    comparison_service = ComparisonService(model_service, chat_service)
    
    # Register with orchestrator
    await register_service()
    
    # Start performance monitoring
    performance_monitor.start_monitoring()
    
    # Start metrics update task
    metrics_task = asyncio.create_task(update_metrics())
    
    logger.info("Chat Comparison Microservice started successfully")
    yield
    
    # Cancel metrics task on shutdown
    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass
    
    # Shutdown
    logger.info("Shutting down Chat Comparison Microservice...")
    if comparison_service:
        await comparison_service.cleanup()
    if model_service:
        await model_service.cleanup()
    performance_monitor.stop_monitoring()
    logger.info("Chat Comparison Microservice shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LLMForge Chat Comparison Service",
    description="Microservice for dual model inference and comparison",
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
        cache_stats = {}
        model_info = {}
        if comparison_service:
            cache_stats = comparison_service.get_cache_stats()
        if model_service:
            model_info = model_service.get_loaded_models_info()
        
        return {
            "status": "healthy",
            "service": "chat_comparison",
            "version": "1.0.0",
            "active_sessions": len(comparison_service.sessions) if comparison_service else 0,
            "loaded_models": len(model_info),
            "cache_stats": cache_stats,
            "websocket_connections": len(connection_manager.active_connections)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return Response(
            content=f'{{"status": "unhealthy", "error": "{str(e)}"}}',
            status_code=503,
            media_type="application/json"
        )


@app.post("/sessions", response_model=SessionInfo)
async def create_session(request: SessionCreateRequest):
    """Create a new chat comparison session"""
    if not comparison_service:
        comparison_errors_total.inc()
        raise HTTPException(status_code=503, detail="Comparison service not initialized")
    
    try:
        session = await comparison_service.create_session(request)
        
        active_sessions.inc()
        
        # Broadcast session creation
        await connection_manager.broadcast({
            "type": "session_created",
            "session": session.dict()
        })
        
        return session
        
    except Exception as e:
        comparison_errors_total.inc()
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/compare", response_model=ComparisonResponse)
async def compare_models(session_id: str, request: ComparisonRequest):
    """Compare responses from dual models"""
    comparison_requests_total.inc()
    
    if not comparison_service:
        comparison_errors_total.inc()
        raise HTTPException(status_code=503, detail="Comparison service not initialized")
    
    try:
        # Update request with session_id
        request.session_id = session_id
        
        response = await comparison_service.generate_comparison(request)
        
        # Broadcast comparison result
        await connection_manager.broadcast({
            "type": "comparison_result",
            "session_id": session_id,
            "response": response.dict()
        })
        
        return response
        
    except KeyError:
        comparison_errors_total.inc()
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as e:
        comparison_errors_total.inc()
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/stream")
async def stream_comparison(session_id: str, request: ComparisonRequest):
    """Stream comparison responses"""
    if not comparison_service:
        raise HTTPException(status_code=503, detail="Comparison service not initialized")
    
    try:
        request.session_id = session_id
        
        async def response_generator():
            async for chunk in comparison_service.generate_comparison_stream(request):
                yield f"data: {chunk}\n\n"
        
        return StreamingResponse(
            response_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as e:
        logger.error(f"Streaming comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions", response_model=List[SessionInfo])
async def list_sessions():
    """List all active sessions"""
    if not comparison_service:
        raise HTTPException(status_code=503, detail="Comparison service not initialized")
    
    try:
        sessions = await comparison_service.list_sessions()
        return sessions
    except Exception as e:
        logger.error(f"List sessions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get session details"""
    if not comparison_service:
        raise HTTPException(status_code=503, detail="Comparison service not initialized")
    
    try:
        session = await comparison_service.get_session(session_id)
        return session
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as e:
        logger.error(f"Get session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if not comparison_service:
        raise HTTPException(status_code=503, detail="Comparison service not initialized")
    
    try:
        success = await comparison_service.delete_session(session_id)
        if success:
            active_sessions.dec()
            
            # Broadcast session deletion
            await connection_manager.broadcast({
                "type": "session_deleted",
                "session_id": session_id
            })
            
            return {"status": "deleted", "session_id": session_id}
        else:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as e:
        logger.error(f"Delete session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/pairs")
async def list_model_pairs():
    """List available model pairs"""
    if not comparison_service:
        raise HTTPException(status_code=503, detail="Comparison service not initialized")
    
    try:
        pairs = await comparison_service.list_model_pairs()
        return {"pairs": pairs}
    except Exception as e:
        logger.error(f"List model pairs failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Chat-prefixed endpoints for frontend compatibility
@app.get("/chat/model-pairs")
async def get_chat_model_pairs():
    """Get all available base model + LoRA adapter pairs for comparison (frontend compatibility)"""
    try:
        pairs_response = await list_model_pairs()
        return {
            "status": "success",
            "model_pairs": pairs_response["pairs"]
        }
    except Exception as e:
        return {
            "status": "error",
            "model_pairs": [],
            "message": str(e)
        }


@app.get("/chat/sessions")
async def list_chat_sessions():
    """List all active comparison chat sessions (frontend compatibility)"""
    try:
        sessions = await list_sessions()
        return {
            "status": "success",
            "sessions": sessions
        }
    except Exception as e:
        return {
            "status": "error", 
            "sessions": [],
            "message": str(e)
        }


@app.post("/chat/sessions")
async def create_chat_session(request: SessionCreateRequest):
    """Create a new comparison chat session (frontend compatibility)"""
    return await create_session(request)


@app.post("/chat/generate")
async def chat_generate(request: ComparisonRequest):
    """Generate chat responses using comparison models (frontend compatibility)"""
    if not comparison_service:
        comparison_errors_total.inc()
        raise HTTPException(status_code=503, detail="Comparison service not initialized")
    
    try:
        # Use the comparison service directly
        response = await comparison_service.generate_comparison(request)
        
        # Broadcast comparison result if session_id is provided
        if hasattr(request, 'session_id') and request.session_id:
            await connection_manager.broadcast({
                "type": "comparison_result",
                "session_id": request.session_id,
                "response": response.model_dump()
            })
        
        return response
    except Exception as e:
        comparison_errors_total.inc()
        logger.error(f"Chat generate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/pairs")
async def create_model_pair(config: DualModelConfig):
    """Create a new model pair"""
    if not comparison_service:
        raise HTTPException(status_code=503, detail="Comparison service not initialized")
    
    try:
        pair_id = await comparison_service.create_model_pair(config)
        return {"pair_id": pair_id, "status": "created"}
    except Exception as e:
        logger.error(f"Create model pair failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    if not comparison_service:
        raise HTTPException(status_code=503, detail="Comparison service not initialized")
    
    try:
        stats = comparison_service.get_cache_stats()
        
        # Update Prometheus metrics
        model_cache_hits.inc(stats.get('hits', 0))
        model_cache_misses.inc(stats.get('misses', 0))
        
        return stats
    except Exception as e:
        logger.error(f"Get cache stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear")
async def clear_cache():
    """Clear model cache"""
    if not comparison_service:
        raise HTTPException(status_code=503, detail="Comparison service not initialized")
    
    try:
        await comparison_service.clear_cache()
        return {"status": "cleared", "message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Clear cache failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time session updates"""
    await connection_manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # Handle WebSocket messages (e.g., typing indicators, live updates)
            message = {
                "type": "websocket_message",
                "session_id": session_id,
                "data": data,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Broadcast to other connections in the same session
            await connection_manager.broadcast(message)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


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
    if not comparison_service:
        raise HTTPException(status_code=503, detail="Comparison service not initialized")
    
    try:
        await comparison_service.optimize_memory()
        return {"status": "optimized", "message": "Memory optimization completed"}
    except Exception as e:
        logger.error(f"Memory optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task for updating metrics
async def update_metrics():
    """Background task to update system metrics"""
    while True:
        try:
            if comparison_service:
                active_sessions.set(len(comparison_service.sessions))
                
                cache_stats = comparison_service.get_cache_stats()
                model_cache_hits.inc(cache_stats.get('hits', 0))
                model_cache_misses.inc(cache_stats.get('misses', 0))
            
            await asyncio.sleep(10)  # Update every 10 seconds
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
            await asyncio.sleep(30)  # Wait longer on error


if __name__ == "__main__":
    # Run the service
    port = int(os.getenv('SERVICE_PORT', '8002'))
    host = os.getenv('SERVICE_HOST', '0.0.0.0')
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )