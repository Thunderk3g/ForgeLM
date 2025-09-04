"""
LLMForge Fine-tuning API - Modular Architecture

Refactored version with clean separation of concerns while maintaining 100% API compatibility.
All existing endpoints work identically to the original monolithic version.
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import controllers
from controllers.health_controller import health_controller
from controllers.model_controller import model_controller
from controllers.training_controller import training_controller
from controllers.chat_controller import chat_controller

# Import schemas for type annotations
from models.schemas import (
    SelectModelRequest, PrepareDataRequest, FineTuneRequest,
    ComparisonRequest, SessionCreateRequest
)

# Import services
from services.model_service import ModelService
from services.training_service import TrainingService
from services.chat_service import ChatService
from services.comparison_service import ComparisonService

# Backward compatibility imports - maintain original import behavior
from model_selection import select_model, SelectModelRequest, scan_available_models
from data_preparation import prepare_data, PrepareDataRequest

# Import fine_tuning components with error handling for TensorFlow issues
try:
    from services.training_service import task_log_queues, task_completion_flags
    from models.schemas import FineTuneRequest, TrainingStatus
    FINE_TUNING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Fine-tuning module import failed: {e}")
    # Create fallback objects for compatibility
    task_log_queues = {}
    task_completion_flags = {}
    FINE_TUNING_AVAILABLE = False

# Import comparison chat service components
try:
    from services.comparison_service import comparison_service
    from models.schemas import ChatMessage, ComparisonRequest, ComparisonResponse, SessionCreateRequest, SessionInfo
    COMPARISON_CHAT_AVAILABLE = True
    print("Comparison chat service loaded (modular)")
except ImportError as e:
    print(f"Warning: Comparison chat service import failed: {e}")
    COMPARISON_CHAT_AVAILABLE = False

# Initialize services
model_service = ModelService()
training_service = TrainingService()
chat_service = ChatService()
comparison_service_instance = ComparisonService(model_service, chat_service)

# Initialize controllers with services
model_controller.model_service = model_service
training_controller.training_service = training_service
chat_controller.comparison_service = comparison_service_instance

# Create FastAPI app
app = FastAPI(
    title="LLMForge Fine-tuning API",
    description="Modular fine-tuning platform for LLM models with advanced optimizations",
    version="1.0.0"
)

# Enable CORS middleware for SSE support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# HEALTH AND SYSTEM ENDPOINTS
# =====================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return await health_controller.health_check()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    endpoints = {
        "/models/available": "GET - List available local models",
        "/select-model": "POST - Download and prepare base model",
        "/prepare-data": "POST - Clean and prepare training data", 
        "/fine-tune": "POST - Start model fine-tuning",
        "/fine-tune/active": "GET - List all active training sessions",
        "/fine-tune/status/{task_id}": "GET - Check training status",
        "/fine-tune/logs/{task_id}": "SSE - Stream training logs",
        "/fine-tune/cancel/{task_id}": "DELETE - Cancel active training",
        "/health": "GET - Health check"
    }
    
    # Add chat endpoints if available
    if COMPARISON_CHAT_AVAILABLE:
        endpoints.update({
            "/chat/model-pairs": "GET - List available model pairs for comparison",
            "/chat/sessions": "POST/GET - Create or list chat sessions",
            "/chat/sessions/{session_id}": "GET/DELETE - Get history or delete session",
            "/chat/generate": "POST - Generate comparison responses",
            "/chat/generate-stream": "SSE - Stream comparison responses in real-time",
            "/chat/metrics": "GET - Get chat service metrics"
        })
    
    return {
        "service": "LLMForge Fine-tuning API",
        "version": "1.0.0 (Modular)",
        "architecture": "microservices",
        "features": {
            "fine_tuning": FINE_TUNING_AVAILABLE,
            "comparison_chat": COMPARISON_CHAT_AVAILABLE,
            "unsloth_optimization": True,
            "fastlibra_cache": True,
            "modular_services": True
        },
        "endpoints": endpoints
    }

# =====================================
# MODEL MANAGEMENT ENDPOINTS
# =====================================

@app.get("/models/available")
async def get_available_models():
    """Get list of available models in the models directory."""
    return await model_controller.get_available_models()

@app.post("/select-model")
async def select_model_endpoint(request: SelectModelRequest):
    """Download and prepare a base model for fine-tuning."""
    return await model_controller.select_model_endpoint(request)

@app.post("/prepare-data")
async def prepare_data_endpoint(request: PrepareDataRequest):
    """Clean and prepare data for fine-tuning."""
    return await model_controller.prepare_data_endpoint(request)

# =====================================
# TRAINING ENDPOINTS
# =====================================

@app.post("/fine-tune")
async def fine_tune_endpoint(request: FineTuneRequest, background_tasks: BackgroundTasks):
    """Fine-tune a model with the prepared data."""
    return await training_controller.fine_tune_endpoint(request, background_tasks)

@app.get("/fine-tune/active")
async def get_active_training_sessions():
    """Get all currently active training sessions."""
    return await training_controller.get_active_training_sessions()

@app.get("/fine-tune/status/{task_id}")
async def get_training_status(task_id: str):
    """Get the status of a fine-tuning task."""
    return await training_controller.get_training_status(task_id)

@app.delete("/fine-tune/cancel/{task_id}")
async def cancel_training(task_id: str):
    """Cancel an active training session safely."""
    return await training_controller.cancel_training(task_id)

@app.get("/fine-tune/logs/{task_id}")
async def stream_training_logs(task_id: str):
    """Stream training logs in real-time using Server-Sent Events."""
    return await training_controller.stream_training_logs(task_id)

# =====================================
# COMPARISON CHAT ENDPOINTS
# =====================================

@app.get("/chat/model-pairs")
async def get_model_pairs():
    """Get all available base model + LoRA adapter pairs for comparison."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller.get_model_pairs()

@app.post("/chat/sessions")
async def create_chat_session(request: SessionCreateRequest):
    """Create a new comparison chat session."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller.create_chat_session(request)

@app.get("/chat/sessions")
async def list_chat_sessions():
    """List all active comparison chat sessions."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller.list_chat_sessions()

@app.get("/chat/sessions/{session_id}")
async def get_chat_session_history(session_id: str):
    """Get conversation history for a specific chat session."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller.get_chat_session_history(session_id)

@app.post("/chat/generate")
async def generate_comparison_response(request: ComparisonRequest):
    """Generate comparison responses from both base and fine-tuned models."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller.generate_comparison_response(request)

@app.post("/chat/generate-stream")
async def generate_comparison_stream(request: ComparisonRequest):
    """Stream comparison responses from both base and fine-tuned models."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller.generate_comparison_stream(request)

@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a comparison chat session and clean up resources."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller.delete_chat_session(session_id)

@app.get("/chat/metrics")
async def get_chat_service_metrics():
    """Get comprehensive chat service performance metrics."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller.get_chat_service_metrics()

# =====================================
# STARTUP AND SHUTDOWN EVENTS
# =====================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Starting LLMForge API (Modular Architecture)")
    
    # Start background tasks for comparison service
    if COMPARISON_CHAT_AVAILABLE:
        await comparison_service_instance.start_background_tasks()
        print("✅ Comparison service background tasks started")
    
    print("✅ LLMForge API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    print("🛑 Shutting down LLMForge API")
    
    # Clean up training service
    training_service.cleanup_all_trainings()
    
    print("✅ LLMForge API shutdown complete")

# =====================================
# MAIN ENTRY POINT
# =====================================

if __name__ == "__main__":
    print("🦥 LLMForge Fine-tuning API - Modular Architecture")
    print("📦 Services: ModelService, TrainingService, ChatService, ComparisonService")
    print("🔧 Optimizations: Unsloth 2x speedup, FastLibra KV caching")
    print("🏗️  Architecture: Clean separation of concerns with backward compatibility")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )