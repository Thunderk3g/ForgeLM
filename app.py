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
from controllers.config_controller import config_controller
from controllers.visualization_controller import VisualizationController

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
from services.visualization_service import VisualizationService

# Backward compatibility imports - maintain original import behavior
from utils.legacy.model_selection import select_model, SelectModelRequest, scan_available_models
from utils.legacy.data_preparation import prepare_data, PrepareDataRequest

# Import fine_tuning components with error handling
try:
    from models.schemas import FineTuneRequest, TrainingStatus
    from services.training_worker import process_manager
    FINE_TUNING_AVAILABLE = True
    print("✅ Fine-tuning with multiprocessing support loaded")
except ImportError as e:
    print(f"Warning: Fine-tuning module import failed: {e}")
    FINE_TUNING_AVAILABLE = False

# Import comparison chat service components
try:
    print("🔄 Attempting to import comparison service components...")
    from services.comparison_service import ComparisonService
    print("✅ ComparisonService imported successfully")
    from models.schemas import ChatMessage, ComparisonRequest, ComparisonResponse, SessionCreateRequest, SessionInfo
    print("✅ Schema imports successful")
    COMPARISON_CHAT_AVAILABLE = True
    print("✅ Comparison chat service loaded (modular)")
except ImportError as e:
    print(f"❌ Warning: Comparison chat service import failed: {e}")
    import traceback
    traceback.print_exc()
    COMPARISON_CHAT_AVAILABLE = False
except Exception as e:
    print(f"❌ Error: Unexpected error during comparison service import: {e}")
    import traceback
    traceback.print_exc()
    COMPARISON_CHAT_AVAILABLE = False

# Initialize services
print("🔄 Initializing core services...")
model_service = ModelService()
print("✅ ModelService initialized")

chat_service = ChatService(model_service)
print("✅ ChatService initialized")

if COMPARISON_CHAT_AVAILABLE:
    try:
        comparison_service_instance = ComparisonService(model_service, chat_service)
        print("✅ ComparisonService initialized")
    except Exception as e:
        print(f"❌ Error initializing ComparisonService: {e}")
        import traceback
        traceback.print_exc()
        COMPARISON_CHAT_AVAILABLE = False
        comparison_service_instance = None
else:
    comparison_service_instance = None
    print("⚠️ ComparisonService not available - skipping initialization")

# Initialize visualization service
try:
    visualization_service = VisualizationService(
        training_service=None,  # Will be set if needed
        comparison_service=comparison_service_instance,
        template_service=None   # Will be integrated later
    )
    visualization_controller = VisualizationController(visualization_service)
    print("✅ VisualizationService and Controller initialized")
    VISUALIZATION_AVAILABLE = True
except Exception as e:
    print(f"❌ Error initializing VisualizationService: {e}")
    import traceback
    traceback.print_exc()
    visualization_service = None
    visualization_controller = None
    VISUALIZATION_AVAILABLE = False

# Initialize data generation service
try:
    from services.data_generation_service import DataGenerationService
    from controllers.data_generation_controller import DataGenerationController
    
    data_generation_service = DataGenerationService(
        model_service=model_service,
        template_service=None  # Will be integrated later
    )
    data_generation_controller = DataGenerationController(data_generation_service)
    print("✅ DataGenerationService and Controller initialized")
    DATA_GENERATION_AVAILABLE = True
except Exception as e:
    print(f"❌ Error initializing DataGenerationService: {e}")
    import traceback
    traceback.print_exc()
    data_generation_service = None
    data_generation_controller = None
    DATA_GENERATION_AVAILABLE = False

# Initialize controllers with services
model_controller.model_service = model_service

# Initialize chat controller properly
if COMPARISON_CHAT_AVAILABLE and comparison_service_instance:
    from controllers.chat_controller import ChatController
    chat_controller_instance = ChatController(comparison_service_instance)
    
    # Replace the imported chat_controller with our properly initialized instance
    import controllers.chat_controller
    controllers.chat_controller.chat_controller = chat_controller_instance
    print("✅ ChatController initialized with ComparisonService")
else:
    chat_controller_instance = None
    print("⚠️ ChatController not available - ComparisonService unavailable")

# Note: TrainingController now uses multiprocessing and doesn't need TrainingService injection

# Create FastAPI app with lifespan for process management
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting LLMForge API (Multiprocessing Architecture)")
    
    if COMPARISON_CHAT_AVAILABLE and comparison_service_instance:
        await comparison_service_instance.start_background_tasks()
        print("✅ Comparison service background tasks started")
    
    print("✅ LLMForge API startup complete")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down LLMForge API")
    
    # Clean up all training processes
    if FINE_TUNING_AVAILABLE:
        process_manager.cleanup_all_processes()
        print("✅ Training processes cleaned up")
    
    # Clean up comparison service
    if COMPARISON_CHAT_AVAILABLE and comparison_service_instance:
        # Add comparison service cleanup here if needed
        pass
    
    print("✅ LLMForge API shutdown complete")

app = FastAPI(
    title="LLMForge Fine-tuning API",
    description="Multiprocessing fine-tuning platform for LLM models with advanced optimizations",
    version="1.1.0",
    lifespan=lifespan
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
        "/config/list": "GET - List available configuration files",
        "/config/{config_name}": "GET - Get specific configuration",
        "/config/create/{config_name}": "POST - Create config from request",
        "/config/validate/{config_name}": "GET - Validate configuration",
        "/config/templates": "GET - List available dataset templates",
        "/config/peft-methods": "GET - List available PEFT methods",
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
        "version": "1.1.0 (Multiprocessing)",
        "architecture": "multiprocessing",
        "features": {
            "fine_tuning": FINE_TUNING_AVAILABLE,
            "comparison_chat": COMPARISON_CHAT_AVAILABLE,
            "unsloth_optimization": True,
            "fastlibra_cache": True,
            "multiprocessing_training": True,
            "non_blocking_api": True,
            "process_isolation": True
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

@app.get("/models/fine-tuned")
async def get_fine_tuned_models():
    """Get list of fine-tuned models with metadata and organization."""
    return await model_controller.get_fine_tuned_models()

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
# CONFIGURATION MANAGEMENT ENDPOINTS  
# =====================================

@app.get("/config/list")
async def list_configs():
    """List all available configuration files."""
    return await config_controller.list_configs()

@app.get("/config/{config_name}")
async def get_config(config_name: str):
    """Get specific configuration by name."""
    return await config_controller.get_config(config_name)

@app.post("/config/create/{config_name}")
async def create_config_from_request(config_name: str, request: FineTuneRequest):
    """Create configuration from training request."""
    return await config_controller.create_config_from_request(request, config_name)

@app.get("/config/validate/{config_name}")
async def validate_config(config_name: str):
    """Validate configuration."""
    return await config_controller.validate_config(config_name)

@app.get("/config/templates")
async def get_available_templates():
    """Get available dataset templates."""
    return await config_controller.get_available_templates()

@app.get("/config/peft-methods")
async def get_available_peft_methods():
    """Get available PEFT methods."""
    return await config_controller.get_available_peft_methods()

# =====================================
# COMPARISON CHAT ENDPOINTS
# =====================================

@app.get("/chat/model-pairs")
async def get_model_pairs():
    """Get all available base model + LoRA adapter pairs for comparison."""
    if not COMPARISON_CHAT_AVAILABLE or not chat_controller_instance:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller_instance.get_model_pairs()

@app.post("/chat/sessions")
async def create_chat_session(request: SessionCreateRequest):
    """Create a new comparison chat session."""
    if not COMPARISON_CHAT_AVAILABLE or not chat_controller_instance:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller_instance.create_chat_session(request)

@app.get("/chat/sessions")
async def list_chat_sessions():
    """List all active comparison chat sessions."""
    if not COMPARISON_CHAT_AVAILABLE or not chat_controller_instance:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller_instance.list_chat_sessions()

@app.get("/chat/sessions/{session_id}")
async def get_chat_session_history(session_id: str):
    """Get conversation history for a specific chat session."""
    if not COMPARISON_CHAT_AVAILABLE or not chat_controller_instance:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller_instance.get_chat_session_history(session_id)

@app.post("/chat/generate")
async def generate_comparison_response(request: ComparisonRequest):
    """Generate comparison responses from both base and fine-tuned models."""
    if not COMPARISON_CHAT_AVAILABLE or not chat_controller_instance:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller_instance.generate_comparison_response(request)

@app.post("/chat/generate-stream")
async def generate_comparison_stream(request: ComparisonRequest):
    """Stream comparison responses from both base and fine-tuned models."""
    if not COMPARISON_CHAT_AVAILABLE or not chat_controller_instance:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller_instance.generate_comparison_stream(request)

@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a comparison chat session and clean up resources."""
    if not COMPARISON_CHAT_AVAILABLE or not chat_controller_instance:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller_instance.delete_chat_session(session_id)

@app.get("/chat/metrics")
async def get_chat_service_metrics():
    """Get comprehensive chat service performance metrics."""
    if not COMPARISON_CHAT_AVAILABLE or not chat_controller_instance:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller_instance.get_chat_service_metrics()

@app.get("/chat/research/analytics")
async def get_research_analytics(session_id: str = None):
    """Get research analytics data for model comparison analysis."""
    if not COMPARISON_CHAT_AVAILABLE or not chat_controller_instance:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller_instance.get_research_analytics(session_id)

@app.get("/chat/research/export")
async def export_research_data(start_date: str = None, end_date: str = None, format: str = "json"):
    """Export research data for external analysis."""
    if not COMPARISON_CHAT_AVAILABLE or not chat_controller_instance:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    return await chat_controller_instance.export_research_data(start_date, end_date, format)

# =====================================
# VISUALIZATION & ANALYTICS ENDPOINTS  
# =====================================

@app.get("/visualization/training/{task_id}")
async def get_training_analytics(task_id: str, limit: int = 100):
    """Get training analytics for visualization dashboard."""
    if not VISUALIZATION_AVAILABLE or not visualization_controller:
        raise HTTPException(status_code=503, detail="Visualization service not available")
    
    return await visualization_controller.get_training_analytics(task_id, limit)

@app.get("/visualization/dataset")
async def get_dataset_analytics(dataset_path: str):
    """Get dataset analytics for data exploration dashboard."""
    if not VISUALIZATION_AVAILABLE or not visualization_controller:
        raise HTTPException(status_code=503, detail="Visualization service not available")
    
    return await visualization_controller.get_dataset_analytics(dataset_path)

@app.get("/visualization/comparisons")
async def get_comparison_analytics(time_range_hours: int = 24):
    """Get model comparison analytics for comparison dashboard."""
    if not VISUALIZATION_AVAILABLE or not visualization_controller:
        raise HTTPException(status_code=503, detail="Visualization service not available")
    
    return await visualization_controller.get_comparison_analytics(time_range_hours)

@app.get("/visualization/system/health")
async def get_system_health():
    """Get current system health metrics for monitoring."""
    if not VISUALIZATION_AVAILABLE or not visualization_controller:
        raise HTTPException(status_code=503, detail="Visualization service not available")
    
    return await visualization_controller.get_system_health()

@app.get("/visualization/dashboard/summary")
async def get_dashboard_summary(
    include_training: bool = True,
    include_comparisons: bool = True,
    include_system: bool = True
):
    """Get comprehensive dashboard summary with all key metrics."""
    if not VISUALIZATION_AVAILABLE or not visualization_controller:
        raise HTTPException(status_code=503, detail="Visualization service not available")
    
    return await visualization_controller.get_dashboard_summary(
        include_training, include_comparisons, include_system
    )

# Enhanced training endpoint with visualization data
@app.get("/fine-tune/visualization/{task_id}")
async def get_training_visualization_data(task_id: str, limit: int = 100):
    """Get enhanced training visualization data including resource metrics."""
    return await training_controller.get_training_visualization_data(task_id, limit)

# =====================================
# DATA GENERATION ENDPOINTS
# =====================================

@app.post("/data-generation/start")
async def start_data_generation(request: dict):
    """Start a new data generation task"""
    if not DATA_GENERATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data generation service not available")
    
    # Convert dict to DataGenerationRequest
    from controllers.data_generation_controller import DataGenerationRequest
    generation_request = DataGenerationRequest(**request)
    return await data_generation_controller.start_generation(generation_request)

@app.get("/data-generation/status/{task_id}")
async def get_generation_status(task_id: str):
    """Get status of a data generation task"""
    if not DATA_GENERATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data generation service not available")
    return await data_generation_controller.get_generation_status(task_id)

@app.post("/data-generation/cancel/{task_id}")
async def cancel_generation(task_id: str):
    """Cancel an active data generation task"""
    if not DATA_GENERATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data generation service not available")
    return await data_generation_controller.cancel_generation(task_id)

@app.get("/data-generation/datasets")
async def list_generated_datasets():
    """List all generated datasets"""
    if not DATA_GENERATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data generation service not available")
    return await data_generation_controller.list_generated_datasets()

@app.get("/data-generation/preview/{dataset_filename}")
async def preview_dataset(dataset_filename: str, limit: int = 5):
    """Preview samples from a generated dataset"""
    if not DATA_GENERATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data generation service not available")
    return await data_generation_controller.preview_dataset(dataset_filename, limit)

@app.get("/data-generation/templates")
async def get_generation_templates():
    """Get available generation templates and configurations"""
    if not DATA_GENERATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data generation service not available")
    return await data_generation_controller.get_generation_templates()

@app.post("/data-generation/validate")
async def validate_generation_config(request: dict):
    """Validate generation configuration without starting generation"""
    if not DATA_GENERATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data generation service not available")
    
    from controllers.data_generation_controller import DataGenerationRequest
    generation_request = DataGenerationRequest(**request)
    return await data_generation_controller.validate_generation_config(generation_request)

@app.get("/data-generation/active")
async def get_active_generations():
    """Get list of currently active generation tasks"""
    if not DATA_GENERATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Data generation service not available")
    return await data_generation_controller.get_active_generations()

# =====================================
# ERROR HANDLERS
# =====================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error reporting"""
    print(f"Unhandled exception in {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )

# =====================================
# MAIN ENTRY POINT
# =====================================

if __name__ == "__main__":
    print("🦥 LLMForge Fine-tuning API - Multiprocessing Architecture")
    print("📦 Services: ModelService, ChatService, ComparisonService")
    print("🏗️ Training: Isolated worker processes for non-blocking ML training")
    print("🔧 Optimizations: Unsloth 2x speedup, FastLibra KV caching")
    print("⚡ Performance: FastAPI stays responsive during training")
    print("🛡️ Isolation: Complete process separation for GPU/CUDA operations")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )