from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
from model_selection import select_model, SelectModelRequest, scan_available_models
from data_preparation import prepare_data, PrepareDataRequest  
# Import fine_tuning components with error handling for TensorFlow issues
try:
    from fine_tuning import fine_tune, FineTuneRequest, TrainingStatus, task_log_queues, task_completion_flags, get_active_sessions
    FINE_TUNING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Fine-tuning module import failed: {e}")
    # Create fallback objects
    task_log_queues = {}
    task_completion_flags = {}
    FINE_TUNING_AVAILABLE = False

# Import comparison chat service
try:
    from comparison_chat_service import (
        ComparisonChatService, ChatMessage, ComparisonRequest, ComparisonResponse,
        SessionCreateRequest, SessionInfo, comparison_chat_service
    )
    COMPARISON_CHAT_AVAILABLE = True
    print("✅ Comparison chat service loaded")
except ImportError as e:
    print(f"Warning: Comparison chat service import failed: {e}")
    COMPARISON_CHAT_AVAILABLE = False
    
    # Define minimal fallback classes
    from pydantic import BaseModel
    from typing import Optional
    
    class FineTuneRequest(BaseModel):
        base_model_dir: str
        dataset_dir: str
        output_dir: str
        ft_type: str = "instruction"
        peft_method: str = "qlora"
        alignment: str = "none"
        epochs: int = 1
        batch_size: int = 20
        learning_rate: float = 0.0001
        gradient_accumulation_steps: int = 4
        max_seq_length: int = 256
        reasoning_effort: Optional[str] = None
    
    class TrainingStatus(BaseModel):
        task_id: str
        status: str
        progress: str
        current_step: Optional[int] = None
        total_steps: Optional[int] = None
        current_epoch: Optional[float] = None
        total_epochs: Optional[int] = None
        loss: Optional[float] = None
        learning_rate: Optional[float] = None
        progress_percentage: Optional[float] = None
        message: Optional[str] = None
        started_at: Optional[str] = None
        updated_at: Optional[str] = None
    
    def get_active_sessions():
        return {}
    
    def fine_tune(request, task_id=None):
        return {"error": "Fine-tuning not available due to import issues"}
import asyncio
from typing import Dict


app = FastAPI(
    title="LLMForge Fine-tuning API",
    description="Simplified fine-tuning platform for LLM models",
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

# Global variable to track background tasks
training_status = {}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "LLMForge Fine-tuning"}


@app.get("/models/available")
async def get_available_models():
    """
    Get list of available models in the models directory.
    
    Returns:
        JSON response with available models and their metadata
    """
    try:
        result = scan_available_models()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan models: {str(e)}")


@app.post("/select-model")
async def select_model_endpoint(request: SelectModelRequest):
    """
    Download and prepare a base model for fine-tuning.
    
    Args:
        request: SelectModelRequest with model_id, output_dir, and quantization options
        
    Returns:
        JSON response with status and model path
    """
    try:
        result = select_model(request)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
            
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model selection failed: {str(e)}")


@app.post("/prepare-data")
async def prepare_data_endpoint(request: PrepareDataRequest):
    """
    Clean and prepare data for fine-tuning.
    
    Args:
        request: PrepareDataRequest with input file, output directory, and processing options
        
    Returns:
        JSON response with status and dataset paths
    """
    try:
        result = prepare_data(request)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
            
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data preparation failed: {str(e)}")


async def run_fine_tuning(request: FineTuneRequest, task_id: str):
    """Run fine-tuning in the background."""
    global training_status
    
    training_status[task_id] = {
        "status": "running", 
        "message": "Fine-tuning in progress..."
    }
    
    # Initialize the task in the log queue system
    import queue
    if task_id not in task_log_queues:
        task_log_queues[task_id] = queue.Queue()
    
    task_completion_flags[task_id] = False
    
    try:
        # Create a modified request with the task_id
        request_dict = request.dict() if hasattr(request, 'dict') else request.__dict__
        
        # Use the new fine_tune function that returns a proper response
        from fine_tuning import fine_tune
        result = fine_tune(request, task_id)  # Pass the API task_id to the fine_tune function
        
        training_status[task_id] = {
            "status": "completed",
            "message": "Fine-tuning completed successfully",
            "task_id": result.get("task_id", task_id)
        }
    except Exception as e:
        training_status[task_id] = {
            "status": "error",
            "message": f"Fine-tuning failed: {str(e)}",
            "task_id": task_id
        }
        # Mark as complete even on error so SSE can close
        task_completion_flags[task_id] = True


@app.post("/fine-tune")
async def fine_tune_endpoint(request: FineTuneRequest, background_tasks: BackgroundTasks):
    """
    Fine-tune a model with the prepared data.
    
    Args:
        request: FineTuneRequest with model path, dataset path, and training parameters
        background_tasks: FastAPI background tasks for async execution
        
    Returns:
        JSON response with task ID for tracking training progress
    """
    import uuid
    
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Start fine-tuning in background
        background_tasks.add_task(run_fine_tuning, request, task_id)
        
        return JSONResponse(content={
            "status": "started",
            "message": "Fine-tuning started in background",
            "task_id": task_id
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start fine-tuning: {str(e)}")


@app.get("/fine-tune/active")
async def get_active_training_sessions():
    """
    Get all currently active training sessions.
    
    Returns:
        JSON response with active training sessions and their status
    """
    try:
        active_sessions = get_active_sessions()
        
        # Enhance session data with current status from status files
        enhanced_sessions = {}
        for task_id, session in active_sessions.items():
            enhanced_sessions[task_id] = session.copy()
            
            # Try to get current status from status file
            try:
                status_file = Path("logs") / f"{task_id}_status.json"
                if status_file.exists():
                    import json
                    with open(status_file, "r") as f:
                        status_data = json.load(f)
                    enhanced_sessions[task_id]["current_status"] = status_data
                else:
                    enhanced_sessions[task_id]["current_status"] = None
            except Exception as e:
                print(f"Error reading status for {task_id}: {e}")
                enhanced_sessions[task_id]["current_status"] = None
        
        return JSONResponse(content={
            "status": "success",
            "active_sessions": enhanced_sessions,
            "total_active": len(enhanced_sessions)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active sessions: {str(e)}")


@app.get("/fine-tune/status/{task_id}")
async def get_training_status(task_id: str):
    """
    Get the status of a fine-tuning task.
    
    Args:
        task_id: The task ID returned from the fine-tune endpoint
        
    Returns:
        JSON response with current training status
    """
    import json
    from pathlib import Path
    global training_status
    
    # Try to read from status file first (more detailed info)
    status_file = Path("logs") / f"{task_id}_status.json"
    if status_file.exists():
        try:
            with open(status_file, "r") as f:
                status_data = json.load(f)
            
            # Enhance with additional computed information
            if status_data.get("current_step") and status_data.get("total_steps"):
                progress_pct = (status_data["current_step"] / status_data["total_steps"]) * 100
                status_data["progress_percentage"] = round(progress_pct, 2)
            elif status_data.get("current_epoch") and status_data.get("total_epochs"):
                progress_pct = (status_data["current_epoch"] / status_data["total_epochs"]) * 100
                status_data["progress_percentage"] = round(progress_pct, 2)
            
            # Add estimated completion time if we have progress
            if status_data.get("started_at") and status_data.get("progress_percentage", 0) > 0:
                try:
                    import time
                    from datetime import datetime, timedelta
                    
                    start_time = datetime.strptime(status_data["started_at"], "%Y-%m-%d %H:%M:%S")
                    current_time = datetime.now()
                    elapsed = (current_time - start_time).total_seconds()
                    
                    if status_data["progress_percentage"] > 0:
                        total_estimated_time = elapsed * (100 / status_data["progress_percentage"])
                        remaining_time = total_estimated_time - elapsed
                        
                        if remaining_time > 0:
                            eta = current_time + timedelta(seconds=remaining_time)
                            status_data["estimated_completion"] = eta.strftime("%Y-%m-%d %H:%M:%S")
                            status_data["estimated_remaining_minutes"] = round(remaining_time / 60, 1)
                except Exception as e:
                    print(f"Error calculating ETA: {e}")
            
            return JSONResponse(content=status_data)
        except Exception as e:
            print(f"Error reading status file: {e}")
    
    # Fallback to global training status
    if task_id not in training_status:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    return JSONResponse(content=training_status[task_id])


@app.delete("/fine-tune/cancel/{task_id}")
async def cancel_training(task_id: str):
    """
    Cancel an active training session safely.
    
    Args:
        task_id: The task ID of the training session to cancel
        
    Returns:
        JSON response confirming cancellation request
    """
    try:
        import time
        import json
        
        # Load active sessions directly from file
        sessions_file = Path("logs") / "active_sessions.json"
        active_sessions = {}
        if sessions_file.exists():
            try:
                with open(sessions_file, "r") as f:
                    active_sessions = json.load(f)
            except:
                pass
        
        if task_id not in active_sessions:
            raise HTTPException(status_code=404, detail=f"Training task {task_id} not found")
        
        # Check if task is in a cancellable state
        session = active_sessions[task_id]
        if session.get("status") not in ["running", "started"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Task {task_id} cannot be cancelled. Current status: {session.get('status', 'unknown')}"
            )
        
        # Force stop the training process
        try:
            # Update status to cancelled immediately
            status_file = Path("logs") / f"{task_id}_status.json"
            if status_file.exists():
                with open(status_file, "r") as f:
                    status_data = json.load(f)
                
                status_data.update({
                    "status": "cancelled",
                    "progress": "Training cancelled by user",
                    "message": "Training was cancelled by user request",
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
                with open(status_file, "w") as f:
                    json.dump(status_data, f, indent=2)
        except Exception as e:
            print(f"Error updating status file: {e}")
        
        # Remove from active sessions
        try:
            sessions_file = Path("logs") / "active_sessions.json"
            if sessions_file.exists():
                with open(sessions_file, "r") as f:
                    sessions = json.load(f)
                
                if task_id in sessions:
                    sessions[task_id]["status"] = "cancelled"
                    sessions[task_id]["last_activity"] = time.strftime("%Y-%m-%d %H:%M:%S")
                
                with open(sessions_file, "w") as f:
                    json.dump(sessions, f, indent=2)
        except Exception as e:
            print(f"Error updating sessions: {e}")
        
        # Update global training status
        global training_status
        training_status[task_id] = {
            "status": "cancelled",
            "message": "Training cancelled by user",
            "task_id": task_id
        }
        
        # Set completion flag to stop SSE streaming
        task_completion_flags[task_id] = True
        
        return JSONResponse(content={
            "status": "cancellation_requested",
            "task_id": task_id,
            "message": "Training cancellation requested. Training will stop at the next safe checkpoint.",
            "details": "The training process will complete the current step/epoch before stopping to ensure data integrity."
        })
        
    except HTTPException:
        # Re-raise HTTP exceptions (they have proper status codes)
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to cancel training: {str(e)}"
        )


@app.get("/fine-tune/logs/{task_id}")
async def stream_training_logs(task_id: str):
    """
    Stream training logs in real-time using Server-Sent Events.
    
    Args:
        task_id: The task ID returned from the fine-tune endpoint
        
    Returns:
        Server-Sent Events stream with training updates
    """
    import json
    import time
    import queue
    from pathlib import Path
    
    async def generate_logs():
        last_heartbeat = time.time()
        last_status_update = None
        
        # Send initial connection message
        yield f"event: connected\n"
        yield f"data: {json.dumps({'message': 'Connected to training logs', 'task_id': task_id})}\n\n"
        
        # Check if task exists in status files
        status_file = Path("logs") / f"{task_id}_status.json"
        if not status_file.exists():
            yield f"event: error\n"
            yield f"data: {json.dumps({'error': 'Task not found', 'task_id': task_id})}\n\n"
            return
        
        while True:
            try:
                # Read status from file instead of queue
                if status_file.exists():
                    try:
                        with open(status_file, "r") as f:
                            status_data = json.load(f)
                        
                        # Check if status was updated since last read
                        current_updated_at = status_data.get("updated_at")
                        if current_updated_at != last_status_update:
                            last_status_update = current_updated_at
                            
                            # Create a log entry from status data
                            if status_data.get("status") == "running":
                                log_entry = {
                                    "timestamp": current_updated_at or time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "type": "training",
                                    "message": status_data.get("progress", "Training in progress"),
                                    "step": status_data.get("current_step"),
                                    "epoch": status_data.get("current_epoch"),
                                    "loss": status_data.get("loss"),
                                    "learning_rate": status_data.get("learning_rate"),
                                    "progress_percentage": status_data.get("progress_percentage")
                                }
                                
                                yield f"event: log\n"
                                yield f"data: {json.dumps(log_entry)}\n\n"
                                
                                # Also send status update
                                yield f"event: status\n"
                                yield f"data: {json.dumps(status_data)}\n\n"
                            
                            elif status_data.get("status") in ["completed", "cancelled", "error"]:
                                # Training completed/cancelled, send final message and exit
                                final_status = status_data.get("status")
                                final_message = status_data.get("message", f"Training {final_status}")
                                
                                final_log = {
                                    "timestamp": current_updated_at or time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "type": "success" if final_status == "completed" else final_status,
                                    "message": final_message
                                }
                                
                                yield f"event: log\n"
                                yield f"data: {json.dumps(final_log)}\n\n"
                                
                                yield f"event: status\n"
                                yield f"data: {json.dumps(status_data)}\n\n"
                                
                                yield f"event: complete\n"
                                yield f"data: {json.dumps({'message': f'Training {final_status}', 'final_status': final_status})}\n\n"
                                return  # Exit the stream
                        
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Error reading status file: {e}")
                
                # Try to get logs from queue if available (fallback)
                elif task_id in task_log_queues:
                    try:
                        log_entry = task_log_queues[task_id].get_nowait()
                        
                        # Determine event type based on log content
                        event_type = "log"
                        if log_entry.get("type") == "training" and log_entry.get("metrics"):
                            event_type = "metrics"
                        elif log_entry.get("type") in ["error", "warning"]:
                            event_type = "error" if log_entry.get("type") == "error" else "warning"
                        
                        yield f"event: {event_type}\n"
                        yield f"data: {json.dumps(log_entry)}\n\n"
                    except queue.Empty:
                        pass
                
                # Send heartbeat every 30 seconds
                current_time = time.time()
                if current_time - last_heartbeat > 30:
                    yield f"event: heartbeat\n"
                    yield f"data: {json.dumps({'timestamp': current_time})}\n\n"
                    last_heartbeat = current_time
                
                # Check if training is complete or cancelled
                if task_completion_flags.get(task_id, False):
                    yield f"event: complete\n"
                    yield f"data: {json.dumps({'message': 'Training stream ended', 'task_id': task_id})}\n\n"
                    return  # Exit the generator instead of break
                
                await asyncio.sleep(3.0)  # Check every 3 seconds to prevent connection spam
                
            except Exception as e:
                print(f"Error in SSE stream: {e}")
                yield f"event: error\n"
                yield f"data: {json.dumps({'error': str(e), 'task_id': task_id})}\n\n"
                break
    
    return StreamingResponse(
        generate_logs(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        }
    )


# Comparison Chat API Endpoints
@app.get("/chat/model-pairs")
async def get_model_pairs():
    """Get all available base model + LoRA adapter pairs for comparison."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    try:
        pairs = await comparison_chat_service.get_available_model_pairs()
        return JSONResponse(content={
            "status": "success",
            "model_pairs": pairs,
            "total": len(pairs)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model pairs: {str(e)}")


@app.post("/chat/sessions")
async def create_chat_session(request: SessionCreateRequest):
    """Create a new comparison chat session."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    try:
        session_info = await comparison_chat_service.create_session(request)
        return JSONResponse(content={
            "status": "success",
            "session": session_info.dict()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@app.get("/chat/sessions")
async def list_chat_sessions():
    """List all active comparison chat sessions."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    try:
        sessions = await comparison_chat_service.list_active_sessions()
        return JSONResponse(content={
            "status": "success",
            "sessions": [session.dict() for session in sessions],
            "total": len(sessions)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@app.get("/chat/sessions/{session_id}")
async def get_chat_session_history(session_id: str):
    """Get conversation history for a specific chat session."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    try:
        history = await comparison_chat_service.get_session_history(session_id)
        return JSONResponse(content={
            "status": "success",
            **history
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session history: {str(e)}")


@app.post("/chat/generate")
async def generate_comparison_response(request: ComparisonRequest):
    """Generate comparison responses from both base and fine-tuned models."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    try:
        response = await comparison_chat_service.generate_comparison(request)
        return JSONResponse(content={
            "status": "success",
            "response": response.dict()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate comparison: {str(e)}")


@app.post("/chat/generate-stream")
async def generate_comparison_stream(request: ComparisonRequest):
    """Stream comparison responses from both base and fine-tuned models."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    async def generate_stream():
        try:
            async for chunk in comparison_chat_service.generate_comparison_stream(request):
                import json
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            error_chunk = {
                "type": "error",
                "data": {"error": str(e)}
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        }
    )


@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a comparison chat session and clean up resources."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    try:
        result = await comparison_chat_service.delete_session(session_id)
        return JSONResponse(content={
            "status": "success",
            **result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@app.get("/chat/metrics")
async def get_chat_service_metrics():
    """Get comprehensive chat service performance metrics."""
    if not COMPARISON_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Comparison chat service not available")
    
    try:
        metrics = await comparison_chat_service.get_service_metrics()
        return JSONResponse(content={
            "status": "success",
            **metrics
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


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
        "version": "1.0.0",
        "features": {
            "fine_tuning": FINE_TUNING_AVAILABLE,
            "comparison_chat": COMPARISON_CHAT_AVAILABLE
        },
        "endpoints": endpoints
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)