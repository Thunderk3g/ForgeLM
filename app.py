from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
from model_selection import select_model, SelectModelRequest, scan_available_models
from data_preparation import prepare_data, PrepareDataRequest  
from fine_tuning import fine_tune, FineTuneRequest, TrainingStatus, task_log_queues, task_completion_flags, get_active_sessions
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
        
        # Send initial connection message
        yield f"event: connected\n"
        yield f"data: {json.dumps({'message': 'Connected to training logs', 'task_id': task_id})}\n\n"
        
        # Check if task exists
        if task_id not in task_log_queues:
            # Try to read from status file as fallback
            status_file = Path("logs") / f"{task_id}_status.json"
            if not status_file.exists():
                yield f"event: error\n"
                yield f"data: {json.dumps({'error': 'Task not found', 'task_id': task_id})}\n\n"
                return
        
        while True:
            try:
                # Try to get logs from queue (non-blocking)
                if task_id in task_log_queues:
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
                
                # Check if training is complete
                if task_completion_flags.get(task_id, False):
                    # Send final status
                    status_file = Path("logs") / f"{task_id}_status.json"
                    if status_file.exists():
                        try:
                            with open(status_file, "r") as f:
                                final_status = json.load(f)
                            yield f"event: status\n"
                            yield f"data: {json.dumps(final_status)}\n\n"
                        except Exception as e:
                            print(f"Error reading final status: {e}")
                    
                    yield f"event: complete\n"
                    yield f"data: {json.dumps({'message': 'Training completed', 'task_id': task_id})}\n\n"
                    break
                
                await asyncio.sleep(0.5)  # Check every 500ms for responsive streaming
                
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


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "LLMForge Fine-tuning API",
        "version": "1.0.0",
        "endpoints": {
            "/models/available": "GET - List available local models",
            "/select-model": "POST - Download and prepare base model",
            "/prepare-data": "POST - Clean and prepare training data", 
            "/fine-tune": "POST - Start model fine-tuning",
            "/fine-tune/active": "GET - List all active training sessions",
            "/fine-tune/status/{task_id}": "GET - Check training status",
            "/fine-tune/logs/{task_id}": "SSE - Stream training logs",
            "/health": "GET - Health check"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)