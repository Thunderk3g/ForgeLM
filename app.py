from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from model_selection import select_model, SelectModelRequest, scan_available_models
from data_preparation import prepare_data, PrepareDataRequest  
from fine_tuning import fine_tune, FineTuneRequest, TrainingStatus
import asyncio
from typing import Dict


app = FastAPI(
    title="LLMForge Fine-tuning API",
    description="Simplified fine-tuning platform for LLM models",
    version="1.0.0"
)

# Disable CORS middleware since nginx handles it
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

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
    
    try:
        result = fine_tune(request)
        training_status[task_id] = result
    except Exception as e:
        training_status[task_id] = {
            "status": "error",
            "message": f"Fine-tuning failed: {str(e)}",
            "model_path": "",
            "eval_results": {}
        }


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
    from pathlib import Path
    
    async def generate_logs():
        last_data = None
        while True:
            status_file = Path("logs") / f"{task_id}_status.json"
            if status_file.exists():
                try:
                    with open(status_file, "r") as f:
                        data = json.load(f)
                    
                    # Only send if data has changed
                    if data != last_data:
                        yield f"data: {json.dumps(data)}\n\n"
                        last_data = data
                        
                        # Stop streaming if completed or failed
                        if data.get('status') in ['completed', 'failed']:
                            break
                            
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            await asyncio.sleep(1)  # Poll every second
    
    return StreamingResponse(
        generate_logs(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
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
            "/fine-tune/status/{task_id}": "GET - Check training status",
            "/health": "GET - Health check"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)