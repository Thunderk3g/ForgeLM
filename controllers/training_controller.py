"""
Training Controller

Handles fine-tuning training endpoints with Unsloth optimization.
"""

import asyncio
import uuid
import json
import time
from pathlib import Path
from typing import Dict, Any
from fastapi import HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse

from models.schemas import FineTuneRequest, TrainingStatus
from services.training_worker import process_manager
from utils.logging import get_logger


class TrainingController:
    """Controller for training and fine-tuning endpoints using multiprocessing"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.process_manager = process_manager
        
        # Track training status for compatibility
        self.training_status = {}
        
        self.logger.info("TrainingController initialized with multiprocessing support")
    
    async def run_fine_tuning(self, request: FineTuneRequest, task_id: str):
        """Run fine-tuning in a separate process (non-blocking)"""
        try:
            self.logger.info(f"Starting multiprocess training for task {task_id}")
            
            # Update initial status
            self.training_status[task_id] = {
                "status": "starting", 
                "message": "Starting fine-tuning process..."
            }
            
            # Convert request to dict for process serialization
            request_data = {
                "base_model_dir": request.base_model_dir,
                "dataset_dir": request.dataset_dir,
                "output_dir": request.output_dir,
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate,
                "max_seq_length": request.max_seq_length,
                "gradient_accumulation_steps": request.gradient_accumulation_steps,
                "peft_method": request.peft_method,
                "ft_type": request.ft_type
            }
            
            # Start training in separate process
            success = self.process_manager.start_training_process(task_id, request_data)
            
            if success:
                self.training_status[task_id] = {
                    "status": "running",
                    "message": "Fine-tuning started in separate process",
                    "task_id": task_id
                }
                self.logger.info(f"Training process started successfully for task {task_id}")
            else:
                self.training_status[task_id] = {
                    "status": "error",
                    "message": "Failed to start training process",
                    "task_id": task_id
                }
                self.logger.error(f"Failed to start training process for task {task_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to start multiprocess training: {e}")
            self.training_status[task_id] = {
                "status": "error",
                "message": f"Failed to start fine-tuning: {str(e)}",
                "task_id": task_id
            }
    
    async def fine_tune_endpoint(self, request: FineTuneRequest, background_tasks: BackgroundTasks) -> JSONResponse:
        """Fine-tune a model with the prepared data"""
        
        try:
            # Generate unique task ID
            task_id = str(uuid.uuid4())
            
            # Start fine-tuning in background
            background_tasks.add_task(self.run_fine_tuning, request, task_id)
            
            return JSONResponse(content={
                "status": "started",
                "message": "Fine-tuning started in background",
                "task_id": task_id
            })
            
        except Exception as e:
            self.logger.error(f"Failed to start fine-tuning: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start fine-tuning: {str(e)}")
    
    async def get_active_training_sessions(self) -> JSONResponse:
        """Get all currently active training sessions"""
        try:
            # Get active processes from process manager
            active_processes = self.process_manager.get_active_processes()
            
            # Enhance with status file data
            enhanced_sessions = {}
            for task_id, process_info in active_processes.items():
                enhanced_sessions[task_id] = process_info.copy()
                
                # Try to get current status from status file
                try:
                    status_file = Path("logs") / f"{task_id}_status.json"
                    if status_file.exists():
                        with open(status_file, "r") as f:
                            status_data = json.load(f)
                        enhanced_sessions[task_id]["current_status"] = status_data
                    else:
                        enhanced_sessions[task_id]["current_status"] = None
                except Exception as e:
                    self.logger.warning(f"Error reading status for {task_id}: {e}")
                    enhanced_sessions[task_id]["current_status"] = None
            
            return JSONResponse(content={
                "status": "success",
                "active_sessions": enhanced_sessions,
                "total_active": len(enhanced_sessions)
            })
            
        except Exception as e:
            self.logger.error(f"Failed to get active sessions: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get active sessions: {str(e)}")
    
    async def get_training_status(self, task_id: str) -> JSONResponse:
        """Get the status of a fine-tuning task"""
        
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
                        self.logger.warning(f"Error calculating ETA: {e}")
                
                return JSONResponse(content=status_data)
                
            except Exception as e:
                self.logger.warning(f"Error reading status file: {e}")
        
        # Fallback to global training status
        if task_id not in self.training_status:
            raise HTTPException(status_code=404, detail="Task ID not found")
        
        return JSONResponse(content=self.training_status[task_id])
    
    async def cancel_training(self, task_id: str) -> JSONResponse:
        """Cancel an active training session safely"""
        try:
            self.logger.info(f"Cancellation requested for task {task_id}")
            
            # Try to cancel using process manager
            success = self.process_manager.cancel_training(task_id)
            
            if success:
                # Update local status
                self.training_status[task_id] = {
                    "status": "cancelling",
                    "message": "Training cancellation requested",
                    "task_id": task_id
                }
                
                return JSONResponse(content={
                    "status": "cancellation_requested",
                    "task_id": task_id,
                    "message": "Training cancellation requested. Process will be terminated gracefully.",
                    "details": "The training process will stop at the next checkpoint to ensure data integrity."
                })
            else:
                # Check if task exists in status files
                status_file = Path("logs") / f"{task_id}_status.json"
                if not status_file.exists():
                    raise HTTPException(status_code=404, detail=f"Training task {task_id} not found")
                else:
                    # Process already finished
                    return JSONResponse(content={
                        "status": "not_running",
                        "task_id": task_id,
                        "message": "Training task is not currently running"
                    })
                
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to cancel training: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to cancel training: {str(e)}")
    
    
    async def stream_training_logs(self, task_id: str) -> StreamingResponse:
        """Stream training logs in real-time using Server-Sent Events"""
        
        async def generate_logs():
            last_heartbeat = time.time()
            last_status_update = None
            connection_start = time.time()
            
            try:
                # Send initial connection message
                yield f"event: connected\n"
                yield f"data: {json.dumps({'message': 'Connected to training logs', 'task_id': task_id})}\n\n"
                
                # Check if task exists in status files
                status_file = Path("logs") / f"{task_id}_status.json"
                if not status_file.exists():
                    self.logger.warning(f"Status file not found for task {task_id}")
                    yield f"event: error\n"
                    yield f"data: {json.dumps({'error': 'Task not found', 'task_id': task_id})}\n\n"
                    return
                
                self.logger.info(f"Starting SSE stream for task {task_id}")
                
            except Exception as e:
                self.logger.error(f"Error in SSE initialization: {e}")
                yield f"event: error\n"
                yield f"data: {json.dumps({'error': str(e), 'task_id': task_id})}\n\n"
                return
            
            while True:
                try:
                    # Read status from file (primary communication method)
                    if status_file.exists():
                        try:
                            with open(status_file, "r") as f:
                                status_data = json.load(f)
                            
                            # Check if status was updated since last read
                            current_updated_at = status_data.get("updated_at")
                            if current_updated_at != last_status_update:
                                last_status_update = current_updated_at
                                
                                # Stream different content based on status
                                current_status = status_data.get("status")
                                
                                if current_status in ["initializing", "loading", "configuring", "loading_data", "preparing"]:
                                    # Initialization phases
                                    log_entry = {
                                        "timestamp": current_updated_at or time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "type": "initialization",
                                        "message": status_data.get("message", "Initializing...")
                                    }
                                    yield f"event: log\n"
                                    yield f"data: {json.dumps(log_entry)}\n\n"
                                
                                elif current_status == "running" or current_status == "training":
                                    # Active training
                                    log_entry = {
                                        "timestamp": current_updated_at or time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "type": "training",
                                        "message": status_data.get("message", "Training in progress"),
                                        "step": status_data.get("current_step"),
                                        "total_steps": status_data.get("total_steps"),
                                        "epoch": status_data.get("current_epoch"),
                                        "total_epochs": status_data.get("total_epochs"),
                                        "loss": status_data.get("loss"),
                                        "learning_rate": status_data.get("learning_rate"),
                                        "progress_percentage": status_data.get("progress_percentage")
                                    }
                                    
                                    yield f"event: log\n"
                                    yield f"data: {json.dumps(log_entry)}\n\n"
                                
                                elif current_status == "saving":
                                    # Model saving phase
                                    log_entry = {
                                        "timestamp": current_updated_at or time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "type": "saving",
                                        "message": status_data.get("message", "Saving model...")
                                    }
                                    yield f"event: log\n"
                                    yield f"data: {json.dumps(log_entry)}\n\n"
                                
                                elif current_status in ["completed", "cancelled", "error"]:
                                    # Final states - send final message and exit
                                    final_status = current_status
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
                                
                                # Always send status update
                                yield f"event: status\n"
                                yield f"data: {json.dumps(status_data)}\n\n"
                            
                        except (json.JSONDecodeError, IOError) as e:
                            self.logger.warning(f"Error reading status file: {e}")
                    else:
                        # Status file doesn't exist - check if process is still running
                        process_status = self.process_manager.get_process_status(task_id)
                        if process_status is None or process_status == "finished":
                            # Process finished but no status file - likely error
                            yield f"event: error\n"
                            yield f"data: {json.dumps({'error': 'Training process ended unexpectedly', 'task_id': task_id})}\n\n"
                            return
                    
                    # Send heartbeat every 30 seconds
                    current_time = time.time()
                    if current_time - last_heartbeat > 30:
                        yield f"event: heartbeat\n"
                        yield f"data: {json.dumps({'timestamp': current_time})}\n\n"
                        last_heartbeat = current_time
                    
                    await asyncio.sleep(2.0)  # Check every 2 seconds for responsive updates
                    
                except Exception as e:
                    self.logger.error(f"Error in SSE stream: {e}")
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


# Create global controller instance
training_controller = TrainingController()