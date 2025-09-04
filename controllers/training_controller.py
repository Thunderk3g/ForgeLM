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
from services.training_service import TrainingService, task_log_queues, task_completion_flags
from utils.logging import get_logger


class TrainingController:
    """Controller for training and fine-tuning endpoints"""
    
    def __init__(self, training_service: TrainingService = None):
        self.logger = get_logger(__name__)
        self.training_service = training_service or TrainingService()
        
        # Global variable to track background tasks (maintain compatibility)
        self.training_status = {}
    
    async def run_fine_tuning(self, request: FineTuneRequest, task_id: str):
        """Run fine-tuning in the background"""
        
        self.training_status[task_id] = {
            "status": "running", 
            "message": "Fine-tuning in progress..."
        }
        
        # Initialize the task in the log queue system
        import queue
        if task_id not in task_log_queues:
            task_log_queues[task_id] = queue.Queue()
        
        task_completion_flags[task_id] = False
        
        try:
            # Use the training service to run fine-tuning
            result = self.training_service.fine_tune(request, task_id)
            
            self.training_status[task_id] = {
                "status": "completed",
                "message": "Fine-tuning completed successfully",
                "task_id": result.get("task_id", task_id)
            }
            
        except Exception as e:
            self.training_status[task_id] = {
                "status": "error",
                "message": f"Fine-tuning failed: {str(e)}",
                "task_id": task_id
            }
            # Mark as complete even on error so SSE can close
            task_completion_flags[task_id] = True
    
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
            active_sessions = self.training_service.get_active_sessions()
            
            # Enhance session data with current status from status files
            enhanced_sessions = {}
            for task_id, session in active_sessions.items():
                enhanced_sessions[task_id] = session.copy()
                
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
            # Try to cancel using training service
            success = self.training_service.cancel_training(task_id)
            
            if not success:
                # Fallback to legacy cancellation logic
                success = await self._legacy_cancel_training(task_id)
            
            if success:
                return JSONResponse(content={
                    "status": "cancellation_requested",
                    "task_id": task_id,
                    "message": "Training cancellation requested. Training will stop at the next safe checkpoint.",
                    "details": "The training process will complete the current step/epoch before stopping to ensure data integrity."
                })
            else:
                raise HTTPException(status_code=404, detail=f"Training task {task_id} not found")
                
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to cancel training: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to cancel training: {str(e)}")
    
    async def _legacy_cancel_training(self, task_id: str) -> bool:
        """Legacy training cancellation logic for backward compatibility"""
        try:
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
                return False
            
            # Check if task is in a cancellable state
            session = active_sessions[task_id]
            if session.get("status") not in ["running", "started"]:
                return False
            
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
                self.logger.warning(f"Error updating status file: {e}")
            
            # Update global training status
            self.training_status[task_id] = {
                "status": "cancelled",
                "message": "Training cancelled by user",
                "task_id": task_id
            }
            
            # Set completion flag to stop SSE streaming
            task_completion_flags[task_id] = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in legacy cancel training: {e}")
            return False
    
    async def stream_training_logs(self, task_id: str) -> StreamingResponse:
        """Stream training logs in real-time using Server-Sent Events"""
        
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
                            self.logger.warning(f"Error reading status file: {e}")
                    
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
                        except:
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