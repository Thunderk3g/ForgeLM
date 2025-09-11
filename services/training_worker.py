"""
Training Worker Process

This module runs ML training in a completely separate process to avoid blocking
the FastAPI main process. Uses file-based IPC for status communication and
proper CUDA context management.
"""

import os
import sys
import json
import time
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional
import signal

# Set multiprocessing start method to spawn for CUDA compatibility
mp.set_start_method('spawn', force=True)

# Environment optimization for worker process
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

def setup_worker_environment():
    """Setup optimal environment for worker process"""
    # Configure CUDA for this specific process
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        # For multiple training sessions, distribute GPU usage
        # Use current process ID to determine GPU assignment if multiple GPUs available
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                # Assign GPU based on process ID for better distribution
                gpu_id = os.getpid() % torch.cuda.device_count()
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                print(f"Worker {os.getpid()}: Assigned to GPU {gpu_id}")
            else:
                # Single GPU or no GPU - let CUDA auto-detect
                pass
        except ImportError:
            # PyTorch not available, skip GPU assignment
            pass
    
    # Optimize memory settings for multiple concurrent processes
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024,expandable_segments:True"
    os.environ["OMP_NUM_THREADS"] = "1"  # Reduce CPU thread contention
    os.environ["MKL_NUM_THREADS"] = "1"  # Reduce Intel MKL thread contention


class TrainingWorker:
    """Worker process for running ML training in complete isolation"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status_file = Path("logs") / f"{task_id}_status.json"
        self.should_stop = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Ensure logs directory exists
        self.status_file.parent.mkdir(exist_ok=True)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"Worker {self.task_id}: Received signal {signum}, stopping training...")
        self.should_stop = True
        self._update_status("cancelled", "Training cancelled due to shutdown signal")
    
    def _update_status(self, status: str, message: str, **kwargs):
        """Update status file with current progress"""
        try:
            # Read existing status if available
            status_data = {}
            if self.status_file.exists():
                try:
                    with open(self.status_file, 'r') as f:
                        status_data = json.load(f)
                except:
                    status_data = {}
            
            # Update with new information
            status_data.update({
                "task_id": self.task_id,
                "status": status,
                "message": message,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                **kwargs
            })
            
            # Write atomically
            temp_file = self.status_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(status_data, f, indent=2)
            temp_file.replace(self.status_file)
            
        except Exception as e:
            print(f"Worker {self.task_id}: Error updating status: {e}")
    
    def run_training(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run training in isolated worker process
        
        Args:
            request_data: Serialized training request data
            
        Returns:
            Training result dictionary
        """
        try:
            setup_worker_environment()
            
            self._update_status("initializing", "Setting up training environment...")
            
            # Import ML libraries only in worker process (after environment setup)
            from models.schemas import FineTuneRequest
            from services.training_service_core import TrainingCore
            
            # Reconstruct request object
            request = FineTuneRequest(**request_data)
            
            # Create isolated training core (no global state)
            training_core = TrainingCore(task_id=self.task_id, worker_mode=True)
            
            # Register cancellation check callback
            training_core.set_cancellation_callback(lambda: self.should_stop)
            
            # Start training
            self._update_status("running", "Training started in worker process...")
            
            result = training_core.execute_training(request)
            
            if not self.should_stop:
                # Extract status and message from result to avoid duplicate arguments
                result_copy = result.copy()
                final_status = result_copy.pop("status", "completed")
                final_message = result_copy.pop("message", "Training completed successfully")
                self._update_status(final_status, final_message, **result_copy)
            
            return result
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            traceback_str = traceback.format_exc()
            
            print(f"Worker {self.task_id}: {error_msg}")
            print(f"Worker {self.task_id}: Traceback:\n{traceback_str}")
            
            self._update_status(
                "error", 
                error_msg,
                error_details=traceback_str
            )
            
            return {"status": "error", "error": error_msg}


def worker_main(task_id: str, request_data: Dict[str, Any], worker_ready_event: mp.Event):
    """
    Main entry point for worker process
    
    Args:
        task_id: Unique task identifier
        request_data: Serialized training request
        worker_ready_event: Event to signal worker is ready
    """
    try:
        print(f"Worker {task_id}: Starting in separate process (PID: {os.getpid()})")
        
        # Signal that worker is ready
        worker_ready_event.set()
        
        # Create and run worker
        worker = TrainingWorker(task_id)
        result = worker.run_training(request_data)
        
        print(f"Worker {task_id}: Training completed with result: {result.get('status', 'unknown')}")
        
        return result
        
    except Exception as e:
        error_msg = f"Worker process failed: {str(e)}"
        print(f"Worker {task_id}: {error_msg}")
        
        # Try to update status file even on catastrophic failure
        try:
            status_file = Path("logs") / f"{task_id}_status.json"
            status_data = {
                "task_id": task_id,
                "status": "error", 
                "message": error_msg,
                "error_details": traceback.format_exc(),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        except:
            pass  # Don't let status update failure mask the original error
        
        return {"status": "error", "error": error_msg}


class ProcessManager:
    """Manages training worker processes"""
    
    def __init__(self):
        self.active_processes: Dict[str, mp.Process] = {}
        self.process_lock = mp.Lock()
    
    def start_training_process(self, task_id: str, request_data: Dict[str, Any]) -> bool:
        """
        Start a new training process
        
        Args:
            task_id: Unique task identifier  
            request_data: Serialized training request data
            
        Returns:
            True if process started successfully, False otherwise
        """
        try:
            with self.process_lock:
                if task_id in self.active_processes:
                    print(f"ProcessManager: Training {task_id} already running")
                    return False
                
                # Create ready event for process synchronization
                worker_ready_event = mp.Event()
                
                # Create and start process
                process = mp.Process(
                    target=worker_main,
                    args=(task_id, request_data, worker_ready_event),
                    name=f"training-{task_id[:8]}"
                )
                
                process.start()
                
                # Wait for worker to be ready (with timeout)
                if worker_ready_event.wait(timeout=30):
                    self.active_processes[task_id] = process
                    print(f"ProcessManager: Started training process {task_id} (PID: {process.pid})")
                    return True
                else:
                    print(f"ProcessManager: Worker {task_id} failed to start within timeout")
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                    return False
                    
        except Exception as e:
            print(f"ProcessManager: Error starting process for {task_id}: {e}")
            return False
    
    def get_process_status(self, task_id: str) -> Optional[str]:
        """Get the status of a training process"""
        with self.process_lock:
            process = self.active_processes.get(task_id)
            if process is None:
                return None
            
            if process.is_alive():
                return "running"
            else:
                # Process finished, clean it up
                process.join(timeout=1)
                del self.active_processes[task_id]
                return "finished"
    
    def cancel_training(self, task_id: str) -> bool:
        """Cancel a running training process"""
        try:
            with self.process_lock:
                process = self.active_processes.get(task_id)
                if process is None or not process.is_alive():
                    return False
                
                print(f"ProcessManager: Cancelling training process {task_id}")
                
                # First try graceful shutdown
                process.terminate()
                
                # Wait for graceful shutdown
                process.join(timeout=10)
                
                # Force kill if still alive
                if process.is_alive():
                    print(f"ProcessManager: Force killing process {task_id}")
                    process.kill()
                    process.join(timeout=5)
                
                # Clean up
                del self.active_processes[task_id]
                
                return True
                
        except Exception as e:
            print(f"ProcessManager: Error cancelling process {task_id}: {e}")
            return False
    
    def cleanup_all_processes(self):
        """Clean up all active processes"""
        print("ProcessManager: Cleaning up all active processes...")
        
        with self.process_lock:
            for task_id, process in list(self.active_processes.items()):
                try:
                    if process.is_alive():
                        print(f"ProcessManager: Terminating process {task_id}")
                        process.terminate()
                        process.join(timeout=5)
                        
                        if process.is_alive():
                            print(f"ProcessManager: Force killing process {task_id}")
                            process.kill()
                            process.join(timeout=2)
                except Exception as e:
                    print(f"ProcessManager: Error cleaning up process {task_id}: {e}")
            
            self.active_processes.clear()
    
    def get_active_processes(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active processes"""
        active = {}
        
        with self.process_lock:
            for task_id, process in list(self.active_processes.items()):
                if process.is_alive():
                    active[task_id] = {
                        "task_id": task_id,
                        "pid": process.pid,
                        "name": process.name,
                        "status": "running"
                    }
                else:
                    # Clean up finished processes
                    try:
                        process.join(timeout=0.1)
                        del self.active_processes[task_id]
                    except:
                        pass
        
        return active


# Global process manager instance
process_manager = ProcessManager()

# Cleanup on module exit
import atexit
atexit.register(process_manager.cleanup_all_processes)