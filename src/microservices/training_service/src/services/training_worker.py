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
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import signal
import asyncio
from concurrent.futures import ThreadPoolExecutor

# FIXED: Don't force spawn method in containers - causes restart loops
# mp.set_start_method('spawn', force=True)
# Use threading instead of multiprocessing for container environments

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


# REMOVED: worker_main function - not used since switching to threading
# This function was causing NameError: name 'mp' is not defined
# The ProcessManager class now handles worker threads directly

class ProcessManager:
    """Manages training worker threads (FIXED: uses threading instead of multiprocessing)"""
    
    def __init__(self):
        self.active_threads: Dict[str, threading.Thread] = {}
        self.thread_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="training-worker")
    
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
            with self.thread_lock:
                if task_id in self.active_threads:
                    print(f"ProcessManager: Training {task_id} already running")
                    return False
                
                # FIXED: Create and start training thread instead of process
                def thread_wrapper():
                    try:
                        worker = TrainingWorker(task_id)
                        return worker.run_training(request_data)
                    except Exception as e:
                        print(f"Training thread {task_id} failed: {e}")
                        return {"status": "error", "error": str(e)}
                
                thread = threading.Thread(
                    target=thread_wrapper,
                    name=f"training-{task_id[:8]}",
                    daemon=False  # Don't make daemon - we want proper cleanup
                )
                
                thread.start()
                self.active_threads[task_id] = thread
                
                print(f"ProcessManager: Started training thread {task_id}")
                return True
                    
        except Exception as e:
            print(f"ProcessManager: Error starting process for {task_id}: {e}")
            return False
    
    def get_process_status(self, task_id: str) -> Optional[str]:
        """Get the status of a training thread"""
        with self.thread_lock:
            thread = self.active_threads.get(task_id)
            if thread is None:
                return None
            
            if thread.is_alive():
                return "running"
            else:
                # Thread finished, clean it up
                thread.join(timeout=1)
                del self.active_threads[task_id]
                return "finished"
    
    def cancel_training(self, task_id: str) -> bool:
        """Cancel a running training thread"""
        try:
            with self.thread_lock:
                thread = self.active_threads.get(task_id)
                if thread is None or not thread.is_alive():
                    return False
                
                print(f"ProcessManager: Cancelling training thread {task_id}")
                
                # Note: Threads can't be forcefully terminated like processes
                # We rely on the TrainingWorker's should_stop flag for cancellation
                # The thread will check this flag and exit gracefully
                
                # Clean up
                del self.active_threads[task_id]
                
                return True
                
        except Exception as e:
            print(f"ProcessManager: Error cancelling process {task_id}: {e}")
            return False
    
    def cleanup_all_processes(self):
        """Clean up all active threads"""
        print("ProcessManager: Cleaning up all active threads...")
        
        with self.thread_lock:
            for task_id, thread in list(self.active_threads.items()):
                try:
                    if thread.is_alive():
                        print(f"ProcessManager: Waiting for thread {task_id} to finish...")
                        thread.join(timeout=5)
                        
                        if thread.is_alive():
                            print(f"ProcessManager: Thread {task_id} still running after timeout")
                except Exception as e:
                    print(f"ProcessManager: Error cleaning up thread {task_id}: {e}")
            
            self.active_threads.clear()
    
    def get_active_processes(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active threads"""
        active = {}
        
        with self.thread_lock:
            for task_id, thread in list(self.active_threads.items()):
                if thread.is_alive():
                    active[task_id] = {
                        "task_id": task_id,
                        "thread_id": thread.ident,
                        "name": thread.name,
                        "status": "running"
                    }
                else:
                    # Clean up finished threads
                    try:
                        thread.join(timeout=0.1)
                        del self.active_threads[task_id]
                    except:
                        pass
        
        return active
    
    def get_active_job_count(self) -> int:
        """Get the number of active training jobs for health checks"""
        try:
            with self.thread_lock:
                # Clean up finished threads first
                finished_tasks = []
                for task_id, thread in list(self.active_threads.items()):
                    if not thread.is_alive():
                        finished_tasks.append(task_id)
                        try:
                            thread.join(timeout=0.1)
                        except:
                            pass
                
                # Remove finished threads
                for task_id in finished_tasks:
                    del self.active_threads[task_id]
                
                return len(self.active_threads)
        except Exception as e:
            print(f"Error getting active job count: {e}")
            return 0


# Global process manager instance
process_manager = ProcessManager()

# Cleanup on module exit
import atexit
atexit.register(process_manager.cleanup_all_processes)