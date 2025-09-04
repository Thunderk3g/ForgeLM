# CRITICAL: Import Unsloth FIRST before any transformers imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Import Unsloth immediately after environment setup
try:
    import unsloth
    from unsloth import FastLanguageModel, is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
    print(f"🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.")
    print(f"🦥 Unsloth Zoo will now patch everything to make training faster!")
    print(f"Unsloth imported successfully (v{unsloth.__version__})")
except (ImportError, NotImplementedError) as e:
    UNSLOTH_AVAILABLE = False
    print(f"WARNING: Unsloth not available ({e})")

# Kubernetes and container optimization settings
KUBERNETES_MODE = os.environ.get("KUBERNETES_MODE", "false").lower() == "true"
MAX_MEMORY_USAGE_GB = float(os.environ.get("MAX_MEMORY_USAGE_GB", "6"))
FORCE_CPU_OFFLOAD = os.environ.get("FORCE_CPU_OFFLOAD", "false").lower() == "true"
CONTAINER_MEMORY_LIMIT_GB = float(os.environ.get("CONTAINER_MEMORY_LIMIT_GB", "8"))

print(f"Container configuration:")
print(f"  Kubernetes mode: {KUBERNETES_MODE}")
print(f"  Max memory usage: {MAX_MEMORY_USAGE_GB}GB")
print(f"  Force CPU offload: {FORCE_CPU_OFFLOAD}")
print(f"  Container memory limit: {CONTAINER_MEMORY_LIMIT_GB}GB")

import queue
import time
import signal
import atexit
import threading
import shutil
import gc
from collections import defaultdict

# Global cleanup registry for Kubernetes graceful shutdown
_cleanup_functions = []

def register_cleanup(func):
    """Register a function to be called on shutdown."""
    _cleanup_functions.append(func)

def cleanup_resources():
    """Clean up resources on shutdown."""
    print("Performing resource cleanup...")
    for func in _cleanup_functions:
        try:
            func()
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    # Clear CUDA cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared")
    except Exception as e:
        print(f"Error clearing CUDA cache: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals in Kubernetes."""
    print(f"Received signal {signum}, initiating graceful shutdown...")
    cleanup_resources()
    exit(0)

# Register signal handlers for Kubernetes graceful shutdown
if KUBERNETES_MODE:
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup_resources)
    print("Kubernetes graceful shutdown handlers registered")

# Unsloth already imported at top of file for optimal performance

import torch
import gc
import json
import logging
import uuid
import re
import inspect
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datasets import load_from_disk
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainerCallback
from trl import SFTTrainer
from pydantic import BaseModel

# Optional imports for standard model setup
try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("WARNING: PEFT not available, standard model setup will not work")

# Import transformers for version checking
import transformers

logger = logging.getLogger(__name__)

# Global log queue system for real-time streaming
task_log_queues = defaultdict(queue.Queue)
task_completion_flags = defaultdict(bool)

# Global training cancellation system
task_cancellation_events = defaultdict(threading.Event)
training_managers = {}

# Active training sessions storage
ACTIVE_SESSIONS_FILE = Path("logs") / "active_sessions.json"

def load_active_sessions():
    """Load active training sessions from persistent storage."""
    try:
        if ACTIVE_SESSIONS_FILE.exists():
            with open(ACTIVE_SESSIONS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading active sessions: {e}")
    return {}

def save_active_sessions(sessions):
    """Save active training sessions to persistent storage."""
    try:
        ACTIVE_SESSIONS_FILE.parent.mkdir(exist_ok=True)
        with open(ACTIVE_SESSIONS_FILE, "w") as f:
            json.dump(sessions, f, indent=2)
    except Exception as e:
        print(f"Error saving active sessions: {e}")

def register_active_session(task_id: str, request_data: dict):
    """Register a new active training session."""
    sessions = load_active_sessions()
    sessions[task_id] = {
        "task_id": task_id,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "request_data": request_data,
        "last_activity": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    save_active_sessions(sessions)

def update_session_activity(task_id: str):
    """Update last activity time for a session."""
    sessions = load_active_sessions()
    if task_id in sessions:
        sessions[task_id]["last_activity"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        save_active_sessions(sessions)

def remove_active_session(task_id: str):
    """Remove a completed/failed training session."""
    sessions = load_active_sessions()
    if task_id in sessions:
        del sessions[task_id]
        save_active_sessions(sessions)

def get_active_sessions():
    """Get all active training sessions."""
    sessions = load_active_sessions()
    # Clean up stale sessions (older than 24 hours without activity)
    current_time = time.time()
    stale_sessions = []
    
    for task_id, session in sessions.items():
        try:
            last_activity = time.strptime(session["last_activity"], "%Y-%m-%d %H:%M:%S")
            last_activity_timestamp = time.mktime(last_activity)
            if current_time - last_activity_timestamp > 24 * 60 * 60:  # 24 hours
                stale_sessions.append(task_id)
        except Exception:
            stale_sessions.append(task_id)  # Remove sessions with invalid timestamps
    
    for task_id in stale_sessions:
        del sessions[task_id]
    
    if stale_sessions:
        save_active_sessions(sessions)
    
    return sessions

print("Environment setup complete:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA devices: {torch.cuda.device_count()}")
    print(f"   Primary GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   CUDA not available - Unsloth optimizations will be used in Docker/NVIDIA environments")

class FineTuneRequest(BaseModel):
    base_model_dir: str = "/app/models/base/Qwen_Qwen2.5-3B"
    dataset_dir: str = "/app/data/prepared"
    output_dir: str = "/app/models/finetuned"
    ft_type: str = "sft"  # unsupervised, sft, instruction
    peft_method: str = "qlora"
    alignment: str = "none"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
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
    
    def save(self):
        """Save status to file for persistence"""
        status_file = Path("logs") / f"{self.task_id}_status.json"
        status_file.parent.mkdir(exist_ok=True)
        with open(status_file, "w") as f:
            json.dump(self.dict(), f, indent=2)
    
    def update(self, status: str = None, progress: str = None, **kwargs):
        """Update status fields"""
        import time
        if status:
            self.status = status
        if progress:
            self.progress = progress
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # Always update the timestamp
        self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.save()


class StreamingLoggingCallback(TrainerCallback):
    """Custom callback to capture training logs, update status, and stream logs via queue."""
    
    def __init__(self, status: TrainingStatus):
        self.status = status
        self.task_id = status.task_id
        
    def _add_log_to_queue(self, message, log_type: str = "info"):
        """Add a log message to the streaming queue."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # Handle both simple string messages and structured log objects
        if isinstance(message, dict) and "message" in message:
            # Structured log with metrics
            formatted_log = {
                "timestamp": timestamp,
                "type": log_type,
                "message": message["message"],
                "step": getattr(self.status, 'current_step', None),
                "epoch": getattr(self.status, 'current_epoch', None),
                "loss": getattr(self.status, 'loss', None),
                "learning_rate": getattr(self.status, 'learning_rate', None),
                "progress_percentage": getattr(self.status, 'progress_percentage', None),
                "metrics": message.get("metrics", {})
            }
        else:
            # Simple string message
            formatted_log = {
                "timestamp": timestamp,
                "type": log_type,
                "message": str(message),
                "step": getattr(self.status, 'current_step', None),
                "epoch": getattr(self.status, 'current_epoch', None),
                "loss": getattr(self.status, 'loss', None),
                "learning_rate": getattr(self.status, 'learning_rate', None),
                "progress_percentage": getattr(self.status, 'progress_percentage', None)
            }
        
        try:
            task_log_queues[self.task_id].put(formatted_log)
        except Exception as e:
            print(f"Failed to add log to queue: {e}")
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Called when training begins."""
        self._add_log_to_queue("Training started successfully", "info")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        epoch = int(state.epoch) if hasattr(state, 'epoch') else state.epoch
        self._add_log_to_queue(f"Starting epoch {epoch}", "info")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs something."""
        if logs is not None:
            # Update status with training metrics
            updates = {}
            
            # Extract training metrics
            if 'loss' in logs:
                updates['loss'] = float(logs['loss'])
            if 'learning_rate' in logs:
                updates['learning_rate'] = float(logs['learning_rate'])
            if 'epoch' in logs:
                updates['current_epoch'] = float(logs['epoch'])
                updates['total_epochs'] = int(args.num_train_epochs)
            
            # Handle step progress - check multiple possible sources
            current_step = None
            if 'step' in logs:
                current_step = int(logs['step'])
            elif hasattr(state, 'global_step'):
                current_step = int(state.global_step)
            
            if current_step is not None:
                updates['current_step'] = current_step
                
            # Calculate total steps properly
            total_steps = None
            if hasattr(state, 'max_steps') and state.max_steps > 0:
                total_steps = int(state.max_steps)
            elif hasattr(args, 'max_steps') and args.max_steps and args.max_steps > 0:
                total_steps = int(args.max_steps)
            elif hasattr(state, 'num_train_epochs') and hasattr(state, 'train_dataloader'):
                # Calculate from epochs and dataset size if max_steps not set
                try:
                    steps_per_epoch = len(state.train_dataloader)
                    total_steps = int(steps_per_epoch * args.num_train_epochs)
                except:
                    pass
                    
            if total_steps is not None and total_steps > 0:
                updates['total_steps'] = total_steps
            
            # Calculate progress percentage
            progress_percentage = 0.0
            if current_step and updates.get('total_steps'):
                progress_percentage = min(100.0, (current_step / updates['total_steps']) * 100)
            elif updates.get('current_epoch') and updates.get('total_epochs'):
                progress_percentage = min(100.0, (updates['current_epoch'] / updates['total_epochs']) * 100)
                
            updates['progress_percentage'] = round(progress_percentage, 2)
            
            # Create comprehensive progress message
            progress_parts = []
            if current_step and updates.get('total_steps'):
                progress_parts.append(f"Step: {current_step}/{updates['total_steps']}")
            if 'current_epoch' in updates and 'total_epochs' in updates:
                progress_parts.append(f"Epoch: {updates['current_epoch']:.2f}/{updates['total_epochs']}")
            if 'loss' in logs:
                progress_parts.append(f"Loss: {logs['loss']:.4f}")
            if 'learning_rate' in logs:
                progress_parts.append(f"LR: {logs['learning_rate']:.2e}")
            if progress_percentage > 0:
                progress_parts.append(f"Progress: {progress_percentage:.1f}%")
            
            progress_msg = " | ".join(progress_parts) if progress_parts else "Training in progress..."
            updates['progress'] = progress_msg
            
            # Update status (save to file)
            self.status.update(**updates)
            
            # Create structured log entry for streaming
            structured_log = {
                "message": progress_msg,
                "metrics": {
                    "loss": updates.get('loss'),
                    "learning_rate": updates.get('learning_rate'),
                    "current_step": updates.get('current_step'),
                    "total_steps": updates.get('total_steps'),
                    "current_epoch": updates.get('current_epoch'),
                    "total_epochs": updates.get('total_epochs'),
                    "progress_percentage": updates.get('progress_percentage')
                }
            }
            
            # Add to streaming queue with metrics
            self._add_log_to_queue(structured_log, "training")
            
            # Print to console for Docker logs
            print(f"Training Update: {progress_msg}")
            
            # Also log structured metrics for debugging and update session activity
            if current_step and current_step % 10 == 0:  # Every 10 steps
                metrics_msg = f"Metrics - Step: {current_step}, Loss: {updates.get('loss', 'N/A')}, Progress: {progress_percentage:.1f}%"
                logger.info(metrics_msg)
                print(f"DEBUG: {metrics_msg}")
                
                # Update session activity every 10 steps
                update_session_activity(self.task_id)
            
    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends."""
        self._add_log_to_queue("Training completed successfully", "success")
        # Mark completion for SSE endpoint
        task_completion_flags[self.task_id] = True


class CancellationCallback(TrainerCallback):
    """Callback to handle safe training cancellation at appropriate checkpoints."""
    
    def __init__(self, task_id: str, cancellation_event: threading.Event, status: TrainingStatus):
        self.task_id = task_id
        self.cancellation_event = cancellation_event
        self.status = status
        self.last_checkpoint_step = None
        
    def _log_cancellation_message(self, message: str):
        """Add cancellation-related message to log queue."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        formatted_log = {
            "timestamp": timestamp,
            "type": "info",
            "message": message,
            "step": getattr(self.status, 'current_step', None),
            "epoch": getattr(self.status, 'current_epoch', None)
        }
        try:
            task_log_queues[self.task_id].put(formatted_log)
        except Exception as e:
            print(f"Failed to add cancellation log: {e}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Check for cancellation after each complete training step (safest cancellation point)."""
        if self.cancellation_event.is_set():
            self._log_cancellation_message(f"Cancellation requested - stopping after step {state.global_step}")
            
            # Update status to indicate cancellation in progress
            self.status.update(
                status="cancelling", 
                progress=f"Cancelling training after step {state.global_step}",
                message="Training cancellation in progress..."
            )
            
            control.should_training_stop = True
            return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Preferred cancellation point - after complete epoch."""
        if self.cancellation_event.is_set():
            epoch = int(state.epoch) if hasattr(state, 'epoch') else state.epoch
            self._log_cancellation_message(f"Cancellation requested - stopping after epoch {epoch}")
            
            # Update status
            self.status.update(
                status="cancelling",
                progress=f"Cancelling training after epoch {epoch}",
                message="Training cancellation in progress..."
            )
            
            control.should_training_stop = True
            return control
    
    def on_save(self, args, state, control, **kwargs):
        """Ensure checkpoint completion before cancellation if requested during save."""
        if self.cancellation_event.is_set():
            self._log_cancellation_message("Completing checkpoint before cancellation...")
            self.last_checkpoint_step = state.global_step
            # Let the checkpoint complete, then we'll stop on next step_end


class TrainingManager:
    """Manages training lifecycle, cancellation, and resource cleanup."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.cancellation_event = threading.Event()
        self.cleanup_lock = threading.Lock()
        self.trainer = None
        self.model = None
        self.tokenizer = None
        self.status = None
        self.is_cleaned_up = False
        
        # Register this manager globally
        training_managers[task_id] = self
        task_cancellation_events[task_id] = self.cancellation_event
        
    def register_resources(self, trainer, model, tokenizer, status):
        """Register training resources for cleanup."""
        self.trainer = trainer
        self.model = model
        self.tokenizer = tokenizer
        self.status = status
        
    def request_cancellation(self):
        """Thread-safe cancellation request."""
        print(f"Cancellation requested for task {self.task_id}")
        self.cancellation_event.set()
        
        # Update session status
        if self.status:
            self.status.update(
                status="cancelling",
                progress="Cancellation requested - stopping at next safe checkpoint",
                message="Training cancellation requested by user"
            )
        
        # Update active sessions
        sessions = load_active_sessions()
        if self.task_id in sessions:
            sessions[self.task_id]["status"] = "cancelling"
            sessions[self.task_id]["last_activity"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_active_sessions(sessions)
    
    def safe_model_cleanup(self):
        """Perform safe model and resource cleanup following Unsloth best practices."""
        if self.is_cleaned_up:
            return
            
        with self.cleanup_lock:
            if self.is_cleaned_up:  # Double-check after acquiring lock
                return
                
            try:
                print(f"Starting resource cleanup for task {self.task_id}")
                
                # Step 1: Disable gradient computation
                if self.model and hasattr(self.model, 'eval'):
                    self.model.eval()
                    print("Model set to eval mode")
                
                # Step 2: Clear any pending CUDA operations
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    print("CUDA operations synchronized")
                
                # Step 3: Save emergency checkpoint if cancellation was requested
                if self.cancellation_event.is_set() and self.model and hasattr(self.model, 'save_pretrained'):
                    self._save_atomic_checkpoint(emergency=True)
                
                # Step 4: Handle PEFT adapter cleanup (Unsloth-specific)
                if self.model:
                    try:
                        # Disable adapters if using PEFT/LoRA
                        if hasattr(self.model, 'disable_adapters'):
                            self.model.disable_adapters()
                            print("PEFT adapters disabled")
                        
                        # Move model to CPU before deletion (frees GPU memory)
                        if hasattr(self.model, 'cpu'):
                            self.model.cpu()
                            print("Model moved to CPU")
                            
                    except Exception as e:
                        print(f"Error during model cleanup: {e}")
                
                # Step 5: Delete references
                if self.model:
                    del self.model
                    self.model = None
                    print("Model reference deleted")
                
                if self.tokenizer:
                    del self.tokenizer
                    self.tokenizer = None
                    print("Tokenizer reference deleted")
                
                if self.trainer:
                    del self.trainer
                    self.trainer = None
                    print("Trainer reference deleted")
                
                # Step 6: Clear CUDA cache (MUST be after deletions)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("CUDA cache cleared and synchronized")
                
                # Step 7: Force garbage collection
                gc.collect()
                print("Garbage collection completed")
                
                self.is_cleaned_up = True
                print(f"Resource cleanup completed for task {self.task_id}")
                
            except Exception as e:
                print(f"Error during resource cleanup for task {self.task_id}: {e}")
                # Force aggressive cleanup on error
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                except:
                    pass
                self.is_cleaned_up = True
    
    def safe_train_with_cancellation(self, trainer, model, tokenizer, status):
        """Execute training with cancellation support and resource management."""
        self.register_resources(trainer, model, tokenizer, status)
        
        try:
            # Clean up any existing corrupted checkpoints
            self._cleanup_corrupted_checkpoints()
            
            # Add cancellation and atomic checkpoint callbacks
            cancellation_callback = CancellationCallback(self.task_id, self.cancellation_event, status)
            atomic_checkpoint_callback = AtomicCheckpointCallback(self.task_id, self)
            
            trainer.add_callback(cancellation_callback)
            trainer.add_callback(atomic_checkpoint_callback)
            
            print(f"Starting training with cancellation and checkpoint safety for task {self.task_id}")
            
            # Start training
            result = trainer.train()
            
            # Check if training was cancelled
            if self.cancellation_event.is_set():
                print(f"Training was cancelled for task {self.task_id}")
                status.update(
                    status="cancelled",
                    progress="Training cancelled by user",
                    message="Training was successfully cancelled"
                )
                
                # Update active sessions
                sessions = load_active_sessions()
                if self.task_id in sessions:
                    sessions[self.task_id]["status"] = "cancelled"
                    sessions[self.task_id]["last_activity"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    save_active_sessions(sessions)
                
                # Mark completion for SSE
                task_completion_flags[self.task_id] = True
                
            return result
            
        except KeyboardInterrupt:
            print(f"Training interrupted by user for task {self.task_id}")
            self._handle_emergency_stop("interrupted")
            raise
        except Exception as e:
            print(f"Training error for task {self.task_id}: {e}")
            self._handle_emergency_stop("error", str(e))
            raise
        finally:
            # Always cleanup resources
            self.safe_model_cleanup()
            
            # Remove from global managers
            if self.task_id in training_managers:
                del training_managers[self.task_id]
    
    def _handle_emergency_stop(self, stop_reason: str, error_msg: str = None):
        """Handle emergency training stop scenarios."""
        if self.status:
            status_msg = f"Training {stop_reason}"
            if error_msg:
                status_msg += f": {error_msg}"
                
            self.status.update(
                status=stop_reason,
                progress=f"Training {stop_reason}",
                message=status_msg
            )
        
        # Update active sessions
        sessions = load_active_sessions()
        if self.task_id in sessions:
            sessions[self.task_id]["status"] = stop_reason
            sessions[self.task_id]["last_activity"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_active_sessions(sessions)
        
        # Mark completion for SSE
        task_completion_flags[self.task_id] = True
    
    def _save_atomic_checkpoint(self, emergency: bool = False):
        """Save checkpoint atomically to prevent corruption."""
        if not self.model or not hasattr(self.model, 'save_pretrained'):
            return
            
        try:
            checkpoint_prefix = "emergency_checkpoint" if emergency else "checkpoint"
            checkpoint_dir = Path("models") / f"{checkpoint_prefix}_{self.task_id}"
            temp_dir = Path("models") / f"{checkpoint_prefix}_{self.task_id}_temp"
            backup_dir = Path("models") / f"{checkpoint_prefix}_{self.task_id}_backup"
            
            # Ensure parent directory exists
            checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
            
            # Remove temp directory if it exists from previous failed attempt
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            print(f"Saving atomic checkpoint to {checkpoint_dir}")
            
            # Step 1: Save to temporary directory first
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(temp_dir))
            
            if self.tokenizer:
                self.tokenizer.save_pretrained(str(temp_dir))
            
            # Step 2: Atomic move operation (OS-level atomic on most filesystems)
            if checkpoint_dir.exists():
                # Backup existing checkpoint
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                shutil.move(str(checkpoint_dir), str(backup_dir))
            
            # Atomic move from temp to final location
            shutil.move(str(temp_dir), str(checkpoint_dir))
            
            print(f"Atomic checkpoint completed: {checkpoint_dir}")
            
            # Clean up backup after successful move
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            
        except Exception as e:
            print(f"Failed to save atomic checkpoint: {e}")
            # Clean up temp directory on failure
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except:
                pass
    
    def _cleanup_corrupted_checkpoints(self):
        """Clean up any corrupted or incomplete checkpoints."""
        try:
            models_dir = Path("models")
            if not models_dir.exists():
                return
                
            # Find and remove temporary checkpoint directories
            temp_patterns = [
                f"checkpoint_{self.task_id}_temp",
                f"emergency_checkpoint_{self.task_id}_temp"
            ]
            
            for pattern in temp_patterns:
                temp_dirs = list(models_dir.glob(pattern))
                for temp_dir in temp_dirs:
                    if temp_dir.is_dir():
                        print(f"Cleaning up incomplete checkpoint: {temp_dir}")
                        shutil.rmtree(temp_dir)
            
            # Find and remove backup directories older than 1 hour
            import time
            current_time = time.time()
            backup_patterns = [
                f"checkpoint_{self.task_id}_backup",
                f"emergency_checkpoint_{self.task_id}_backup"
            ]
            
            for pattern in backup_patterns:
                backup_dirs = list(models_dir.glob(pattern))
                for backup_dir in backup_dirs:
                    if backup_dir.is_dir():
                        dir_time = backup_dir.stat().st_mtime
                        if current_time - dir_time > 3600:  # 1 hour
                            print(f"Cleaning up old backup: {backup_dir}")
                            shutil.rmtree(backup_dir)
                            
        except Exception as e:
            print(f"Error during checkpoint cleanup: {e}")


class AtomicCheckpointCallback(TrainerCallback):
    """Callback to handle atomic checkpoint operations during training."""
    
    def __init__(self, task_id: str, training_manager):
        self.task_id = task_id
        self.training_manager = training_manager
        self.checkpoint_lock = threading.Lock()
    
    def on_save(self, args, state, control, **kwargs):
        """Ensure atomic checkpoint operations."""
        if self.training_manager.cancellation_event.is_set():
            with self.checkpoint_lock:
                print(f"Completing checkpoint before cancellation for task {self.task_id}")
                # The normal checkpoint save will complete, then cancellation will proceed
                # This ensures the checkpoint is not corrupted by premature cancellation
    
    def on_save_end(self, args, state, control, **kwargs):
        """Called after checkpoint save completes."""
        if self.training_manager.cancellation_event.is_set():
            print(f"Checkpoint completed, ready for safe cancellation")


def detect_hardware():
    """Detect available hardware and optimize settings accordingly."""
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    total_vram = 0
    
    if gpu_count > 0:
        for i in range(gpu_count):
            total_vram += torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
    
    # Enhanced hardware detection for Unsloth optimizations
    has_ampere_or_newer = False
    if gpu_count > 0:
        # Check for Ampere (RTX 30 series) or newer for optimal Unsloth performance
        compute_capability = torch.cuda.get_device_capability(0)
        has_ampere_or_newer = compute_capability[0] >= 8  # Ampere is compute capability 8.x
    
    return {
        "gpu_count": gpu_count,
        "total_vram_gb": total_vram,
        "has_bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "has_fp16": torch.cuda.is_available(),
        "has_ampere_or_newer": has_ampere_or_newer,
        "compute_capability": torch.cuda.get_device_capability(0) if gpu_count > 0 else None
    }


def get_optimal_device_map(hardware_info: dict):
    """Get optimal device map for limited memory scenarios and Kubernetes environments."""
    if hardware_info["gpu_count"] == 0:
        return "cpu"
    
    # Check if we should force CPU offloading for Kubernetes or resource constraints
    available_vram = min(hardware_info["total_vram_gb"], MAX_MEMORY_USAGE_GB)
    should_offload = (
        available_vram < 8 or 
        FORCE_CPU_OFFLOAD or 
        (KUBERNETES_MODE and available_vram < CONTAINER_MEMORY_LIMIT_GB * 0.7)
    )
    
    if should_offload:
        # For low VRAM or Kubernetes constraints, use sequential device mapping to avoid conflicts
        print(f"Using CPU offloading - Available VRAM: {available_vram:.1f}GB, K8s mode: {KUBERNETES_MODE}")
        if available_vram < 6:
            # Very conservative mapping for <6GB VRAM
            return "sequential"
        else:
            # Balanced approach for 6-8GB VRAM (like RTX 4060)
            return "balanced_low_0"
    elif hardware_info["gpu_count"] == 1:
        return "auto"
    else:
        return "balanced"


def validate_training_args_compatibility():
    """Validate TrainingArguments parameter compatibility with current transformers version."""
    
    # Get supported parameters for current TrainingArguments
    signature = inspect.signature(TrainingArguments.__init__)
    supported_params = set(signature.parameters.keys()) - {'self'}
    
    return {
        "supported_params": supported_params,
        "has_eval_strategy": "eval_strategy" in supported_params,
        "has_evaluation_strategy": "evaluation_strategy" in supported_params,
        "has_use_gradient_checkpointing": "use_gradient_checkpointing" in supported_params,
        "has_optim": "optim" in supported_params,
    }


def get_optimal_lora_config(model_name: str, hardware_info: dict):
    """Get optimal LoRA configuration based on model and hardware."""
    # Enhanced configuration with memory-aware optimization
    if hardware_info["total_vram_gb"] <= 8:
        # Conservative settings for 8GB cards like RTX 4060
        lora_r = 16  # Reduced for memory efficiency
        lora_alpha = 16  # Maintain ratio with r
    elif hardware_info["total_vram_gb"] <= 16:
        lora_r = 32
        lora_alpha = 32
    else:
        lora_r = 64
        lora_alpha = 64  # Scale alpha with r for consistency
    
    config = {
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,  # 0.0 is optimal for most cases
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
    }
    
    # Model-specific target modules
    if "llama" in model_name.lower():
        config["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", 
                                   "gate_proj", "up_proj", "down_proj"]
    elif "mistral" in model_name.lower():
        config["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", 
                                   "gate_proj", "up_proj", "down_proj"]
    elif "phi" in model_name.lower():
        config["target_modules"] = ["q_proj", "k_proj", "v_proj", "dense"]
    elif "gemma" in model_name.lower():
        config["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", 
                                   "gate_proj", "up_proj", "down_proj"]
    else:
        # Default for most models
        config["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    return config


def setup_unsloth_model(model_dir: str, max_seq_length: int, hardware_info: dict = None):
    """Setup model using Unsloth for efficient training."""
    if not UNSLOTH_AVAILABLE:
        print("Unsloth not available, falling back to standard setup")
        return None, None, False, max_seq_length
        
    try:
        if hardware_info is None:
            hardware_info = detect_hardware()
        
        # Enhanced dtype optimization for Unsloth
        if is_bfloat16_supported() and hardware_info["has_bf16"] and hardware_info["has_ampere_or_newer"]:
            dtype = None  # Auto-detect, likely bfloat16 (best for Ampere+)
            print("Using bfloat16 precision - optimal for Ampere+ GPUs")
        elif hardware_info["has_fp16"]:
            dtype = torch.float16
            print("Using float16 precision")
        else:
            dtype = torch.float32
            print("Using float32 precision")
        
        # Enhanced max_seq_length optimization based on VRAM and model
        if hardware_info["total_vram_gb"] < 6:
            max_seq_length = min(max_seq_length, 1024)
            print(f"Limited VRAM detected, reducing max_seq_length to {max_seq_length}")
        elif hardware_info["total_vram_gb"] < 12:
            max_seq_length = min(max_seq_length, 2048)
            print(f"Medium VRAM detected, setting max_seq_length to {max_seq_length}")
        elif hardware_info["total_vram_gb"] < 24:
            max_seq_length = min(max_seq_length, 4096)
            print(f"High VRAM detected, setting max_seq_length to {max_seq_length}")
        else:
            max_seq_length = min(max_seq_length, 8192)
            print(f"Very high VRAM detected, setting max_seq_length to {max_seq_length}")
        
        # Enhanced Unsloth model loading with all optimizations
        optimal_device_map = get_optimal_device_map(hardware_info)
        print(f"Loading model with Unsloth optimizations: {model_dir}")
        print(f"Using device map: {optimal_device_map}")
        print(f"Hardware info: {hardware_info['total_vram_gb']:.1f}GB VRAM")
        
        # Use additional memory optimization for limited VRAM or Kubernetes
        additional_kwargs = {}
        available_vram = min(hardware_info["total_vram_gb"], MAX_MEMORY_USAGE_GB)
        needs_optimization = (
            available_vram < 8 or 
            FORCE_CPU_OFFLOAD or 
            KUBERNETES_MODE
        )
        
        if needs_optimization:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            additional_kwargs.update({
                "low_cpu_mem_usage": True,
                "quantization_config": quantization_config,
            })
            optimization_reason = "Kubernetes" if KUBERNETES_MODE else f"low VRAM ({available_vram:.1f}GB)"
            print(f"Applied aggressive memory optimization for {optimization_reason}")
        else:
            # Standard 4-bit quantization - let Unsloth handle it automatically
            additional_kwargs.update({
                "low_cpu_mem_usage": True,
            })
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=True,  # Always use 4-bit for maximum memory savings
            device_map=optimal_device_map,
            # Additional Unsloth optimizations
            token=None,  # Add HF token if needed
            trust_remote_code=False,  # Security setting
            **additional_kwargs
        )
        
        # Verify tokenizer setup
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token for proper tokenization")
        
        # Get optimal LoRA configuration
        lora_config = get_optimal_lora_config(model_dir, hardware_info)
        print(f"Applying LoRA with config: rank={lora_config['r']}, alpha={lora_config['lora_alpha']}")
        
        # Setup LoRA with optimized parameters
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config["r"],
            target_modules=lora_config["target_modules"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            use_gradient_checkpointing=lora_config["use_gradient_checkpointing"],
            random_state=lora_config["random_state"],
            use_rslora=False,  # Standard LoRA for compatibility
            loftq_config=None,  # No quantization-aware training by default
        )
        
        print("Unsloth model setup completed successfully")
        return model, tokenizer, True, max_seq_length
        
    except ImportError as e:
        print(f"Unsloth import error: {e}")
        return None, None, False, max_seq_length
    except Exception as e:
        print(f"Unsloth setup failed: {e}. Falling back to standard setup.")
        return None, None, False, max_seq_length


def setup_standard_model(model_dir: str, peft_method: str, hardware_info: dict = None, xformers_available: bool = False):
    """Setup model using standard transformers + PEFT with optimizations."""
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT is required for standard model setup")
    
    from transformers import BitsAndBytesConfig
    
    if hardware_info is None:
        hardware_info = detect_hardware()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Optimize dtype
    if hardware_info["has_bf16"]:
        compute_dtype = torch.bfloat16
        dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
        dtype = torch.float16
    
    # Setup optimized quantization
    if peft_method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=dtype,
            llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offload for limited GPU memory
        )
        optimal_device_map = get_optimal_device_map(hardware_info)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_config,
            device_map=optimal_device_map,
            dtype=dtype,
            attn_implementation="flash_attention_2" if (hardware_info["gpu_count"] > 0 and xformers_available) else None,
        )
    else:
        optimal_device_map = get_optimal_device_map(hardware_info)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map=optimal_device_map, 
            dtype=dtype,
            attn_implementation="flash_attention_2" if (hardware_info["gpu_count"] > 0 and xformers_available) else None,
        )
    
    # Get optimal LoRA configuration
    lora_params = get_optimal_lora_config(model_dir, hardware_info)
    lora_config = LoraConfig(
        r=lora_params["r"],
        lora_alpha=lora_params["lora_alpha"],
        target_modules=lora_params["target_modules"],
        lora_dropout=lora_params["lora_dropout"],
        bias=lora_params["bias"],
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def get_chat_template(tokenizer, model_name: str):
    """Get appropriate chat template for the model."""
    try:
        from unsloth.chat_templates import get_chat_template
        
        # Use Unsloth's optimized chat templates
        if "llama-3" in model_name.lower():
            tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
        elif "llama" in model_name.lower():
            tokenizer = get_chat_template(tokenizer, chat_template="llama-2")
        elif "mistral" in model_name.lower():
            tokenizer = get_chat_template(tokenizer, chat_template="mistral")
        elif "phi" in model_name.lower():
            tokenizer = get_chat_template(tokenizer, chat_template="phi-3")
        elif "gemma" in model_name.lower():
            tokenizer = get_chat_template(tokenizer, chat_template="gemma")
        else:
            # Default chat template
            tokenizer = get_chat_template(tokenizer, chat_template="chatml")
            
    except ImportError:
        # Fallback to basic template if Unsloth not available
        pass
    
    return tokenizer


def format_instruction_data_unsloth(examples, tokenizer, model_name: str):
    """Format instruction data using Unsloth's optimized formatting."""
    try:
        from unsloth.chat_templates import get_chat_template, standardize_sharegpt
        
        # Convert to ShareGPT format for optimal training
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples["output"]
        
        # Ensure all are lists and convert to strings
        if not isinstance(instructions, list):
            instructions = [instructions]
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(outputs, list):
            outputs = [outputs]
        
        # Ensure same lengths
        max_len = max(len(instructions), len(inputs), len(outputs))
        if len(instructions) < max_len:
            instructions = instructions + [""] * (max_len - len(instructions))
        if len(inputs) < max_len:
            inputs = inputs + [""] * (max_len - len(inputs))
        if len(outputs) < max_len:
            outputs = outputs + [""] * (max_len - len(outputs))
        
        conversations = []
        for i, (instruction, input_text, output) in enumerate(zip(instructions, inputs, outputs)):
            try:
                # Convert all to strings and handle None values
                instruction = str(instruction) if instruction is not None else ""
                input_text = str(input_text) if input_text is not None else ""
                output = str(output) if output is not None else ""
                
                user_message = instruction + (f"\n\n{input_text}" if input_text.strip() else "")
                conversation = [
                    {"from": "human", "value": user_message},
                    {"from": "gpt", "value": output}
                ]
                conversations.append({"conversations": conversation})
                
            except Exception as e:
                print(f"ERROR: Failed to format Unsloth example {i}: {e}")
                print(f"  Instruction: {repr(instruction)} (type: {type(instruction)})")
                print(f"  Input: {repr(input_text)} (type: {type(input_text)})")
                print(f"  Output: {repr(output)} (type: {type(output)})")
                # Skip this example but continue
                continue
        
        if not conversations:
            raise ValueError("No valid conversations could be formatted")
        
        # Apply chat template
        tokenizer = get_chat_template(tokenizer, model_name)
        
        return {"conversations": conversations}
        
    except ImportError:
        # Fallback to basic formatting
        return format_instruction_data_basic(examples, tokenizer)
    except Exception as e:
        print(f"ERROR in format_instruction_data_unsloth: {e}")
        # Fallback to basic formatting
        return format_instruction_data_basic(examples, tokenizer)


def format_instruction_data_basic(examples, tokenizer):
    """Basic instruction data formatting (fallback)."""
    try:
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples["output"]
        
        # Ensure all are lists and convert to strings
        if not isinstance(instructions, list):
            instructions = [instructions]
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(outputs, list):
            outputs = [outputs]
        
        # Ensure same lengths
        max_len = max(len(instructions), len(inputs), len(outputs))
        if len(instructions) < max_len:
            instructions = instructions + [""] * (max_len - len(instructions))
        if len(inputs) < max_len:
            inputs = inputs + [""] * (max_len - len(inputs))  
        if len(outputs) < max_len:
            outputs = outputs + [""] * (max_len - len(outputs))
        
        texts = []
        for i, (instruction, input_text, output) in enumerate(zip(instructions, inputs, outputs)):
            # Convert all to strings and handle None values
            instruction = str(instruction) if instruction is not None else ""
            input_text = str(input_text) if input_text is not None else ""
            output = str(output) if output is not None else ""
            
            try:
                if input_text.strip():
                    text = f"<|user|>\n{instruction}\n\n{input_text}<|end|>\n<|assistant|>\n{output}<|end|>"
                else:
                    text = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{output}<|end|>"
                texts.append(text)
            except Exception as e:
                print(f"ERROR: Failed to format example {i}: {e}")
                print(f"  Instruction: {repr(instruction)} (type: {type(instruction)})")
                print(f"  Input: {repr(input_text)} (type: {type(input_text)})")
                print(f"  Output: {repr(output)} (type: {type(output)})")
                # Skip this example but continue
                continue
        
        if not texts:
            raise ValueError("No valid examples could be formatted")
            
        return {"text": texts}
        
    except Exception as e:
        print(f"ERROR in format_instruction_data_basic: {e}")
        print(f"Examples type: {type(examples)}")
        print(f"Examples keys: {list(examples.keys()) if hasattr(examples, 'keys') else 'N/A'}")
        if hasattr(examples, 'keys'):
            for key in examples.keys():
                value = examples[key]
                print(f"  {key}: type={type(value)}, len={len(value) if hasattr(value, '__len__') else 'N/A'}")
                if hasattr(value, '__len__') and len(value) > 0:
                    print(f"    First item: {repr(value[0])} (type: {type(value[0])})")
        raise

# -----------------------
# Formatting functions
# -----------------------
# -----------------------
# Formatting functions
# -----------------------
def format_plain_text(examples, tokenizer=None):
    """For unsupervised/plain text datasets."""
    if "text" in examples:
        return [str(x) for x in examples["text"]]
    elif "question" in examples and "answer" in examples:
        return [f"### Question:\n{q}\n\n### Answer:\n{a}<|endoftext|>" for q, a in zip(examples["question"], examples["answer"])]
    else:
        raise KeyError("Dataset must contain either 'text' field or 'question'/'answer' fields")

def format_instruction(examples, tokenizer=None):
    """For Alpaca/Instruction datasets."""
    if "instruction" in examples and "output" in examples:
        return [
            f"### Instruction:\n{instr}\n\n### Response:\n{out}"
            for instr, out in zip(examples["instruction"], examples["output"])
        ]
    return [str(x) for x in examples.get("text", [])]

def format_sharegpt(examples, tokenizer=None):
    """For ShareGPT-style datasets."""
    return [
        "".join(
            f"<|im_start|>{c['from']}\n{c['value']}<|im_end|>\n"
            for c in conv
        )
        for conv in examples["conversations"]
    ]

# -----------------------
# Fine-tuning function
# -----------------------

def fine_tune(request: FineTuneRequest, task_id: str = None) -> dict:
    """
    Launch fine-tuning with Unsloth on prepared datasets.
    Auto-detects dataset format (ShareGPT, Instruction, Plain Text).
    """
    if task_id is None:
        task_id = str(uuid.uuid4())
    
    import time
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    status = TrainingStatus(
        task_id=task_id, 
        status="running", 
        progress="Starting fine-tuning...",
        started_at=current_time,
        updated_at=current_time
    )
    status.save()
    
    # Register active training session
    register_active_session(task_id, request.dict())

    def run_training():
        try:
            logger.info("Starting fine-tuning process...")
            
            # Initialize streaming for this task
            status._add_log_to_queue = lambda msg, log_type="info": task_log_queues[task_id].put({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "type": log_type,
                "message": msg,
                "step": None,
                "epoch": None,
                "loss": None,
                "learning_rate": None
            })
            
            # Add initial logs
            status._add_log_to_queue("Fine-tuning process initialized", "info")
            
            # Load datasets from their respective directories
            logger.info(f"Loading datasets from {request.dataset_dir}")
            status._add_log_to_queue(f"Loading datasets from {request.dataset_dir}", "info")
            
            train_dataset = load_from_disk(f"{request.dataset_dir}/train")
            val_dataset = load_from_disk(f"{request.dataset_dir}/val")
            
            dataset_msg = f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
            logger.info(dataset_msg)
            status._add_log_to_queue(dataset_msg, "info")

            # Detect dataset type
            if "conversations" in train_dataset.column_names:
                formatting_func = format_sharegpt
                format_msg = "Detected ShareGPT dataset format."
                logger.info(format_msg)
                status._add_log_to_queue(format_msg, "info")
            elif "instruction" in train_dataset.column_names:
                formatting_func = format_instruction
                format_msg = "Detected Instruction dataset format."
                logger.info(format_msg)
                status._add_log_to_queue(format_msg, "info")
            elif "question" in train_dataset.column_names and "answer" in train_dataset.column_names:
                formatting_func = format_plain_text
                format_msg = "Detected Question-Answer dataset format."
                logger.info(format_msg)
                status._add_log_to_queue(format_msg, "info")
            else:
                formatting_func = format_plain_text
                format_msg = "Detected Plain Text dataset format."
                logger.info(format_msg)
                status._add_log_to_queue(format_msg, "info")

            # Detect hardware capabilities
            hardware_info = detect_hardware()
            hardware_msg = f"Hardware detected: {hardware_info['gpu_count']} GPU(s), {hardware_info['total_vram_gb']:.1f}GB VRAM"
            logger.info(hardware_msg)
            status._add_log_to_queue(hardware_msg, "info")
            
            # Load base model with Unsloth optimizations
            model_load_msg = f"Loading model from {request.base_model_dir}"
            logger.info(model_load_msg)
            status._add_log_to_queue(model_load_msg, "info")
            
            optimal_device_map = get_optimal_device_map(hardware_info)
            device_map_msg = f"Using device map: {optimal_device_map}"
            logger.info(device_map_msg)
            status._add_log_to_queue(device_map_msg, "info")
            
            # Use additional memory optimization for limited VRAM or Kubernetes
            additional_kwargs = {}
            available_vram = min(hardware_info["total_vram_gb"], MAX_MEMORY_USAGE_GB)
            needs_optimization = (
                available_vram < 8 or 
                FORCE_CPU_OFFLOAD or 
                KUBERNETES_MODE
            )
            
            # Configure quantization properly using BitsAndBytesConfig approach
            if needs_optimization:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading for memory
                )
                additional_kwargs.update({
                    "low_cpu_mem_usage": True,
                    "quantization_config": quantization_config,
                })
                optimization_reason = "Kubernetes" if KUBERNETES_MODE else f"low VRAM ({available_vram:.1f}GB)"
                memory_opt_msg = f"Applied aggressive memory optimization for {optimization_reason}"
                logger.info(memory_opt_msg)
                status._add_log_to_queue(memory_opt_msg, "info")
            else:
                # Standard 4-bit quantization - let Unsloth handle it automatically
                additional_kwargs.update({
                    "low_cpu_mem_usage": True,
                })
            
            # Clear CUDA cache before loading model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=request.base_model_dir,
                    load_in_4bit=True,
                    max_seq_length=request.max_seq_length,
                    dtype=None,
                    device_map=optimal_device_map,
                    **additional_kwargs
                )
            except Exception as e:
                if "Some modules are dispatched on the CPU or the disk" in str(e):
                    error_msg = f"GPU memory insufficient. Retrying with more aggressive CPU offloading..."
                    logger.warning(error_msg)
                    status._add_log_to_queue(error_msg, "warning")
                    
                    # Force more aggressive CPU offloading with proper BitsAndBytesConfig
                    from transformers import BitsAndBytesConfig
                    aggressive_quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        llm_int8_enable_fp32_cpu_offload=True,
                    )
                    additional_kwargs.update({
                        "low_cpu_mem_usage": True,
                        "quantization_config": aggressive_quantization_config,
                    })
                    
                    # Use sequential device mapping for maximum memory efficiency
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=request.base_model_dir,
                        load_in_4bit=True,
                        max_seq_length=min(request.max_seq_length, 1024),  # Reduce sequence length
                        dtype=None,
                        device_map="sequential",
                        **additional_kwargs
                    )
                else:
                    raise e
            
            model_success_msg = "Model loaded successfully"
            logger.info(model_success_msg)
            status._add_log_to_queue(model_success_msg, "success")

            # Apply LoRA with hardware-optimized configuration
            lora_msg = "Applying LoRA configuration..."
            logger.info(lora_msg)
            status._add_log_to_queue(lora_msg, "info")
            
            lora_config = get_optimal_lora_config(request.base_model_dir, hardware_info)
            lora_config_msg = f"LoRA config: r={lora_config['r']}, alpha={lora_config['lora_alpha']}"
            logger.info(lora_config_msg)
            status._add_log_to_queue(lora_config_msg, "info")
            
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_config["r"],
                target_modules=lora_config["target_modules"],
                lora_alpha=lora_config["lora_alpha"],
                lora_dropout=lora_config["lora_dropout"],
                bias=lora_config["bias"],
                use_gradient_checkpointing=lora_config["use_gradient_checkpointing"],
                random_state=lora_config["random_state"],
            )
            
            lora_success_msg = "LoRA applied successfully"
            logger.info(lora_success_msg)
            status._add_log_to_queue(lora_success_msg, "success")

            # Training arguments (hardware-optimized)
            args_msg = "Setting up training arguments..."
            logger.info(args_msg)
            status._add_log_to_queue(args_msg, "info")
            
            # Optimize batch size and gradient accumulation based on VRAM
            if hardware_info["total_vram_gb"] < 6:
                batch_size = 1
                gradient_accumulation = 32  # More aggressive accumulation for very low VRAM
            elif hardware_info["total_vram_gb"] < 10:
                # Optimized for RTX 4060 (8GB VRAM) - conservative but efficient
                batch_size = 1
                gradient_accumulation = 16
            else:
                batch_size = min(request.batch_size, 2)
                gradient_accumulation = request.gradient_accumulation_steps
            
            # Use bf16 only on supported hardware, fallback to fp16
            use_bf16 = hardware_info["has_bf16"] and hardware_info["has_ampere_or_newer"]
            use_fp16 = not use_bf16 and hardware_info["has_fp16"]
            
            training_args = TrainingArguments(
                output_dir=request.output_dir,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation,
                learning_rate=request.learning_rate,
                num_train_epochs=request.epochs,
                logging_steps=10,
                save_strategy="steps",
                save_steps=100,
                save_total_limit=2,
                eval_strategy="steps" if val_dataset and len(val_dataset) > 0 else "no",
                eval_steps=100 if val_dataset and len(val_dataset) > 0 else None,
                bf16=use_bf16,
                fp16=use_fp16,
                dataloader_pin_memory=False,  # Reduce memory usage
                gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
                report_to="none"
            )
            
            training_config_msg = f"Training config: batch_size={batch_size}, grad_accum={gradient_accumulation}, bf16={use_bf16}, fp16={use_fp16}"
            logger.info(training_config_msg)
            status._add_log_to_queue(training_config_msg, "info")

            # Create streaming logging callback
            logging_callback = StreamingLoggingCallback(status)
            
            # Use SFTTrainer for instruction fine-tuning
            trainer_msg = "Creating SFTTrainer..."
            logger.info(trainer_msg)
            status._add_log_to_queue(trainer_msg, "info")
            
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset if len(val_dataset) > 0 else None,
                tokenizer=tokenizer,
                formatting_func=formatting_func,
                max_seq_length=request.max_seq_length,
                callbacks=[logging_callback],
            )
            
            trainer_success_msg = "SFTTrainer created successfully"
            logger.info(trainer_success_msg)
            status._add_log_to_queue(trainer_success_msg, "success")

            # Start training with cancellation support
            training_start_msg = "Starting training with cancellation support..."
            logger.info(training_start_msg)
            status._add_log_to_queue(training_start_msg, "info")
            
            # Create TrainingManager for this task
            training_manager = TrainingManager(task_id)
            
            # Use TrainingManager for safe training with cancellation support
            result = training_manager.safe_train_with_cancellation(trainer, model, tokenizer, status)
            
            # Only proceed with saving if training wasn't cancelled
            if not training_manager.cancellation_event.is_set():
                save_msg = "Training completed, saving model..."
                logger.info(save_msg)
                status._add_log_to_queue(save_msg, "info")
                
                trainer.save_model(request.output_dir)
                tokenizer.save_pretrained(request.output_dir)
                
                final_success_msg = "Model saved successfully"
                logger.info(final_success_msg)
                status._add_log_to_queue(final_success_msg, "success")

                status.update(status="completed", progress="Fine-tuning completed successfully.")
                
                # Mark completion and clean up session
                task_completion_flags[task_id] = True
                remove_active_session(task_id)
            else:
                # Training was cancelled, status already updated by TrainingManager
                logger.info(f"Training was cancelled for task {task_id}")
                # Note: completion flag and session cleanup are handled by TrainingManager

        except Exception as e:
            logger.exception("Fine-tuning failed")
            error_msg = f"Fine-tuning failed: {str(e)}"
            status.update(status="failed", progress=error_msg)
            status._add_log_to_queue(error_msg, "error")
            
            # Clean up failed session
            remove_active_session(task_id)
            task_completion_flags[task_id] = True

    threading.Thread(target=run_training, daemon=True).start()
    return {"task_id": task_id, "status": "started"}