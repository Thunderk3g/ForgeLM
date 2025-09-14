"""
Training Service with Unsloth Optimization

Handles fine-tuning operations with advanced ML optimizations including:
- Unsloth 2x faster training with reduced VRAM
- LoRA/QLoRA parameter-efficient fine-tuning
- Gradient accumulation and memory optimization
- Real-time progress tracking and logging
"""

import os
import time
import asyncio
import threading
import queue
import json
import uuid
import signal
import atexit
import gc
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import torch
from datasets import load_from_disk

from ..models.schemas import FineTuneRequest, TrainingStatus
from ..utils.logging import get_logger
from ..utils.performance import performance_monitor, timing_context, timed
from ..config.settings import settings

# Environment optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Import common dependencies first
try:
    from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainerCallback
    from trl import SFTTrainer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: Transformers/TRL not available")

# Import Unsloth with proper error handling
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

# Optional PEFT import
try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("WARNING: PEFT not available")

# Container configuration
KUBERNETES_MODE = os.environ.get("KUBERNETES_MODE", "false").lower() == "true"
MAX_MEMORY_USAGE_GB = float(os.environ.get("MAX_MEMORY_USAGE_GB", "6"))
FORCE_CPU_OFFLOAD = os.environ.get("FORCE_CPU_OFFLOAD", "false").lower() == "true"
CONTAINER_MEMORY_LIMIT_GB = float(os.environ.get("CONTAINER_MEMORY_LIMIT_GB", "8"))


# Global task management
task_log_queues: Dict[str, queue.Queue] = {}
task_completion_flags: Dict[str, bool] = {}
active_sessions: Dict[str, Dict[str, Any]] = {}

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


# Define callback class only if transformers is available
if TRANSFORMERS_AVAILABLE:
    class UnslothProgressCallback(TrainerCallback):
        """Custom callback for Unsloth training progress tracking"""
    
    def __init__(self, task_id: str, log_queue: queue.Queue):
        self.task_id = task_id
        self.log_queue = log_queue
        self.start_time = time.time()
        self.logger = get_logger(f"training.{task_id}")
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        try:
            message = f"Starting training with {state.max_steps} total steps"
            
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "train_start",
                "message": message,
                "total_steps": state.max_steps,
                "total_epochs": args.num_train_epochs
            }
            
            self.log_queue.put(log_entry)
            self.logger.info(message)
        except Exception as e:
            self.logger.error(f"Error in train begin callback: {e}")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch"""
        try:
            epoch = state.epoch
            message = f"Starting epoch {epoch}/{args.num_train_epochs}"
            
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "epoch_start",
                "message": message,
                "epoch": epoch,
                "total_epochs": args.num_train_epochs
            }
            
            self.log_queue.put(log_entry)
            self.logger.info(message)
        except Exception as e:
            self.logger.error(f"Error in epoch begin callback: {e}")
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training step"""
        # We don't need to log every step begin to avoid spam
        # This method exists just to satisfy the trainer callback interface
        pass
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        try:
            current_step = state.global_step
            total_steps = state.max_steps
            current_epoch = state.epoch
            
            # Get loss from logs if available
            loss = None
            learning_rate = None
            if state.log_history:
                last_log = state.log_history[-1]
                loss = last_log.get('train_loss', last_log.get('loss'))
                learning_rate = last_log.get('learning_rate')
            
            # Calculate progress
            progress_pct = (current_step / total_steps * 100) if total_steps > 0 else 0
            
            # Log every 10 steps to avoid spam
            if current_step % 10 == 0 or current_step == total_steps:
                message = f"Step {current_step}/{total_steps} ({progress_pct:.1f}%) - Epoch: {current_epoch:.2f}"
                if loss is not None:
                    message += f" - Loss: {loss:.4f}"
                if learning_rate is not None:
                    message += f" - LR: {learning_rate:.2e}"
                
                log_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "training",
                    "message": message,
                    "step": current_step,
                    "total_steps": total_steps,
                    "epoch": current_epoch,
                    "total_epochs": args.num_train_epochs,
                    "loss": loss,
                    "learning_rate": learning_rate,
                    "progress_percentage": progress_pct
                }
                
                self.log_queue.put(log_entry)
                
                # Also update status file
                self._update_status_file(current_step, total_steps, current_epoch, args.num_train_epochs, loss, learning_rate, progress_pct)
                
        except Exception as e:
            self.logger.error(f"Error in step end callback: {e}")
    
    def _update_status_file(self, current_step, total_steps, current_epoch, total_epochs, loss, learning_rate, progress_pct):
        """Update status file with current training progress"""
        try:
            status_file = Path("logs") / f"{self.task_id}_status.json"
            status_data = {
                "task_id": self.task_id,
                "status": "running",
                "progress": f"Step {current_step}/{total_steps} ({progress_pct:.1f}%)",
                "current_step": current_step,
                "total_steps": total_steps,
                "current_epoch": float(current_epoch),
                "total_epochs": total_epochs,
                "loss": float(loss) if loss is not None else None,
                "learning_rate": float(learning_rate) if learning_rate is not None else None,
                "progress_percentage": progress_pct,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time))
            }
            
            with open(status_file, "w") as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error updating status file: {e}")

else:
    # Dummy callback class when transformers is not available
    class UnslothProgressCallback:
        def __init__(self, task_id: str, log_queue: queue.Queue):
            self.task_id = task_id
            self.log_queue = log_queue


class TrainingService:
    """Advanced training service with Unsloth optimization"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.active_trainings: Dict[str, Dict[str, Any]] = {}
        self.cleanup_lock = threading.Lock()
        
        # Register cleanup
        register_cleanup(self.cleanup_all_trainings)
        
        self.logger.info(f"TrainingService initialized")
        self.logger.info(f"Unsloth available: {UNSLOTH_AVAILABLE}")
        self.logger.info(f"Container memory limit: {CONTAINER_MEMORY_LIMIT_GB}GB")
    
    @timed("fine_tune")
    def fine_tune(self, request: FineTuneRequest, task_id: str = None) -> Dict[str, Any]:
        """
        Fine-tune a model with Unsloth optimization
        
        Args:
            request: Fine-tuning parameters
            task_id: Unique task identifier
            
        Returns:
            Training result dictionary
        """
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        # Initialize logging
        log_queue = queue.Queue()
        task_log_queues[task_id] = log_queue
        task_completion_flags[task_id] = False
        
        # Create status file
        self._initialize_status_file(task_id, request)
        
        try:
            self.logger.info(f"Starting fine-tuning task {task_id}")
            
            if UNSLOTH_AVAILABLE:
                return self._fine_tune_with_unsloth(request, task_id, log_queue)
            elif TRANSFORMERS_AVAILABLE:
                return self._fine_tune_with_transformers(request, task_id, log_queue)
            else:
                raise RuntimeError("No fine-tuning backend available")
                
        except Exception as e:
            self.logger.error(f"Fine-tuning failed: {e}")
            self._update_status_file_error(task_id, str(e))
            task_completion_flags[task_id] = True
            raise
        finally:
            # Cleanup
            self._cleanup_training_resources(task_id)
    
    def _fine_tune_with_unsloth(self, request: FineTuneRequest, task_id: str, log_queue: queue.Queue) -> Dict[str, Any]:
        """Fine-tune using Unsloth for 2x speedup"""
        
        self.logger.info("Using Unsloth optimization for fine-tuning")
        
        try:
            # Load model and tokenizer with Unsloth
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=request.base_model_dir,
                max_seq_length=request.max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
            )
            
            # Apply LoRA adapter for parameter-efficient fine-tuning
            model = FastLanguageModel.get_peft_model(
                model,
                r=16 if request.peft_method == "lora" else 64,  # Rank
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha=16,
                lora_dropout=0.0,  # Supports any, but = 0 is optimized
                bias="none",    # Supports any, but = "none" is optimized
                use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
                random_state=3407,
                use_rslora=False,  # We support rank stabilized LoRA
                loftq_config=None, # And LoftQ
            )
            
            # Load dataset
            dataset = load_from_disk(request.dataset_dir)
            
            # Setup training arguments
            training_args = self._create_training_args(request, task_id)
            
            # Create progress callback
            progress_callback = UnslothProgressCallback(task_id, log_queue)
            
            # Create trainer with Unsloth optimization
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset["train"] if "train" in dataset else dataset,
                eval_dataset=dataset.get("val"),
                dataset_text_field="text",  # Assuming text field
                max_seq_length=request.max_seq_length,
                data_collator=None,
                dataset_num_proc=2,
                packing=False,
                args=training_args,
            )
            
            # Add custom callback for progress tracking
            trainer.add_callback(progress_callback)
            
            # Register this training for cleanup
            self.active_trainings[task_id] = {
                "trainer": trainer,
                "model": model,
                "tokenizer": tokenizer,
                "start_time": time.time()
            }
            
            # Start training
            self.logger.info(f"Starting Unsloth-optimized training for task {task_id}")
            trainer.train()
            
            # Save the fine-tuned model
            output_dir = Path(request.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save with Unsloth
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            
            # Also save in 16bit for inference
            if request.ft_type == "instruction":
                model.save_pretrained_merged(
                    str(output_dir / "merged_16bit"), 
                    tokenizer, 
                    save_method="merged_16bit"
                )
            
            # Mark as complete
            self._update_status_file_complete(task_id)
            task_completion_flags[task_id] = True
            
            self.logger.info(f"Unsloth training completed for task {task_id}")
            
            return {
                "status": "completed",
                "message": "Fine-tuning completed successfully with Unsloth optimization",
                "task_id": task_id,
                "output_dir": request.output_dir,
                "optimization": "unsloth_2x_speedup"
            }
            
        except Exception as e:
            self.logger.error(f"Unsloth training failed: {e}")
            raise
    
    def _fine_tune_with_transformers(self, request: FineTuneRequest, task_id: str, log_queue: queue.Queue) -> Dict[str, Any]:
        """Fallback fine-tuning with standard Transformers + PEFT"""
        
        self.logger.info("Using Transformers + PEFT fallback for fine-tuning")
        
        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(request.base_model_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                request.base_model_dir,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            
            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.resize_token_embeddings(len(tokenizer))
            
            # Apply PEFT if available
            if PEFT_AVAILABLE and request.peft_method in ["lora", "qlora"]:
                peft_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, peft_config)
            
            # Load dataset
            dataset = load_from_disk(request.dataset_dir)
            
            # Setup training arguments  
            training_args = self._create_training_args(request, task_id)
            
            # Create trainer
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset["train"] if "train" in dataset else dataset,
                eval_dataset=dataset.get("val"),
                dataset_text_field="text",
                max_seq_length=request.max_seq_length,
                args=training_args,
            )
            
            # Register this training for cleanup
            self.active_trainings[task_id] = {
                "trainer": trainer,
                "model": model,
                "tokenizer": tokenizer,
                "start_time": time.time()
            }
            
            # Start training
            self.logger.info(f"Starting standard training for task {task_id}")
            trainer.train()
            
            # Save the fine-tuned model
            output_dir = Path(request.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            
            # Mark as complete
            self._update_status_file_complete(task_id)
            task_completion_flags[task_id] = True
            
            self.logger.info(f"Standard training completed for task {task_id}")
            
            return {
                "status": "completed",
                "message": "Fine-tuning completed successfully with standard Transformers",
                "task_id": task_id,
                "output_dir": request.output_dir,
                "optimization": "standard_transformers"
            }
            
        except Exception as e:
            self.logger.error(f"Standard training failed: {e}")
            raise
    
    def _create_training_args(self, request: FineTuneRequest, task_id: str):
        """Create training arguments"""
        
        output_dir = Path(request.output_dir) / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        args = {
            "per_device_train_batch_size": request.batch_size,
            "gradient_accumulation_steps": request.gradient_accumulation_steps,
            "warmup_steps": 5,
            "num_train_epochs": request.epochs,
            "learning_rate": request.learning_rate,
            "fp16": torch.cuda.is_available() and not is_bfloat16_supported(),
            "bf16": torch.cuda.is_available() and is_bfloat16_supported() if UNSLOTH_AVAILABLE else False,
            "logging_steps": 10,
            "optim": "adamw_8bit" if UNSLOTH_AVAILABLE else "adamw_hf",
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "seed": 3407,
            "output_dir": str(output_dir),
            "save_steps": 100,
            "save_total_limit": 2,
            "dataloader_num_workers": 0,  # Avoid multiprocessing issues
            "remove_unused_columns": False,
            "run_name": f"llmforge_{task_id[:8]}",
        }
        
        # Memory optimization for containers
        if KUBERNETES_MODE or MAX_MEMORY_USAGE_GB < 8:
            args.update({
                "per_device_train_batch_size": min(request.batch_size, 2),
                "gradient_accumulation_steps": max(request.gradient_accumulation_steps, 8),
                "dataloader_pin_memory": False,
            })
        
        if UNSLOTH_AVAILABLE:
            from transformers import TrainingArguments
            return TrainingArguments(**args)
        else:
            from transformers import TrainingArguments
            return TrainingArguments(**args)
    
    def _initialize_status_file(self, task_id: str, request: FineTuneRequest):
        """Initialize status file for tracking"""
        status_file = Path("logs") / f"{task_id}_status.json"
        status_file.parent.mkdir(exist_ok=True)
        
        status_data = {
            "task_id": task_id,
            "status": "starting",
            "progress": "Initializing fine-tuning...",
            "current_step": 0,
            "total_steps": None,
            "current_epoch": 0.0,
            "total_epochs": request.epochs,
            "loss": None,
            "learning_rate": request.learning_rate,
            "progress_percentage": 0.0,
            "message": "Starting fine-tuning process",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "optimization": "unsloth" if UNSLOTH_AVAILABLE else "transformers"
        }
        
        with open(status_file, "w") as f:
            json.dump(status_data, f, indent=2)
    
    def _update_status_file_complete(self, task_id: str):
        """Update status file on completion"""
        status_file = Path("logs") / f"{task_id}_status.json"
        
        if status_file.exists():
            with open(status_file, "r") as f:
                status_data = json.load(f)
        else:
            status_data = {"task_id": task_id}
        
        status_data.update({
            "status": "completed",
            "progress": "Fine-tuning completed successfully",
            "progress_percentage": 100.0,
            "message": "Fine-tuning completed successfully",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        with open(status_file, "w") as f:
            json.dump(status_data, f, indent=2)
    
    def _update_status_file_error(self, task_id: str, error_message: str):
        """Update status file on error"""
        status_file = Path("logs") / f"{task_id}_status.json"
        
        if status_file.exists():
            with open(status_file, "r") as f:
                status_data = json.load(f)
        else:
            status_data = {"task_id": task_id}
        
        status_data.update({
            "status": "error",
            "progress": f"Error: {error_message}",
            "message": f"Fine-tuning failed: {error_message}",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        with open(status_file, "w") as f:
            json.dump(status_data, f, indent=2)
    
    def _cleanup_training_resources(self, task_id: str):
        """Clean up resources for a specific training"""
        with self.cleanup_lock:
            if task_id in self.active_trainings:
                training_info = self.active_trainings[task_id]
                
                try:
                    # Clear model and trainer references
                    del training_info["trainer"]
                    del training_info["model"] 
                    del training_info["tokenizer"]
                    
                    # Remove from active trainings
                    del self.active_trainings[task_id]
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    self.logger.info(f"Cleaned up resources for training {task_id}")
                    
                except Exception as e:
                    self.logger.error(f"Error cleaning up training {task_id}: {e}")
    
    def cleanup_all_trainings(self):
        """Clean up all active trainings - called on shutdown"""
        self.logger.info("Cleaning up all active trainings...")
        
        with self.cleanup_lock:
            for task_id in list(self.active_trainings.keys()):
                self._cleanup_training_resources(task_id)
    
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active training sessions"""
        sessions = {}
        
        for task_id, training_info in self.active_trainings.items():
            sessions[task_id] = {
                "task_id": task_id,
                "start_time": training_info["start_time"],
                "status": "running",
                "duration_seconds": time.time() - training_info["start_time"]
            }
        
        return sessions
    
    def cancel_training(self, task_id: str) -> bool:
        """Cancel an active training session"""
        if task_id in self.active_trainings:
            try:
                # Set completion flag to stop SSE streaming
                task_completion_flags[task_id] = True
                
                # Update status file
                self._update_status_file_error(task_id, "Training cancelled by user")
                
                # Clean up resources
                self._cleanup_training_resources(task_id)
                
                self.logger.info(f"Cancelled training {task_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error cancelling training {task_id}: {e}")
                return False
        
        return False


# Create global service instance
training_service = TrainingService()