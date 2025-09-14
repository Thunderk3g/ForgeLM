"""
Training Service Core

Isolated training logic that runs in worker processes. This module contains
the core ML training functionality extracted from training_service.py but
designed to run in complete process isolation.
"""

import os
import time
import json
import uuid
import gc
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import torch
from datasets import load_from_disk

from ..models.schemas import FineTuneRequest
from utils.logging import get_logger

# Import ML dependencies with error handling
try:
    from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, TrainerCallback
    from trl import SFTTrainer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: Transformers/TRL not available in worker")

try:
    import unsloth
    from unsloth import FastLanguageModel, is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
    print(f"Worker: Unsloth available (v{unsloth.__version__})")
except (ImportError, NotImplementedError) as e:
    UNSLOTH_AVAILABLE = False
    print(f"Worker: Unsloth not available ({e})")

try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Worker: PEFT not available")


class WorkerProgressCallback(TrainerCallback):
    """Progress callback for worker process training"""
    
    def __init__(self, task_id: str, status_updater: Callable, cancellation_callback: Callable):
        self.task_id = task_id
        self.status_updater = status_updater
        self.cancellation_callback = cancellation_callback
        self.start_time = time.time()
        self.logger = get_logger(f"worker.training.{task_id}")
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        try:
            message = f"Starting training with {state.max_steps} total steps"
            self.status_updater(
                "running",
                message,
                total_steps=state.max_steps,
                total_epochs=args.num_train_epochs,
                current_step=0,
                current_epoch=0.0,
                progress_percentage=0.0
            )
            self.logger.info(message)
        except Exception as e:
            self.logger.error(f"Error in train begin callback: {e}")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch"""
        try:
            # Check for cancellation
            if self.cancellation_callback():
                control.should_training_stop = True
                return
                
            epoch = state.epoch
            message = f"Starting epoch {epoch}/{args.num_train_epochs}"
            self.logger.info(message)
        except Exception as e:
            self.logger.error(f"Error in epoch begin callback: {e}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        try:
            # Check for cancellation first
            if self.cancellation_callback():
                self.logger.info("Cancellation requested, stopping training")
                control.should_training_stop = True
                return
                
            current_step = state.global_step
            total_steps = state.max_steps
            current_epoch = state.epoch
            
            # Get metrics from logs
            loss = None
            learning_rate = None
            if state.log_history:
                last_log = state.log_history[-1]
                loss = last_log.get('train_loss', last_log.get('loss'))
                learning_rate = last_log.get('learning_rate')
            
            # Calculate progress
            progress_pct = (current_step / total_steps * 100) if total_steps > 0 else 0
            
            # Update status every 10 steps or at completion
            if current_step % 10 == 0 or current_step == total_steps:
                message = f"Step {current_step}/{total_steps} ({progress_pct:.1f}%) - Epoch: {current_epoch:.2f}"
                if loss is not None:
                    message += f" - Loss: {loss:.4f}"
                if learning_rate is not None:
                    message += f" - LR: {learning_rate:.2e}"
                
                self.status_updater(
                    "running",
                    message,
                    current_step=current_step,
                    total_steps=total_steps,
                    current_epoch=float(current_epoch),
                    total_epochs=args.num_train_epochs,
                    loss=float(loss) if loss is not None else None,
                    learning_rate=float(learning_rate) if learning_rate is not None else None,
                    progress_percentage=progress_pct
                )
                
        except Exception as e:
            self.logger.error(f"Error in step end callback: {e}")


class TrainingCore:
    """Core training logic for worker processes"""
    
    def __init__(self, task_id: str, worker_mode: bool = True):
        self.task_id = task_id
        self.worker_mode = worker_mode
        self.logger = get_logger(f"worker.core.{task_id}")
        self.status_file = Path("logs") / f"{task_id}_status.json"
        self.cancellation_callback: Optional[Callable] = None
        
        self.logger.info(f"TrainingCore initialized for task {task_id} in worker mode")
        
    def set_cancellation_callback(self, callback: Callable):
        """Set callback to check for cancellation"""
        self.cancellation_callback = callback
    
    def _should_cancel(self) -> bool:
        """Check if training should be cancelled"""
        if self.cancellation_callback:
            return self.cancellation_callback()
        return False
    
    def _update_status(self, status: str, message: str, **kwargs):
        """Update status file with current progress"""
        try:
            # Read existing status
            status_data = {}
            if self.status_file.exists():
                try:
                    with open(self.status_file, 'r') as f:
                        status_data = json.load(f)
                except:
                    status_data = {}
            
            # Update with new data
            status_data.update({
                "task_id": self.task_id,
                "status": status,
                "message": message,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                **kwargs
            })
            
            # Ensure started_at is preserved
            if "started_at" not in status_data:
                status_data["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Atomic write
            temp_file = self.status_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(status_data, f, indent=2)
            temp_file.replace(self.status_file)
            
        except Exception as e:
            self.logger.error(f"Error updating status: {e}")
    
    def execute_training(self, request: FineTuneRequest) -> Dict[str, Any]:
        """
        Execute training with the given request
        
        Args:
            request: Training parameters
            
        Returns:
            Training result dictionary
        """
        try:
            self.logger.info(f"Starting training execution for task {self.task_id}")
            
            # Initialize status
            self._update_status("initializing", "Setting up training...")
            
            # Choose training backend
            if UNSLOTH_AVAILABLE:
                return self._train_with_unsloth(request)
            elif TRANSFORMERS_AVAILABLE:
                return self._train_with_transformers(request)
            else:
                raise RuntimeError("No training backend available in worker process")
                
        except Exception as e:
            self.logger.error(f"Training execution failed: {e}")
            raise
    
    def _train_with_unsloth(self, request: FineTuneRequest) -> Dict[str, Any]:
        """Execute training using Unsloth optimization"""
        
        self.logger.info("Using Unsloth optimization for training")
        self._update_status("loading", "Loading model with Unsloth optimization...")
        
        try:
            # Load model and tokenizer with Unsloth
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=request.base_model_dir,
                max_seq_length=request.max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=True,  # Use 4-bit quantization
            )
            
            # Check for cancellation after model loading
            if self._should_cancel():
                raise InterruptedError("Training cancelled during model loading")
            
            self._update_status("configuring", "Applying LoRA configuration...")
            
            # Apply LoRA adapter
            model = FastLanguageModel.get_peft_model(
                model,
                r=16 if request.peft_method == "lora" else 64,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
            
            # Check for cancellation
            if self._should_cancel():
                raise InterruptedError("Training cancelled during model configuration")
                
            self._update_status("loading_data", "Loading training dataset...")
            
            # Load dataset
            dataset = load_from_disk(request.dataset_dir)
            
            # Check for cancellation
            if self._should_cancel():
                raise InterruptedError("Training cancelled during data loading")
            
            self._update_status("preparing", "Setting up training arguments...")
            
            # Setup training arguments
            training_args = self._create_training_args(request)
            
            # Create progress callback
            progress_callback = WorkerProgressCallback(
                self.task_id, 
                self._update_status,
                self._should_cancel
            )
            
            # Create trainer
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset["train"] if "train" in dataset else dataset,
                eval_dataset=dataset.get("val"),
                dataset_text_field="text",
                max_seq_length=request.max_seq_length,
                data_collator=None,
                dataset_num_proc=2,
                packing=False,
                args=training_args,
            )
            
            # Add progress callback
            trainer.add_callback(progress_callback)
            
            # Start training
            self.logger.info("Starting Unsloth training...")
            self._update_status("training", "Training started...")
            
            # Check for cancellation one more time before training
            if self._should_cancel():
                raise InterruptedError("Training cancelled before start")
            
            trainer.train()
            
            # Check if training was cancelled
            if self._should_cancel():
                raise InterruptedError("Training was cancelled")
            
            # Save model
            self._update_status("saving", "Saving trained model...")
            
            output_dir = Path(request.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save with Unsloth
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            
            # Also save merged model for inference if needed
            if request.ft_type == "instruction":
                try:
                    merged_dir = output_dir / "merged_16bit"
                    model.save_pretrained_merged(
                        str(merged_dir), 
                        tokenizer, 
                        save_method="merged_16bit"
                    )
                    self.logger.info("Saved merged 16-bit model for inference")
                except Exception as e:
                    self.logger.warning(f"Could not save merged model: {e}")
            
            # Final cleanup
            self._cleanup_resources()
            
            result = {
                "status": "completed",
                "message": "Training completed successfully with Unsloth optimization",
                "task_id": self.task_id,
                "output_dir": request.output_dir,
                "optimization": "unsloth_2x_speedup",
                "backend": "unsloth"
            }
            
            self.logger.info("Unsloth training completed successfully")
            return result
            
        except InterruptedError as e:
            self.logger.info(f"Training cancelled: {e}")
            self._update_status("cancelled", f"Training cancelled: {str(e)}")
            return {"status": "cancelled", "message": str(e)}
        except Exception as e:
            self.logger.error(f"Unsloth training failed: {e}")
            raise
    
    def _train_with_transformers(self, request: FineTuneRequest) -> Dict[str, Any]:
        """Fallback training using standard Transformers + PEFT"""
        
        self.logger.info("Using Transformers + PEFT for training")
        self._update_status("loading", "Loading model with Transformers...")
        
        try:
            # Load tokenizer and model
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
            
            # Check for cancellation
            if self._should_cancel():
                raise InterruptedError("Training cancelled during model loading")
            
            # Apply PEFT if available
            if PEFT_AVAILABLE and request.peft_method in ["lora", "qlora"]:
                self._update_status("configuring", "Applying PEFT configuration...")
                
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
            self._update_status("loading_data", "Loading training dataset...")
            dataset = load_from_disk(request.dataset_dir)
            
            # Check for cancellation
            if self._should_cancel():
                raise InterruptedError("Training cancelled during data loading")
            
            # Setup training
            training_args = self._create_training_args(request)
            
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset["train"] if "train" in dataset else dataset,
                eval_dataset=dataset.get("val"),
                dataset_text_field="text",
                max_seq_length=request.max_seq_length,
                args=training_args,
            )
            
            # Start training
            self.logger.info("Starting standard training...")
            self._update_status("training", "Training started...")
            
            if self._should_cancel():
                raise InterruptedError("Training cancelled before start")
            
            trainer.train()
            
            if self._should_cancel():
                raise InterruptedError("Training was cancelled")
            
            # Save model
            self._update_status("saving", "Saving trained model...")
            output_dir = Path(request.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            
            # Cleanup
            self._cleanup_resources()
            
            result = {
                "status": "completed",
                "message": "Training completed successfully with Transformers",
                "task_id": self.task_id,
                "output_dir": request.output_dir,
                "optimization": "standard_transformers",
                "backend": "transformers"
            }
            
            self.logger.info("Standard training completed successfully")
            return result
            
        except InterruptedError as e:
            self.logger.info(f"Training cancelled: {e}")
            self._update_status("cancelled", f"Training cancelled: {str(e)}")
            return {"status": "cancelled", "message": str(e)}
        except Exception as e:
            self.logger.error(f"Standard training failed: {e}")
            raise
    
    def _create_training_args(self, request: FineTuneRequest):
        """Create training arguments optimized for worker process"""
        
        output_dir = Path(request.output_dir) / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        args = {
            "per_device_train_batch_size": request.batch_size,
            "gradient_accumulation_steps": request.gradient_accumulation_steps,
            "warmup_steps": 5,
            "num_train_epochs": request.epochs,
            "learning_rate": request.learning_rate,
            "fp16": torch.cuda.is_available() and not (UNSLOTH_AVAILABLE and is_bfloat16_supported()),
            "bf16": torch.cuda.is_available() and UNSLOTH_AVAILABLE and is_bfloat16_supported(),
            "logging_steps": 10,
            "optim": "adamw_8bit" if UNSLOTH_AVAILABLE else "adamw_hf",
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "seed": 3407,
            "output_dir": str(output_dir),
            "save_steps": 100,
            "save_total_limit": 2,
            "dataloader_num_workers": 0,  # Avoid multiprocessing issues in worker
            "remove_unused_columns": False,
            "run_name": f"llmforge_worker_{self.task_id[:8]}",
        }
        
        # Worker-specific optimizations
        worker_memory_gb = float(os.environ.get("WORKER_MEMORY_GB", "6"))
        if worker_memory_gb < 8:
            args.update({
                "per_device_train_batch_size": min(request.batch_size, 2),
                "gradient_accumulation_steps": max(request.gradient_accumulation_steps, 8),
                "dataloader_pin_memory": False,
            })
        
        return TrainingArguments(**args)
    
    def _cleanup_resources(self):
        """Clean up GPU/CPU resources in worker process"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("Cleared CUDA cache in worker process")
                
        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")