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
os.environ["TRITON_CACHE_DIR"] = "/tmp/.triton_cache"  # or any path you know is writable

try:
    import unsloth
    from unsloth import FastLanguageModel, is_bfloat16_supported

    UNSLOTH_AVAILABLE = True
    print(f"Worker: Unsloth available (v{unsloth.__version__})")
except (ImportError, NotImplementedError) as e:
    UNSLOTH_AVAILABLE = False
    print(f"Worker: Unsloth not available ({e})")
from models.schemas import FineTuneRequest
from utils.logging import get_logger

# Import ML dependencies with error handling
try:
    from transformers import (
        TrainingArguments,
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainerCallback,
    )
    from trl import SFTTrainer, SFTConfig

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: Transformers/TRL not available in worker")


try:
    from peft import get_peft_model, LoraConfig

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Worker: PEFT not available")


class WorkerProgressCallback(TrainerCallback):
    """Progress callback for worker process training"""

    def __init__(
        self, task_id: str, status_updater: Callable, cancellation_callback: Callable
    ):
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
                progress_percentage=0.0,
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
                loss = last_log.get("train_loss", last_log.get("loss"))
                learning_rate = last_log.get("learning_rate")

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
                    learning_rate=(
                        float(learning_rate) if learning_rate is not None else None
                    ),
                    progress_percentage=progress_pct,
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
                    with open(self.status_file, "r") as f:
                        status_data = json.load(f)
                except:
                    status_data = {}

            # Update with new data
            status_data.update(
                {
                    "task_id": self.task_id,
                    "status": status,
                    "message": message,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    **kwargs,
                }
            )

            # Ensure started_at is preserved
            if "started_at" not in status_data:
                status_data["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            # Atomic write
            temp_file = self.status_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
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
        """Execute training using Unsloth optimization with Qwen2.5 optimizations"""

        self.logger.info("Using Unsloth optimization for training with Qwen2.5 optimizations")
        self._update_status("loading", "Loading model with Unsloth optimization...")

        try:
            # Load model and tokenizer with Unsloth and Qwen2.5-specific configurations
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=request.base_model_dir,
                max_seq_length=request.max_seq_length,
                dtype=None,  # Auto-detect optimal dtype
                load_in_4bit=True,  # Critical for memory efficiency
                trust_remote_code=True,  # Required for Qwen2.5
                # Qwen2.5-specific optimizations
                device_map="balanced",  # Better than "auto" for Qwen2.5
                attn_implementation="flash_attention_2",  # Enable if available
            )
            
            # CRITICAL FIX: Proper Qwen2.5 tokenizer configuration
            self.logger.info("Applying Qwen2.5 tokenizer fixes...")
            
            # Fix tokenizer special tokens for Qwen2.5
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                self.logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
            
            # Ensure BOS token is properly configured for Qwen2.5
            if tokenizer.bos_token_id is None:
                if hasattr(tokenizer, 'im_start_id') and tokenizer.im_start_id is not None:
                    tokenizer.bos_token_id = tokenizer.im_start_id
                    self.logger.info(f"Set bos_token_id to im_start_id: {tokenizer.bos_token_id}")
                else:
                    # Fallback for Qwen2.5 - use the actual BOS token ID from vocab
                    if '<|im_start|>' in tokenizer.get_vocab():
                        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
                        tokenizer.bos_token = '<|im_start|>'
                        self.logger.info(f"Set bos_token_id to <|im_start|>: {tokenizer.bos_token_id}")
                    else:
                        # Ultimate fallback - use eos_token_id
                        tokenizer.bos_token_id = tokenizer.eos_token_id
                        tokenizer.bos_token = tokenizer.eos_token
                        self.logger.warning(f"Fallback: Set bos_token_id to eos_token_id: {tokenizer.bos_token_id}")
            
            # Verify tokenizer configuration
            self.logger.info(f"Final tokenizer config - BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}, PAD: {tokenizer.pad_token_id}")
            
            # Resize token embeddings if needed (for safety)
            model.resize_token_embeddings(len(tokenizer))
            
            # Set up chat template for Qwen2.5 if not present
            if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
                qwen_chat_template = """{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
                tokenizer.chat_template = qwen_chat_template
                self.logger.info("Applied Qwen2.5 chat template")

            # Check for cancellation after model loading
            if self._should_cancel():
                raise InterruptedError("Training cancelled during model loading")

            self._update_status("configuring", "Applying optimized LoRA configuration for Qwen2.5...")

            # Memory-optimized LoRA configuration for Qwen2.5
            # Adjust rank based on available memory and model size
            worker_memory_gb = float(os.environ.get("WORKER_MEMORY_GB", "6"))
            if worker_memory_gb <= 8:
                # Conservative settings for limited memory
                lora_rank = 8 if request.peft_method == "lora" else 16
                lora_alpha = 16
                self.logger.info(f"Using memory-optimized LoRA config: r={lora_rank}, alpha={lora_alpha}")
            else:
                # Standard settings for higher memory
                lora_rank = 16 if request.peft_method == "lora" else 32
                lora_alpha = 32
                self.logger.info(f"Using standard LoRA config: r={lora_rank}, alpha={lora_alpha}")

            # Qwen2.5-optimized target modules (includes all attention and MLP layers)
            qwen_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj",      # MLP layers
                # Add output projection for better performance
                "lm_head",
            ]
            
            # Apply LoRA adapter with Qwen2.5 optimizations
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_rank,
                target_modules=qwen_target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,  # Slight dropout for regularization
                bias="none",
                use_gradient_checkpointing="unsloth",  # Critical for memory efficiency
                random_state=3407,
                use_rslora=False,  # Keep disabled for stability
                loftq_config=None,
                # Qwen2.5-specific optimizations
                modules_to_save=None,  # Don't save embedding layers to save memory
            )
            
            self.logger.info(f"Applied LoRA with rank {lora_rank}, targeting {len(qwen_target_modules)} module types")

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
                self.task_id, self._update_status, self._should_cancel
            )

            # Create memory-optimized SFT trainer for Qwen2.5
            self._update_status("configuring_trainer", "Setting up optimized SFT trainer...")

            # Memory-optimized SFT configuration
            worker_memory_gb = float(os.environ.get("WORKER_MEMORY_GB", "6"))
            
            sft_config = SFTConfig(
                dataset_text_field="text",
                max_seq_length=min(request.max_seq_length, 2048 if worker_memory_gb <= 8 else request.max_seq_length),
                packing=True,  # Enable packing for 30% faster training
                dataset_num_proc=1 if worker_memory_gb <= 8 else 2,  # Reduce for memory
                # Qwen2.5-specific optimizations
                chars_per_token=3.5,  # Qwen2.5 average
                dataset_kwargs={
                    "add_special_tokens": True,
                    "split": "train"
                },
                # Memory optimizations
                dataset_batch_size=1000 if worker_memory_gb <= 8 else 2000,
                neftune_noise_alpha=None,  # Disable NEFTune to save memory
            )

            # CRITICAL: Validate and adjust precision settings for Unsloth compatibility
            training_args = self._validate_precision_settings(model, training_args)
            
            # Prepare dataset for training
            train_dataset = dataset["train"] if "train" in dataset else dataset
            eval_dataset = dataset.get("val") if worker_memory_gb > 8 else None  # Disable eval for memory
            
            # Log dataset info
            self.logger.info(f"Training dataset size: {len(train_dataset)}")
            if eval_dataset:
                self.logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
            else:
                self.logger.info("Evaluation disabled for memory optimization")

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=None,
                args=training_args,
                **sft_config.to_dict(),
            )
            
            self.logger.info(f"SFT trainer configured with max_seq_length={sft_config.max_seq_length}, packing={sft_config.packing}")

            # Add progress callback
            trainer.add_callback(progress_callback)

            # Start training with enhanced error handling
            self.logger.info("Starting Unsloth training with enhanced monitoring...")
            self._update_status("training", "Training started with memory monitoring...")

            # Check for cancellation one more time before training
            if self._should_cancel():
                raise InterruptedError("Training cancelled before start")

            # Enhanced training with memory monitoring and recovery
            try:
                # Log initial GPU memory state
                if torch.cuda.is_available():
                    gpu_mem_before = torch.cuda.memory_allocated() / 1024**3
                    gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    self.logger.info(f"GPU memory before training: {gpu_mem_before:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved")
                
                # Training with automatic recovery
                trainer.train()
                
                # Log GPU memory after training
                if torch.cuda.is_available():
                    gpu_mem_after = torch.cuda.memory_allocated() / 1024**3
                    self.logger.info(f"GPU memory after training: {gpu_mem_after:.2f}GB allocated")
                
            except torch.cuda.OutOfMemoryError as oom_error:
                self.logger.error(f"GPU Out of Memory during training: {oom_error}")
                
                # Emergency memory cleanup
                self._emergency_memory_cleanup()
                
                # Try to resume with even smaller batch size
                if training_args.per_device_train_batch_size > 1:
                    self.logger.info("Attempting recovery with smaller batch size...")
                    
                    # Update training args for recovery
                    recovery_args = training_args
                    recovery_args.per_device_train_batch_size = 1
                    recovery_args.gradient_accumulation_steps = min(32, recovery_args.gradient_accumulation_steps * 2)
                    recovery_args.dataloader_pin_memory = False
                    
                    # Recreate trainer with recovery settings
                    trainer = SFTTrainer(
                        model=model,
                        tokenizer=tokenizer,
                        train_dataset=train_dataset,
                        eval_dataset=None,  # Disable eval for recovery
                        data_collator=None,
                        args=recovery_args,
                        **sft_config.to_dict(),
                    )
                    trainer.add_callback(progress_callback)
                    
                    self._update_status("recovering", "Recovering training with reduced memory settings...")
                    trainer.train()
                    self.logger.info("Successfully recovered from OOM error")
                else:
                    # Cannot reduce batch size further
                    raise RuntimeError("Cannot recover from OOM - batch size already at minimum") from oom_error
                    
            except Exception as training_error:
                self.logger.error(f"Training failed with error: {training_error}")
                
                # Check if it's a tokenizer-related error
                if any(keyword in str(training_error).lower() for keyword in 
                       ["bos_token", "eos_token", "pad_token", "tokenizer", "token_id"]):
                    self.logger.error("Detected tokenizer configuration error - applying emergency fixes")
                    
                    # Emergency tokenizer fixes
                    if tokenizer.bos_token_id is None:
                        tokenizer.bos_token_id = tokenizer.eos_token_id
                        tokenizer.bos_token = tokenizer.eos_token
                        self.logger.info("Emergency fix: Set bos_token_id to eos_token_id")
                    
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token_id = tokenizer.eos_token_id
                        tokenizer.pad_token = tokenizer.eos_token
                        self.logger.info("Emergency fix: Set pad_token_id to eos_token_id")
                    
                    # Resize embeddings again
                    model.resize_token_embeddings(len(tokenizer))
                    self.logger.info("Emergency fix: Resized token embeddings")
                    
                    # Try training again with fixed tokenizer
                    self._update_status("retrying", "Retrying training with fixed tokenizer...")
                    trainer.train()
                    self.logger.info("Successfully recovered from tokenizer error")
                else:
                    # Re-raise non-tokenizer errors
                    raise

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
                        str(merged_dir), tokenizer, save_method="merged_16bit"
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
                "backend": "unsloth",
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
            tokenizer = AutoTokenizer.from_pretrained(
                request.base_model_dir, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                request.base_model_dir,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
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

            sft_config = SFTConfig(
                dataset_text_field="text",
                max_seq_length=request.max_seq_length,
                packing=False,
                dataset_num_proc=2,
            )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset["train"] if "train" in dataset else dataset,
                eval_dataset=dataset.get("val"),
                data_collator=None,
                args=training_args,
                **sft_config.to_dict(),
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
                "backend": "transformers",
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
        """Create memory-optimized training arguments for Qwen2.5 in worker process"""

        output_dir = Path(request.output_dir) / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Base configuration optimized for Qwen2.5
        args = {
            "per_device_train_batch_size": request.batch_size,
            "gradient_accumulation_steps": request.gradient_accumulation_steps,
            "warmup_steps": max(5, min(50, request.epochs * 10)),  # Dynamic warmup
            "num_train_epochs": request.epochs,
            "learning_rate": request.learning_rate,
            # Optimized precision for Qwen2.5
            "fp16": torch.cuda.is_available() and not (UNSLOTH_AVAILABLE and is_bfloat16_supported()),
            "bf16": torch.cuda.is_available() and UNSLOTH_AVAILABLE and is_bfloat16_supported(),
            "logging_steps": 5,  # More frequent logging for better monitoring
            "optim": "adamw_8bit" if UNSLOTH_AVAILABLE else "adamw_hf",
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",  # Better for Qwen2.5
            "seed": 3407,
            "output_dir": str(output_dir),
            "save_steps": 50,  # More frequent saves for recovery
            "save_total_limit": 3,  # Keep more checkpoints
            "dataloader_num_workers": 0,  # Avoid multiprocessing issues
            "remove_unused_columns": False,
            "run_name": f"llmforge_qwen25_{self.task_id[:8]}",
            # Memory optimization settings
            "dataloader_pin_memory": False,
            "torch_compile": False,  # Disable for stability
            "group_by_length": False,  # Disabled for memory consistency
            # Gradient optimization
            "max_grad_norm": 1.0,
            "prediction_loss_only": True,
            # Evaluation settings
            "eval_strategy": "no",  # Disable evaluation to save memory
            "save_safetensors": True,  # Use safetensors format
        }

        # Memory-specific optimizations based on container constraints
        worker_memory_gb = float(os.environ.get("WORKER_MEMORY_GB", "6"))
        container_memory_gb = 8.0  # Your container limit
        
        if worker_memory_gb <= container_memory_gb:
            # Aggressive memory optimization for 8GB container
            memory_optimized_args = {
                "per_device_train_batch_size": min(request.batch_size, 1),  # Ultra-small batch
                "gradient_accumulation_steps": max(request.gradient_accumulation_steps, 16),  # Compensate with accumulation
                "dataloader_pin_memory": False,
                "save_steps": 25,  # More frequent saves
                "logging_steps": 5,
                # Additional memory optimizations
                "report_to": [],  # Disable wandb/tensorboard
                "load_best_model_at_end": False,  # Don't keep best model in memory
                "metric_for_best_model": None,
                "greater_is_better": None,
                "label_smoothing_factor": 0.0,  # Disable label smoothing to save memory
            }
            args.update(memory_optimized_args)
            self.logger.info(f"Applied aggressive memory optimization for {worker_memory_gb}GB limit")
            
        elif worker_memory_gb < 12:
            # Moderate optimization for medium memory
            moderate_args = {
                "per_device_train_batch_size": min(request.batch_size, 2),
                "gradient_accumulation_steps": max(request.gradient_accumulation_steps, 8),
                "dataloader_pin_memory": False,
            }
            args.update(moderate_args)
            self.logger.info(f"Applied moderate memory optimization for {worker_memory_gb}GB")
        
        # Qwen2.5-specific training optimizations
        qwen_optimizations = {
            # Learning rate optimization for Qwen2.5
            "warmup_ratio": 0.05,  # 5% warmup
            # Stability improvements
            "ddp_find_unused_parameters": False,
            "ddp_bucket_cap_mb": 25,  # Smaller buckets for memory
        }
        args.update(qwen_optimizations)
        
        # Log final configuration
        effective_batch_size = args["per_device_train_batch_size"] * args["gradient_accumulation_steps"]
        self.logger.info(f"Training config: batch_size={args['per_device_train_batch_size']}, "
                        f"grad_accum={args['gradient_accumulation_steps']}, "
                        f"effective_batch={effective_batch_size}, "
                        f"lr={args['learning_rate']}, "
                        f"precision={'bf16' if args.get('bf16') else 'fp16' if args.get('fp16') else 'fp32'}")

        return TrainingArguments(**args)
    
    def _validate_precision_settings(self, model, training_args):
        """Validate and adjust precision settings for Unsloth compatibility"""
        try:
            # Detect model's actual precision
            model_dtype = None
            if hasattr(model, 'dtype'):
                model_dtype = model.dtype
            elif hasattr(model, 'model') and hasattr(model.model, 'dtype'):
                model_dtype = model.model.dtype
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'dtype'):
                model_dtype = model.base_model.dtype
            
            # Extract dtype string
            dtype_str = str(model_dtype) if model_dtype else "unknown"
            self.logger.info(f"Detected model precision: {dtype_str}")
            
            # Check current training args precision settings
            current_fp16 = getattr(training_args, 'fp16', False)
            current_bf16 = getattr(training_args, 'bf16', False)
            
            self.logger.info(f"Training args precision: fp16={current_fp16}, bf16={current_bf16}")
            
            # Determine optimal precision based on model and hardware
            if "bfloat16" in dtype_str.lower():
                # Model is in bfloat16, adjust training args accordingly
                if current_fp16 and not current_bf16:
                    self.logger.warning("Model is in bfloat16 but training args use fp16. Adjusting to bf16...")
                    # Create new training args with corrected precision
                    args_dict = training_args.to_dict()
                    args_dict['fp16'] = False
                    args_dict['bf16'] = True
                    training_args = type(training_args)(**args_dict)
                    self.logger.info("Adjusted training precision to bfloat16 for Unsloth compatibility")
            elif "float16" in dtype_str.lower():
                # Model is in float16, ensure training args match
                if current_bf16 and not current_fp16:
                    self.logger.warning("Model is in float16 but training args use bf16. Adjusting to fp16...")
                    args_dict = training_args.to_dict()
                    args_dict['fp16'] = True
                    args_dict['bf16'] = False
                    training_args = type(training_args)(**args_dict)
                    self.logger.info("Adjusted training precision to float16 for Unsloth compatibility")
            
            # Final validation
            final_fp16 = getattr(training_args, 'fp16', False)
            final_bf16 = getattr(training_args, 'bf16', False)
            self.logger.info(f"Final training precision: fp16={final_fp16}, bf16={final_bf16}")
            
            return training_args
            
        except Exception as e:
            self.logger.error(f"Error validating precision settings: {e}")
            self.logger.info("Continuing with original training args...")
            return training_args

    def _emergency_memory_cleanup(self):
        """Emergency memory cleanup during OOM recovery"""
        try:
            self.logger.warning("Performing emergency memory cleanup...")
            
            # Force garbage collection multiple times
            for i in range(3):
                gc.collect()
            
            # Clear all CUDA caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                
                # Log memory status
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                gpu_reserved = torch.cuda.memory_reserved() / 1024**3
                self.logger.info(f"After emergency cleanup: {gpu_mem:.2f}GB allocated, {gpu_reserved:.2f}GB reserved")
            
            # Brief pause to let cleanup complete
            import time
            time.sleep(2)
            
        except Exception as e:
            self.logger.error(f"Error during emergency cleanup: {e}")

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
