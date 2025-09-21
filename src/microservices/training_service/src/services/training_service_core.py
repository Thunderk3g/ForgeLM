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
        # Initialize hardware detection and adaptive configuration
        from utils.hardware_detection import hardware_detector, AdaptiveTrainingConfig

        self.hardware_profile = hardware_detector.detect_full_profile()
        self.adaptive_config = AdaptiveTrainingConfig(self.hardware_profile)

        # Register cleanup

        # Log hardware profile and optimizations
        hardware_summary = self.adaptive_config.get_hardware_summary()
        self.logger.info(
            f"TrainingService initialized with adaptive hardware optimization"
        )
        self.logger.info(
            f"Hardware Profile: GPU={hardware_summary['gpu']}, RAM={hardware_summary['memory']}, CPU={hardware_summary['cpu']}"
        )
        self.logger.info(
            f"Environment: {hardware_summary['environment']}, Storage={hardware_summary['storage']}"
        )
        self.logger.info(f"Unsloth available: {UNSLOTH_AVAILABLE}")
        
        

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
            # Optimized model loading for RTX 4060 8GB with Unsloth 2025.9.5+
            worker_memory_gb = float(os.environ.get("WORKER_MEMORY_GB", "6"))
            
            # RTX 4060 8GB optimal loading configuration
            loading_config = {
                "model_name": request.base_model_dir,
                "max_seq_length": 2048 if worker_memory_gb <= 8 else request.max_seq_length,
                "dtype": None,  # Let Unsloth auto-detect for optimal performance
                "load_in_4bit": True,  # Essential for 8GB VRAM - 4x memory reduction
                "trust_remote_code": True,  # Required for Qwen2.5
            }
            
            # Hardware-specific optimizations
            if worker_memory_gb <= 8:
                loading_config.update({
                    "device_map": "auto",  # Let Unsloth handle device mapping
                    "attn_implementation": "flash_attention_2",  # Better memory efficiency
                    # Additional 8GB optimizations
                    "torch_dtype": torch.float16,  # Explicit for compatibility
                })
            else:
                loading_config.update({
                    "device_map": "balanced",  # Better for multi-GPU setups
                    "attn_implementation": "flash_attention_2",
                })
            
            self.logger.info(f"Loading model with RTX 4060 8GB optimized config: 4bit={loading_config['load_in_4bit']}, max_seq={loading_config['max_seq_length']}")
            
            model, tokenizer = FastLanguageModel.from_pretrained(**loading_config)
            
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

            # Optimized LoRA configuration for Qwen2.5 based on latest Unsloth best practices
            # RTX 4060 8GB optimal settings based on community benchmarks
            worker_memory_gb = float(os.environ.get("WORKER_MEMORY_GB", "6"))
            if worker_memory_gb <= 8:
                # Optimal settings for RTX 4060 8GB - based on Unsloth 2025.9.5+ benchmarks
                lora_rank = 16  # Sweet spot for 8GB VRAM - proven optimal in community testing
                lora_alpha = 32  # 2x rank ratio for stable training
                self.logger.info(f"Using RTX 4060 8GB optimized LoRA config: r={lora_rank}, alpha={lora_alpha}")
            else:
                # Higher memory configurations can use more aggressive settings
                lora_rank = 32 if request.peft_method == "lora" else 64
                lora_alpha = 64
                self.logger.info(f"Using high-memory LoRA config: r={lora_rank}, alpha={lora_alpha}")

            # Qwen2.5-optimized target modules based on latest Unsloth recommendations
            # Research shows targeting all major layers is crucial for matching full fine-tuning performance
            qwen_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # All attention layers (essential)
                "gate_proj", "up_proj", "down_proj",      # All MLP layers (recommended for Qwen2.5)
                # Note: embedding layers excluded to save memory on 8GB GPUs
                # "embed_tokens", "lm_head"  # Uncomment for higher VRAM setups
            ]
            
            # Apply LoRA adapter with Qwen2.5 optimizations
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_rank,
                target_modules=qwen_target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,  # Optimized for Unsloth fast patching  # Slight dropout for regularization
                bias="none",
                use_gradient_checkpointing="unsloth",  # 30% less VRAM + 2x larger batch sizes
                random_state=3407,
                use_rslora=False,  # Keep disabled for stability
                loftq_config=None,
                # Qwen2.5-specific optimizations
                modules_to_save=None,  # Exclude embeddings for 8GB VRAM optimization
            )
            
            self.logger.info(f"Applied LoRA with rank {lora_rank}, alpha {lora_alpha}, targeting {len(qwen_target_modules)} module types for optimal 8GB VRAM performance")

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

            # Setup optimized training arguments for RTX 4060 8GB
            training_args = self._create_adaptive_training_args(request)

            # Create progress callback
            progress_callback = WorkerProgressCallback(
                self.task_id, self._update_status, self._should_cancel
            )

            # Create memory-optimized SFT trainer for Qwen2.5
            self._update_status("configuring_trainer", "Setting up optimized SFT trainer...")

            # Memory-optimized SFT configuration
            worker_memory_gb = float(os.environ.get("WORKER_MEMORY_GB", "6"))
            
            # Optimized SFT configuration for RTX 4060 8GB based on Unsloth benchmarks
            if worker_memory_gb <= 8:
                # RTX 4060 8GB optimal settings
                optimal_seq_length = 2048  # Unsloth recommended for 8GB testing
                dataset_num_proc = 1  # Single process to avoid memory overhead
                dataset_batch_size = 1000  # Conservative for memory
            else:
                # Higher memory configurations
                optimal_seq_length = min(request.max_seq_length, 4096)
                dataset_num_proc = 2
                dataset_batch_size = 2000
            
            sft_config = SFTConfig(
                dataset_text_field="text",
                max_seq_length=optimal_seq_length,
                packing=True,  # Critical: 30% faster training with Unsloth
                dataset_num_proc=dataset_num_proc,
                # Qwen2.5-specific optimizations
                chars_per_token=3.5,  # Qwen2.5 tokenizer average
                dataset_kwargs={
                    "add_special_tokens": True,
                    "split": "train"
                },
                # Memory optimizations
                dataset_batch_size=dataset_batch_size,
                neftune_noise_alpha=None,  # Disable NEFTune to save memory
            )
            
            self.logger.info(f"SFT Config optimized for {worker_memory_gb}GB: max_seq_length={optimal_seq_length}, packing=True")

            # CRITICAL: Validate and adjust precision settings for Unsloth compatibility
            # This prevents "Unsloth: Model is in bfloat16 precision but you want to use float16 precision" errors
            self.logger.info("Validating precision settings to prevent Unsloth compatibility errors...")
            training_args = self._validate_precision_settings(model, training_args)
            
            # Prepare dataset for training
            train_dataset = dataset["train"] if "train" in dataset else dataset
            eval_dataset = dataset["val"] if "val" in dataset and worker_memory_gb > 8 else None  # Disable eval for memory
            
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
            
            # SFT configuration logging moved above for clarity

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
                    lora_dropout=0.0,  # Optimized for Unsloth fast patching
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
            training_args = self._create_adaptive_training_args(request)

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
                eval_dataset=dataset["val"] if "val" in dataset else None,
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
    def _create_adaptive_training_args(
        self, request: FineTuneRequest
    ):
        """Create adaptive training arguments optimized for hardware profile
        
        Args:
            request: Fine-tuning parameters
        """

        output_dir = Path(request.output_dir) / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get adaptive configuration from hardware profile
        adaptive_config = self.adaptive_config.generate_optimal_config(request)
        adaptive_args = adaptive_config["training_args"]
        gpu_opts = adaptive_config["gpu_optimizations"]

        # CRITICAL FIX: Use Unsloth's built-in precision detection instead of manual guessing
        optimal_precision = self._get_optimal_precision_settings()
        self.logger.info(
            f"Precision detection for training args: {optimal_precision['reason']}\n"
            f"  Setting fp16={optimal_precision['fp16']}, bf16={optimal_precision['bf16']}"
        )

        # Merge base args with adaptive configuration
        args = {
            "per_device_train_batch_size": base_args.get("per_device_train_batch_size", adaptive_args["per_device_train_batch_size"]),
            "gradient_accumulation_steps": base_args.get("gradient_accumulation_steps", adaptive_args["gradient_accumulation_steps"]),
            "warmup_steps": base_args.get("warmup_steps", 5),
            "num_train_epochs": request.epochs,
            "learning_rate": base_args.get("learning_rate", request.learning_rate),
            # FIXED: Use optimal precision detection instead of flawed gpu_opts guessing
            "fp16": optimal_precision["fp16"],
            "bf16": optimal_precision["bf16"],
            # Adaptive optimization
            "logging_steps": base_args.get("logging_steps", adaptive_args.get("logging_steps", 10)),
            "optim": (
                "adamw_8bit"
                if UNSLOTH_AVAILABLE and gpu_opts["enabled"]
                else "adamw_hf"
            ),
            "weight_decay": base_args.get("weight_decay", 0.01),
            "lr_scheduler_type": base_args.get("lr_scheduler_type", "linear"),
            "seed": 3407,
            "output_dir": str(output_dir),
            # Adaptive checkpoint and evaluation settings
            "save_steps": base_args.get("save_steps", adaptive_args.get("save_steps", 100)),
            "save_total_limit": adaptive_args.get("save_total_limit", 2),
            "eval_strategy": adaptive_args.get("eval_strategy", "no"),
            # Adaptive data loading
            "dataloader_num_workers": base_args.get("dataloader_num_workers", adaptive_args.get("dataloader_num_workers", 0)),
            "dataloader_pin_memory": base_args.get("dataloader_pin_memory", adaptive_args.get("dataloader_pin_memory", False)),
            # System settings
            "remove_unused_columns": base_args.get("remove_unused_columns", False),
            "run_name": f"llmforge_{training_type}_{task_id[:8]}",
            "report_to": adaptive_args.get("report_to", []),
        }

        # Apply missing Unsloth optimizations
        if UNSLOTH_AVAILABLE:
            unsloth_optimizations = {
                "gradient_checkpointing": True,
                "max_grad_norm": 0.3,
                # Remove invalid TrainingArguments parameters - these should be set elsewhere
                # "adam_beta1": 0.9,  # Not a TrainingArguments parameter
                # "adam_beta2": 0.95,  # Not a TrainingArguments parameter  
                # "adam_epsilon": 1e-5,  # Not a TrainingArguments parameter
                # "dataloader_persistent_workers": False,  # Not a TrainingArguments parameter
                # "auto_find_batch_size": False,  # Not a TrainingArguments parameter
            }
            args.update(unsloth_optimizations)

        # Apply Unsloth-specific optimizations for RTX 4060 8GB
        if UNSLOTH_AVAILABLE:
            args.update({
                "gradient_checkpointing": True,  # Essential for memory efficiency
                "max_grad_norm": 0.3,  # Unsloth recommended for stability
                "lr_scheduler_type": "cosine",  # Better than linear for Qwen2.5
                "warmup_ratio": 0.05,  # 5% warmup for stable training
            })

        # Log final optimization summary
        precision_str = 'bf16' if args['bf16'] else 'fp16' if args['fp16'] else 'fp32'
        effective_batch = args['per_device_train_batch_size'] * args['gradient_accumulation_steps']
        
        self.logger.info(
            f"🚀 RTX 4060 8GB Optimized Configuration:"
            f"\n  📊 Batch: {args['per_device_train_batch_size']} × {args['gradient_accumulation_steps']} = {effective_batch} effective"
            f"\n  🎯 Precision: {precision_str} ({optimal_precision['reason']})"
            f"\n  🧠 Optimizer: {args['optim']}"
            f"\n  ⚡ Unsloth: {'Enabled' if UNSLOTH_AVAILABLE else 'Disabled'}"
        )

        # Final validation to prevent precision errors
        if args.get('fp16') and args.get('bf16'):
            # Fix conflict by using optimal precision
            args['fp16'] = not optimal_precision['bf16']
            args['bf16'] = optimal_precision['bf16']
            self.logger.warning("Fixed precision conflict: disabled conflicting settings")

        return TrainingArguments(**args)

    def _get_instruction_training_args(self, adaptive_config: Dict[str, Any], request: FineTuneRequest) -> Dict[str, Any]:
        """Get training arguments optimized for instruction tuning"""
        adaptive_args = adaptive_config["training_args"]
        return {
            "per_device_train_batch_size": adaptive_args["per_device_train_batch_size"],
            "gradient_accumulation_steps": adaptive_args["gradient_accumulation_steps"],
            "warmup_steps": 5,
            "learning_rate": request.learning_rate,
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "logging_steps": adaptive_args.get("logging_steps", 10),
            "save_steps": adaptive_args.get("save_steps", 100),
            "dataloader_num_workers": adaptive_args.get("dataloader_num_workers", 0),
            "dataloader_pin_memory": adaptive_args.get("dataloader_pin_memory", False),
            "remove_unused_columns": False,
        }

    def _get_chat_completion_args(self, adaptive_config: Dict[str, Any], request: FineTuneRequest) -> Dict[str, Any]:
        """Get training arguments optimized for chat completion"""
        adaptive_args = adaptive_config["training_args"]
        return {
            "per_device_train_batch_size": max(1, adaptive_args["per_device_train_batch_size"] // 2),
            "gradient_accumulation_steps": adaptive_args["gradient_accumulation_steps"] * 2,
            "warmup_steps": 10,
            "learning_rate": request.learning_rate * 0.8,
            "weight_decay": 0.005,
            "lr_scheduler_type": "cosine",
            "logging_steps": adaptive_args.get("logging_steps", 5),
            "save_steps": adaptive_args.get("save_steps", 200),
            "dataloader_num_workers": min(2, adaptive_args.get("dataloader_num_workers", 1)),
            "dataloader_pin_memory": False,
            "remove_unused_columns": False,
            "group_by_length": True,
            "prediction_loss_only": False,
        }

    def _get_reasoning_training_args(self, adaptive_config: Dict[str, Any], request: FineTuneRequest) -> Dict[str, Any]:
        """Get training arguments optimized for reasoning/GRPO training"""
        adaptive_args = adaptive_config["training_args"]
        return {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": adaptive_args["gradient_accumulation_steps"] * 4,
            "warmup_steps": 20,
            "learning_rate": request.learning_rate * 0.5,
            "weight_decay": 0.001,
            "lr_scheduler_type": "cosine",
            "logging_steps": adaptive_args.get("logging_steps", 5),
            "save_steps": adaptive_args.get("save_steps", 500),
            "dataloader_num_workers": 1,
            "dataloader_pin_memory": False,
            "remove_unused_columns": False,
            "group_by_length": False,
            "prediction_loss_only": False,
        }

    def _apply_training_type_optimizations(
        self, args: Dict[str, Any], training_type: str, gpu_opts: Dict[str, Any], adaptive_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply training type specific Unsloth optimizations"""
        
        if not UNSLOTH_AVAILABLE:
            return args
        
        if training_type == "chat":
            chat_optimizations = {
                "gradient_checkpointing": True,
                # "eval_accumulation_steps": 4,  # Not a TrainingArguments parameter
            }
            
            # GRPO specific settings if reasoning is enabled
            if adaptive_config.get("enable_reasoning", False):
                chat_optimizations.update({
                    "group_by_length": False,
                    "prediction_loss_only": False,
                })
            
            args.update(chat_optimizations)
            
        elif training_type == "reasoning":
            reasoning_optimizations = {
                "gradient_checkpointing": True,
                # "eval_accumulation_steps": 8,  # Not a TrainingArguments parameter
                "group_by_length": False,
                "prediction_loss_only": False,
            }
            args.update(reasoning_optimizations)
            
        return args

    def get_chat_sft_config(self, request: FineTuneRequest, adaptive_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get SFTConfig optimized for chat completion"""
        
        chat_config = adaptive_config.get("chat_config", {})
        
        sft_config = {
            "dataset_text_field": "text",
            "max_seq_length": min(request.max_seq_length, 4096),
            "packing": True,
            "dataset_num_proc": 2,
            "chars_per_token": 3.5,
        }
        
        # Chat specific configurations
        if UNSLOTH_AVAILABLE:
            sft_config.update({
                "dataset_kwargs": {
                    "add_special_tokens": False,
                    "chat_template": chat_config.get("template_type", "chatml"),
                },
                "use_unsloth_template": True,
                "assistant_only_loss": chat_config.get("assistant_only_loss", True),
            })
        
        return sft_config

    def _detect_model_precision(self, model) -> str:
        """Detect the actual precision of the loaded model"""
        try:
            # Try multiple ways to get model dtype
            model_dtype = None
            
            # Method 1: Direct dtype attribute
            if hasattr(model, 'dtype'):
                model_dtype = model.dtype
            # Method 2: Through model.model (for wrapped models)
            elif hasattr(model, 'model') and hasattr(model.model, 'dtype'):
                model_dtype = model.model.dtype
            # Method 3: Check first parameter dtype
            elif hasattr(model, 'parameters'):
                try:
                    model_dtype = next(model.parameters()).dtype
                except (StopIteration, AttributeError):
                    pass
            # Method 4: For Unsloth models
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'dtype'):
                model_dtype = model.base_model.dtype
            
            # Normalize dtype to string
            if model_dtype is not None:
                if model_dtype == torch.bfloat16:
                    return "bfloat16"
                elif model_dtype == torch.float16:
                    return "float16"
                elif model_dtype == torch.float32:
                    return "float32"
                else:
                    return str(model_dtype).split('.')[-1]  # Extract dtype name
            
            # Fallback: assume float16 for safety
            self.logger.warning("Could not detect model precision, defaulting to float16")
            return "float16"
            
        except Exception as e:
            self.logger.error(f"Error detecting model precision: {e}")
            return "float16"  # Safe fallback

    def _get_optimal_precision_settings(self, model=None, gpu_opts=None) -> Dict[str, Any]:
        """Get optimal precision settings for Unsloth training"""
        try:
            # Detect model precision if model is available
            model_precision = None
            if model is not None:
                model_precision = self._detect_model_precision(model)
                self.logger.info(f"Detected model precision: {model_precision}")
            
            # Check Unsloth hardware support
            if UNSLOTH_AVAILABLE:
                try:
                    supports_bf16 = is_bfloat16_supported()
                    self.logger.info(f"Unsloth hardware bf16 support: {supports_bf16}")
                except Exception as e:
                    self.logger.warning(f"Could not check Unsloth bf16 support: {e}")
                    supports_bf16 = torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
            else:
                supports_bf16 = torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
            
            # Decision logic based on model and hardware
            if model_precision == "bfloat16":
                if supports_bf16:
                    return {
                        "fp16": False,
                        "bf16": True,
                        "reason": f"Model is bf16, hardware supports bf16"
                    }
                else:
                    self.logger.warning("Model is bf16 but hardware doesn't support bf16, using fp16 with potential conversion")
                    return {
                        "fp16": True,
                        "bf16": False,
                        "reason": "Model is bf16 but hardware doesn't support bf16"
                    }
            elif model_precision == "float16":
                return {
                    "fp16": True,
                    "bf16": False,
                    "reason": "Model is fp16"
                }
            else:
                # Model is fp32 or unknown, choose based on hardware
                if supports_bf16:
                    return {
                        "fp16": False,
                        "bf16": True,
                        "reason": f"Model is {model_precision}, using hardware optimal bf16"
                    }
                else:
                    return {
                        "fp16": True,
                        "bf16": False,
                        "reason": f"Model is {model_precision}, hardware optimal fp16"
                    }
                    
        except Exception as e:
            self.logger.error(f"Error determining optimal precision: {e}")
            # Emergency fallback
            return {
                "fp16": True,
                "bf16": False,
                "reason": "Emergency fallback to fp16"
            }

    def _validate_precision_settings(self, training_args, model=None) -> TrainingArguments:
        """Validate and fix precision settings to prevent Unsloth errors"""
        try:
            # Get optimal settings
            optimal_settings = self._get_optimal_precision_settings(model)
            
            # Check current settings
            current_fp16 = getattr(training_args, 'fp16', False)
            current_bf16 = getattr(training_args, 'bf16', False)
            
            # Detect issues
            needs_adjustment = False
            
            # Issue 1: Both fp16 and bf16 enabled
            if current_fp16 and current_bf16:
                self.logger.warning("Both fp16 and bf16 are enabled, this will cause conflicts")
                needs_adjustment = True
            
            # Issue 2: Model-training precision mismatch
            if model is not None:
                model_precision = self._detect_model_precision(model)
                if model_precision == "bfloat16" and current_fp16 and not current_bf16:
                    self.logger.warning("Model is bf16 but training args specify fp16 - this causes Unsloth errors")
                    needs_adjustment = True
                elif model_precision == "float16" and current_bf16 and not current_fp16:
                    self.logger.warning("Model is fp16 but training args specify bf16 - potential inefficiency")
                    needs_adjustment = True
            
            # Apply fixes if needed
            if needs_adjustment:
                self.logger.info(f"Adjusting precision settings: {optimal_settings['reason']}")
                
                # Convert training args to dict, modify, and recreate
                args_dict = training_args.to_dict()
                args_dict['fp16'] = optimal_settings['fp16']
                args_dict['bf16'] = optimal_settings['bf16']
                
                # Recreate training arguments
                training_args = TrainingArguments(**args_dict)
                
                self.logger.info(f"Fixed precision settings - fp16: {optimal_settings['fp16']}, bf16: {optimal_settings['bf16']}")
            else:
                self.logger.info(f"Precision settings look good - fp16: {current_fp16}, bf16: {current_bf16}")
            
            return training_args
            
        except Exception as e:
            self.logger.error(f"Error validating precision settings: {e}")
            return training_args  # Return original if validation fails

    def _create_memory_optimized_training_args(self, request: FineTuneRequest) -> TrainingArguments:
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
    
    def _detect_model_precision(self, model):
        """Detect the actual precision of the model with comprehensive checking"""
        try:
            model_dtype = None
            dtype_source = "unknown"
            
            # Try multiple ways to get model dtype
            if hasattr(model, 'dtype'):
                model_dtype = model.dtype
                dtype_source = "model.dtype"
            elif hasattr(model, 'model') and hasattr(model.model, 'dtype'):
                model_dtype = model.model.dtype
                dtype_source = "model.model.dtype"
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'dtype'):
                model_dtype = model.base_model.dtype
                dtype_source = "model.base_model.dtype"
            elif hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'dtype'):
                model_dtype = model.model.model.dtype
                dtype_source = "model.model.model.dtype"
            else:
                # Try to get dtype from first parameter
                try:
                    first_param = next(model.parameters())
                    model_dtype = first_param.dtype
                    dtype_source = "first_parameter.dtype"
                except (StopIteration, AttributeError):
                    pass
            
            if model_dtype:
                dtype_str = str(model_dtype).lower()
                self.logger.info(f"Detected model precision: {model_dtype} (source: {dtype_source})")
                
                # Normalize dtype detection
                if "bfloat16" in dtype_str or "bf16" in dtype_str:
                    return "bfloat16"
                elif "float16" in dtype_str or "fp16" in dtype_str or "half" in dtype_str:
                    return "float16"
                elif "float32" in dtype_str or "fp32" in dtype_str or "float" in dtype_str:
                    return "float32"
                else:
                    self.logger.warning(f"Unknown dtype detected: {model_dtype}")
                    return "unknown"
            else:
                self.logger.warning("Could not detect model dtype")
                return "unknown"
                
        except Exception as e:
            self.logger.error(f"Error detecting model precision: {e}")
            return "unknown"
    
    def _get_optimal_precision_settings(self, model=None):
        """Get optimal precision settings using Unsloth's built-in detection"""
        try:
            # First, use Unsloth's hardware detection if available
            if UNSLOTH_AVAILABLE:
                supports_bf16 = is_bfloat16_supported()
                self.logger.info(f"Unsloth hardware check - supports bfloat16: {supports_bf16}")
                
                # If model is provided, also check model precision
                model_precision = None
                if model is not None:
                    model_precision = self._detect_model_precision(model)
                    self.logger.info(f"Model precision detected: {model_precision}")
                
                # Decision logic based on both hardware and model
                if model_precision == "bfloat16":
                    if supports_bf16:
                        return {"fp16": False, "bf16": True, "reason": "Model in bf16, hardware supports bf16"}
                    else:
                        self.logger.warning("Model is bf16 but hardware doesn't support bf16. This may cause issues.")
                        return {"fp16": True, "bf16": False, "reason": "Model in bf16 but hardware limited to fp16"}
                
                elif model_precision == "float16":
                    return {"fp16": True, "bf16": False, "reason": "Model in fp16"}
                
                else:
                    # Use Unsloth's recommended settings for unknown/fp32 models
                    return {
                        "fp16": not supports_bf16,
                        "bf16": supports_bf16,
                        "reason": f"Hardware-optimized: bf16_supported={supports_bf16}"
                    }
            else:
                # Fallback when Unsloth not available
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    # Try to detect bf16 support manually
                    try:
                        # Test bf16 support
                        test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device='cuda')
                        supports_bf16 = True
                        del test_tensor
                    except:
                        supports_bf16 = False
                    
                    return {
                        "fp16": not supports_bf16,
                        "bf16": supports_bf16,
                        "reason": f"Manual detection: bf16_supported={supports_bf16}"
                    }
                else:
                    return {"fp16": False, "bf16": False, "reason": "No CUDA available"}
                    
        except Exception as e:
            self.logger.error(f"Error determining optimal precision: {e}")
            # Safe fallback
            return {"fp16": torch.cuda.is_available(), "bf16": False, "reason": "Error fallback"}
    
    def _validate_precision_settings(self, model, training_args):
        """Validate and adjust precision settings for Unsloth compatibility with comprehensive detection"""
        try:
            # Get optimal precision settings
            optimal_settings = self._get_optimal_precision_settings(model)
            
            # Check current training args precision settings
            current_fp16 = getattr(training_args, 'fp16', False)
            current_bf16 = getattr(training_args, 'bf16', False)
            
            self.logger.info(
                f"Current training args: fp16={current_fp16}, bf16={current_bf16}\n"
                f"Optimal settings: fp16={optimal_settings['fp16']}, bf16={optimal_settings['bf16']}\n"
                f"Reason: {optimal_settings['reason']}"
            )
            
            # Check if adjustment is needed
            needs_adjustment = (
                current_fp16 != optimal_settings['fp16'] or 
                current_bf16 != optimal_settings['bf16']
            )
            
            if needs_adjustment:
                self.logger.warning(
                    f"Precision mismatch detected! Adjusting training args to prevent Unsloth errors:\n"
                    f"  Changing fp16: {current_fp16} -> {optimal_settings['fp16']}\n"
                    f"  Changing bf16: {current_bf16} -> {optimal_settings['bf16']}"
                )
                
                # Create new training args with corrected precision
                try:
                    args_dict = training_args.to_dict()
                    args_dict['fp16'] = optimal_settings['fp16']
                    args_dict['bf16'] = optimal_settings['bf16']
                    training_args = type(training_args)(**args_dict)
                    
                    self.logger.info(
                        f"✅ Successfully adjusted precision settings for Unsloth compatibility\n"
                        f"   Final settings: fp16={optimal_settings['fp16']}, bf16={optimal_settings['bf16']}"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to create new training args: {e}")
                    # Try alternative approach
                    try:
                        training_args.fp16 = optimal_settings['fp16']
                        training_args.bf16 = optimal_settings['bf16']
                        self.logger.info("✅ Successfully adjusted precision using direct assignment")
                    except Exception as e2:
                        self.logger.error(f"Failed to adjust precision settings: {e2}")
                        raise RuntimeError(
                            f"Could not adjust precision settings. Original error: {e}, Direct assignment error: {e2}"
                        )
            else:
                self.logger.info("✅ Precision settings are already optimal for Unsloth")
            
            # Final validation and logging
            final_fp16 = getattr(training_args, 'fp16', False)
            final_bf16 = getattr(training_args, 'bf16', False)
            
            # Validate final settings make sense
            if final_fp16 and final_bf16:
                self.logger.error("❌ Both fp16 and bf16 are enabled - this will cause errors!")
                # Force to optimal single precision
                if UNSLOTH_AVAILABLE and is_bfloat16_supported():
                    training_args.fp16 = False
                    training_args.bf16 = True
                    self.logger.info("🔧 Fixed: Disabled fp16, kept bf16")
                else:
                    training_args.fp16 = True
                    training_args.bf16 = False
                    self.logger.info("🔧 Fixed: Disabled bf16, kept fp16")
                    
            elif not final_fp16 and not final_bf16 and torch.cuda.is_available():
                self.logger.warning("⚠️  Neither fp16 nor bf16 enabled on CUDA - training will be slower")
            
            # Log final configuration
            final_fp16 = getattr(training_args, 'fp16', False)
            final_bf16 = getattr(training_args, 'bf16', False)
            self.logger.info(
                f"🎯 Final precision configuration: fp16={final_fp16}, bf16={final_bf16}\n"
                f"   This should prevent 'Unsloth: Model is in bfloat16 precision but you want to use float16 precision' errors"
            )
            
            return training_args
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in precision validation: {e}")
            self.logger.error("This may cause Unsloth precision mismatch errors during training!")
            
            # Emergency fallback - use Unsloth recommended settings
            try:
                if UNSLOTH_AVAILABLE:
                    emergency_bf16 = is_bfloat16_supported()
                    training_args.fp16 = not emergency_bf16
                    training_args.bf16 = emergency_bf16
                    self.logger.info(f"🚨 Applied emergency precision settings: fp16={not emergency_bf16}, bf16={emergency_bf16}")
                else:
                    training_args.fp16 = torch.cuda.is_available()
                    training_args.bf16 = False
                    self.logger.info(f"🚨 Applied fallback precision settings: fp16={torch.cuda.is_available()}, bf16=False")
            except Exception as e2:
                self.logger.error(f"❌ Emergency fallback failed: {e2}")
                
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
