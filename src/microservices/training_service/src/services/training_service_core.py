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
from utils.factory_bridge import causal_lm_load_plan, resolve_peft_config, torch_device_and_dtype
from services.training_common import TrainingArgsMixin

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
            
            # FIXED: Enhanced logging for epoch begin to detect hangs
            if epoch < 1:  # Only for first epoch
                self.logger.info(f"🚨 Epoch {epoch} begin - about to enter first training loop iteration")
                if torch.cuda.is_available():
                    epoch_mem = torch.cuda.memory_allocated() / 1024**3
                    self.logger.info(f"🚨 Memory at epoch begin {epoch}: {epoch_mem:.2f}GB allocated")
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


class TrainingCore(TrainingArgsMixin):
    """Core training logic for worker processes

    The per-training-type argument helpers (``_get_instruction_training_args``,
    ``_get_chat_completion_args``, ``_get_reasoning_training_args``,
    ``_apply_training_type_optimizations``, ``get_chat_sft_config``) come from
    ``TrainingArgsMixin``. ``TrainingService`` inherits the same ones.
    """

    def _unsloth_available(self) -> bool:
        # The mixin asks the class instead of importing Unsloth itself, because
        # this module and training_service.py decide the flag differently (see
        # training_common.py). Read on each call, as the helpers always did.
        return UNSLOTH_AVAILABLE

    def __init__(self, task_id: str, worker_mode: bool = True):
        self.task_id = task_id
        self.worker_mode = worker_mode
        self.logger = get_logger(f"worker.core.{task_id}")
        self.status_file = Path("logs") / f"{task_id}_status.json"
        self.cancellation_callback: Optional[Callable] = None
        
        # FIXED: Add signal handlers to prevent unexpected container restarts
        try:
            import signal
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            self.logger.info("Signal handlers registered for graceful shutdown")
        except Exception as e:
            self.logger.warning(f"Could not register signal handlers: {e}")
        
        # FIXED: Add memory monitoring flag
        self._memory_monitoring_active = False
        self._memory_monitor_thread = None
        
        # Initialize hardware detection and adaptive configuration
        from utils.hardware_detection import hardware_detector, AdaptiveTrainingConfig

        self.hardware_profile = hardware_detector.detect_full_profile()
        self.adaptive_config = AdaptiveTrainingConfig(self.hardware_profile)

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

        # Log container memory configuration for debugging restarts
        container_limit = os.environ.get("CONTAINER_MEMORY_LIMIT_GB", "8")
        worker_memory = os.environ.get("WORKER_MEMORY_GB", "6")
        self.logger.info(f"🐳 Container Memory Config: Limit={container_limit}GB, Worker={worker_memory}GB")

        # Log initial memory state
        try:
            import psutil
            mem = psutil.virtual_memory()
            self.logger.info(f"📊 Initial Memory: {mem.used / 1024**3:.1f}GB used / {mem.total / 1024**3:.1f}GB total ({mem.percent}%)")
        except:
            pass
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self._update_status("cancelled", f"Training cancelled due to signal {signum}")
        # Don't raise exception, just log and let the process exit gracefully
        
    def _detect_container_stress(self) -> bool:
        """Detect container stress signals that could cause restart"""
        try:
            # Check OOM score
            try:
                with open('/proc/self/oom_score', 'r') as f:
                    oom_score = int(f.read().strip())
                    if oom_score > 200:  # Even more aggressive for 8GB container
                        self.logger.error(f"🚨 CRITICAL OOM score: {oom_score} - container may be killed!")
                        return True
                    elif oom_score > 100:
                        self.logger.warning(f"⚠️ Elevated OOM score: {oom_score}")
            except (FileNotFoundError, PermissionError):
                # /proc may not be available in some container environments
                pass

            # Check memory pressure using psutil
            try:
                import psutil
                memory = psutil.virtual_memory()
                memory_percent = memory.percent

                # Get container limits
                container_limit_gb = float(os.environ.get("CONTAINER_MEMORY_LIMIT_GB", "8"))
                memory_used_gb = (memory.total - memory.available) / (1024**3)
                memory_usage_percent = (memory_used_gb / container_limit_gb) * 100

                # Log current memory state periodically
                if hasattr(self, '_last_memory_log_time'):
                    if time.time() - self._last_memory_log_time > 30:  # Log every 30s
                        self.logger.info(f"Memory: {memory_used_gb:.1f}GB used ({memory_usage_percent:.1f}% of {container_limit_gb}GB limit), system: {memory_percent:.1f}%")
                        self._last_memory_log_time = time.time()
                else:
                    self._last_memory_log_time = time.time()

                # Critical threshold - prevent OOM kill
                if memory_usage_percent > 85:
                    self.logger.error(f"🚨 CRITICAL: Memory at {memory_usage_percent:.1f}% of container limit! OOM kill imminent!")
                    return True
                elif memory_usage_percent > 70:
                    self.logger.warning(f"⚠️ HIGH: Memory at {memory_usage_percent:.1f}% of container limit - approaching danger zone")
                    return True
                elif memory.percent > 80:
                    self.logger.warning(f"⚠️ System memory pressure: {memory.percent}%")
                    return True

            except ImportError:
                self.logger.warning("psutil not available for memory monitoring")

            return False

        except Exception as e:
            self.logger.error(f"Error detecting container stress: {e}")
            return False
    
    def _start_memory_monitoring(self):
        """Start memory monitoring thread during training"""
        if self._memory_monitoring_active:
            return
            
        self._memory_monitoring_active = True
        import threading
        
        def memory_monitor():
            while self._memory_monitoring_active:
                try:
                    if self._detect_container_stress():
                        self.logger.error("🚨 CRITICAL: Memory limit approaching - stopping training to prevent container restart")
                        self._update_status("cancelled", "Training stopped due to memory pressure")
                        self._memory_monitoring_active = False
                        # Force cleanup and exit
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        break
                    
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    self.logger.warning(f"Memory monitoring error: {e}")
                    time.sleep(10)
        
        self._memory_monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        self._memory_monitor_thread.start()
        self.logger.info("Memory monitoring started")
    
    def _stop_memory_monitoring(self):
        """Stop memory monitoring thread"""
        self._memory_monitoring_active = False
        if self._memory_monitor_thread:
            self._memory_monitor_thread.join(timeout=2)
        self.logger.info("Memory monitoring stopped")

    def _enable_emergency_mode(self, request: FineTuneRequest):
        """Enable emergency mode for container stability"""
        self.logger.warning("🚨 Enabling emergency stability mode for RTX 3050 4GB Laptop (32GB RAM)")

        # Conservative settings optimized for 4GB VRAM (RAM is not the bottleneck)
        emergency_config = {
            "max_seq_length": 1024,  # Can use longer sequences with 32GB RAM
            "batch_size": 1,  # Still 1 for 4GB VRAM
            "gradient_accumulation_steps": max(32, request.gradient_accumulation_steps),  # Moderate accumulation
            "reduced_precision": True,
            "dataloader_num_workers": 0,  # Force single worker
            "dataset_num_proc": 1  # Force single process
        }

        self.logger.info(f"Emergency config for 4GB GPU with 32GB RAM: {emergency_config}")
        self.logger.warning("🚨 GPU is the bottleneck (4GB VRAM), RAM is plentiful (32GB)")
        return emergency_config
        

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
            # CRITICAL: Validate all required paths before proceeding
            if not request.base_model_dir:
                raise ValueError("base_model_dir is None or empty. Please provide a valid model directory path.")
            if not request.dataset_dir:
                raise ValueError("dataset_dir is None or empty. Please provide a valid dataset directory path.")
            if not request.output_dir:
                raise ValueError("output_dir is None or empty. Please provide a valid output directory path.")

            self.logger.info(f"Validated paths - model: {request.base_model_dir}, dataset: {request.dataset_dir}, output: {request.output_dir}")

            # FIXED: Always apply emergency mode for RTX 4060 8GB to prevent restarts
            self.logger.warning("🚨 Applying emergency stability mode for RTX 4060 8GB")
            emergency_config = self._enable_emergency_mode(request)
            # Override request parameters with emergency settings
            request.max_seq_length = emergency_config["max_seq_length"]
            request.batch_size = emergency_config["batch_size"]
            request.gradient_accumulation_steps = emergency_config["gradient_accumulation_steps"]

            # Additional memory pressure check
            if self._detect_container_stress():
                self.logger.error("Container stress detected - applying additional emergency measures")
                # Even more aggressive settings
                request.batch_size = 1
                request.gradient_accumulation_steps = max(32, request.gradient_accumulation_steps)
                request.max_seq_length = min(1024, request.max_seq_length)

            # FIXED: Validate precision BEFORE model loading to prevent race conditions
            optimal_precision = self._get_optimal_precision_settings()
            self.logger.info(f"Pre-validated precision: {optimal_precision['reason']}")

            # Optimized model loading for RTX 3050 4GB Laptop with Unsloth
            worker_memory_gb = float(os.environ.get("WORKER_MEMORY_GB", "4"))

            # CRITICAL FIX: Qwen models have bias terms that prevent Unsloth from patching any layers
            # When Unsloth can't patch (0 QKV, 0 O, 0 MLP), using it causes segfault
            # Solution: Use standard HuggingFace BitsAndBytes 4-bit instead of Unsloth

            self.logger.warning("⚠️ Qwen models incompatible with Unsloth (bias terms prevent patching)")
            self.logger.info("Using standard BitsAndBytes 4-bit quantization instead")

            from transformers import AutoModelForCausalLM, AutoTokenizer

            # How to load depends on the hardware. NVIDIA keeps the 4-bit loading this worker was
            # tuned with; AMD, Intel, Apple and CPU follow the hardware configuration instead of
            # always forcing bitsandbytes 4-bit (see factory_bridge.causal_lm_load_plan).
            load_plan = causal_lm_load_plan(
                self.adaptive_config.generate_optimal_config(request).get("gpu_optimizations")
                if getattr(self, "adaptive_config", None) is not None else None
            )
            self.logger.info(f"Model loading plan: {load_plan['reason']}")
            if load_plan["accelerator"] == "cuda":
                from transformers import BitsAndBytesConfig

                # BitsAndBytes 4-bit config for RTX 3050 4GB
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
                    bnb_4bit_compute_dtype=torch.bfloat16 if optimal_precision['bf16'] else torch.float16,
                    bnb_4bit_use_double_quant=True,  # Nested quantization for memory
                )

                self.logger.info(f"Loading model with BitsAndBytes 4-bit: compute_dtype={'bf16' if optimal_precision['bf16'] else 'fp16'}, double_quant=True")

                model = AutoModelForCausalLM.from_pretrained(
                    request.base_model_dir,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )

            else:
                model = AutoModelForCausalLM.from_pretrained(
                    request.base_model_dir, trust_remote_code=True, **load_plan["kwargs"]
                )
                if load_plan["move_to"]:
                    model = model.to(load_plan["move_to"])

            tokenizer = AutoTokenizer.from_pretrained(
                request.base_model_dir,
                trust_remote_code=True,
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

            # Conservative LoRA for containers to prevent OOM
            is_container = self.hardware_profile.is_container
            if is_container:
                lora_rank = 16
                lora_alpha = 32
                self.logger.info(f"Container detected - using conservative LoRA: r={lora_rank}, alpha={lora_alpha}")
            else:
                worker_memory_gb = float(os.environ.get("WORKER_MEMORY_GB", "6"))
                if worker_memory_gb <= 8:
                    lora_rank = 16
                    lora_alpha = 32
                    self.logger.info(f"Using low-memory LoRA config: r={lora_rank}, alpha={lora_alpha}")
                else:
                    lora_rank = 32 if request.peft_method == "lora" else 64
                    lora_alpha = 64
                    self.logger.info(f"Using high-memory LoRA config: r={lora_rank}, alpha={lora_alpha}")

            # Conservative target modules excluding embeddings for memory safety
            qwen_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention only
                "gate_proj", "up_proj", "down_proj",     # MLP
                # Embeddings excluded to reduce trainable params ~1.67%
            ]
            
            # Apply standard PEFT LoRA (not Unsloth) for Qwen compatibility
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            # Prepare model for k-bit training (only when it was loaded quantised)
            if load_plan["kbit"]:
                model = prepare_model_for_kbit_training(model)

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=qwen_target_modules,
                lora_dropout=0.05,  # Standard PEFT dropout (not 0.0 like Unsloth)
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
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

            # Conservative training args for containers
            if self.hardware_profile.is_container:
                request.gradient_accumulation_steps = max(16, request.gradient_accumulation_steps)
                self.logger.info(f"Container: Increased grad_accum to {request.gradient_accumulation_steps} for effective batch {request.batch_size * request.gradient_accumulation_steps}")

            # Setup training arguments
            training_args = self._create_adaptive_training_args(request, optimal_precision)

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
                dataset_batch_size = 1000  # Conservative for memory - handled via tokenizer batching
            else:
                # Higher memory configurations
                optimal_seq_length = min(request.max_seq_length, 4096)
                dataset_num_proc = 1  # FIXED: Force single process to prevent semaphore leaks
                dataset_batch_size = 2000  # Conservative for memory - handled via tokenizer batching

            # FIXED: Remove invalid 'dataset_batch_size' from SFTConfig - use dataset_kwargs for tokenization batching if needed
            self.logger.info(f"Creating SFTConfig without invalid 'dataset_batch_size' - max_seq_length={optimal_seq_length}, dataset_num_proc={dataset_num_proc}, packing=True, chars_per_token=3.5")
            
            sft_config = SFTConfig(
                dataset_text_field="text",
                max_seq_length=optimal_seq_length,
                packing=False,  # FIXED: Temporarily disable packing to test if collation is causing hang
                dataset_num_proc=1,  # FIXED: Reduce to 1 to prevent semaphore leaks
                # Qwen2.5-specific optimizations
                chars_per_token=3.5,  # Qwen2.5 tokenizer average
                dataset_kwargs={
                    "add_special_tokens": True,
                    "split": "train",
                    "batched": True,  # Enable batched tokenization
                    "batch_size": dataset_batch_size,  # Move batch_size here for tokenizer if needed
                    "remove_columns": ["text"]  # Clean up after tokenization
                },
                neftune_noise_alpha=None,  # Disable NEFTune to save memory
            )
            
            self.logger.info(f"SFT Config optimized for {worker_memory_gb}GB: max_seq_length={optimal_seq_length}, packing=True")

            # FIXED: Precision already validated before model loading - just log configuration
            self.logger.info(f"Model loaded with precision: {optimal_precision['reason']}")
            # No need to re-validate since model was loaded with correct precision
            
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
                eval_dataset=None,  # Disable eval in containers for memory
                data_collator=None,
                args=training_args,
                **sft_config.to_dict(),
            )

            # FIXED: Add log after trainer creation to confirm dataloader builds successfully
            self.logger.info("🚨 Trainer created successfully - testing dataloader construction")
            try:
                train_dataloader = trainer.get_train_dataloader()
                self.logger.info(f"🚨 Dataloader created: len={len(train_dataloader)}, batch_size={trainer.args.per_device_train_batch_size}")
                # Test first batch without consuming it
                first_batch_iterator = iter(train_dataloader)
                sample_batch = next(first_batch_iterator)
                self.logger.info(f"🚨 Sample batch keys: {list(sample_batch.keys())}, input_ids shape: {sample_batch['input_ids'].shape if 'input_ids' in sample_batch else 'N/A'}")
                batch_mem_peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                self.logger.info(f"🚨 Memory peak during dataloader test: {batch_mem_peak:.2f}GB")
            except Exception as dl_error:
                self.logger.error(f"🚨 DATALOADER CONSTRUCTION FAILED: {dl_error}")
                import traceback
                self.logger.error(f"🚨 Dataloader traceback: {traceback.format_exc()}")
                raise
            
            # Aggressive cleanup post-trainer creation, pre-train
            self.logger.info("Post-trainer cleanup to prevent OOM...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            self._emergency_memory_cleanup()
            
            # SFT configuration logging moved above for clarity

            # Add progress callback
            trainer.add_callback(progress_callback)

            # Start training with enhanced error handling
            self.logger.info("Starting Unsloth training with enhanced monitoring...")
            self._update_status("training", "Training started with memory monitoring...")

            # FIXED: Start memory monitoring before training
            self._start_memory_monitoring()

            # Check for cancellation one more time before training
            if self._should_cancel():
                self._stop_memory_monitoring()
                raise InterruptedError("Training cancelled before start")

            # Enhanced training with memory monitoring and recovery
            try:
                # Log initial GPU memory state
                if torch.cuda.is_available():
                    gpu_mem_before = torch.cuda.memory_allocated() / 1024**3
                    gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    self.logger.info(f"GPU memory before training: {gpu_mem_before:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved")
                
                # FIXED: Force memory cleanup before training to prevent container restarts
                self._emergency_memory_cleanup()
                
                # Training with monitoring
                self.logger.info("🚨 About to create first training batch - monitoring memory closely")
                if torch.cuda.is_available():
                    pre_train_mem = torch.cuda.memory_allocated() / 1024**3
                    self.logger.info(f"🚨 Pre-first-batch memory: {pre_train_mem:.2f}GB allocated")
                
                # Test dataloader iteration manually before full train
                self.logger.info("🚨 Testing first batch from dataloader...")
                try:
                    first_batch = next(iter(trainer.get_train_dataloader()))
                    self.logger.info(f"🚨 First batch shape: inputs={first_batch['input_ids'].shape if 'input_ids' in first_batch else 'N/A'}, labels={first_batch.get('labels', 'N/A').shape if 'labels' in first_batch else 'N/A'}")
                    batch_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    self.logger.info(f"🚨 Memory after first batch load: {batch_mem:.2f}GB allocated")
                except Exception as batch_error:
                    self.logger.error(f"🚨 ERROR loading first batch: {batch_error}")
                    import traceback
                    self.logger.error(f"🚨 Batch load traceback: {traceback.format_exc()}")
                    raise
                
                self.logger.info("🚨 First batch loaded successfully - starting trainer.train()")
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
                
                # FIXED: Add comprehensive error handling to prevent container restarts
                error_msg = str(training_error).lower()
                
                # Check for memory-related errors
                if any(keyword in error_msg for keyword in ["out of memory", "oom", "cuda error", "memory"]):
                    self.logger.error("🚨 Memory-related error detected - applying emergency recovery")
                    self._emergency_memory_cleanup()
                    
                    # Try with even more conservative settings
                    self.logger.warning("🚨 Attempting recovery with ultra-conservative settings")
                    request.batch_size = 1
                    request.gradient_accumulation_steps = 128  # Very high accumulation
                    request.max_seq_length = 256  # Very small sequence length
                    
                    # Recreate trainer with emergency settings
                    emergency_args = self._create_adaptive_training_args(request, optimal_precision)
                    emergency_trainer = SFTTrainer(
                        model=model,
                        tokenizer=tokenizer,
                        train_dataset=train_dataset,
                        eval_dataset=None,  # Disable eval
                        data_collator=None,
                        args=emergency_args,
                        **sft_config.to_dict(),
                    )
                    emergency_trainer.add_callback(progress_callback)
                    
                    self._update_status("emergency_recovery", "Attempting emergency recovery...")
                    emergency_trainer.train()
                    self.logger.info("✅ Successfully recovered from memory error")
                    
                # Check if it's a tokenizer-related error
                elif any(keyword in error_msg for keyword in 
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
                    # Log the full error for debugging
                    import traceback
                    self.logger.error(f"Full error traceback: {traceback.format_exc()}")
                    # Re-raise non-recoverable errors
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
            # CRITICAL: Validate all required paths before proceeding
            if not request.base_model_dir:
                raise ValueError("base_model_dir is None or empty. Please provide a valid model directory path.")
            if not request.dataset_dir:
                raise ValueError("dataset_dir is None or empty. Please provide a valid dataset directory path.")
            if not request.output_dir:
                raise ValueError("output_dir is None or empty. Please provide a valid output directory path.")

            self.logger.info(f"Validated paths - model: {request.base_model_dir}, dataset: {request.dataset_dir}, output: {request.output_dir}")

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                request.base_model_dir, trust_remote_code=True
            )
            # Device and precision for any accelerator vendor, not only CUDA.
            _device, _dtype = torch_device_and_dtype()
            self.logger.info(f"Loading on device={_device} dtype={_dtype}")
            model = AutoModelForCausalLM.from_pretrained(
                request.base_model_dir,
                trust_remote_code=True,
                device_map="auto" if _device == "cuda" else None,
                torch_dtype=_dtype,
            )
            if _device in ("mps", "xpu"):
                model = model.to(_device)

            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.resize_token_embeddings(len(tokenizer))

            # Check for cancellation
            if self._should_cancel():
                raise InterruptedError("Training cancelled during model loading")

            # Apply PEFT if available
            # The adapter follows the request (method, rank, alpha, dropout, target modules).
            # Unsupported method names raise instead of silently training the full model.
            peft_config = resolve_peft_config(request) if PEFT_AVAILABLE else None
            if peft_config is not None:
                self._update_status("configuring", f"Applying PEFT configuration ({request.peft_method})...")
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
                dataset_num_proc=1,  # FIXED: Force single process to prevent semaphore leaks
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
        self, request: FineTuneRequest, optimal_precision: Dict[str, Any] = None
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

        # FIXED: Use pre-validated precision if provided, otherwise detect
        if optimal_precision is None:
            optimal_precision = self._get_optimal_precision_settings()
            
        self.logger.info(
            f"Precision for training args: {optimal_precision['reason']}\n"
            f"  Setting fp16={optimal_precision['fp16']}, bf16={optimal_precision['bf16']}"
        )

        # Merge base args with adaptive configuration
        args = {
            "per_device_train_batch_size": adaptive_args["per_device_train_batch_size"],
            "gradient_accumulation_steps": adaptive_args["gradient_accumulation_steps"],
            "warmup_steps": 5,
            "num_train_epochs": request.epochs,
            "learning_rate": request.learning_rate,
            # FIXED: Use optimal precision detection instead of flawed gpu_opts guessing
            "fp16": optimal_precision["fp16"],
            "bf16": optimal_precision["bf16"],
            # Adaptive optimization
            "logging_steps": adaptive_args.get("logging_steps", 10),
            "optim": (
                "adamw_8bit"
                if UNSLOTH_AVAILABLE and gpu_opts["enabled"]
                else "adamw_hf"
            ),
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "seed": 3407,
            "output_dir": str(output_dir),
            # Adaptive checkpoint and evaluation settings
            "save_steps": adaptive_args.get("save_steps", 100),
            "save_total_limit": adaptive_args.get("save_total_limit", 2),
            "eval_strategy": adaptive_args.get("eval_strategy", "no"),
            # FIXED: Adaptive data loading - force single worker to prevent semaphore leaks
            "dataloader_num_workers": 0,  # Always 0 to prevent multiprocessing issues
            "dataloader_pin_memory": False,  # Disabled for container stability
            # System settings
            "remove_unused_columns": False,
            "run_name": f"llmforge_{self.task_id[:8]}",
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
            
            # FIXED: Stop memory monitoring during emergency cleanup
            self._stop_memory_monitoring()
            
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
            # FIXED: Stop memory monitoring
            self._stop_memory_monitoring()
            
            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("Cleared CUDA cache in worker process")

        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")
