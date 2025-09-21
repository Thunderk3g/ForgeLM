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

from models.schemas import FineTuneRequest, TrainingStatus
from utils.logging import get_logger
from utils.performance import performance_monitor, timing_context, timed
from config.settings import settings

# Setup proper Triton cache directory with correct permissions
import tempfile

triton_cache_dir = tempfile.mkdtemp(prefix="triton_cache_")
os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
try:
    os.chmod(triton_cache_dir, 0o777)
except Exception as e:
    print(f"Warning: Could not set Triton cache permissions: {e}")

# Environment optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
# Import Unsloth with proper error handling
try:
    # Work around Unsloth v2025.9.5 KTO trainer bug
    import sys
    import types
    
    # First, let's patch the exec function to handle the specific AttributeError
    original_exec = exec
    
    def patched_exec(code, globals_dict=None, locals_dict=None):
        try:
            return original_exec(code, globals_dict, locals_dict)
        except AttributeError as e:
            if "UnslothKTOTrainer" in str(e) and "UnslothKTOTrainer" in str(code):
                # This is the problematic line - just skip it
                print(f"Skipping problematic Unsloth KTO trainer assignment: {e}")
                return None
            else:
                raise
    
    # Temporarily replace exec
    builtins_exec = __builtins__['exec'] if isinstance(__builtins__, dict) else __builtins__.exec
    if isinstance(__builtins__, dict):
        __builtins__['exec'] = patched_exec
    else:
        __builtins__.exec = patched_exec
    
    try:
        import unsloth
        from unsloth import FastLanguageModel, is_bfloat16_supported
    finally:
        # Restore original exec
        if isinstance(__builtins__, dict):
            __builtins__['exec'] = builtins_exec
        else:
            __builtins__.exec = builtins_exec
    
    UNSLOTH_AVAILABLE = True
    print(f"🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.")
    print(f"🦥 Unsloth Zoo will now patch everything to make training faster!")
    print(f"Unsloth imported successfully (v{unsloth.__version__})")
except (ImportError, NotImplementedError, AttributeError) as e:
    UNSLOTH_AVAILABLE = False
    print(f"WARNING: Unsloth not available ({e})")
# Import common dependencies first
try:
    from transformers import (
        TrainingArguments,
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainerCallback,
    )
    from trl import SFTTrainer, SFTConfig

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: Transformers/TRL not available")


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
                    "total_epochs": args.num_train_epochs,
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
                    "total_epochs": args.num_train_epochs,
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
                    loss = last_log.get("train_loss", last_log.get("loss"))
                    learning_rate = last_log.get("learning_rate")

                # Calculate progress
                progress_pct = (
                    (current_step / total_steps * 100) if total_steps > 0 else 0
                )

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
                        "progress_percentage": progress_pct,
                    }

                    self.log_queue.put(log_entry)

                    # Also update status file
                    self._update_status_file(
                        current_step,
                        total_steps,
                        current_epoch,
                        args.num_train_epochs,
                        loss,
                        learning_rate,
                        progress_pct,
                    )

            except Exception as e:
                self.logger.error(f"Error in step end callback: {e}")

        def _update_status_file(
            self,
            current_step,
            total_steps,
            current_epoch,
            total_epochs,
            loss,
            learning_rate,
            progress_pct,
        ):
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
                    "learning_rate": (
                        float(learning_rate) if learning_rate is not None else None
                    ),
                    "progress_percentage": progress_pct,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "started_at": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)
                    ),
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
    """Advanced training service with adaptive hardware optimization"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.active_trainings: Dict[str, Dict[str, Any]] = {}
        self.cleanup_lock = threading.Lock()

        # Initialize hardware detection and adaptive configuration
        from utils.hardware_detection import hardware_detector, AdaptiveTrainingConfig

        self.hardware_profile = hardware_detector.detect_full_profile()
        self.adaptive_config = AdaptiveTrainingConfig(self.hardware_profile)

        # Register cleanup
        register_cleanup(self.cleanup_all_trainings)

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

        # Apply system-level optimizations
        system_opts = self.adaptive_config._get_system_optimizations()
        for env_var, value in system_opts.items():
            if env_var.startswith(("OMP_", "MKL_", "PYTORCH_")):
                os.environ[env_var] = value
                self.logger.info(f"Applied system optimization: {env_var}={value}")

    @timed("fine_tune")
    def fine_tune(
        self, request: FineTuneRequest, task_id: str = None
    ) -> Dict[str, Any]:
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

    def fix_qwen_tokenizer(self, tokenizer):
        """Fix Qwen2.5 tokenizer configuration for training compatibility"""
        try:
            # Qwen2.5 has bos_token=None by design, but training frameworks need it
            if tokenizer.bos_token_id is None:
                self.logger.warning("Fixing bos_token_id for Qwen2.5 compatibility")
                tokenizer.bos_token_id = tokenizer.eos_token_id
                tokenizer.bos_token = tokenizer.eos_token

            # Ensure padding token is properly set
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.pad_token = tokenizer.eos_token

            # Log final tokenizer state
            self.logger.info(
                f"Tokenizer config: bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}"
            )
            return tokenizer

        except Exception as e:
            self.logger.error(f"Error fixing tokenizer: {e}")
            return tokenizer

    def _validate_lora_dropout(self, model, expected_dropout: float):
        """Validate that LoRA dropout was applied correctly"""
        try:
            # Check if model has LoRA modules
            if hasattr(model, 'peft_config'):
                for adapter_name, config in model.peft_config.items():
                    if hasattr(config, 'lora_dropout'):
                        actual_dropout = config.lora_dropout
                        if abs(actual_dropout - expected_dropout) > 1e-6:
                            self.logger.warning(
                                f"LoRA dropout mismatch: expected {expected_dropout}, got {actual_dropout}"
                            )
                        else:
                            self.logger.info(f"✅ LoRA dropout validated: {actual_dropout}")
                            
            # Also check individual LoRA layers
            dropout_found = False
            for name, module in model.named_modules():
                if hasattr(module, 'lora_dropout') and hasattr(module.lora_dropout, 'p'):
                    actual_dropout = module.lora_dropout.p
                    dropout_found = True
                    if abs(actual_dropout - expected_dropout) > 1e-6:
                        self.logger.warning(
                            f"Layer {name} dropout mismatch: expected {expected_dropout}, got {actual_dropout}"
                        )
                        
            if dropout_found:
                self.logger.info(f"✅ LoRA layer dropouts validated: {expected_dropout}")
            else:
                self.logger.info("No LoRA dropout layers found in model - this is expected for dropout=0.0")
                
        except Exception as e:
            self.logger.error(f"Error validating LoRA dropout: {e}")

    def _load_and_validate_dataset(self, dataset_dir: str):
        """Load and validate dataset with comprehensive error handling"""
        from pathlib import Path
        import os
        
        dataset_path = Path(dataset_dir)
        self.logger.info(f"Loading dataset from: {dataset_path}")
        
        # Check if dataset directory exists
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")
        
        # Check for dataset files
        expected_files = [
            "dataset_info.json",
            "state.json", 
            "train/data-00000-of-00001.arrow",  # Common Arrow format
            "train/dataset_info.json",
            "train/state.json"
        ]
        
        missing_files = []
        for file_path in expected_files:
            full_path = dataset_path / file_path
            if not full_path.exists():
                missing_files.append(str(full_path))
        
        if missing_files:
            self.logger.warning(f"Missing dataset files: {missing_files}")
            self.logger.info("This may indicate a corrupted or incomplete dataset")
        
        try:
            # Attempt primary dataset loading
            self.logger.info("Attempting to load dataset with load_from_disk...")
            dataset = load_from_disk(str(dataset_path))
            
            # Validate loaded dataset
            self._validate_dataset_schema(dataset)
            
            self.logger.info(f"✅ Dataset loaded successfully: {len(dataset)} samples")
            return dataset
            
        except Exception as primary_error:
            self.logger.error(f"Primary dataset loading failed: {primary_error}")
            
            # Check if this is the schema error we're trying to fix
            if "schema message" in str(primary_error).lower() or "null or length 0" in str(primary_error).lower():
                self.logger.warning("🔧 Detected schema error - attempting recovery...")
                return self._recover_from_schema_error(dataset_path, primary_error)
            else:
                # Try alternative loading methods
                return self._try_alternative_dataset_loading(dataset_path, primary_error)

    def _validate_dataset_schema(self, dataset):
        """Validate dataset schema and structure"""
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        
        # Check if it's a DatasetDict or Dataset
        if hasattr(dataset, 'keys'):  # DatasetDict
            if 'train' not in dataset:
                raise ValueError("Dataset must contain a 'train' split")
            train_dataset = dataset['train']
        else:  # Single Dataset
            train_dataset = dataset
        
        # Check for text field (most important for SFT)
        if 'text' not in train_dataset.column_names:
            available_columns = train_dataset.column_names
            self.logger.warning(f"No 'text' column found. Available columns: {available_columns}")
            
            # Look for alternative text columns
            text_candidates = ['instruction', 'input', 'prompt', 'conversation', 'messages']
            found_alternative = None
            for candidate in text_candidates:
                if candidate in available_columns:
                    found_alternative = candidate
                    break
            
            if found_alternative:
                self.logger.info(f"Found alternative text column: {found_alternative}")
            else:
                raise ValueError(f"No suitable text column found. Available: {available_columns}")
        
        # Check sample data
        sample = train_dataset[0]
        self.logger.info(f"Dataset schema validation passed. Sample keys: {list(sample.keys())}")

    def _recover_from_schema_error(self, dataset_path: Path, original_error):
        """Attempt to recover from dataset schema errors"""
        self.logger.info("🔧 Attempting dataset schema recovery...")
        
        recovery_methods = [
            self._try_reload_with_cache_clear,
            self._try_recreate_from_arrow_files,
            self._try_fallback_json_loading,
        ]
        
        for i, recovery_method in enumerate(recovery_methods, 1):
            try:
                self.logger.info(f"Recovery attempt {i}/{len(recovery_methods)}: {recovery_method.__name__}")
                dataset = recovery_method(dataset_path)
                if dataset is not None:
                    self.logger.info(f"✅ Dataset recovery successful with method {i}")
                    return dataset
            except Exception as recovery_error:
                self.logger.warning(f"Recovery method {i} failed: {recovery_error}")
                continue
        
        # All recovery methods failed
        raise ValueError(
            f"Dataset schema error recovery failed. Original error: {original_error}. "
            f"Please regenerate the dataset or check the source files."
        )

    def _try_reload_with_cache_clear(self, dataset_path: Path):
        """Try reloading after clearing caches"""
        import datasets
        
        # Clear datasets cache
        try:
            datasets.disable_caching()
            dataset = load_from_disk(str(dataset_path))
            datasets.enable_caching()
            return dataset
        except Exception:
            datasets.enable_caching()  # Ensure caching is re-enabled
            raise

    def _try_recreate_from_arrow_files(self, dataset_path: Path):
        """Try to recreate dataset from raw Arrow files"""
        from datasets import Dataset, DatasetDict
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Look for Arrow files
        arrow_files = list(dataset_path.rglob("*.arrow"))
        parquet_files = list(dataset_path.rglob("*.parquet"))
        
        if arrow_files:
            self.logger.info(f"Found {len(arrow_files)} Arrow files, attempting direct load...")
            # Try loading the first Arrow file directly
            table = pa.ipc.open_file(str(arrow_files[0])).read_all()
            dataset = Dataset(table)
            return dataset
        elif parquet_files:
            self.logger.info(f"Found {len(parquet_files)} Parquet files, attempting direct load...")
            table = pq.read_table(str(parquet_files[0]))
            dataset = Dataset(table)
            return dataset
        else:
            raise FileNotFoundError("No Arrow or Parquet files found for recovery")

    def _try_fallback_json_loading(self, dataset_path: Path):
        """Try loading from JSON files if available"""
        from datasets import Dataset
        import json
        
        json_files = list(dataset_path.rglob("*.json"))
        jsonl_files = list(dataset_path.rglob("*.jsonl"))
        
        data_files = json_files + jsonl_files
        if not data_files:
            raise FileNotFoundError("No JSON files found for fallback loading")
        
        # Try loading from the first JSON file
        data_file = data_files[0]
        self.logger.info(f"Attempting fallback loading from: {data_file}")
        
        if data_file.suffix == '.jsonl':
            # Load JSONL
            data = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            # Load JSON
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        return Dataset.from_list(data)

    def _try_alternative_dataset_loading(self, dataset_path: Path, original_error):
        """Try alternative dataset loading methods"""
        self.logger.info("Trying alternative dataset loading methods...")
        
        # Method 1: Direct datasets.load_dataset
        try:
            from datasets import load_dataset
            dataset = load_dataset(str(dataset_path))
            return dataset
        except Exception as e:
            self.logger.warning(f"load_dataset failed: {e}")
        
        # Method 2: Load specific format files
        try:
            return self._try_recreate_from_arrow_files(dataset_path)
        except Exception as e:
            self.logger.warning(f"Arrow file loading failed: {e}")
        
        # Method 3: JSON fallback
        try:
            return self._try_fallback_json_loading(dataset_path)
        except Exception as e:
            self.logger.warning(f"JSON fallback failed: {e}")
        
        # All methods failed
        raise ValueError(
            f"All dataset loading methods failed. Original error: {original_error}. "
            f"Please check dataset integrity and regenerate if necessary."
        )

    def _fine_tune_with_unsloth(
        self, request: FineTuneRequest, task_id: str, log_queue: queue.Queue
    ) -> Dict[str, Any]:
        """Fine-tune using Unsloth for 2x speedup"""

        self.logger.info("Using Unsloth optimization for fine-tuning")

        try:
            # Determine training type first
            training_type = getattr(request, 'ft_type', 'instruction')
            if training_type == 'instruction':
                training_type = 'instruction'
            elif training_type in ['chat', 'conversation']:
                training_type = 'chat'
            elif training_type in ['reasoning', 'grpo']:
                training_type = 'reasoning'
                
            # Generate enhanced adaptive configuration for this hardware and training type
            adaptive_config = self.adaptive_config.generate_enhanced_adaptive_config(request, training_type)
            self.logger.info(
                f"Using adaptive config: {adaptive_config['effective_memory_gb']:.1f}GB effective memory"
            )

            # Get hardware-optimized model loading parameters
            gpu_opts = adaptive_config["gpu_optimizations"]
            model_config = adaptive_config["model_loading"]
            lora_config = adaptive_config["lora_config"]

            # Load model and tokenizer with adaptive optimization
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=request.base_model_dir,
                max_seq_length=min(
                    request.max_seq_length, model_config["max_seq_length"]
                ),
                dtype=None,  # Auto-detect optimal dtype
                load_in_4bit=gpu_opts["load_in_4bit"],
                load_in_8bit=gpu_opts.get("load_in_8bit", False),
                device_map=gpu_opts["device_map"],
                trust_remote_code=model_config["trust_remote_code"],
            )

            # CRITICAL FIX: Fix Qwen2.5 tokenizer configuration
            tokenizer = self.fix_qwen_tokenizer(tokenizer)

            # Apply adaptive LoRA configuration with explicit dropout enforcement
            enforced_dropout = 0.0  # CRITICAL: Must be 0.0 for Unsloth QKV optimization
            self.logger.info(
                f"Applying LoRA: r={lora_config['r']}, alpha={lora_config['lora_alpha']}, "
                f"dropout={enforced_dropout} (enforced for Unsloth), targets={len(lora_config['target_modules'])}"
            )
            self.logger.info(f"LoRA target modules: {lora_config['target_modules']}")
            self.logger.warning(f"CRITICAL: LoRA dropout ENFORCED to {enforced_dropout} for Unsloth QKV optimization")
            
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_config["r"],
                target_modules=lora_config["target_modules"],
                lora_alpha=lora_config["lora_alpha"],
                lora_dropout=enforced_dropout,  # Explicitly enforce 0.0
                bias="none",
                use_gradient_checkpointing=(
                    "unsloth" if gpu_opts.get("gradient_checkpointing", True) else "no"
                ),
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
            
            # Validate dropout was applied correctly
            self._validate_lora_dropout(model, enforced_dropout)

            # Load dataset with comprehensive error handling and schema validation
            dataset = self._load_and_validate_dataset(request.dataset_dir)

            # Setup adaptive training arguments with training type (already defined above)
            training_args = self._create_adaptive_training_args(
                request, task_id, adaptive_config, training_type
            )

            # Create progress callback
            progress_callback = UnslothProgressCallback(task_id, log_queue)

            # Create trainer with adaptive SFT configuration
            adaptive_training_args = adaptive_config["training_args"]
            effective_seq_length = min(
                request.max_seq_length, model_config["max_seq_length"]
            )

            # Get training type specific SFT configuration
            if training_type == "chat":
                sft_config_dict = self.get_chat_sft_config(request, adaptive_config)
                sft_config = SFTConfig(
                    **sft_config_dict
                )
            else:
                sft_config = SFTConfig(
                    dataset_text_field="text",
                    max_seq_length=effective_seq_length,
                    dataset_num_proc=adaptive_training_args.get(
                        "dataloader_num_workers", 1
                    ),
                    packing=True,  # Enable packing for efficiency
                )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset["train"] if "train" in dataset else dataset,
                eval_dataset=dataset["val"] if "val" in dataset else None,
                args=training_args,
                data_collator=None,
                **sft_config.to_dict(),
            )

            # Add custom callback for progress tracking
            trainer.add_callback(progress_callback)
            
            # CRITICAL: Validate precision settings to prevent Unsloth errors
            training_args = self._validate_precision_settings(training_args, model)

            # Register this training for cleanup
            self.active_trainings[task_id] = {
                "trainer": trainer,
                "model": model,
                "tokenizer": tokenizer,
                "start_time": time.time(),
            }

            # Start training with error recovery
            self.logger.info(f"Starting Unsloth-optimized training for task {task_id}")
            try:
                trainer.train()
            except torch.cuda.OutOfMemoryError as oom_error:
                self.logger.error(f"OOM Error during training: {oom_error}")
                # Emergency memory cleanup
                torch.cuda.empty_cache()
                gc.collect()

                # Reduce batch size and try again
                if training_args.per_device_train_batch_size > 1:
                    self.logger.info("Attempting recovery with reduced batch size")
                    training_args.per_device_train_batch_size = 1
                    training_args.gradient_accumulation_steps = min(
                        32, training_args.gradient_accumulation_steps * 2
                    )

                    # Create new trainer with reduced memory settings
                    trainer = SFTTrainer(
                        model=model,
                        tokenizer=tokenizer,
                        train_dataset=(
                            dataset["train"] if "train" in dataset else dataset
                        ),
                        eval_dataset=None,  # Disable eval for recovery
                        args=training_args,
                        data_collator=None,
                        **sft_config.to_dict(),
                    )
                    trainer.add_callback(progress_callback)
                    trainer.train()
                else:
                    raise
            except Exception as training_error:
                # Check for tokenizer-related errors
                error_msg = str(training_error).lower()
                if any(
                    keyword in error_msg
                    for keyword in ["bos_token", "eos_token", "tokenizer", "nan"]
                ):
                    self.logger.error(f"Tokenizer error detected: {training_error}")
                    # Apply emergency tokenizer fixes and retry
                    if tokenizer.bos_token_id is None:
                        tokenizer.bos_token_id = tokenizer.eos_token_id
                        tokenizer.bos_token = tokenizer.eos_token
                    trainer.train()  # Retry with fixed tokenizer
                else:
                    raise

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
                    save_method="merged_16bit",
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
                "optimization": "unsloth_2x_speedup",
            }

        except Exception as e:
            self.logger.error(f"Unsloth training failed: {e}")
            raise

    def _fine_tune_with_transformers(
        self, request: FineTuneRequest, task_id: str, log_queue: queue.Queue
    ) -> Dict[str, Any]:
        """Fallback fine-tuning with standard Transformers + PEFT"""

        self.logger.info("Using Transformers + PEFT fallback for fine-tuning")

        try:
            # Load model and tokenizer
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

            # Apply PEFT if available
            if PEFT_AVAILABLE and request.peft_method in ["lora", "qlora"]:
                peft_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.0,  # Optimized for Unsloth fast patching
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, peft_config)

            # Determine training type first  
            training_type = getattr(request, 'ft_type', 'instruction')
            if training_type == 'instruction':
                training_type = 'instruction'
            elif training_type in ['chat', 'conversation']:
                training_type = 'chat'
            elif training_type in ['reasoning', 'grpo']:
                training_type = 'reasoning'

            # Load dataset
            dataset = load_from_disk(request.dataset_dir)
            adaptive_config = self.adaptive_config.generate_enhanced_adaptive_config(request, training_type)
            self.logger.info(
                f"Using adaptive config: {adaptive_config['effective_memory_gb']:.1f}GB effective memory"
            )

            # Setup training arguments with training type (already defined above)
            training_args = self._create_adaptive_training_args(
                request, task_id, adaptive_config, training_type
            )
            sft_config = SFTConfig(
                dataset_text_field="text",
                max_seq_length=request.max_seq_length,
                packing=False,
                dataset_num_proc=2,
            )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=dataset["train"] if "train" in dataset else dataset,
                eval_dataset=dataset["val"] if "val" in dataset else None,
                data_collator=None,
                **sft_config.to_dict(),
            )

            # Register this training for cleanup
            self.active_trainings[task_id] = {
                "trainer": trainer,
                "model": model,
                "tokenizer": tokenizer,
                "start_time": time.time(),
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
                "optimization": "standard_transformers",
            }

        except Exception as e:
            self.logger.error(f"Standard training failed: {e}")
            raise

    def _create_adaptive_training_args(
        self, request: FineTuneRequest, task_id: str, adaptive_config: Dict[str, Any], training_type: str = "instruction"
    ):
        """Create adaptive training arguments based on hardware profile and training type
        
        Args:
            request: Fine-tuning parameters
            task_id: Unique task identifier
            adaptive_config: Hardware-optimized configuration
            training_type: Type of training ("instruction", "chat", "reasoning")
        """

        output_dir = Path(request.output_dir) / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get training type specific configuration
        if training_type == "chat":
            base_args = self._get_chat_completion_args(adaptive_config, request)
        elif training_type == "reasoning":
            base_args = self._get_reasoning_training_args(adaptive_config, request)
        else:
            base_args = self._get_instruction_training_args(adaptive_config, request)

        adaptive_args = adaptive_config["training_args"]
        gpu_opts = adaptive_config["gpu_optimizations"]
        
        # CRITICAL FIX: Replace flawed manual precision detection with Unsloth's built-in detection
        # This prevents the "Unsloth: Model is in bfloat16 precision but you want to use float16 precision" error
        from utils.hardware_detection import hardware_detector
        
        optimal_precision = self._get_optimal_precision_settings()
        self.logger.info(
            f"Precision detection for main training service: {optimal_precision['reason']}\n"
            f"  Setting fp16={optimal_precision['fp16']}, bf16={optimal_precision['bf16']}"
        )
        
        fp16 = optimal_precision['fp16']
        bf16 = optimal_precision['bf16']
        
        # Allow adaptive_config to explicitly override precision if needed
        if "precision" in adaptive_config:
            if adaptive_config["precision"] == "bf16":
                bf16, fp16 = True, False
                self.logger.info("Precision overridden by adaptive_config to bf16")
            elif adaptive_config["precision"] == "fp16":
                fp16, bf16 = True, False
                self.logger.info("Precision overridden by adaptive_config to fp16")
        # Merge base args with adaptive configuration
        args = {
            "per_device_train_batch_size": base_args.get("per_device_train_batch_size", adaptive_args["per_device_train_batch_size"]),
            "gradient_accumulation_steps": base_args.get("gradient_accumulation_steps", adaptive_args["gradient_accumulation_steps"]),
            "warmup_steps": base_args.get("warmup_steps", 5),
            "num_train_epochs": request.epochs,
            "learning_rate": base_args.get("learning_rate", request.learning_rate),
            "fp16": fp16,
            "bf16": bf16,
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
            "save_steps": base_args.get("save_steps", adaptive_args.get("save_steps", 100)),
            "save_total_limit": adaptive_args.get("save_total_limit", 2),
            "eval_strategy": adaptive_args.get("eval_strategy", "no"),
            "dataloader_num_workers": base_args.get("dataloader_num_workers", adaptive_args.get("dataloader_num_workers", 0)),
            "dataloader_pin_memory": base_args.get("dataloader_pin_memory", adaptive_args.get("dataloader_pin_memory", False)),
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

        # Apply training type specific Unsloth optimizations
        args = self._apply_training_type_optimizations(args, training_type, gpu_opts, adaptive_config)

        # Enhanced precision logging and validation
        precision_info = adaptive_config.get("precision_info", {})
        final_precision = 'bf16' if args['bf16'] else 'fp16' if args['fp16'] else 'fp32'
        
        self.logger.info(
            f"Precision analysis - Model: {precision_info.get('model_family', 'unknown')}, "
            f"Selected: {final_precision}, Reason: {optimal_precision['reason']}"
        )

        if UNSLOTH_AVAILABLE:
            model_family = precision_info.get("model_family", "")
            if args.get('bf16', False):
                self.logger.info("✅ Using bfloat16 precision - optimal for Unsloth")
            elif args.get('fp16', False):
                self.logger.info("✅ Using float16 precision - compatible with Unsloth")
                if model_family in ["qwen", "llama", "mistral", "gemma"] and is_bfloat16_supported():
                    self.logger.info(
                        f"Note: Model family '{model_family}' could potentially use bfloat16 on this hardware, "
                        f"but fp16 should work fine with Unsloth"
                    )
            else:
                self.logger.warning("⚠️  Using fp32 precision - training will be slower")

        self.logger.info(
            f"Adaptive training config: batch_size={args['per_device_train_batch_size']}, "
            f"grad_accum={args['gradient_accumulation_steps']}, precision={final_precision}\n"
            f"This configuration should prevent Unsloth precision mismatch errors during training"
        )

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
    
    def _get_optimal_precision_settings(self, model=None):
        """Get optimal precision settings using Unsloth's built-in detection"""
        try:
            # First, use Unsloth's hardware detection if available
            if UNSLOTH_AVAILABLE:
                supports_bf16 = is_bfloat16_supported()
                self.logger.info(f"Unsloth hardware check - supports bfloat16: {supports_bf16}")
                
                # Use Unsloth's recommended settings
                return {
                    "fp16": not supports_bf16,
                    "bf16": supports_bf16,
                    "reason": f"Unsloth hardware detection: bf16_supported={supports_bf16}"
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

    def _validate_precision_settings(self, training_args, model=None) -> TrainingArguments:
        """Validate and fix precision settings to prevent Unsloth errors"""
        try:
            # Detect model precision
            model_precision = None
            if model is not None:
                model_precision = self._detect_model_precision(model)
                self.logger.info(f"Detected model precision: {model_precision}")
            
            # Get optimal settings based on model and hardware
            optimal_settings = self._get_optimal_precision_settings()
            
            # Check current settings
            current_fp16 = getattr(training_args, 'fp16', False)
            current_bf16 = getattr(training_args, 'bf16', False)
            
            # Detect precision mismatches that cause Unsloth errors
            needs_adjustment = False
            adjustment_reason = ""
            
            # Issue 1: Both fp16 and bf16 enabled
            if current_fp16 and current_bf16:
                needs_adjustment = True
                adjustment_reason = "Both fp16 and bf16 enabled - will cause conflicts"
            
            # Issue 2: Model is bf16 but training args specify fp16 (the exact error we're fixing)
            elif model_precision == "bfloat16" and current_fp16 and not current_bf16:
                needs_adjustment = True
                adjustment_reason = "Model is bf16 but training args specify fp16 - this causes Unsloth errors"
            
            # Issue 3: Model is fp16 but training args specify bf16
            elif model_precision == "float16" and current_bf16 and not current_fp16:
                needs_adjustment = True
                adjustment_reason = "Model is fp16 but training args specify bf16 - potential inefficiency"
            
            # Apply fixes if needed
            if needs_adjustment:
                self.logger.warning(f"🔧 PRECISION MISMATCH DETECTED: {adjustment_reason}")
                self.logger.info(f"🔧 Applying automatic fix: {optimal_settings['reason']}")
                
                # Convert training args to dict, modify, and recreate
                args_dict = training_args.to_dict()
                args_dict['fp16'] = optimal_settings['fp16']
                args_dict['bf16'] = optimal_settings['bf16']
                
                # Recreate training arguments
                training_args = TrainingArguments(**args_dict)
                
                self.logger.info(f"✅ FIXED: fp16={optimal_settings['fp16']}, bf16={optimal_settings['bf16']}")
                self.logger.info("✅ This should prevent the 'Unsloth: Model is in bfloat16 precision but you want to use float16 precision' error")
            else:
                self.logger.info(f"✅ Precision settings validated - fp16: {current_fp16}, bf16: {current_bf16}")
            
            return training_args
            
        except Exception as e:
            self.logger.error(f"Error validating precision settings: {e}")
            return training_args  # Return original if validation fails

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
            "optimization": "unsloth" if UNSLOTH_AVAILABLE else "transformers",
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

        status_data.update(
            {
                "status": "completed",
                "progress": "Fine-tuning completed successfully",
                "progress_percentage": 100.0,
                "message": "Fine-tuning completed successfully",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

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

        status_data.update(
            {
                "status": "error",
                "progress": f"Error: {error_message}",
                "message": f"Fine-tuning failed: {error_message}",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

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
                "duration_seconds": time.time() - training_info["start_time"],
            }

        return sessions

    async def start_fine_tuning(self, request: FineTuneRequest) -> str:
        """Async wrapper for fine_tune method"""
        import asyncio
        import concurrent.futures

        # Generate task ID
        task_id = str(uuid.uuid4())

        # Run the fine_tune method in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start training in background and return task_id immediately
            future = executor.submit(self.fine_tune, request, task_id)
            # Don't wait for completion, just return the task_id

        return task_id

    async def list_training_jobs(self) -> List[Dict[str, Any]]:
        """List training jobs with minimal status info.
        Matches the API expectation used by the /jobs and /fine-tune/active endpoints.
        """
        jobs: List[Dict[str, Any]] = []
        # Aggregate running jobs from active_trainings
        for task_id, training_info in self.active_trainings.items():
            jobs.append(
                {
                    "task_id": task_id,
                    "status": "running",
                    "started_at": time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(training_info.get("start_time", time.time())),
                    ),
                    "duration_seconds": time.time()
                    - training_info.get("start_time", time.time()),
                }
            )
        # Optionally, enrich from status files if present
        try:
            logs_dir = Path("logs")
            if logs_dir.exists():
                for status_file in logs_dir.glob("*_status.json"):
                    try:
                        with open(status_file, "r") as f:
                            status_data = json.load(f)
                        task_id = status_data.get("task_id")
                        # Avoid duplicates for running tasks
                        if task_id and not any(
                            j.get("task_id") == task_id for j in jobs
                        ):
                            jobs.append(
                                {
                                    "task_id": task_id,
                                    "status": status_data.get("status", "unknown"),
                                    "progress": status_data.get("progress"),
                                    "progress_percentage": status_data.get(
                                        "progress_percentage"
                                    ),
                                    "updated_at": status_data.get("updated_at"),
                                }
                            )
                    except Exception:
                        # Ignore corrupt status files
                        pass
        except Exception:
            pass
        return jobs

    def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status for health checks"""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = []

                for i in range(gpu_count):
                    device_props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                    memory_total = device_props.total_memory / 1024**3  # GB

                    gpu_info.append(
                        {
                            "device_id": i,
                            "name": device_props.name,
                            "memory_allocated_gb": round(memory_allocated, 2),
                            "memory_reserved_gb": round(memory_reserved, 2),
                            "memory_total_gb": round(memory_total, 2),
                            "memory_free_gb": round(memory_total - memory_reserved, 2),
                            "utilization_percent": round(
                                (memory_reserved / memory_total) * 100, 1
                            ),
                            "available": True,
                        }
                    )

                return {
                    "cuda_available": True,
                    "gpu_count": gpu_count,
                    "gpus": gpu_info,
                    "total_allocated_gb": round(
                        sum(g["memory_allocated_gb"] for g in gpu_info), 2
                    ),
                    "total_reserved_gb": round(
                        sum(g["memory_reserved_gb"] for g in gpu_info), 2
                    ),
                }
            else:
                return {
                    "cuda_available": False,
                    "gpu_count": 0,
                    "gpus": [],
                    "message": "CUDA not available or no GPUs detected",
                }
        except Exception as e:
            self.logger.error(f"Error getting GPU status: {e}")
            return {
                "cuda_available": False,
                "gpu_count": 0,
                "gpus": [],
                "error": str(e),
            }

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

    def estimate_training_time(self, request: FineTuneRequest) -> Dict[str, Any]:
        """
        Estimate training time based on dataset size, model parameters, and hardware

        Args:
            request: Fine-tuning parameters

        Returns:
            Dictionary with time estimates in various formats
        """
        try:
            self.logger.info(
                f"Estimating training time for model: {request.base_model_dir}"
            )

            # Default estimates (these are rough approximations)
            base_time_per_epoch_minutes = 15  # Base estimate for small models

            # Adjust based on epochs
            total_epochs = request.epochs

            # Adjust based on batch size (smaller batch = more steps = longer time)
            batch_size_factor = max(1.0, 4.0 / request.batch_size)

            # Adjust based on sequence length (longer sequences = more computation)
            seq_length_factor = max(1.0, request.max_seq_length / 512.0)

            # Adjust based on PEFT method
            peft_factor = 0.7 if request.peft_method in ["lora", "qlora"] else 1.0

            # Adjust based on optimization backend
            optimization_factor = 0.5 if UNSLOTH_AVAILABLE else 1.0

            # Calculate estimated time per epoch
            estimated_time_per_epoch = (
                base_time_per_epoch_minutes
                * batch_size_factor
                * seq_length_factor
                * peft_factor
                * optimization_factor
            )

            # Total estimated time
            total_estimated_minutes = estimated_time_per_epoch * total_epochs

            # Convert to various formats
            total_hours = total_estimated_minutes / 60
            total_seconds = total_estimated_minutes * 60

            # Format human readable
            if total_estimated_minutes < 60:
                human_readable = f"{int(total_estimated_minutes)} minutes"
            elif total_hours < 24:
                hours = int(total_hours)
                minutes = int((total_hours - hours) * 60)
                human_readable = f"{hours}h {minutes}m"
            else:
                days = int(total_hours / 24)
                hours = int(total_hours % 24)
                human_readable = f"{days}d {hours}h"

            result = {
                "estimated_total_minutes": round(total_estimated_minutes, 1),
                "estimated_total_hours": round(total_hours, 2),
                "estimated_total_seconds": round(total_seconds, 1),
                "estimated_time_per_epoch_minutes": round(estimated_time_per_epoch, 1),
                "human_readable": human_readable,
                "total_epochs": total_epochs,
                "factors": {
                    "batch_size_factor": round(batch_size_factor, 2),
                    "sequence_length_factor": round(seq_length_factor, 2),
                    "peft_factor": round(peft_factor, 2),
                    "optimization_factor": round(optimization_factor, 2),
                },
                "optimization_backend": (
                    "unsloth" if UNSLOTH_AVAILABLE else "transformers"
                ),
                "gpu_acceleration": torch.cuda.is_available(),
                "note": "These are rough estimates and actual time may vary significantly based on dataset complexity, hardware, and system load",
            }

            self.logger.info(f"Estimated training time: {human_readable}")
            return result

        except Exception as e:
            self.logger.error(f"Error estimating training time: {e}")
            return {
                "error": str(e),
                "estimated_total_minutes": None,
                "human_readable": "Unable to estimate",
                "note": "Error occurred during estimation",
            }

    async def get_training_status(self, task_id: str) -> Dict[str, Any]:
        """Get training status for a specific task"""
        try:
            # Check if task is in active trainings
            if task_id in self.active_trainings:
                training_info = self.active_trainings[task_id]
                return {
                    "task_id": task_id,
                    "status": "running",
                    "start_time": training_info["start_time"],
                    "duration_seconds": time.time() - training_info["start_time"],
                }

            # Check status file if exists
            status_file = Path("logs") / f"{task_id}_status.json"
            if status_file.exists():
                with open(status_file, "r") as f:
                    status_data = json.load(f)
                return status_data

            # Task not found
            raise KeyError(f"Task {task_id} not found")

        except Exception as e:
            self.logger.error(f"Error getting training status for {task_id}: {e}")
            raise

    async def stream_logs(self, task_id: str):
        """Stream training logs for a task"""
        try:
            if task_id not in task_log_queues:
                raise KeyError(f"Task {task_id} not found")

            log_queue = task_log_queues[task_id]

            # Stream existing logs first
            logs = []
            while not log_queue.empty():
                try:
                    log_entry = log_queue.get_nowait()
                    logs.append(log_entry)
                except queue.Empty:
                    break

            # Yield existing logs
            for log in logs:
                yield json.dumps(log)

            # Stream new logs until task completion
            while not task_completion_flags.get(task_id, False):
                try:
                    # Wait for new log with timeout
                    log_entry = log_queue.get(timeout=1.0)
                    yield json.dumps(log_entry)
                except queue.Empty:
                    # Yield heartbeat
                    yield json.dumps(
                        {
                            "type": "heartbeat",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )

        except Exception as e:
            self.logger.error(f"Error streaming logs for {task_id}: {e}")
            yield json.dumps({"type": "error", "message": str(e)})

    async def stop_training(self, task_id: str) -> bool:
        """Stop a training job"""
        return self.cancel_training(task_id)

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List available base models"""
        try:
            models = []
            models_dir = Path("/app/models/base")

            if models_dir.exists():
                for model_path in models_dir.iterdir():
                    if model_path.is_dir():
                        # Check if it's a valid model directory
                        config_file = model_path / "config.json"
                        if config_file.exists():
                            try:
                                with open(config_file, "r") as f:
                                    config = json.load(f)

                                models.append(
                                    {
                                        "model_name": model_path.name,
                                        "model_path": str(model_path),
                                        "model_type": config.get(
                                            "model_type", "unknown"
                                        ),
                                        "vocab_size": config.get("vocab_size", 0),
                                        "hidden_size": config.get("hidden_size", 0),
                                        "num_layers": config.get(
                                            "num_hidden_layers", 0
                                        ),
                                        "status": "available",
                                    }
                                )
                            except Exception as e:
                                self.logger.warning(
                                    f"Error reading config for {model_path.name}: {e}"
                                )
                                models.append(
                                    {
                                        "model_name": model_path.name,
                                        "model_path": str(model_path),
                                        "status": "available",
                                        "note": "Config file could not be read",
                                    }
                                )

            if not models:
                # Return fallback if no models found
                models.append(
                    {
                        "model_name": "No models found",
                        "model_path": "/app/models/base",
                        "status": "unavailable",
                        "note": "Please add models to /app/models/base directory",
                    }
                )

            return models

        except Exception as e:
            self.logger.error(f"Error listing available models: {e}")
            return [{"error": str(e)}]

    async def list_available_datasets(self) -> List[Dict[str, Any]]:
        """List available datasets"""
        try:
            datasets = []
            datasets_dir = Path("/app/datasets")

            if datasets_dir.exists():
                for dataset_path in datasets_dir.iterdir():
                    if dataset_path.is_dir():
                        # Check for dataset files
                        train_files = list(dataset_path.glob("train.*"))
                        val_files = list(dataset_path.glob("val.*")) or list(
                            dataset_path.glob("validation.*")
                        )

                        datasets.append(
                            {
                                "dataset_name": dataset_path.name,
                                "dataset_path": str(dataset_path),
                                "has_train": len(train_files) > 0,
                                "has_validation": len(val_files) > 0,
                                "train_files": [str(f) for f in train_files],
                                "validation_files": [str(f) for f in val_files],
                                "status": "available",
                            }
                        )

            if not datasets:
                # Return fallback if no datasets found
                datasets.append(
                    {
                        "dataset_name": "No datasets found",
                        "dataset_path": "/app/datasets",
                        "status": "unavailable",
                        "note": "Please add datasets to /app/datasets directory",
                    }
                )

            return datasets

        except Exception as e:
            self.logger.error(f"Error listing available datasets: {e}")
            return [{"error": str(e)}]

    async def list_training_templates(self) -> List[Dict[str, Any]]:
        """List available training configuration templates"""
        try:
            # Return predefined templates for common scenarios
            templates = [
                {
                    "template_name": "quick_test",
                    "description": "Quick test training with minimal resources",
                    "config": {
                        "epochs": 1,
                        "batch_size": 1,
                        "learning_rate": 5e-5,
                        "max_seq_length": 512,
                        "peft_method": "lora",
                        "gradient_accumulation_steps": 4,
                    },
                },
                {
                    "template_name": "standard_lora",
                    "description": "Standard LoRA fine-tuning configuration",
                    "config": {
                        "epochs": 3,
                        "batch_size": 2,
                        "learning_rate": 2e-4,
                        "max_seq_length": 1024,
                        "peft_method": "lora",
                        "gradient_accumulation_steps": 8,
                    },
                },
                {
                    "template_name": "high_quality",
                    "description": "High quality training with more epochs",
                    "config": {
                        "epochs": 5,
                        "batch_size": 4,
                        "learning_rate": 1e-4,
                        "max_seq_length": 2048,
                        "peft_method": "lora",
                        "gradient_accumulation_steps": 4,
                    },
                },
                {
                    "template_name": "qlora_efficient",
                    "description": "Memory-efficient QLoRA configuration",
                    "config": {
                        "epochs": 3,
                        "batch_size": 1,
                        "learning_rate": 1e-4,
                        "max_seq_length": 1024,
                        "peft_method": "qlora",
                        "gradient_accumulation_steps": 16,
                    },
                },
            ]

            return templates

        except Exception as e:
            self.logger.error(f"Error listing training templates: {e}")
            return [{"error": str(e)}]

    async def optimize_memory(self):
        """Optimize memory usage"""
        try:
            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("CUDA cache cleared")

            # Clean up completed training resources
            completed_tasks = []
            for task_id in list(self.active_trainings.keys()):
                if task_completion_flags.get(task_id, False):
                    completed_tasks.append(task_id)

            for task_id in completed_tasks:
                self._cleanup_training_resources(task_id)

            self.logger.info(
                f"Memory optimization completed. Cleaned up {len(completed_tasks)} completed tasks"
            )

        except Exception as e:
            self.logger.error(f"Error during memory optimization: {e}")
            raise

    async def cleanup(self):
        """Cleanup service on shutdown"""
        try:
            self.cleanup_all_trainings()
            self.logger.info("Training service cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def get_hardware_profile(self) -> Dict[str, Any]:
        """Get detailed hardware profile for debugging and optimization"""
        try:
            hardware_summary = self.adaptive_config.get_hardware_summary()
            sample_config = self.adaptive_config.generate_optimal_config(
                type(
                    "MockRequest",
                    (),
                    {
                        "batch_size": 2,
                        "gradient_accumulation_steps": 4,
                        "epochs": 1,
                        "learning_rate": 2e-4,
                        "max_seq_length": 1024,
                        "peft_method": "lora",
                    },
                )()
            )

            return {
                "hardware_summary": hardware_summary,
                "detected_profile": {
                    "gpu_name": self.hardware_profile.gpu_name,
                    "gpu_memory_gb": self.hardware_profile.gpu_memory_gb,
                    "system_memory_gb": self.hardware_profile.system_memory_gb,
                    "container_memory_gb": self.hardware_profile.container_memory_gb,
                    "cpu_cores": self.hardware_profile.cpu_cores,
                    "cpu_threads": self.hardware_profile.cpu_threads,
                    "storage_type": self.hardware_profile.storage_type,
                    "is_container": self.hardware_profile.is_container,
                    "cuda_available": self.hardware_profile.cuda_available,
                },
                "sample_adaptive_config": {
                    "effective_memory_gb": sample_config["effective_memory_gb"],
                    "gpu_optimizations": sample_config["gpu_optimizations"],
                    "training_args": sample_config["training_args"],
                    "lora_config": sample_config["lora_config"],
                },
                "optimization_tier": self._get_optimization_tier(),
            }
        except Exception as e:
            self.logger.error(f"Error getting hardware profile: {e}")
            return {"error": str(e)}

    def _get_optimization_tier(self) -> str:
        """Get human-readable optimization tier"""
        memory_gb = (
            self.hardware_profile.container_memory_gb
            or self.hardware_profile.available_memory_gb
        )
        gpu_memory = self.hardware_profile.gpu_memory_gb

        if not self.hardware_profile.cuda_available:
            return "CPU_ONLY"
        elif gpu_memory <= 4:
            return "LOW_VRAM"  # GTX 1650, RTX 3050 4GB
        elif gpu_memory <= 8:
            return "MID_VRAM"  # RTX 4060, RTX 3070
        elif gpu_memory <= 16:
            return "HIGH_VRAM"  # RTX 4080, RTX 3080 Ti
        else:
            return "ENTHUSIAST"  # RTX 4090, A100


# Create global service instance
training_service = TrainingService()
