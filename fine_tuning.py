# CRITICAL: Import Unsloth FIRST
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

try:
    import unsloth
    from unsloth import FastLanguageModel, is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
    print(f"Unsloth imported successfully (v{unsloth.__version__})")
except (ImportError, NotImplementedError) as e:
    UNSLOTH_AVAILABLE = False
    print(f"WARNING: Unsloth not available ({e})")

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

print("Environment setup complete:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA devices: {torch.cuda.device_count()}")
    print(f"   Primary GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   CUDA not available - Unsloth optimizations will be used in Docker/NVIDIA environments")

class FineTuneRequest(BaseModel):
    base_model_dir: str = "/app/models/base/unsloth_llama-3.1-8b-unsloth-bnb-4bit"
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
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    message: Optional[str] = None
    
    def save(self):
        """Save status to file for persistence"""
        status_file = Path("logs") / f"{self.task_id}_status.json"
        status_file.parent.mkdir(exist_ok=True)
        with open(status_file, "w") as f:
            json.dump(self.dict(), f, indent=2)
    
    def update(self, status: str = None, progress: str = None, **kwargs):
        """Update status fields"""
        if status:
            self.status = status
        if progress:
            self.progress = progress
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.save()


class LoggingCallback(TrainerCallback):
    """Custom callback to capture training logs and update status."""
    
    def __init__(self, status: TrainingStatus):
        self.status = status
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs something."""
        if logs is not None:
            # Update status with training metrics
            updates = {}
            if 'loss' in logs:
                updates['loss'] = logs['loss']
            if 'learning_rate' in logs:
                updates['learning_rate'] = logs['learning_rate']
            if 'epoch' in logs:
                updates['current_epoch'] = int(logs['epoch']) if logs['epoch'] else None
                updates['total_epochs'] = args.num_train_epochs
            if 'step' in logs or hasattr(state, 'global_step'):
                updates['current_step'] = state.global_step
                updates['total_steps'] = state.max_steps
            
            # Create progress message
            progress_parts = []
            if 'loss' in logs:
                progress_parts.append(f"Loss: {logs['loss']:.4f}")
            if 'learning_rate' in logs:
                progress_parts.append(f"LR: {logs['learning_rate']:.2e}")
            if 'epoch' in logs:
                progress_parts.append(f"Epoch: {logs['epoch']:.2f}")
            
            progress_msg = " | ".join(progress_parts) if progress_parts else "Training in progress..."
            updates['progress'] = progress_msg
            
            self.status.update(**updates)
            
            # Print to console for Docker logs
            print(f"Training Update: {progress_msg}")


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
        print(f"Loading model with Unsloth optimizations: {model_dir}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=True,  # Always use 4-bit for maximum memory savings
            device_map="balanced" if hardware_info["gpu_count"] > 1 else "auto",
            # Additional Unsloth optimizations
            token=None,  # Add HF token if needed
            trust_remote_code=False,  # Security setting
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
        torch_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
        torch_dtype = torch.float16
    
    # Setup optimized quantization
    if peft_method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_config,
            device_map="balanced" if hardware_info["gpu_count"] > 1 else "auto",
            dtype=torch_dtype,
            attn_implementation="flash_attention_2" if (hardware_info["gpu_count"] > 0 and xformers_available) else None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="balanced" if hardware_info["gpu_count"] > 1 else "auto", 
            torch_dtype=torch_dtype,
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

def fine_tune(request: FineTuneRequest) -> dict:
    """
    Launch fine-tuning with Unsloth on prepared datasets.
    Auto-detects dataset format (ShareGPT, Instruction, Plain Text).
    """
    task_id = str(uuid.uuid4())
    status = TrainingStatus(task_id=task_id, status="running", progress="Starting fine-tuning...")
    status.save()

    def run_training():
        try:
            logger.info("Starting fine-tuning process...")
            
            # Load datasets from their respective directories
            logger.info(f"Loading datasets from {request.dataset_dir}")
            train_dataset = load_from_disk(f"{request.dataset_dir}/train")
            val_dataset = load_from_disk(f"{request.dataset_dir}/val")
            logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

            # Detect dataset type
            if "conversations" in train_dataset.column_names:
                formatting_func = format_sharegpt
                logger.info("Detected ShareGPT dataset format.")
            elif "instruction" in train_dataset.column_names:
                formatting_func = format_instruction
                logger.info("Detected Instruction dataset format.")
            elif "question" in train_dataset.column_names and "answer" in train_dataset.column_names:
                formatting_func = format_plain_text
                logger.info("Detected Question-Answer dataset format.")
            else:
                formatting_func = format_plain_text
                logger.info("Detected Plain Text dataset format.")

            # Load base model with Unsloth optimizations
            logger.info(f"Loading model from {request.base_model_dir}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=request.base_model_dir,
                load_in_4bit=True,
                max_seq_length=512,
                dtype=None,
                device_map="auto"
            )
            logger.info("Model loaded successfully")

            # Apply LoRA
            logger.info("Applying LoRA configuration...")
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "v_proj"],
                lora_alpha=16,
                lora_dropout=0.0,  # Set to 0 for optimal Unsloth performance
                bias="none",
            )
            logger.info("LoRA applied successfully")

            # Training arguments (Unsloth defaults)
            logger.info("Setting up training arguments...")
            training_args = TrainingArguments(
                output_dir=request.output_dir,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=16,
                learning_rate=2e-4,
                num_train_epochs=1,
                logging_steps=10,
                save_strategy="steps",
                save_steps=50,
                save_total_limit=2,
                eval_strategy="steps",
                eval_steps=50,
                bf16=True,  # Ampere+ GPUs
                report_to="none"
            )

            # Create logging callback
            logging_callback = LoggingCallback(status)
            
            # Use SFTTrainer for instruction fine-tuning
            logger.info("Creating SFTTrainer...")
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset if len(val_dataset) > 0 else None,
                tokenizer=tokenizer,
                formatting_func=formatting_func,
                max_seq_length=512,
                callbacks=[logging_callback],
            )
            logger.info("SFTTrainer created successfully")

            # Start training
            logger.info("Starting training...")
            trainer.train()
            logger.info("Training completed, saving model...")
            trainer.save_model(request.output_dir)
            tokenizer.save_pretrained(request.output_dir)
            logger.info("Model saved successfully")

            status.update(status="completed", progress="Fine-tuning completed successfully.")

        except Exception as e:
            logger.exception("Fine-tuning failed")
            status.update(status="failed", progress=f"Fine-tuning failed: {str(e)}")

    threading.Thread(target=run_training, daemon=True).start()
    return {"task_id": task_id, "status": "started"}