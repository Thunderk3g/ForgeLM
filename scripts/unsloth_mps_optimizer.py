#!/usr/bin/env python3
"""
Unsloth MPS Optimizer for LLMForge
Optimizes Unsloth training and inference for MPS GPU sharing environment.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnslothMPSOptimizer:
    """Optimizes Unsloth configuration for MPS environment"""
    
    def __init__(self):
        self.service_type = os.getenv('SERVICE_TYPE', 'unknown')
        self.mps_thread_percentage = float(os.getenv('MPS_ACTIVE_THREAD_PERCENTAGE', '50'))
        self.mps_memory_limit = int(os.getenv('MPS_PINNED_DEVICE_MEM_LIMIT', '4096'))  # MB
        self.gpu_sharing_mode = os.getenv('GPU_SHARING_MODE', 'exclusive')
        
        # Detect service type from container name if not set
        if self.service_type == 'unknown':
            container_name = os.getenv('HOSTNAME', '')
            if 'training' in container_name:
                self.service_type = 'training'
            elif 'chat-comparison' in container_name:
                self.service_type = 'inference'
            elif 'model-management' in container_name:
                self.service_type = 'model_management'
    
    def get_optimized_unsloth_config(self) -> Dict[str, Any]:
        """Get optimized Unsloth configuration for current service and MPS setup"""
        
        base_config = {
            # Core Unsloth settings
            'load_in_4bit': True,
            'use_gradient_checkpointing': "unsloth",
            'random_state': 3407,
            'use_rslora': False,  # Set to True if rank > 256
            'loftq_config': None,
            
            # Memory optimizations for MPS
            'max_seq_length': self._get_optimal_sequence_length(),
            'device_map': self._get_device_map(),
            
            # LoRA settings optimized for shared GPU
            'r': self._get_optimal_lora_rank(),
            'target_modules': self._get_target_modules(),
            'lora_alpha': None,  # Will be set to r if None
            'lora_dropout': 0.0,
            'bias': 'none',
            
            # Training arguments optimized for MPS
            'training_args': self._get_training_arguments(),
            
            # Quantization config
            'bnb_4bit_config': self._get_quantization_config(),
        }
        
        # Service-specific optimizations
        if self.service_type == 'training':
            base_config.update(self._get_training_optimizations())
        elif self.service_type in ['inference', 'model_management']:
            base_config.update(self._get_inference_optimizations())
        
        return base_config
    
    def _get_optimal_sequence_length(self) -> int:
        """Get optimal sequence length based on MPS memory allocation"""
        if self.service_type == 'training':
            # Training service gets priority - longer sequences
            if self.mps_memory_limit >= 8192:  # 8GB+
                return 4096
            elif self.mps_memory_limit >= 4096:  # 4GB+
                return 2048
            else:
                return 1024
        else:
            # Inference services - shorter sequences for faster response
            if self.mps_memory_limit >= 4096:
                return 2048
            else:
                return 1024
    
    def _get_device_map(self) -> str:
        """Get device mapping strategy for MPS"""
        if self.gpu_sharing_mode == 'mps':
            # For MPS, use auto device mapping but constrain to single GPU
            return "auto"
        else:
            return "balanced"
    
    def _get_optimal_lora_rank(self) -> int:
        """Get optimal LoRA rank based on memory constraints"""
        if self.service_type == 'training':
            # Higher rank for training if memory allows
            if self.mps_memory_limit >= 8192:
                return 64
            elif self.mps_memory_limit >= 4096:
                return 32
            else:
                return 16
        else:
            # Lower rank for inference to save memory
            return 16
    
    def _get_target_modules(self) -> list:
        """Get optimal target modules for LoRA based on service type"""
        if self.service_type == 'training':
            # Full attention for training
            return ["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]
        else:
            # Reduced modules for inference
            return ["q_proj", "v_proj", "o_proj"]
    
    def _get_training_arguments(self) -> Dict[str, Any]:
        """Get optimized training arguments for MPS environment"""
        
        # Base settings optimized for MPS
        args = {
            'per_device_train_batch_size': self._get_optimal_batch_size(),
            'gradient_accumulation_steps': self._get_accumulation_steps(),
            'learning_rate': 2e-4,
            'warmup_steps': 5,
            'max_steps': 60,
            'logging_steps': 1,
            'save_steps': 30,
            'save_total_limit': 2,
            'optim': "adamw_8bit",
            'weight_decay': 0.01,
            'lr_scheduler_type': "linear",
            'seed': 3407,
            'output_dir': "./outputs",
            'report_to': "none",  # Disable wandb in shared environment
            
            # Memory optimizations
            'fp16': True,
            'bf16': False,  # Use fp16 for better compatibility
            'gradient_checkpointing': True,
            'dataloader_pin_memory': False,  # Reduce memory pressure
            'remove_unused_columns': False,
            
            # MPS-specific settings
            'dataloader_num_workers': 1,  # Reduce workers in shared environment
            'group_by_length': True,
            'ddp_find_unused_parameters': False,
        }
        
        # Service-specific adjustments
        if self.service_type == 'training':
            if self.mps_thread_percentage >= 60:  # High priority training
                args.update({
                    'per_device_train_batch_size': max(args['per_device_train_batch_size'], 2),
                    'max_steps': 100,
                })
        
        return args
    
    def _get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on memory allocation"""
        if self.service_type == 'training':
            if self.mps_memory_limit >= 8192:
                return 2
            elif self.mps_memory_limit >= 4096:
                return 1
            else:
                return 1
        else:
            # Inference always uses batch size 1
            return 1
    
    def _get_accumulation_steps(self) -> int:
        """Get gradient accumulation steps to maintain effective batch size"""
        batch_size = self._get_optimal_batch_size()
        target_effective_batch = 8  # Target effective batch size
        
        return max(1, target_effective_batch // batch_size)
    
    def _get_quantization_config(self) -> Dict[str, Any]:
        """Get optimized 4-bit quantization config for MPS"""
        return {
            'bnb_4bit_compute_dtype': 'float16',
            'bnb_4bit_quant_type': 'nf4',
            'bnb_4bit_use_double_quant': True,
            'bnb_4bit_quant_storage_type': 'uint8',
        }
    
    def _get_training_optimizations(self) -> Dict[str, Any]:
        """Get training-specific optimizations"""
        return {
            'max_steps': 100 if self.mps_thread_percentage >= 60 else 60,
            'save_strategy': 'steps',
            'evaluation_strategy': 'no',  # Disable eval to save memory
            'load_best_model_at_end': False,
            'metric_for_best_model': None,
            
            # Unsloth-specific training optimizations
            'packing': True,  # Enable packing for efficiency
            'dataset_text_field': "text",
            'max_seq_length': self._get_optimal_sequence_length(),
        }
    
    def _get_inference_optimizations(self) -> Dict[str, Any]:
        """Get inference-specific optimizations"""
        return {
            # Inference-specific settings
            'torch_dtype': 'float16',
            'attn_implementation': 'flash_attention_2',
            
            # Memory optimizations for inference
            'low_cpu_mem_usage': True,
            'use_cache': True,
            
            # Reduced precision for faster inference
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'max_new_tokens': 512,
        }
    
    def apply_environment_optimizations(self):
        """Apply environment-level optimizations for Unsloth with MPS"""
        
        # PyTorch CUDA allocator optimizations for MPS
        cuda_alloc_conf = [
            f"max_split_size_mb:{128 if self.service_type != 'training' else 256}",
            "expandable_segments:True",
        ]
        
        if self.service_type == 'training' and self.mps_thread_percentage >= 60:
            cuda_alloc_conf.append("backend:cudaMallocAsync")
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ','.join(cuda_alloc_conf)
        
        # CUDA optimizations
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Triton optimizations for Unsloth
        os.environ['TRITON_CACHE_DIR'] = '/tmp/.triton_cache'
        
        # OMP settings for shared environment
        if self.service_type == 'training':
            os.environ['OMP_NUM_THREADS'] = '4'
        else:
            os.environ['OMP_NUM_THREADS'] = '2'
        
        # Memory mapping optimizations
        os.environ['MMAP_MAX_SIZE'] = str(self.mps_memory_limit * 1024 * 1024)  # Convert MB to bytes
        
        logger.info(f"Applied environment optimizations for {self.service_type} service with MPS")
    
    def get_unsloth_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get Unsloth model loading configuration"""
        
        config = {
            'model_name': model_name,
            'max_seq_length': self._get_optimal_sequence_length(),
            'dtype': None,  # Auto-detect
            'load_in_4bit': True,
            'device_map': self._get_device_map(),
        }
        
        # Add quantization config
        config.update(self._get_quantization_config())
        
        return config
    
    def get_sft_trainer_config(self) -> Dict[str, Any]:
        """Get optimized SFTTrainer configuration for MPS"""
        
        config = {
            'model': None,  # Will be set by caller
            'tokenizer': None,  # Will be set by caller
            'train_dataset': None,  # Will be set by caller
            'dataset_text_field': "text",
            'max_seq_length': self._get_optimal_sequence_length(),
            'dataset_num_proc': 1,  # Reduced for shared environment
            'packing': True,
            'args': self._get_training_arguments(),
        }
        
        return config
    
    def save_config_to_file(self, output_path: str = "/tmp/unsloth_mps_config.json"):
        """Save the complete configuration to a file"""
        
        config = {
            'service_type': self.service_type,
            'mps_config': {
                'thread_percentage': self.mps_thread_percentage,
                'memory_limit_mb': self.mps_memory_limit,
                'sharing_mode': self.gpu_sharing_mode,
            },
            'unsloth_config': self.get_optimized_unsloth_config(),
            'environment_vars': {
                'PYTORCH_CUDA_ALLOC_CONF': os.getenv('PYTORCH_CUDA_ALLOC_CONF'),
                'CUDA_LAUNCH_BLOCKING': os.getenv('CUDA_LAUNCH_BLOCKING'),
                'TOKENIZERS_PARALLELISM': os.getenv('TOKENIZERS_PARALLELISM'),
                'OMP_NUM_THREADS': os.getenv('OMP_NUM_THREADS'),
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved Unsloth MPS configuration to {output_path}")
        return output_path

def create_optimized_training_script(service_type: str = 'training') -> str:
    """Create an optimized training script template for MPS environment"""
    
    script_template = '''#!/usr/bin/env python3
"""
Optimized Unsloth Training Script for MPS Environment
Auto-generated configuration for {service_type} service
"""

import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth_mps_optimizer import UnslothMPSOptimizer

def main():
    # Initialize MPS optimizer
    optimizer = UnslothMPSOptimizer()
    optimizer.apply_environment_optimizations()
    
    # Get optimized configuration
    config = optimizer.get_optimized_unsloth_config()
    model_config = optimizer.get_unsloth_model_config("unsloth/llama-3-8b-bnb-4bit")
    
    print(f"Starting {{optimizer.service_type}} with MPS optimization...")
    print(f"Memory limit: {{optimizer.mps_memory_limit}}MB")
    print(f"Thread percentage: {{optimizer.mps_thread_percentage}}%")
    
    # Load model with optimized settings
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config['model_name'],
        max_seq_length=model_config['max_seq_length'],
        dtype=model_config['dtype'],
        load_in_4bit=model_config['load_in_4bit'],
        device_map=model_config['device_map'],
    )
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['r'],
        target_modules=config['target_modules'],
        lora_alpha=config['lora_alpha'] or config['r'],
        lora_dropout=config['lora_dropout'],
        bias=config['bias'],
        use_gradient_checkpointing=config['use_gradient_checkpointing'],
        random_state=config['random_state'],
        use_rslora=config['use_rslora'],
        loftq_config=config['loftq_config'],
    )
    
    # Your dataset loading code here
    # dataset = load_dataset(...)
    
    # Create trainer with optimized settings
    trainer_config = optimizer.get_sft_trainer_config()
    trainer_config.update({{
        'model': model,
        'tokenizer': tokenizer,
        # 'train_dataset': dataset,  # Add your dataset here
    }})
    
    trainer = SFTTrainer(**trainer_config)
    
    # Start training
    print("Starting optimized training...")
    trainer.train()
    
    # Save model
    model.save_pretrained("./outputs/lora_model")
    tokenizer.save_pretrained("./outputs/lora_model")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
'''.format(service_type=service_type)
    
    return script_template

def main():
    """CLI interface for the optimizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unsloth MPS Optimizer')
    parser.add_argument('--service-type', choices=['training', 'inference', 'model_management'], 
                       default='training', help='Service type to optimize for')
    parser.add_argument('--save-config', help='Save configuration to file')
    parser.add_argument('--create-script', help='Create optimized training script')
    parser.add_argument('--apply-env', action='store_true', help='Apply environment optimizations')
    
    args = parser.parse_args()
    
    # Override service type if provided
    if args.service_type:
        os.environ['SERVICE_TYPE'] = args.service_type
    
    optimizer = UnslothMPSOptimizer()
    
    if args.apply_env:
        optimizer.apply_environment_optimizations()
        print("Applied environment optimizations")
    
    if args.save_config:
        optimizer.save_config_to_file(args.save_config)
    
    if args.create_script:
        script_content = create_optimized_training_script(optimizer.service_type)
        with open(args.create_script, 'w') as f:
            f.write(script_content)
        print(f"Created optimized training script: {args.create_script}")
    
    # Print configuration summary
    config = optimizer.get_optimized_unsloth_config()
    print(f"\\n=== Unsloth MPS Configuration for {optimizer.service_type.upper()} ===")
    print(f"Service Type: {optimizer.service_type}")
    print(f"MPS Thread %: {optimizer.mps_thread_percentage}")
    print(f"Memory Limit: {optimizer.mps_memory_limit}MB")
    print(f"Sequence Length: {config['max_seq_length']}")
    print(f"LoRA Rank: {config['r']}")
    print(f"Batch Size: {config['training_args']['per_device_train_batch_size']}")
    print(f"Device Map: {config['device_map']}")
    print()

if __name__ == '__main__':
    main()