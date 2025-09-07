
"""
CPU-Optimized Model Service for LLMForge

This version removes GPU dependencies and TensorFlow conflicts.
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    print("Transformers available (CPU mode)")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Transformers not available: {e}")

from models.entities import ModelPair
from utils.logging import get_logger
from config.settings import settings

class CPUModelService:
    """CPU-optimized model service without GPU dependencies"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.loaded_models_cache = {}
        self.logger.info("CPU-optimized ModelService initialized")
    
    async def load_models_cpu_only(self, pair: ModelPair):
        """Load models in CPU-only mode with minimal dependencies"""
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library not available")
        
        base_model_path = pair.base_model_path
        adapter_path = pair.adapter_path
        
        self.logger.info(f"Loading models in CPU mode...")
        self.logger.info(f"Base model: {base_model_path}")
        self.logger.info(f"Adapter: {adapter_path}")
        
        # Load with minimal CPU configuration
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, 
            trust_remote_code=True,
            use_fast=True
        )
        
        # Fix pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with CPU-optimized settings
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",           # Force CPU
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        
        # For now, use base model for both (proper adapter loading can be added later)
        finetuned_model = base_model
        adapter_tokenizer = tokenizer
        
        self.logger.info("CPU models loaded successfully")
        
        return base_model, tokenizer, finetuned_model, adapter_tokenizer
    
    async def generate_cpu_safe(self, model, tokenizer, prompt: str, max_new_tokens: int = 50):
        """CPU-safe generation method"""
        
        if model is None or tokenizer is None:
            raise RuntimeError("Model or tokenizer is None")
        
        # Tokenize
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512  # Smaller for CPU
        )
        
        # Generate with CPU-optimized parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
