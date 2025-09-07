#!/usr/bin/env python3
"""
LLMForge Model Generation Diagnostic Tool

This script helps diagnose model loading and generation issues.
"""

import os
import sys
import asyncio
import torch
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.model_service import ModelService, model_service
from models.entities import ModelPair
from utils.logging import get_logger
from config.settings import settings

logger = get_logger(__name__)

async def diagnose_environment():
    """Diagnose the environment setup"""
    logger.info("=== ENVIRONMENT DIAGNOSTICS ===")
    
    # Check CUDA availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}, Memory: {props.total_memory / (1024**3):.1f}GB")
    
    # Check model directories
    logger.info(f"Base models directory: {settings.models_base_dir}")
    logger.info(f"Base models exist: {Path(settings.models_base_dir).exists()}")
    
    logger.info(f"Fine-tuned models directory: {settings.models_finetuned_dir}")
    logger.info(f"Fine-tuned models exist: {Path(settings.models_finetuned_dir).exists()}")
    
    # Check available model files
    base_dir = Path(settings.models_base_dir)
    if base_dir.exists():
        for model_dir in base_dir.iterdir():
            if model_dir.is_dir():
                logger.info(f"Found base model: {model_dir.name}")
                safetensors_files = list(model_dir.glob("*.safetensors"))
                logger.info(f"  - SafeTensor files: {len(safetensors_files)}")
                config_files = list(model_dir.glob("config.json"))
                logger.info(f"  - Config files: {len(config_files)}")
    
    # Check fine-tuned directory
    ft_dir = Path(settings.models_finetuned_dir)
    if ft_dir.exists():
        adapter_files = list(ft_dir.glob("adapter_model.safetensors"))
        logger.info(f"Fine-tuned adapter files: {len(adapter_files)}")
        
        checkpoints_dir = ft_dir / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.iterdir())
            logger.info(f"Available checkpoints: {len(checkpoints)}")
            for cp in checkpoints:
                if cp.is_dir():
                    logger.info(f"  - {cp.name}")

async def test_model_discovery():
    """Test model pair discovery"""
    logger.info("\n=== MODEL DISCOVERY TEST ===")
    
    try:
        service = ModelService()
        pairs = await service.discover_model_pairs()
        logger.info(f"Discovered {len(pairs)} model pairs:")
        
        for pair in pairs:
            logger.info(f"  - Pair ID: {pair.pair_id}")
            logger.info(f"    Base: {pair.base_model_name} ({pair.base_model_path})")
            logger.info(f"    Adapter: {pair.adapter_name} ({pair.adapter_path})")
            logger.info(f"    Base exists: {Path(pair.base_model_path).exists()}")
            logger.info(f"    Adapter exists: {Path(pair.adapter_path).exists()}")
        
        return pairs
        
    except Exception as e:
        logger.error(f"Model discovery failed: {e}")
        return []

async def test_model_loading(pairs):
    """Test model loading for discovered pairs"""
    logger.info("\n=== MODEL LOADING TEST ===")
    
    if not pairs:
        logger.warning("No model pairs to test")
        return
    
    service = ModelService()
    test_pair = pairs[0]  # Test first pair
    
    try:
        logger.info(f"Testing model loading for pair: {test_pair.pair_id}")
        
        # Try to load models
        models = await service._load_models_for_generation(test_pair)
        base_model, tokenizer, finetuned_model, adapter_tokenizer = models
        
        logger.info(f"Model loading results:")
        logger.info(f"  - Base model: {type(base_model)} (None: {base_model is None})")
        logger.info(f"  - Tokenizer: {type(tokenizer)} (None: {tokenizer is None})")
        logger.info(f"  - Fine-tuned model: {type(finetuned_model)} (None: {finetuned_model is None})")
        logger.info(f"  - Adapter tokenizer: {type(adapter_tokenizer)} (None: {adapter_tokenizer is None})")
        
        if base_model is not None:
            logger.info(f"  - Base model device: {next(base_model.parameters()).device}")
        if finetuned_model is not None:
            logger.info(f"  - Fine-tuned model device: {next(finetuned_model.parameters()).device}")
            
        return models
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.error(f"Exception type: {type(e)}")
        return None

async def test_tokenization(models):
    """Test tokenization process"""
    logger.info("\n=== TOKENIZATION TEST ===")
    
    if models is None:
        logger.warning("No models to test tokenization")
        return
    
    base_model, tokenizer, finetuned_model, adapter_tokenizer = models
    
    test_prompt = "Human: Hello, how are you today?\n\nAssistant:"
    
    try:
        logger.info(f"Testing tokenization with prompt: {test_prompt}")
        
        # Test base tokenizer
        if tokenizer is not None:
            inputs = tokenizer([test_prompt], return_tensors="pt", padding=True, truncation=True)
            logger.info(f"Base tokenizer results:")
            logger.info(f"  - Input IDs shape: {inputs['input_ids'].shape}")
            logger.info(f"  - Attention mask shape: {inputs.get('attention_mask', 'None').shape if inputs.get('attention_mask') is not None else 'None'}")
            logger.info(f"  - Pad token ID: {tokenizer.pad_token_id}")
            logger.info(f"  - EOS token ID: {tokenizer.eos_token_id}")
        
        # Test adapter tokenizer
        if adapter_tokenizer is not None and adapter_tokenizer != tokenizer:
            inputs = adapter_tokenizer([test_prompt], return_tensors="pt", padding=True, truncation=True)
            logger.info(f"Adapter tokenizer results:")
            logger.info(f"  - Input IDs shape: {inputs['input_ids'].shape}")
            logger.info(f"  - Attention mask shape: {inputs.get('attention_mask', 'None').shape if inputs.get('attention_mask') is not None else 'None'}")
            logger.info(f"  - Pad token ID: {adapter_tokenizer.pad_token_id}")
            logger.info(f"  - EOS token ID: {adapter_tokenizer.eos_token_id}")
            
    except Exception as e:
        logger.error(f"Tokenization test failed: {e}")

async def test_simple_generation(models):
    """Test simple model generation"""
    logger.info("\n=== GENERATION TEST ===")
    
    if models is None:
        logger.warning("No models to test generation")
        return
    
    base_model, tokenizer, finetuned_model, adapter_tokenizer = models
    
    test_prompt = "Human: What is the capital of France?\n\nAssistant:"
    
    service = ModelService()
    
    try:
        # Test base model generation
        logger.info("Testing base model generation...")
        base_result = await service._generate_single_response_simple(
            base_model, tokenizer, test_prompt, {"max_new_tokens": 20}, "base"
        )
        logger.info(f"Base model response: {base_result}")
        
    except Exception as e:
        logger.error(f"Base model generation failed: {e}")
        logger.error(f"Exception type: {type(e)}")
    
    try:
        # Test fine-tuned model generation
        logger.info("Testing fine-tuned model generation...")
        ft_result = await service._generate_single_response_simple(
            finetuned_model, adapter_tokenizer, test_prompt, {"max_new_tokens": 20}, "finetuned"
        )
        logger.info(f"Fine-tuned model response: {ft_result}")
        
    except Exception as e:
        logger.error(f"Fine-tuned model generation failed: {e}")
        logger.error(f"Exception type: {type(e)}")

async def main():
    """Main diagnostic function"""
    logger.info("Starting LLMForge Model Generation Diagnostics")
    logger.info("=" * 50)
    
    # Run diagnostics
    await diagnose_environment()
    
    pairs = await test_model_discovery()
    
    models = await test_model_loading(pairs)
    
    await test_tokenization(models)
    
    await test_simple_generation(models)
    
    logger.info("\n" + "=" * 50)
    logger.info("Diagnostics completed")

if __name__ == "__main__":
    asyncio.run(main())