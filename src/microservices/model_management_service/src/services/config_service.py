"""
Configuration Service for Model Management

Handles configuration management, validation, and template operations.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from models.schemas import FineTuneRequest
from utils.logging import get_logger

logger = get_logger(__name__)


class ConfigService:
    """Service for configuration management"""
    
    def __init__(self, config_dir: str = "/app/configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    async def list_configs(self) -> Dict[str, Any]:
        """List all available configuration files"""
        try:
            configs = []
            for config_file in self.config_dir.glob("*.json"):
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    configs.append({
                        "name": config_file.stem,
                        "path": str(config_file),
                        "type": config_data.get("ft_type", "unknown"),
                        "peft_method": config_data.get("peft_method", "unknown"),
                        "created_at": config_data.get("created_at", "unknown"),
                        "description": config_data.get("description", "")
                    })
                except Exception as e:
                    logger.warning(f"Failed to read config {config_file}: {e}")
                    continue
            
            return {
                "status": "success",
                "configs": configs,
                "total_count": len(configs)
            }
        except Exception as e:
            logger.error(f"Failed to list configs: {e}")
            return {
                "status": "error",
                "message": str(e),
                "configs": [],
                "total_count": 0
            }
    
    async def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get specific configuration by name"""
        try:
            config_path = self.config_dir / f"{config_name}.json"
            
            if not config_path.exists():
                return {
                    "status": "error",
                    "message": f"Configuration '{config_name}' not found"
                }
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            return {
                "status": "success",
                "config": config_data,
                "name": config_name,
                "path": str(config_path)
            }
        except Exception as e:
            logger.error(f"Failed to get config {config_name}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def create_config_from_request(self, request: FineTuneRequest, config_name: str) -> Dict[str, Any]:
        """Create configuration from training request"""
        try:
            from datetime import datetime
            
            config_data = {
                "name": config_name,
                "description": f"Configuration created from fine-tuning request",
                "created_at": datetime.now().isoformat(),
                "ft_type": request.ft_type,
                "peft_method": request.peft_method,
                "alignment": request.alignment,
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate,
                "gradient_accumulation_steps": request.gradient_accumulation_steps,
                "max_seq_length": request.max_seq_length,
                "peft_rank": request.peft_rank,
                "peft_alpha": request.peft_alpha,
                "peft_dropout": request.peft_dropout,
                "target_modules": request.target_modules,
                "template_name": request.template_name,
                "system_message": request.system_message,
                "reasoning_effort": request.reasoning_effort
            }
            
            config_path = self.config_dir / f"{config_name}.json"
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Created config {config_name} at {config_path}")
            
            return {
                "status": "success",
                "message": f"Configuration '{config_name}' created successfully",
                "config": config_data,
                "path": str(config_path)
            }
        except Exception as e:
            logger.error(f"Failed to create config {config_name}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def validate_config(self, config_name: str) -> Dict[str, Any]:
        """Validate configuration"""
        try:
            config_result = await self.get_config(config_name)
            if config_result["status"] == "error":
                return config_result
            
            config_data = config_result["config"]
            
            # Validation checks
            errors = []
            warnings = []
            
            # Check required fields
            required_fields = ["ft_type", "peft_method", "epochs", "batch_size", "learning_rate"]
            for field in required_fields:
                if field not in config_data:
                    errors.append(f"Missing required field: {field}")
            
            # Validate values
            if "epochs" in config_data:
                if not isinstance(config_data["epochs"], int) or config_data["epochs"] <= 0:
                    errors.append("epochs must be a positive integer")
            
            if "batch_size" in config_data:
                if not isinstance(config_data["batch_size"], int) or config_data["batch_size"] <= 0:
                    errors.append("batch_size must be a positive integer")
            
            if "learning_rate" in config_data:
                lr = config_data["learning_rate"]
                if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                    errors.append("learning_rate must be between 0 and 1")
            
            if "ft_type" in config_data:
                valid_types = ["instruction", "supervised", "unsupervised", "sft"]
                if config_data["ft_type"] not in valid_types:
                    errors.append(f"ft_type must be one of: {valid_types}")
            
            if "peft_method" in config_data:
                valid_methods = ["lora", "qlora", "dora", "oft", "pissa", "none"]
                if config_data["peft_method"] not in valid_methods:
                    errors.append(f"peft_method must be one of: {valid_methods}")
            
            # Performance warnings
            if "batch_size" in config_data and config_data["batch_size"] > 32:
                warnings.append("Large batch size may cause memory issues")
            
            if "max_seq_length" in config_data and config_data["max_seq_length"] > 2048:
                warnings.append("Large sequence length may cause memory issues")
            
            is_valid = len(errors) == 0
            
            return {
                "status": "success",
                "config_name": config_name,
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "validation_summary": {
                    "total_checks": len(required_fields) + 4,  # Additional validation checks
                    "passed_checks": len(required_fields) + 4 - len(errors),
                    "failed_checks": len(errors),
                    "warning_count": len(warnings)
                }
            }
        except Exception as e:
            logger.error(f"Failed to validate config {config_name}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_available_templates(self) -> Dict[str, Any]:
        """Get available dataset templates"""
        try:
            templates = [
                {
                    "name": "alpaca",
                    "description": "Alpaca-style instruction format",
                    "format": "instruction/input/output",
                    "use_cases": ["instruction tuning", "chat models"]
                },
                {
                    "name": "chatml",
                    "description": "ChatML conversation format",
                    "format": "<|im_start|>user<|im_end|> style",
                    "use_cases": ["chat models", "conversation"]
                },
                {
                    "name": "sharegpt",
                    "description": "ShareGPT conversation format",
                    "format": "conversations array with from/value",
                    "use_cases": ["chat models", "multi-turn conversations"]
                },
                {
                    "name": "vicuna",
                    "description": "Vicuna conversation format",
                    "format": "USER: ... ASSISTANT: ...",
                    "use_cases": ["chat models", "instruction following"]
                },
                {
                    "name": "llama2",
                    "description": "Llama-2 chat format",
                    "format": "[INST] ... [/INST]",
                    "use_cases": ["Llama-2 models", "instruction tuning"]
                },
                {
                    "name": "openai",
                    "description": "OpenAI API format",
                    "format": "prompt/completion pairs",
                    "use_cases": ["completion models", "API-style training"]
                }
            ]
            
            return {
                "status": "success",
                "templates": templates,
                "total_count": len(templates)
            }
        except Exception as e:
            logger.error(f"Failed to get templates: {e}")
            return {
                "status": "error",
                "message": str(e),
                "templates": []
            }
    
    async def get_available_peft_methods(self) -> Dict[str, Any]:
        """Get available PEFT methods"""
        try:
            peft_methods = [
                {
                    "name": "lora",
                    "full_name": "Low-Rank Adaptation",
                    "description": "Efficient adaptation using low-rank matrices",
                    "memory_efficient": True,
                    "training_speed": "fast",
                    "recommended_for": ["general fine-tuning", "resource-constrained environments"]
                },
                {
                    "name": "qlora",
                    "full_name": "Quantized Low-Rank Adaptation", 
                    "description": "LoRA with 4-bit quantization for memory efficiency",
                    "memory_efficient": True,
                    "training_speed": "fast",
                    "recommended_for": ["large models", "limited GPU memory"]
                },
                {
                    "name": "dora",
                    "full_name": "Weight-Decomposed Low-Rank Adaptation",
                    "description": "Enhanced LoRA with magnitude and direction decomposition",
                    "memory_efficient": True,
                    "training_speed": "medium",
                    "recommended_for": ["improved performance over LoRA"]
                },
                {
                    "name": "oft",
                    "full_name": "Orthogonal Fine-Tuning",
                    "description": "Fine-tuning with orthogonal transformations",
                    "memory_efficient": True,
                    "training_speed": "medium",
                    "recommended_for": ["preserving pre-trained knowledge"]
                },
                {
                    "name": "pissa",
                    "full_name": "Principal Singular Values and Singular Vectors Adaptation",
                    "description": "SVD-based parameter-efficient fine-tuning",
                    "memory_efficient": True,
                    "training_speed": "medium",
                    "recommended_for": ["mathematical modeling tasks"]
                },
                {
                    "name": "none",
                    "full_name": "Full Fine-Tuning",
                    "description": "Traditional full parameter fine-tuning",
                    "memory_efficient": False,
                    "training_speed": "slow",
                    "recommended_for": ["maximum performance", "abundant resources"]
                }
            ]
            
            return {
                "status": "success",
                "peft_methods": peft_methods,
                "total_count": len(peft_methods),
                "recommended": "qlora"  # Default recommendation
            }
        except Exception as e:
            logger.error(f"Failed to get PEFT methods: {e}")
            return {
                "status": "error",
                "message": str(e),
                "peft_methods": []
            }