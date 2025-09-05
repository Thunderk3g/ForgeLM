"""
Configuration Controller

Provides API endpoints for YAML configuration management
"""

from typing import List, Dict, Any
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from config.yaml_config import yaml_config_manager, YAMLConfig
from models.schemas import FineTuneRequest
from utils.logging import get_logger


class ConfigController:
    """Controller for configuration management endpoints"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config_manager = yaml_config_manager
        
        # Create default configs on initialization
        try:
            self.config_manager.create_default_configs()
        except Exception as e:
            self.logger.warning(f"Failed to create default configs: {e}")
    
    async def list_configs(self) -> JSONResponse:
        """List all available configuration files"""
        try:
            configs = self.config_manager.list_configs()
            
            return JSONResponse(content={
                "status": "success",
                "configs": configs,
                "total": len(configs)
            })
            
        except Exception as e:
            self.logger.error(f"Failed to list configurations: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list configurations: {str(e)}")
    
    async def get_config(self, config_name: str) -> JSONResponse:
        """Get specific configuration"""
        try:
            config = self.config_manager.load_config(config_name)
            
            # Convert to dict for JSON response
            config_dict = {
                "config_name": config.config_name,
                "description": config.description,
                "version": config.version,
                "created_at": config.created_at,
                "author": config.author,
                "model": {
                    "model_name": config.model.model_name,
                    "model_path": config.model.model_path,
                    "quantization": config.model.quantization,
                    "trust_remote_code": config.model.trust_remote_code,
                    "use_fast_tokenizer": config.model.use_fast_tokenizer,
                    "model_revision": config.model.model_revision
                },
                "dataset": {
                    "path": config.dataset.path,
                    "format": config.dataset.format,
                    "template": config.dataset.template,
                    "system_message": config.dataset.system_message,
                    "max_samples": config.dataset.max_samples,
                    "test_size": config.dataset.test_size
                },
                "training": {
                    "output_dir": config.training.output_dir,
                    "num_train_epochs": config.training.num_train_epochs,
                    "per_device_train_batch_size": config.training.per_device_train_batch_size,
                    "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
                    "learning_rate": config.training.learning_rate,
                    "max_seq_length": config.training.max_seq_length,
                    "lr_scheduler_type": config.training.lr_scheduler_type,
                    "warmup_ratio": config.training.warmup_ratio,
                    "logging_steps": config.training.logging_steps,
                    "save_steps": config.training.save_steps,
                    "evaluation_strategy": config.training.evaluation_strategy
                },
                "peft": {
                    "method": config.peft.method.value,
                    "rank": config.peft.rank,
                    "alpha": config.peft.alpha,
                    "dropout": config.peft.dropout,
                    "target_modules": config.peft.target_modules,
                    "bias": config.peft.bias
                }
            }
            
            return JSONResponse(content={
                "status": "success",
                "config": config_dict
            })
            
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Configuration not found: {config_name}")
        except Exception as e:
            self.logger.error(f"Failed to get configuration: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")
    
    async def create_config_from_request(self, request: FineTuneRequest, config_name: str) -> JSONResponse:
        """Create configuration from training request"""
        try:
            # Create config from request
            config = self.config_manager.create_config_from_request(request, config_name)
            
            # Validate config
            validation = self.config_manager.validate_config(config)
            
            if not validation["valid"]:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "validation_failed",
                        "errors": validation["errors"],
                        "warnings": validation["warnings"]
                    }
                )
            
            # Save config
            config_path = self.config_manager.save_config(config)
            
            return JSONResponse(content={
                "status": "success",
                "message": f"Configuration created successfully",
                "config_name": config_name,
                "config_path": config_path,
                "warnings": validation.get("warnings", [])
            })
            
        except Exception as e:
            self.logger.error(f"Failed to create configuration: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create configuration: {str(e)}")
    
    async def validate_config(self, config_name: str) -> JSONResponse:
        """Validate configuration"""
        try:
            config = self.config_manager.load_config(config_name)
            validation = self.config_manager.validate_config(config)
            
            return JSONResponse(content={
                "status": "success",
                "validation": validation
            })
            
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Configuration not found: {config_name}")
        except Exception as e:
            self.logger.error(f"Failed to validate configuration: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to validate configuration: {str(e)}")
    
    async def get_available_templates(self) -> JSONResponse:
        """Get available dataset templates"""
        try:
            from models.templates import list_available_templates
            templates = list_available_templates()
            
            return JSONResponse(content={
                "status": "success",
                "templates": templates
            })
            
        except Exception as e:
            self.logger.error(f"Failed to get templates: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")
    
    async def get_available_peft_methods(self) -> JSONResponse:
        """Get available PEFT methods"""
        try:
            from models.peft_methods import PEFTMethod
            methods = [method.value for method in PEFTMethod]
            
            return JSONResponse(content={
                "status": "success",
                "peft_methods": methods
            })
            
        except Exception as e:
            self.logger.error(f"Failed to get PEFT methods: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get PEFT methods: {str(e)}")


# Global controller instance  
config_controller = ConfigController()