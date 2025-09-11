"""
Configuration Controller for Chat Configuration Panel
Handles all configuration-related API endpoints
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import HTTPException, status
from models.schemas import (
    DualModelConfig, ModelConfigRequest, ConfigPreset, ConfigPresetRequest,
    ConfigValidationResponse, ConfigPresetListResponse, ModelCapabilities,
    GenerationConfig
)
from services.comparison_service import ComparisonService
from services.model_service import ModelService
import logging

logger = logging.getLogger(__name__)


class ConfigurationController:
    """Controller for managing chat configuration settings"""
    
    def __init__(self):
        self.comparison_service = ComparisonService()
        self.model_service = ModelService()
        self._session_configs: Dict[str, DualModelConfig] = {}
        self._config_presets: Dict[str, ConfigPreset] = {}
        self._initialize_default_presets()
    
    def _initialize_default_presets(self):
        """Initialize built-in configuration presets"""
        default_presets = [
            {
                "preset_id": "creative",
                "name": "Creative Writing",
                "description": "High temperature for creative, diverse responses",
                "category": "creative",
                "base_config": GenerationConfig(
                    temperature=1.2,
                    top_p=0.9,
                    top_k=50,
                    max_tokens=512,
                    repetition_penalty=1.1
                ),
                "finetuned_config": GenerationConfig(
                    temperature=1.2,
                    top_p=0.9,
                    top_k=50,
                    max_tokens=512,
                    repetition_penalty=1.1
                ),
                "system_prompts": {
                    "base": "You are a creative assistant focused on generating imaginative and engaging content.",
                    "finetuned": "You are a creative assistant focused on generating imaginative and engaging content."
                },
                "is_built_in": True
            },
            {
                "preset_id": "factual",
                "name": "Factual & Precise",
                "description": "Low temperature for factual, consistent responses",
                "category": "factual",
                "base_config": GenerationConfig(
                    temperature=0.3,
                    top_p=0.8,
                    top_k=20,
                    max_tokens=1024,
                    repetition_penalty=1.0
                ),
                "finetuned_config": GenerationConfig(
                    temperature=0.3,
                    top_p=0.8,
                    top_k=20,
                    max_tokens=1024,
                    repetition_penalty=1.0
                ),
                "system_prompts": {
                    "base": "You are a factual assistant focused on providing accurate and precise information.",
                    "finetuned": "You are a factual assistant focused on providing accurate and precise information."
                },
                "is_built_in": True
            },
            {
                "preset_id": "code",
                "name": "Code Generation",
                "description": "Optimized for code generation and technical explanations",
                "category": "code",
                "base_config": GenerationConfig(
                    temperature=0.2,
                    top_p=0.95,
                    top_k=40,
                    max_tokens=1024,
                    repetition_penalty=1.05
                ),
                "finetuned_config": GenerationConfig(
                    temperature=0.2,
                    top_p=0.95,
                    top_k=40,
                    max_tokens=1024,
                    repetition_penalty=1.05
                ),
                "system_prompts": {
                    "base": "You are a programming assistant focused on generating clean, efficient code with clear explanations.",
                    "finetuned": "You are a programming assistant focused on generating clean, efficient code with clear explanations."
                },
                "is_built_in": True
            },
            {
                "preset_id": "balanced",
                "name": "Balanced",
                "description": "Well-balanced parameters for general-purpose use",
                "category": "general",
                "base_config": GenerationConfig(),  # Uses defaults
                "finetuned_config": GenerationConfig(),  # Uses defaults
                "system_prompts": {
                    "base": "You are a helpful assistant.",
                    "finetuned": "You are a helpful assistant."
                },
                "is_built_in": True
            }
        ]
        
        for preset_data in default_presets:
            preset = ConfigPreset(**preset_data)
            self._config_presets[preset.preset_id] = preset
        
        logger.info(f"Initialized {len(default_presets)} default configuration presets")
    
    async def get_session_config(self, session_id: str) -> DualModelConfig:
        """Get configuration for a specific session"""
        if session_id not in self._session_configs:
            # Create default configuration for new session
            config = DualModelConfig(session_id=session_id)
            self._session_configs[session_id] = config
            logger.info(f"Created default configuration for session {session_id}")
        
        return self._session_configs[session_id]
    
    async def update_session_config(self, request: ModelConfigRequest) -> DualModelConfig:
        """Update configuration for a specific session"""
        try:
            # Validate the configuration
            validation = await self.validate_config(request.config)
            if not validation.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid configuration: {'; '.join(validation.errors)}"
                )
            
            # Update timestamp
            request.config.updated_at = datetime.now().isoformat()
            
            # Store the configuration
            self._session_configs[request.session_id] = request.config
            
            logger.info(f"Updated configuration for session {request.session_id}")
            return request.config
            
        except Exception as e:
            logger.error(f"Error updating session config: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update configuration: {str(e)}"
            )
    
    async def validate_config(self, config: DualModelConfig) -> ConfigValidationResponse:
        """Validate a configuration and return warnings/errors"""
        warnings = []
        errors = []
        estimated_performance = {}
        memory_requirements = {}
        
        try:
            # Validate base model config
            base_config = config.base_model.generation_config
            finetuned_config = config.finetuned_model.generation_config
            
            # Check for potential performance issues
            if base_config.max_tokens > 2048:
                warnings.append("Base model max_tokens > 2048 may impact performance")
            if finetuned_config.max_tokens > 2048:
                warnings.append("Fine-tuned model max_tokens > 2048 may impact performance")
            
            if base_config.temperature > 1.5:
                warnings.append("Base model high temperature may produce unpredictable results")
            if finetuned_config.temperature > 1.5:
                warnings.append("Fine-tuned model high temperature may produce unpredictable results")
            
            # Estimate memory requirements
            base_memory = self._estimate_memory_usage(base_config)
            finetuned_memory = self._estimate_memory_usage(finetuned_config)
            
            memory_requirements = {
                "base_model_mb": base_memory,
                "finetuned_model_mb": finetuned_memory,
                "total_mb": base_memory + finetuned_memory
            }
            
            # Estimate performance impact
            estimated_performance = {
                "base_model_speed_factor": self._estimate_speed_factor(base_config),
                "finetuned_model_speed_factor": self._estimate_speed_factor(finetuned_config)
            }
            
            return ConfigValidationResponse(
                is_valid=len(errors) == 0,
                warnings=warnings,
                errors=errors,
                estimated_performance=estimated_performance,
                memory_requirements=memory_requirements
            )
            
        except Exception as e:
            logger.error(f"Error validating configuration: {str(e)}")
            return ConfigValidationResponse(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"]
            )
    
    def _estimate_memory_usage(self, config: GenerationConfig) -> int:
        """Estimate memory usage in MB for a configuration"""
        base_memory = 1000  # Base model memory
        
        # Context length impact
        context_factor = config.max_tokens / 1024  # Normalize to 1024 tokens
        memory_usage = int(base_memory * (1 + context_factor * 0.2))
        
        return memory_usage
    
    def _estimate_speed_factor(self, config: GenerationConfig) -> float:
        """Estimate relative speed factor (1.0 = baseline)"""
        speed_factor = 1.0
        
        # Max tokens impact
        if config.max_tokens > 512:
            speed_factor *= 0.8
        
        # Sampling complexity impact
        if config.do_sample:
            if config.top_k and config.top_k < 20:
                speed_factor *= 1.1  # Faster with limited top-k
            if config.temperature < 0.3:
                speed_factor *= 1.05  # Slightly faster with low temperature
        else:
            speed_factor *= 1.2  # Greedy is faster
        
        return speed_factor
    
    async def get_model_capabilities(self, model_id: str) -> ModelCapabilities:
        """Get capabilities and constraints for a specific model"""
        try:
            # This would typically query the model service for actual capabilities
            # For now, return reasonable defaults
            default_config = GenerationConfig()
            
            capabilities = ModelCapabilities(
                model_id=model_id,
                model_type="unknown",  # Would be determined by model service
                max_context_length=4096,
                supports_system_prompt=True,
                default_parameters=default_config,
                parameter_constraints={
                    "temperature": {"min": 0.1, "max": 2.0, "step": 0.1},
                    "top_p": {"min": 0.1, "max": 1.0, "step": 0.05},
                    "top_k": {"min": 1, "max": 100, "step": 1},
                    "max_tokens": {"min": 50, "max": 4096, "step": 1},
                    "presence_penalty": {"min": -2.0, "max": 2.0, "step": 0.1},
                    "frequency_penalty": {"min": -2.0, "max": 2.0, "step": 0.1},
                    "repetition_penalty": {"min": 0.5, "max": 2.0, "step": 0.05}
                },
                recommended_presets=["balanced", "factual"],
                memory_usage_mb=2048,
                estimated_tokens_per_second=25.0
            )
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Error getting model capabilities: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get model capabilities: {str(e)}"
            )
    
    async def save_config_preset(self, request: ConfigPresetRequest) -> ConfigPreset:
        """Save a new configuration preset"""
        try:
            preset_id = str(uuid.uuid4())
            preset = ConfigPreset(
                preset_id=preset_id,
                name=request.name,
                description=request.description,
                category=request.category,
                base_config=request.base_config,
                finetuned_config=request.finetuned_config,
                system_prompts=request.system_prompts or {},
                is_built_in=False,
                created_by="user"  # Would be actual user ID in production
            )
            
            self._config_presets[preset_id] = preset
            logger.info(f"Saved new preset: {preset.name} ({preset_id})")
            
            return preset
            
        except Exception as e:
            logger.error(f"Error saving preset: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save preset: {str(e)}"
            )
    
    async def list_config_presets(self, category: Optional[str] = None) -> ConfigPresetListResponse:
        """List available configuration presets"""
        try:
            presets = list(self._config_presets.values())
            
            if category:
                presets = [p for p in presets if p.category == category]
            
            # Get unique categories
            categories = list(set(p.category for p in self._config_presets.values()))
            categories.sort()
            
            return ConfigPresetListResponse(
                presets=presets,
                categories=categories,
                total_count=len(presets)
            )
            
        except Exception as e:
            logger.error(f"Error listing presets: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list presets: {str(e)}"
            )
    
    async def get_config_preset(self, preset_id: str) -> ConfigPreset:
        """Get a specific configuration preset"""
        if preset_id not in self._config_presets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset {preset_id} not found"
            )
        
        preset = self._config_presets[preset_id]
        preset.usage_count += 1  # Track usage
        return preset
    
    async def delete_config_preset(self, preset_id: str) -> dict:
        """Delete a configuration preset"""
        if preset_id not in self._config_presets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset {preset_id} not found"
            )
        
        preset = self._config_presets[preset_id]
        if preset.is_built_in:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete built-in presets"
            )
        
        del self._config_presets[preset_id]
        logger.info(f"Deleted preset: {preset.name} ({preset_id})")
        
        return {"message": f"Preset {preset.name} deleted successfully"}
    
    async def apply_preset_to_session(self, session_id: str, preset_id: str) -> DualModelConfig:
        """Apply a preset to a session's configuration"""
        try:
            preset = await self.get_config_preset(preset_id)
            
            # Get or create session config
            session_config = await self.get_session_config(session_id)
            
            # Apply preset settings
            session_config.base_model.generation_config = preset.base_config
            session_config.finetuned_model.generation_config = preset.finetuned_config
            
            # Apply system prompts if provided
            if "base" in preset.system_prompts:
                session_config.base_model.system_prompt = preset.system_prompts["base"]
            if "finetuned" in preset.system_prompts:
                session_config.finetuned_model.system_prompt = preset.system_prompts["finetuned"]
            
            session_config.config_name = f"Preset: {preset.name}"
            session_config.updated_at = datetime.now().isoformat()
            
            # Save updated configuration
            self._session_configs[session_id] = session_config
            
            logger.info(f"Applied preset {preset.name} to session {session_id}")
            return session_config
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error applying preset: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to apply preset: {str(e)}"
            )