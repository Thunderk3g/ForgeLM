"""
Data Generation Controller

Handles API endpoints for automated data generation functionality.
"""

import asyncio
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.data_generation_service import DataGenerationService, GenerationConfig, DataGenerationError
from utils.logging import get_logger


class DataGenerationRequest(BaseModel):
    """Request model for data generation"""
    task_type: str = Field(..., description="Type of task: instruction_tuning, conversation, domain_specific")
    template_format: str = Field(..., description="Template format: alpaca, chatml, sharegpt, qwen")
    num_samples: int = Field(..., ge=1, le=10000, description="Number of samples to generate")
    model_pair_id: Optional[str] = Field(None, description="Specific model pair to use")
    domain: str = Field("general", description="Domain for generation: general, code, math, science")
    difficulty_level: str = Field("mixed", description="Difficulty level: easy, medium, hard, mixed")
    generation_strategy: str = Field("self_supervised", description="Strategy: self_supervised, iterative_refinement, multi_model_consensus")
    quality_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Quality threshold for validation")
    max_tokens: int = Field(512, ge=50, le=2048, description="Maximum tokens per response")
    temperature: float = Field(0.8, ge=0.1, le=1.5, description="Generation temperature")
    diversity_factor: float = Field(0.5, ge=0.0, le=1.0, description="Diversity factor for generation")
    seed_prompts: Optional[List[str]] = Field(None, description="Custom seed prompts for generation")


class DataGenerationController:
    """Controller for data generation endpoints"""
    
    def __init__(self, data_generation_service: Optional[DataGenerationService] = None):
        self.logger = get_logger(__name__)
        self.data_generation_service = data_generation_service
        
        if not self.data_generation_service:
            self.logger.warning("DataGenerationService not provided - data generation endpoints will be limited")
        
        self.logger.info("DataGenerationController initialized")
    
    async def start_generation(self, request: DataGenerationRequest) -> JSONResponse:
        """Start a new data generation task"""
        try:
            if not self.data_generation_service:
                raise HTTPException(
                    status_code=503, 
                    detail="Data generation service not available"
                )
            
            # Convert request to GenerationConfig
            config = GenerationConfig(
                task_type=request.task_type,
                template_format=request.template_format,
                num_samples=request.num_samples,
                model_pair_id=request.model_pair_id,
                domain=request.domain,
                difficulty_level=request.difficulty_level,
                generation_strategy=request.generation_strategy,
                quality_threshold=request.quality_threshold,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                diversity_factor=request.diversity_factor,
                seed_prompts=request.seed_prompts
            )
            
            # Start generation task
            task_id = await self.data_generation_service.start_generation(config)
            
            self.logger.info(f"Started data generation task {task_id}")
            
            return JSONResponse(
                status_code=202,  # Accepted
                content={
                    "status": "accepted",
                    "task_id": task_id,
                    "message": "Data generation task started",
                    "config": {
                        "task_type": request.task_type,
                        "template_format": request.template_format,
                        "num_samples": request.num_samples,
                        "domain": request.domain,
                        "generation_strategy": request.generation_strategy
                    }
                }
            )
            
        except DataGenerationError as e:
            self.logger.error(f"Data generation error: {e.message}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": e.error_type,
                    "message": e.message,
                    "details": e.details
                }
            )
        except Exception as e:
            self.logger.error(f"Unexpected error starting data generation: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start data generation: {str(e)}"
            )
    
    async def get_generation_status(self, task_id: str) -> JSONResponse:
        """Get status of a data generation task"""
        try:
            if not self.data_generation_service:
                raise HTTPException(
                    status_code=503,
                    detail="Data generation service not available"
                )
            
            status = await self.data_generation_service.get_generation_status(task_id)
            
            return JSONResponse(content={
                "task_id": task_id,
                "status": status["status"],
                "progress": status.get("progress", 0),
                "samples_generated": status.get("samples_generated", 0),
                "samples_validated": status.get("samples_validated", 0),
                "started_at": status.get("started_at"),
                "completed_at": status.get("completed_at"),
                "dataset_path": status.get("dataset_path"),
                "config": {
                    "task_type": status["config"].task_type if hasattr(status.get("config"), "task_type") else "unknown",
                    "template_format": status["config"].template_format if hasattr(status.get("config"), "template_format") else "unknown",
                    "num_samples": status["config"].num_samples if hasattr(status.get("config"), "num_samples") else 0
                },
                "errors": status.get("errors", []),
                "error": status.get("error")
            })
            
        except DataGenerationError as e:
            self.logger.error(f"Data generation error: {e.message}")
            raise HTTPException(
                status_code=404 if e.error_type == "task_not_found" else 400,
                detail={
                    "error": e.error_type,
                    "message": e.message,
                    "details": e.details
                }
            )
        except Exception as e:
            self.logger.error(f"Error getting generation status: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get generation status: {str(e)}"
            )
    
    async def cancel_generation(self, task_id: str) -> JSONResponse:
        """Cancel an active data generation task"""
        try:
            if not self.data_generation_service:
                raise HTTPException(
                    status_code=503,
                    detail="Data generation service not available"
                )
            
            success = await self.data_generation_service.cancel_generation(task_id)
            
            if success:
                return JSONResponse(content={
                    "status": "cancelled",
                    "task_id": task_id,
                    "message": "Data generation task cancelled successfully"
                })
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Generation task {task_id} not found or cannot be cancelled"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error cancelling generation: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to cancel generation: {str(e)}"
            )
    
    async def list_generated_datasets(self) -> JSONResponse:
        """List all generated datasets"""
        try:
            if not self.data_generation_service:
                raise HTTPException(
                    status_code=503,
                    detail="Data generation service not available"
                )
            
            datasets = await self.data_generation_service.list_generated_datasets()
            
            return JSONResponse(content={
                "datasets": datasets,
                "total_count": len(datasets)
            })
            
        except Exception as e:
            self.logger.error(f"Error listing generated datasets: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list generated datasets: {str(e)}"
            )
    
    async def preview_dataset(self, dataset_filename: str, limit: int = 5) -> JSONResponse:
        """Preview samples from a generated dataset"""
        try:
            if not self.data_generation_service:
                raise HTTPException(
                    status_code=503,
                    detail="Data generation service not available"
                )
            
            dataset_path = self.data_generation_service.output_dir / dataset_filename
            
            if not dataset_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset {dataset_filename} not found"
                )
            
            # Load dataset and return preview
            import json
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            samples = dataset.get("samples", [])
            preview_samples = samples[:min(limit, len(samples))]
            
            return JSONResponse(content={
                "dataset_filename": dataset_filename,
                "metadata": dataset.get("metadata", {}),
                "total_samples": len(samples),
                "preview_samples": preview_samples,
                "preview_count": len(preview_samples)
            })
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error previewing dataset: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to preview dataset: {str(e)}"
            )
    
    async def get_generation_templates(self) -> JSONResponse:
        """Get available generation templates and configurations"""
        try:
            templates = {
                "task_types": [
                    {
                        "id": "instruction_tuning",
                        "name": "Instruction Tuning",
                        "description": "Generate instruction-response pairs for fine-tuning",
                        "example": "Explain the concept of machine learning"
                    },
                    {
                        "id": "conversation",
                        "name": "Conversation",
                        "description": "Generate conversational exchanges",
                        "example": "Hi, I need help with understanding Python functions"
                    },
                    {
                        "id": "domain_specific",
                        "name": "Domain Specific",
                        "description": "Generate domain-specific training data",
                        "example": "Write a function to sort an array"
                    }
                ],
                "template_formats": [
                    {
                        "id": "alpaca",
                        "name": "Alpaca",
                        "description": "Stanford Alpaca format with instruction/input/output"
                    },
                    {
                        "id": "chatml",
                        "name": "ChatML",
                        "description": "OpenAI ChatML format with role-based messages"
                    },
                    {
                        "id": "sharegpt",
                        "name": "ShareGPT",
                        "description": "ShareGPT conversation format"
                    },
                    {
                        "id": "qwen",
                        "name": "Qwen",
                        "description": "Qwen chat format with system/user/assistant roles"
                    }
                ],
                "domains": [
                    {
                        "id": "general",
                        "name": "General Knowledge",
                        "description": "General purpose knowledge and reasoning"
                    },
                    {
                        "id": "code",
                        "name": "Programming",
                        "description": "Code generation and programming assistance"
                    },
                    {
                        "id": "math",
                        "name": "Mathematics",
                        "description": "Mathematical problems and explanations"
                    },
                    {
                        "id": "science",
                        "name": "Science",
                        "description": "Scientific concepts and explanations"
                    }
                ],
                "generation_strategies": [
                    {
                        "id": "self_supervised",
                        "name": "Self-Supervised",
                        "description": "Generate using available models with seed prompts"
                    },
                    {
                        "id": "iterative_refinement",
                        "name": "Iterative Refinement",
                        "description": "Generate and iteratively improve responses"
                    },
                    {
                        "id": "multi_model_consensus",
                        "name": "Multi-Model Consensus",
                        "description": "Use multiple models and select best responses"
                    }
                ]
            }
            
            return JSONResponse(content=templates)
            
        except Exception as e:
            self.logger.error(f"Error getting generation templates: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get generation templates: {str(e)}"
            )
    
    async def validate_generation_config(self, request: DataGenerationRequest) -> JSONResponse:
        """Validate generation configuration without starting generation"""
        try:
            # Create config to validate
            config = GenerationConfig(
                task_type=request.task_type,
                template_format=request.template_format,
                num_samples=request.num_samples,
                model_pair_id=request.model_pair_id,
                domain=request.domain,
                difficulty_level=request.difficulty_level,
                generation_strategy=request.generation_strategy,
                quality_threshold=request.quality_threshold,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                diversity_factor=request.diversity_factor,
                seed_prompts=request.seed_prompts
            )
            
            # Validate the configuration
            if self.data_generation_service:
                self.data_generation_service._validate_config(config)
            
            # Estimate generation time and resources
            estimated_time_minutes = max(1, request.num_samples // 20)  # Rough estimate
            estimated_dataset_size_mb = (request.num_samples * request.max_tokens * 4) / (1024 * 1024)  # Rough estimate
            
            return JSONResponse(content={
                "valid": True,
                "config": {
                    "task_type": request.task_type,
                    "template_format": request.template_format,
                    "num_samples": request.num_samples,
                    "domain": request.domain,
                    "generation_strategy": request.generation_strategy,
                    "quality_threshold": request.quality_threshold
                },
                "estimates": {
                    "estimated_time_minutes": estimated_time_minutes,
                    "estimated_dataset_size_mb": round(estimated_dataset_size_mb, 2),
                    "expected_valid_samples": int(request.num_samples * request.quality_threshold)
                },
                "seed_prompts_count": len(config.seed_prompts),
                "message": "Configuration is valid and ready for generation"
            })
            
        except DataGenerationError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "valid": False,
                    "error": e.error_type,
                    "message": e.message,
                    "details": e.details
                }
            )
        except Exception as e:
            self.logger.error(f"Error validating generation config: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to validate generation config: {str(e)}"
            )
    
    async def get_active_generations(self) -> JSONResponse:
        """Get list of currently active generation tasks"""
        try:
            if not self.data_generation_service:
                return JSONResponse(content={"active_generations": [], "total_count": 0})
            
            active_tasks = []
            for task_id, status in self.data_generation_service.active_generations.items():
                if status["status"] in ["initializing", "running"]:
                    active_tasks.append({
                        "task_id": task_id,
                        "status": status["status"],
                        "progress": status.get("progress", 0),
                        "samples_generated": status.get("samples_generated", 0),
                        "started_at": status.get("started_at"),
                        "task_type": status["config"].task_type if hasattr(status.get("config"), "task_type") else "unknown",
                        "num_samples": status["config"].num_samples if hasattr(status.get("config"), "num_samples") else 0
                    })
            
            return JSONResponse(content={
                "active_generations": active_tasks,
                "total_count": len(active_tasks)
            })
            
        except Exception as e:
            self.logger.error(f"Error getting active generations: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get active generations: {str(e)}"
            )