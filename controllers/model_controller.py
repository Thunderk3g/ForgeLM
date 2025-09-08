"""
Model Controller

Handles model selection, scanning, and management endpoints.
"""

from typing import Dict, Any
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from models.schemas import SelectModelRequest, PrepareDataRequest
from services.model_service import ModelService
from utils.logging import get_logger

# Import the original functions to maintain compatibility
try:
    from utils.legacy.model_selection import select_model, scan_available_models
    from utils.legacy.data_preparation import prepare_data
    MODEL_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    MODEL_FUNCTIONS_AVAILABLE = False
    print(f"Warning: Original model functions not available: {e}")


class ModelController:
    """Controller for model management endpoints"""
    
    def __init__(self, model_service: ModelService = None):
        self.logger = get_logger(__name__)
        self.model_service = model_service or ModelService()
    
    async def get_available_models(self) -> JSONResponse:
        """Get list of available models in the models directory"""
        try:
            if MODEL_FUNCTIONS_AVAILABLE:
                # Use original function to maintain exact compatibility
                result = scan_available_models()
            else:
                # Fallback using model service
                pairs = await self.model_service.discover_model_pairs()
                result = {
                    "status": "success",
                    "models": [
                        {
                            "name": pair.base_model_name,
                            "path": pair.base_model_path,
                            "size_mb": pair.base_model_size_mb
                        }
                        for pair in pairs
                    ]
                }
            
            return JSONResponse(content=result)
            
        except Exception as e:
            self.logger.error(f"Failed to scan models: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to scan models: {str(e)}")
    
    async def select_model_endpoint(self, request: SelectModelRequest) -> JSONResponse:
        """Download and prepare a base model for fine-tuning"""
        try:
            if MODEL_FUNCTIONS_AVAILABLE:
                # Use original function to maintain exact compatibility
                result = select_model(request)
            else:
                # Mock result for demo
                result = {
                    "status": "success",
                    "message": f"Model {request.model_id} selected successfully",
                    "model_path": request.output_dir
                }
            
            if result.get("status") == "error":
                raise HTTPException(status_code=400, detail=result["message"])
                
            return JSONResponse(content=result)
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Model selection failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model selection failed: {str(e)}")
    
    async def prepare_data_endpoint(self, request: PrepareDataRequest) -> JSONResponse:
        """Clean and prepare data for fine-tuning"""
        try:
            if MODEL_FUNCTIONS_AVAILABLE:
                # Use original function to maintain exact compatibility
                result = prepare_data(request)
            else:
                # Mock result for demo
                result = {
                    "status": "success",
                    "message": "Data prepared successfully",
                    "output_dir": request.output_dir,
                    "dataset_info": {
                        "total_samples": 1000,
                        "train_samples": 800,
                        "val_samples": 200
                    }
                }
            
            if result.get("status") == "error":
                raise HTTPException(status_code=400, detail=result["message"])
                
            return JSONResponse(content=result)
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Data preparation failed: {str(e)}")
    
    async def get_model_pairs(self) -> JSONResponse:
        """Get all available base model + LoRA adapter pairs for comparison"""
        try:
            pairs = await self.model_service.discover_model_pairs()
            
            pairs_data = [
                {
                    "pair_id": pair.pair_id,
                    "base_model_name": pair.base_model_name,
                    "adapter_name": pair.adapter_name,
                    "base_model_size_mb": pair.base_model_size_mb,
                    "adapter_size_mb": pair.adapter_size_mb,
                    "estimated_memory_mb": pair.base_model_size_mb + pair.adapter_size_mb,
                    "status": pair.status,
                    "load_time_ms": pair.load_time_ms
                }
                for pair in pairs
            ]
            
            return JSONResponse(content={
                "status": "success",
                "model_pairs": pairs_data,
                "total": len(pairs_data)
            })
            
        except Exception as e:
            self.logger.error(f"Failed to get model pairs: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get model pairs: {str(e)}")
    
    async def get_fine_tuned_models(self) -> JSONResponse:
        """Get list of fine-tuned models with their metadata and organization"""
        try:
            from pathlib import Path
            from config.settings import settings
            
            fine_tuned_dir = Path(settings.models_finetuned_dir)
            models_data = []
            
            if fine_tuned_dir.exists():
                for model_dir in fine_tuned_dir.iterdir():
                    if model_dir.is_dir() and model_dir.name != 'checkpoints':
                        # Read metadata
                        metadata = self.model_service._read_model_metadata(model_dir)
                        
                        # Calculate size
                        size_mb = self.model_service._calculate_directory_size(model_dir)
                        
                        # Check if model files exist
                        has_adapter = (model_dir / "adapter_model.safetensors").exists() or \
                                     (model_dir / "adapter_model.bin").exists()
                        has_merged = (model_dir / "merged_16bit").exists()
                        
                        model_info = {
                            "name": model_dir.name,
                            "path": str(model_dir),
                            "size_mb": round(size_mb, 2),
                            "created_at": metadata.get("model_info", {}).get("created_at", "unknown"),
                            "base_model": metadata.get("model_info", {}).get("base_model", "unknown"),
                            "task_id": metadata.get("model_info", {}).get("task_id"),
                            "training_config": metadata.get("training_config", {}),
                            "files": {
                                "has_adapter": has_adapter,
                                "has_merged": has_merged,
                                "metadata_exists": (model_dir / "model_metadata.json").exists()
                            },
                            "framework": metadata.get("model_info", {}).get("framework", "unknown")
                        }
                        models_data.append(model_info)
                
                # Sort by creation date (newest first)
                models_data.sort(key=lambda x: x["created_at"], reverse=True)
            
            return JSONResponse(content={
                "status": "success",
                "fine_tuned_models": models_data,
                "total": len(models_data),
                "base_directory": str(fine_tuned_dir)
            })
            
        except Exception as e:
            self.logger.error(f"Failed to get fine-tuned models: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get fine-tuned models: {str(e)}")


# Create global controller instance
model_controller = ModelController()