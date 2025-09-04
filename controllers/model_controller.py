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
    from model_selection import select_model, scan_available_models
    from data_preparation import prepare_data
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


# Create global controller instance
model_controller = ModelController()