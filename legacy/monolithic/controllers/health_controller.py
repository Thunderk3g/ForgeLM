"""
Health Controller

Handles health check and system status endpoints.
"""

from typing import Dict, Any
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from utils.logging import get_logger
from config.settings import settings


class HealthController:
    """Controller for health and system status endpoints"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "LLMForge Fine-tuning",
            "version": "1.0.0"
        }
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information and configuration"""
        try:
            return {
                "service": "LLMForge Fine-tuning API",
                "version": "1.0.0",
                "configuration": {
                    "max_hbm_gb": settings.max_hbm_gb,
                    "max_sessions": settings.max_sessions,
                    "unsloth_optimization": settings.unsloth_optimization,
                    "cache_optimization": settings.cache_optimization
                },
                "features": {
                    "fine_tuning": True,
                    "comparison_chat": True,
                    "model_management": True,
                    "data_preparation": True
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")


# Create global controller instance
health_controller = HealthController()