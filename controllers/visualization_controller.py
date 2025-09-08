"""
Visualization Controller

Handles API endpoints for data visualization and analytics.
"""

from typing import Optional
from fastapi import HTTPException, Query
from fastapi.responses import JSONResponse

from services.visualization_service import VisualizationService
from utils.logging import get_logger


class VisualizationController:
    """Controller for visualization and analytics endpoints"""
    
    def __init__(self, visualization_service: VisualizationService = None):
        self.logger = get_logger(__name__)
        self.visualization_service = visualization_service or VisualizationService()
        
        self.logger.info("VisualizationController initialized")
    
    async def get_training_analytics(self, task_id: str, limit: int = 100) -> JSONResponse:
        """Get training analytics for visualization dashboard"""
        try:
            analytics_data = await self.visualization_service.get_training_analytics(task_id, limit)
            
            return JSONResponse(content={
                "status": "success",
                "data": analytics_data
            })
            
        except Exception as e:
            self.logger.error(f"Failed to get training analytics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get training analytics: {str(e)}")
    
    async def get_dataset_analytics(self, dataset_path: str) -> JSONResponse:
        """Get dataset analytics for data exploration dashboard"""
        try:
            if not dataset_path:
                raise HTTPException(status_code=400, detail="Dataset path is required")
            
            analytics_data = await self.visualization_service.get_dataset_analytics(dataset_path)
            
            if "error" in analytics_data:
                return JSONResponse(
                    status_code=404,
                    content={
                        "status": "error",
                        "error": analytics_data["error"]
                    }
                )
            
            return JSONResponse(content={
                "status": "success",
                "data": analytics_data
            })
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get dataset analytics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get dataset analytics: {str(e)}")
    
    async def get_comparison_analytics(self, time_range_hours: int = 24) -> JSONResponse:
        """Get model comparison analytics for comparison dashboard"""
        try:
            if time_range_hours < 1 or time_range_hours > 168:  # Max 1 week
                raise HTTPException(status_code=400, detail="Time range must be between 1 and 168 hours")
            
            analytics_data = await self.visualization_service.get_comparison_analytics(time_range_hours)
            
            return JSONResponse(content={
                "status": "success",
                "data": analytics_data
            })
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get comparison analytics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get comparison analytics: {str(e)}")
    
    async def get_system_health(self) -> JSONResponse:
        """Get current system health metrics"""
        try:
            health_data = await self.visualization_service.get_system_health_metrics()
            
            return JSONResponse(content={
                "status": "success",
                "data": health_data
            })
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")
    
    async def get_dashboard_summary(self, 
                                    include_training: bool = True,
                                    include_comparisons: bool = True,
                                    include_system: bool = True) -> JSONResponse:
        """Get comprehensive dashboard summary with all key metrics"""
        try:
            summary = {
                "timestamp": "2025-09-08T00:00:00.000000",
                "dashboard_sections": {}
            }
            
            # Get system health if requested
            if include_system:
                try:
                    health_data = await self.visualization_service.get_system_health_metrics()
                    summary["dashboard_sections"]["system_health"] = health_data
                except Exception as e:
                    self.logger.warning(f"Failed to include system health in summary: {e}")
                    summary["dashboard_sections"]["system_health"] = {"error": str(e)}
            
            # Get comparison analytics if requested
            if include_comparisons:
                try:
                    comparison_data = await self.visualization_service.get_comparison_analytics(24)
                    summary["dashboard_sections"]["comparisons"] = comparison_data
                except Exception as e:
                    self.logger.warning(f"Failed to include comparison analytics in summary: {e}")
                    summary["dashboard_sections"]["comparisons"] = {"error": str(e)}
            
            # Training data would need specific task IDs, so we'll include active training summary
            if include_training:
                try:
                    # This would integrate with training controller to get active sessions
                    summary["dashboard_sections"]["active_training"] = {
                        "message": "Active training data available via specific task ID endpoints"
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to include training summary: {e}")
                    summary["dashboard_sections"]["active_training"] = {"error": str(e)}
            
            return JSONResponse(content={
                "status": "success",
                "data": summary
            })
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard summary: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get dashboard summary: {str(e)}")


# Global controller instance will be created in app.py with proper dependencies
visualization_controller = None