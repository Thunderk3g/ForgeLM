"""
Visualization Controller

Handles API endpoints for data visualization and analytics.
"""

from typing import Optional
from fastapi import HTTPException, Query
from fastapi.responses import JSONResponse
from datetime import datetime

from services.visualization_service import VisualizationService
from utils.logging import get_logger


class VisualizationController:
    """Controller for visualization and analytics endpoints"""
    
    def __init__(self, visualization_service: VisualizationService = None, training_controller = None):
        self.logger = get_logger(__name__)
        self.visualization_service = visualization_service or VisualizationService()
        self.training_controller = training_controller  # Will be injected from app.py
        
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
            # Initialize counters
            active_trainings = 0
            completed_trainings = 0
            recent_comparisons = 0
            avg_training_loss = 0.0
            avg_response_time = 0.0
            model_performance_score = 0.0
            system_status = "unknown"
            
            # Get training data if requested
            if include_training and self.training_controller:
                try:
                    # Call the training controller method directly to get Python dict
                    from services.training_service import TrainingService
                    training_service = TrainingService()
                    active_sessions_data = training_service.get_active_sessions()
                    
                    active_trainings = len(active_sessions_data)
                    completed_trainings = 3  # Placeholder - would count from logs directory
                    
                    # Calculate average loss from active sessions
                    if active_sessions_data:
                        losses = []
                        for session_id, session_info in active_sessions_data.items():
                            if isinstance(session_info, dict) and 'loss' in session_info:
                                losses.append(session_info['loss'])
                        avg_training_loss = sum(losses) / len(losses) if losses else 0.0
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get training data: {e}")
                    # Use system health data to detect actual model activity
                    try:
                        health_data = await self.visualization_service.get_system_health_metrics()
                        gpu_usage = health_data.get("system_resources", {}).get("gpu", {}).get("allocated_mb", 0)
                        
                        # More intelligent detection - check if GPU usage indicates actual training vs just models loaded
                        if gpu_usage > 6000:  # > 6GB suggests models + training
                            active_trainings = 1
                        elif gpu_usage > 3000:  # 3-6GB might be just loaded models
                            active_trainings = 0  # Models loaded but not training
                        else:  # < 3GB means clean slate
                            active_trainings = 0
                            
                        # Log for debugging
                        self.logger.info(f"GPU usage: {gpu_usage}MB, detected active trainings: {active_trainings}")
                    except:
                        active_trainings = 0
            
            # Get comparison analytics if requested
            if include_comparisons:
                try:
                    comparison_data = await self.visualization_service.get_comparison_analytics(24)
                    comparison_stats = comparison_data.get("comparison_stats", {})
                    recent_comparisons = comparison_stats.get("total_comparisons", 0)
                    avg_response_time = comparison_stats.get("average_finetuned_time_ms", 0)
                    
                    # Calculate performance score from comparison data
                    performance_advantage = comparison_stats.get("performance_advantage", 0)
                    model_performance_score = max(0.0, min(1.0, (performance_advantage + 100) / 200))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get comparison data: {e}")
            
            # Get system health if requested
            if include_system:
                try:
                    health_data = await self.visualization_service.get_system_health_metrics()
                    health_score = health_data.get("health_score", 0.5)
                    
                    if health_score > 0.8:
                        system_status = "healthy"
                    elif health_score > 0.6:
                        system_status = "warning"
                    else:
                        system_status = "critical"
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get system health: {e}")
                    system_status = "unknown"
            
            # Build summary data structure that matches frontend expectations
            summary_data = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "active_trainings": active_trainings,
                    "completed_trainings": completed_trainings,
                    "recent_comparisons": recent_comparisons,
                    "system_status": system_status,
                    "quick_stats": {
                        "avg_training_loss": avg_training_loss,
                        "avg_response_time": avg_response_time,
                        "model_performance_score": model_performance_score
                    }
                },
                "dashboard_sections": {}
            }
            
            # Add detailed sections
            if include_system:
                try:
                    health_data = await self.visualization_service.get_system_health_metrics()
                    summary_data["dashboard_sections"]["system_health"] = health_data
                except Exception as e:
                    summary_data["dashboard_sections"]["system_health"] = {"error": str(e)}
            
            if include_comparisons:
                try:
                    comparison_data = await self.visualization_service.get_comparison_analytics(24)
                    summary_data["dashboard_sections"]["comparisons"] = comparison_data
                except Exception as e:
                    summary_data["dashboard_sections"]["comparisons"] = {"error": str(e)}
            
            if include_training:
                summary_data["dashboard_sections"]["active_training"] = {
                    "active_sessions": active_trainings,
                    "message": f"Found {active_trainings} active training sessions"
                }
            
            return JSONResponse(content={
                "status": "success",
                "data": summary_data
            })
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard summary: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get dashboard summary: {str(e)}")


# Global controller instance will be created in app.py with proper dependencies
visualization_controller = None