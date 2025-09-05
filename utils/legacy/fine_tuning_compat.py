"""
Backward Compatibility Wrapper for fine_tuning.py

This file maintains the original API surface while delegating to the new TrainingService.
Existing code that imports from fine_tuning should continue to work unchanged.
"""

# Import all the original classes and functions from their new locations
from models.schemas import FineTuneRequest, TrainingStatus
from services.training_service import (
    TrainingService, task_log_queues, task_completion_flags,
    UNSLOTH_AVAILABLE, TRANSFORMERS_AVAILABLE,
    KUBERNETES_MODE, MAX_MEMORY_USAGE_GB, FORCE_CPU_OFFLOAD, CONTAINER_MEMORY_LIMIT_GB
)

# Create global training service instance
training_service = TrainingService()

# Maintain the original function signatures for backward compatibility
def fine_tune(request, task_id=None):
    """
    Fine-tune a model with Unsloth optimization
    
    This is the original function signature maintained for backward compatibility.
    """
    return training_service.fine_tune(request, task_id)

def get_active_sessions():
    """
    Get information about active training sessions
    
    This is the original function signature maintained for backward compatibility.
    """
    return training_service.get_active_sessions()

# Export all classes and functions for backward compatibility
__all__ = [
    'FineTuneRequest',
    'TrainingStatus', 
    'fine_tune',
    'get_active_sessions',
    'task_log_queues',
    'task_completion_flags',
    'UNSLOTH_AVAILABLE',
    'TRANSFORMERS_AVAILABLE',
    'KUBERNETES_MODE',
    'MAX_MEMORY_USAGE_GB',
    'FORCE_CPU_OFFLOAD', 
    'CONTAINER_MEMORY_LIMIT_GB'
]