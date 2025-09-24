"""
Utility functions for LLMForge
"""

# Import logging first as it's safe
from .logging import get_logger, log_performance

# Import performance utilities
from .performance import performance_monitor, timing_context

# Conditionally import memory utilities
def get_memory_functions():
    """Lazily import memory functions to avoid torch dependency issues"""
    try:
        from .memory import get_gpu_memory_info, cleanup_gpu_memory, estimate_model_memory, get_system_resource_info
        return get_gpu_memory_info, cleanup_gpu_memory, estimate_model_memory, get_system_resource_info
    except ImportError as e:
        # Provide fallback implementations
        def get_gpu_memory_info():
            return {"gpu_available": False, "error": "torch_not_available"}
        def cleanup_gpu_memory():
            pass
        def estimate_model_memory(model_path):
            return 0.0
        def get_system_resource_info():
            return {"error": "torch_not_available"}
        return get_gpu_memory_info, cleanup_gpu_memory, estimate_model_memory, get_system_resource_info

# Initialize memory functions
get_gpu_memory_info, cleanup_gpu_memory, estimate_model_memory, get_system_resource_info = get_memory_functions()

__all__ = [
    'get_gpu_memory_info',
    'cleanup_gpu_memory', 
    'estimate_model_memory',
    'get_system_resource_info',
    'performance_monitor',
    'timing_context',
    'get_logger',
    'log_performance'
]