"""
Memory management utilities
"""

import torch
import gc
import psutil
from typing import Dict, Any
from pathlib import Path
import os


def get_gpu_memory_info() -> Dict[str, Any]:
    """Get GPU memory usage information"""
    if not torch.cuda.is_available():
        return {
            "gpu_available": False,
            "allocated_mb": 0,
            "cached_mb": 0,
            "total_mb": 0
        }
    
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    cached = torch.cuda.memory_reserved() / (1024 * 1024)
    
    return {
        "gpu_available": True,
        "allocated_mb": allocated,
        "cached_mb": cached,
        "total_mb": torch.cuda.get_device_properties(0).total_memory / (1024 * 1024),
        "free_mb": (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)) - cached
    }


def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def estimate_model_memory(model_path: str) -> float:
    """Estimate model memory usage in MB"""
    if not Path(model_path).exists():
        return 0.0
    
    total_size = 0
    for file_path in Path(model_path).rglob('*.safetensors'):
        total_size += file_path.stat().st_size
    
    for file_path in Path(model_path).rglob('*.bin'):
        total_size += file_path.stat().st_size
        
    # Convert to MB and add overhead estimate
    size_mb = total_size / (1024 * 1024)
    return size_mb * 1.2  # 20% overhead estimate


def get_system_resource_info() -> Dict[str, Any]:
    """Get comprehensive system resource information for visualization"""
    try:
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_info = {
            "total_mb": memory.total / (1024 * 1024),
            "available_mb": memory.available / (1024 * 1024),
            "used_mb": memory.used / (1024 * 1024),
            "percent": memory.percent
        }
        
        # Process-specific information
        process = psutil.Process()
        process_info = {
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / (1024 * 1024),
            "memory_percent": process.memory_percent()
        }
        
        # GPU information
        gpu_info = get_gpu_memory_info()
        
        return {
            "timestamp": psutil.boot_time(),
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count
            },
            "memory": memory_info,
            "process": process_info,
            "gpu": gpu_info,
            "disk": {
                "usage_percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
            }
        }
        
    except Exception as e:
        # Fallback with basic information
        return {
            "timestamp": 0,
            "cpu": {"percent": 0, "count": 1},
            "memory": {"total_mb": 0, "available_mb": 0, "used_mb": 0, "percent": 0},
            "process": {"cpu_percent": 0, "memory_mb": 0, "memory_percent": 0},
            "gpu": get_gpu_memory_info(),
            "disk": {"usage_percent": 0},
            "error": str(e)
        }


def get_training_resource_snapshot(task_id: str) -> Dict[str, Any]:
    """Get a comprehensive resource snapshot specifically for training visualization"""
    resource_info = get_system_resource_info()
    
    return {
        "task_id": task_id,
        "timestamp": psutil.time.time(),
        "resources": resource_info,
        "gpu_utilization": resource_info["gpu"]["allocated_mb"] / max(resource_info["gpu"]["total_mb"], 1) * 100 if resource_info["gpu"]["gpu_available"] else 0,
        "memory_pressure": resource_info["memory"]["percent"],
        "cpu_load": resource_info["cpu"]["percent"]
    }