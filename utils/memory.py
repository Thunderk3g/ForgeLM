"""
Memory management utilities
"""

import torch
import gc
from typing import Dict, Any
from pathlib import Path


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