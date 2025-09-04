"""
Performance monitoring utilities
"""

import time
from contextlib import contextmanager
from typing import Dict, Any
import functools


class PerformanceMonitor:
    """Performance monitoring utility"""
    
    def __init__(self):
        self.metrics = {}
    
    def record(self, operation: str, duration_ms: float, **metadata):
        """Record performance metric"""
        if operation not in self.metrics:
            self.metrics[operation] = {
                "total_calls": 0,
                "total_time_ms": 0,
                "min_time_ms": float('inf'),
                "max_time_ms": 0,
                "avg_time_ms": 0
            }
        
        metric = self.metrics[operation]
        metric["total_calls"] += 1
        metric["total_time_ms"] += duration_ms
        metric["min_time_ms"] = min(metric["min_time_ms"], duration_ms)
        metric["max_time_ms"] = max(metric["max_time_ms"], duration_ms)
        metric["avg_time_ms"] = metric["total_time_ms"] / metric["total_calls"]
        
        # Add metadata
        for key, value in metadata.items():
            metric[key] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics"""
        return self.metrics.copy()
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()


# Global performance monitor
performance_monitor = PerformanceMonitor()


@contextmanager
def timing_context(operation_name: str, **metadata):
    """Context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        performance_monitor.record(operation_name, duration_ms, **metadata)


def timed(operation_name: str = None):
    """Decorator for timing function calls"""
    def decorator(func):
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with timing_context(op_name):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)  
        def sync_wrapper(*args, **kwargs):
            with timing_context(op_name):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator