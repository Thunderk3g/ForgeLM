"""
Legacy modules for backward compatibility.

This package contains legacy modules that were originally in the root directory
but have been moved to maintain a clean project structure while preserving
all existing functionality.
"""

# Re-export commonly used functions for easier imports
from .model_selection import select_model, SelectModelRequest, scan_available_models
from .data_preparation import prepare_data, PrepareDataRequest

__all__ = [
    'select_model', 
    'SelectModelRequest', 
    'scan_available_models',
    'prepare_data', 
    'PrepareDataRequest'
]