"""
Services layer for LLMForge

Provides business logic services for model management, training, and chat functionality.
"""

# Try to import torch-dependent services
try:
    from .model_service import ModelService
    from .training_service import TrainingService  
    from .chat_service import ChatService
    from .comparison_service import ComparisonService
    
    __all__ = [
        'ModelService',
        'TrainingService', 
        'ChatService',
        'ComparisonService'
    ]
except ImportError:
    # For services that don't have torch, only import non-torch dependent services
    # Import data_generation_service if it exists
    try:
        from .data_generation_service import DataGenerationService
        from .visualization_service import VisualizationService
        
        __all__ = [
            'DataGenerationService',
            'VisualizationService'
        ]
    except ImportError:
        __all__ = []