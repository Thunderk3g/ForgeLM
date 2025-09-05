"""
Application configuration settings
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration"""
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Model Management
    max_hbm_gb: float = 12.0
    max_cached_pairs: int = 5
    max_sessions: int = 20
    
    # Paths
    models_base_dir: str = "models/base"
    models_finetuned_dir: str = "models/fine_tuned"
    data_dir: str = "data"
    logs_dir: str = "logs"
    
    # Performance
    unsloth_optimization: bool = True
    cache_optimization: bool = True
    background_optimization: bool = True
    
    # Training
    default_max_seq_length: int = 1024  # Reduced for better memory usage
    default_batch_size: int = 20
    default_learning_rate: float = 0.0001
    
    # Session Management
    session_cleanup_interval: int = 300  # 5 minutes
    inactive_session_threshold: int = 3600  # 1 hour


# Global settings instance
settings = AppConfig(
    # Override with environment variables if available
    max_hbm_gb=float(os.getenv("MAX_HBM_GB", "12.0")),
    max_sessions=int(os.getenv("MAX_SESSIONS", "20")),
    api_host=os.getenv("API_HOST", "0.0.0.0"),
    api_port=int(os.getenv("API_PORT", "8000")),
    unsloth_optimization=os.getenv("UNSLOTH_OPTIMIZATION", "true").lower() == "true",
    cache_optimization=os.getenv("CACHE_OPTIMIZATION", "true").lower() == "true",
)