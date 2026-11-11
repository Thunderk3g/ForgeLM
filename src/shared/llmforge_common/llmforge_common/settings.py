"""
Application configuration settings
"""

import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration"""
    # Model Management
    max_hbm_gb: float = 12.0
    max_cached_pairs: int = 5
    max_sessions: int = 20
    
    # Paths
    models_finetuned_dir: str = "models/fine_tuned"
    logs_dir: str = "logs"
    
    # Performance
    background_optimization: bool = True
    
    # Training
    default_max_seq_length: int = 1024  # Balance between memory usage and sequence handling
    
    # Session Management
    session_cleanup_interval: int = 300  # 5 minutes
    inactive_session_threshold: int = 3600  # 1 hour


# Global settings instance
settings = AppConfig(
    # Override with environment variables if available
    max_hbm_gb=float(os.getenv("MAX_HBM_GB", "12.0")),
    max_sessions=int(os.getenv("MAX_SESSIONS", "20")),
)