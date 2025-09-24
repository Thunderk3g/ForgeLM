"""
Logging utilities
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Get a configured logger"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create handler
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Set level
        level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(level)
        handler.setLevel(level)
        
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger


def setup_file_logging(log_file: Optional[str] = None):
    """Setup file logging"""
    if log_file is None:
        log_file = "logs/llmforge.log"
    
    # Ensure logs directory exists
    Path(log_file).parent.mkdir(exist_ok=True)
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


def log_performance(logger: logging.Logger, operation: str, duration_ms: float, **metadata):
    """Log performance metrics"""
    metadata_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
    logger.info(f"PERF: {operation} took {duration_ms:.2f}ms ({metadata_str})")