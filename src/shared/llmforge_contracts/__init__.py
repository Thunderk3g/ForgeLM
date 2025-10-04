"""
LLMForge Contracts Package

Minimal shared contracts for inter-service communication.
Contains only essential schemas and API contracts.
"""

__version__ = "1.0.0"

from .schemas import (
    ServiceHealth,
    ServiceInfo,
    ErrorResponse,
    SuccessResponse
)

__all__ = [
    "ServiceHealth", 
    "ServiceInfo",
    "ErrorResponse", 
    "SuccessResponse"
]