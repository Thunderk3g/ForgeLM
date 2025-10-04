"""
Shared schemas for inter-service communication.
Keep this minimal - only essential contracts that services need to communicate.
"""

from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime


class ServiceHealth(BaseModel):
    """Health check response schema"""
    status: str  # "healthy", "unhealthy", "degraded"
    timestamp: str
    service_name: str
    version: str


class ServiceInfo(BaseModel):
    """Service information schema"""
    name: str
    version: str
    description: str
    status: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Standard error response schema"""
    error: str
    message: str
    status_code: int
    timestamp: str
    service: str


class SuccessResponse(BaseModel):
    """Standard success response schema"""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str