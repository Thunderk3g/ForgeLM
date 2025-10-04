"""
Pydantic schemas for Model Management Service
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class ModelLoadRequest(BaseModel):
    model_name: str
    model_path: str
    device: Optional[str] = None
    quantization: Optional[str] = None
    max_memory: Optional[int] = None


class ModelInfo(BaseModel):
    model_name: str
    model_path: str
    model_type: str
    size_gb: float
    loaded: bool
    last_used: str
    device: str


class ModelConfig(BaseModel):
    model_name: str
    config: Dict[str, Any]
    created_at: str
    updated_at: str


class CacheStats(BaseModel):
    total_models: int
    loaded_models: int
    cache_usage_gb: float
    cache_limit_gb: float
    hit_rate: float


class ModelUploadRequest(BaseModel):
    model_name: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelVersionInfo(BaseModel):
    model_name: str
    version: str
    created_at: str
    size_gb: float
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class FineTuneRequest(BaseModel):
    base_model_dir: str
    dataset_dir: str
    output_dir: str = "/app/models/fine_tuned"
    ft_type: str = "instruction"
    peft_method: str = "qlora"  # lora, qlora, dora, oft, pissa
    alignment: str = "none"
    epochs: int = 1
    batch_size: int = 20
    learning_rate: float = 0.0001
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 256
    reasoning_effort: Optional[str] = None
    
    # Advanced PEFT parameters
    peft_rank: int = 8
    peft_alpha: int = 16
    peft_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    
    # Template system parameters  
    template_name: str = "alpaca"
    system_message: Optional[str] = None