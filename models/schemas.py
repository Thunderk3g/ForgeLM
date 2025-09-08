"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, field_validator
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid


class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[str] = None


class ComparisonRequest(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    generation_params: Optional[Dict[str, Any]] = None
    custom_system_prompt: Optional[str] = None
    use_agent_orchestrator: bool = False


class ComparisonResponse(BaseModel):
    base_response: str
    finetuned_response: str
    generation_time_ms: Dict[str, float]
    model_info: Dict[str, str]
    cache_metrics: Dict[str, int]
    session_id: str
    timestamp: str


class SessionCreateRequest(BaseModel):
    pair_id: str
    session_name: Optional[str] = None
    custom_system_prompt: Optional[str] = None
    use_agent_orchestrator: bool = False


class SessionInfo(BaseModel):
    session_id: str
    session_name: str
    pair_id: str
    model_info: Dict[str, str]
    created_at: str
    last_activity: str
    message_count: int
    status: str  # "active", "inactive", "error"


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


class TrainingStatus(BaseModel):
    task_id: str
    status: str
    progress: str
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    current_epoch: Optional[float] = None
    total_epochs: Optional[int] = None
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    progress_percentage: Optional[float] = None
    message: Optional[str] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None


class SelectModelRequest(BaseModel):
    model_id: str
    output_dir: str
    quantization: Optional[str] = None


class PrepareDataRequest(BaseModel):
    input_file: str
    output_dir: str
    format: Optional[str] = None


class ResearchLogEntry(BaseModel):
    """Research log entry for detailed analysis of model comparisons"""
    log_id: str = None
    timestamp: str = None
    session_id: str
    model_pair: Dict[str, str]
    user_message: str
    base_response: str
    finetuned_response: str
    base_generation_time_ms: float
    finetuned_generation_time_ms: float
    base_token_count: int
    finetuned_token_count: int
    generation_params: Dict[str, Any]
    model_versions: Dict[str, str]
    response_similarity: Optional[float] = None
    length_ratio: Optional[float] = None
    
    def __init__(self, **data):
        if 'log_id' not in data or data['log_id'] is None:
            data['log_id'] = str(uuid.uuid4())
        if 'timestamp' not in data or data['timestamp'] is None:
            data['timestamp'] = datetime.now().isoformat()
        super().__init__(**data)