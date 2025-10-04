"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, field_validator, Field
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


# ============================
# Configuration Schemas for Chat Configuration Panel
# ============================

class GenerationConfig(BaseModel):
    """Configuration for model generation parameters"""
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Controls randomness in generation")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Nucleus sampling threshold")
    top_k: Optional[int] = Field(50, ge=1, le=100, description="Top-k sampling limit")
    max_tokens: int = Field(256, ge=50, le=4096, description="Maximum response length")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Penalty for token presence")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Penalty for token frequency")
    repetition_penalty: float = Field(1.0, ge=0.5, le=2.0, description="Repetition penalty (transformers)")
    do_sample: bool = Field(True, description="Use sampling vs greedy decoding")
    early_stopping: bool = Field(False, description="Stop early when possible")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")
    pad_token_id: Optional[int] = Field(None, description="Override padding token ID")
    eos_token_id: Optional[int] = Field(None, description="Override end-of-sequence token ID")


class ModelSpecificConfig(BaseModel):
    """Configuration specific to a single model in the pair"""
    system_prompt: str = Field("", description="Custom system prompt for this model")
    context_length: int = Field(2048, ge=512, le=8192, description="Maximum context length")
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)
    model_id: Optional[str] = Field(None, description="Identifier for this model")
    enabled: bool = Field(True, description="Whether this model is enabled for generation")


class DualModelConfig(BaseModel):
    """Configuration for both models in a comparison pair"""
    session_id: str = Field(..., description="Chat session identifier")
    base_model: ModelSpecificConfig = Field(default_factory=ModelSpecificConfig)
    finetuned_model: ModelSpecificConfig = Field(default_factory=ModelSpecificConfig)
    sync_parameters: bool = Field(False, description="Sync generation parameters between models")
    config_name: Optional[str] = Field(None, description="Name for this configuration")
    created_at: Optional[str] = Field(None, description="Configuration creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    
    def __init__(self, **data):
        if 'created_at' not in data or data['created_at'] is None:
            data['created_at'] = datetime.now().isoformat()
        if 'updated_at' not in data or data['updated_at'] is None:
            data['updated_at'] = datetime.now().isoformat()
        super().__init__(**data)


class ConfigPreset(BaseModel):
    """Predefined configuration preset"""
    preset_id: str = Field(..., description="Unique preset identifier")
    name: str = Field(..., description="Human-readable preset name")
    description: str = Field("", description="Description of this preset's purpose")
    category: str = Field("general", description="Preset category (creative, factual, code, etc.)")
    base_config: GenerationConfig = Field(default_factory=GenerationConfig)
    finetuned_config: GenerationConfig = Field(default_factory=GenerationConfig)
    system_prompts: Dict[str, str] = Field(default_factory=dict, description="Default system prompts")
    is_built_in: bool = Field(False, description="Whether this is a built-in preset")
    created_by: Optional[str] = Field(None, description="User who created this preset")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = Field(0, description="Number of times this preset has been used")


class ModelCapabilities(BaseModel):
    """Information about what a model can do and its constraints"""
    model_id: str
    model_type: str  # "base", "finetuned", "unknown"
    max_context_length: int
    supports_system_prompt: bool
    default_parameters: GenerationConfig
    parameter_constraints: Dict[str, Dict[str, Any]]  # min/max values for each param
    recommended_presets: List[str]
    memory_usage_mb: Optional[int] = None
    estimated_tokens_per_second: Optional[float] = None


# Request/Response schemas for configuration endpoints

class ModelConfigRequest(BaseModel):
    """Request to update model configuration"""
    session_id: str
    config: DualModelConfig


class ConfigPresetRequest(BaseModel):
    """Request to create or update a configuration preset"""
    name: str
    description: Optional[str] = ""
    category: str = "general"
    base_config: GenerationConfig
    finetuned_config: GenerationConfig
    system_prompts: Optional[Dict[str, str]] = None


class ConfigValidationResponse(BaseModel):
    """Response for configuration validation"""
    is_valid: bool
    warnings: List[str] = []
    errors: List[str] = []
    estimated_performance: Optional[Dict[str, Any]] = None
    memory_requirements: Optional[Dict[str, int]] = None


class ConfigPresetListResponse(BaseModel):
    """Response containing list of available presets"""
    presets: List[ConfigPreset]
    categories: List[str]
    total_count: int


# Enhanced comparison request with configuration support
class EnhancedComparisonRequest(BaseModel):
    """Enhanced comparison request with full configuration support"""
    session_id: str
    messages: List[ChatMessage]
    config: Optional[DualModelConfig] = None
    use_agent_orchestrator: bool = False
    return_debug_info: bool = False