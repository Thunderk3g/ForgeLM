"""
Domain entities for the LLMForge platform
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from enum import Enum
import time


class CacheState(Enum):
    """States for cache management"""
    ACTIVE = "active"       # Currently in HBM
    CACHED = "cached"       # In main memory  
    LOADING = "loading"     # Being loaded to HBM
    EVICTING = "evicting"   # Being moved to main memory


@dataclass
class LoRAMetadata:
    """Metadata for LoRA adapters"""
    adapter_id: str
    adapter_path: str
    base_model_name: str
    size_mb: float
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    load_time_ms: float = 0.0
    state: CacheState = CacheState.CACHED


@dataclass
class KVCacheEntry:
    """KV Cache entry with dependency tracking"""
    cache_id: str
    sequence_length: int
    memory_usage_mb: float
    associated_loras: Set[str] = field(default_factory=set)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    state: CacheState = CacheState.CACHED


@dataclass 
class ModelPair:
    """Represents a base model + LoRA adapter pair"""
    pair_id: str
    base_model_name: str
    base_model_path: str
    adapter_name: str
    adapter_path: str
    base_model_size_mb: float
    adapter_size_mb: float
    memory_usage_mb: float = 0.0
    status: str = "unloaded"  # "loaded", "loading", "unloaded", "error"
    load_time_ms: float = 0.0
    last_inference_time: float = 0.0
    
    @property
    def name(self) -> str:
        """Compatibility property that returns a descriptive name for the model pair"""
        return f"{self.base_model_name}_{self.adapter_name}"


@dataclass
class ChatSession:
    session_id: str
    session_name: str
    pair_id: str
    model_pair: ModelPair
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "pair_id": self.pair_id,
            "model_info": {
                "base_model": self.model_pair.base_model_name,
                "adapter_name": self.model_pair.adapter_name,
                "pair_id": self.model_pair.pair_id
            },
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "message_count": len(self.messages),
            "status": self.status
        }