# FastLibra-Inspired Dual Model Manager with KV Cache Optimization
# Based on "FastLibra: Scalable Multi-LoRA Inference with HBM-Aware Cache Management"

import os
import time
import asyncio
import threading
import gc
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import torch
import json
from pathlib import Path
import weakref
from enum import Enum

# Import Unsloth with proper error handling
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    print("Unsloth available for optimization")
except (ImportError, NotImplementedError) as e:
    UNSLOTH_AVAILABLE = False
    print(f"Warning: Unsloth not available: {e}")
    print("Falling back to standard transformers loading")
    
    # Fallback imports
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        TRANSFORMERS_AVAILABLE = True
        print("Transformers fallback available")
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        print("Neither Unsloth nor Transformers available")
else:
    # If Unsloth is available, we still want transformers as backup
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False


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
    associated_loras: set = field(default_factory=set)
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


class DependencyGraph:
    """Tree-based dependency tracking for LoRAs and KV caches (FastLibra approach)"""
    
    def __init__(self):
        self.lora_to_kv_cache = defaultdict(set)  # LoRA -> set of KV cache IDs
        self.kv_cache_to_loras = defaultdict(set)  # KV cache -> set of LoRA IDs
        self.access_history = defaultdict(list)   # Track access patterns
    
    def add_dependency(self, lora_id: str, kv_cache_id: str):
        """Add dependency between LoRA and KV cache"""
        self.lora_to_kv_cache[lora_id].add(kv_cache_id)
        self.kv_cache_to_loras[kv_cache_id].add(lora_id)
        self.access_history[lora_id].append(time.time())
    
    def remove_dependency(self, lora_id: str, kv_cache_id: str):
        """Remove dependency between LoRA and KV cache"""
        self.lora_to_kv_cache[lora_id].discard(kv_cache_id)
        self.kv_cache_to_loras[kv_cache_id].discard(lora_id)
    
    def get_dependent_caches(self, lora_id: str) -> set:
        """Get all KV caches dependent on a LoRA"""
        return self.lora_to_kv_cache.get(lora_id, set())
    
    def get_dependent_loras(self, kv_cache_id: str) -> set:
        """Get all LoRAs dependent on a KV cache"""
        return self.kv_cache_to_loras.get(kv_cache_id, set())
    
    def calculate_usage_score(self, item_id: str, current_time: float) -> float:
        """Calculate usage score based on access patterns (FastLibra cost model)"""
        history = self.access_history.get(item_id, [])
        if not history:
            return 0.0
        
        # Recent access weight (higher = more recent)
        recent_weight = max(0, 1.0 - (current_time - max(history)) / 3600)  # 1 hour decay
        
        # Frequency weight
        frequency_weight = len(history) / (current_time - min(history) + 1)
        
        return recent_weight * 0.7 + frequency_weight * 0.3


class PerformanceDrivenCacheSwapper:
    """Performance-driven cache swapper implementing FastLibra cost model"""
    
    def __init__(self, max_hbm_gb: float = 12.0):
        self.max_hbm_bytes = int(max_hbm_gb * 1024 * 1024 * 1024)
        self.current_hbm_usage = 0
        self.swap_costs = {}  # Cache swap operation costs
        
    def calculate_swap_cost(self, item_size_mb: float, is_swap_in: bool) -> float:
        """Calculate cost of swapping item in/out of HBM"""
        base_cost = item_size_mb * 0.1  # Base transfer cost
        
        if is_swap_in:
            # Cost increases with current memory pressure
            memory_pressure = self.current_hbm_usage / self.max_hbm_bytes
            return base_cost * (1.0 + memory_pressure)
        else:
            # Swap-out cost is lower
            return base_cost * 0.5
    
    def should_swap_in(self, lora_metadata: LoRAMetadata, kv_entries: List[KVCacheEntry]) -> bool:
        """Decide if LoRA should be swapped into HBM"""
        current_time = time.time()
        
        # Calculate expected benefit
        access_frequency = lora_metadata.access_count / max(1, current_time - (current_time - 3600))
        recency_score = max(0, 1.0 - (current_time - lora_metadata.last_access_time) / 1800)  # 30 min decay
        
        benefit_score = access_frequency * recency_score
        swap_cost = self.calculate_swap_cost(lora_metadata.size_mb, is_swap_in=True)
        
        return benefit_score > swap_cost
    
    def select_eviction_candidates(self, required_memory_mb: float, 
                                 active_loras: Dict[str, LoRAMetadata],
                                 active_kv_caches: Dict[str, KVCacheEntry]) -> List[str]:
        """Select items to evict based on cost model"""
        candidates = []
        current_time = time.time()
        
        # Score all active items
        for lora_id, metadata in active_loras.items():
            recency = current_time - metadata.last_access_time
            frequency = metadata.access_count
            size_penalty = metadata.size_mb / 100  # Prefer evicting larger items
            
            eviction_score = recency + size_penalty - frequency
            candidates.append((eviction_score, lora_id, metadata.size_mb, 'lora'))
        
        for cache_id, entry in active_kv_caches.items():
            recency = current_time - entry.last_access_time  
            frequency = entry.access_count
            size_penalty = entry.memory_usage_mb / 100
            
            eviction_score = recency + size_penalty - frequency
            candidates.append((eviction_score, cache_id, entry.memory_usage_mb, 'kv'))
        
        # Sort by eviction score (higher = more likely to evict)
        candidates.sort(reverse=True)
        
        # Select candidates until we free enough memory
        selected = []
        freed_memory = 0
        
        for score, item_id, size_mb, item_type in candidates:
            selected.append(item_id)
            freed_memory += size_mb
            
            if freed_memory >= required_memory_mb:
                break
        
        return selected


class DualModelManager:
    """Advanced dual model manager with FastLibra-inspired KV cache optimization"""
    
    def __init__(self, max_hbm_gb: float = 12.0, max_cached_pairs: int = 5):
        self.max_hbm_gb = max_hbm_gb
        self.max_cached_pairs = max_cached_pairs
        
        # Core storage
        self.base_models: Dict[str, Tuple[Any, Any]] = {}  # name -> (model, tokenizer)
        self.lora_adapters: Dict[str, LoRAMetadata] = {}  # adapter_id -> metadata
        self.kv_caches: Dict[str, KVCacheEntry] = {}      # cache_id -> entry
        self.active_pairs: Dict[str, ModelPair] = {}       # session_id -> model pair
        
        # FastLibra components
        self.dependency_graph = DependencyGraph()
        self.cache_swapper = PerformanceDrivenCacheSwapper(max_hbm_gb)
        
        # Synchronization
        self.model_lock = threading.RLock()
        self.cache_lock = threading.RLock()
        
        # Performance tracking
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "swap_operations": 0,
            "total_inference_time_ms": 0,
            "memory_usage_mb": 0
        }
        
        print(f"DualModelManager initialized with {max_hbm_gb}GB HBM limit")
        print(f"KV cache optimization: FastLibra-inspired architecture")
    
    async def discover_model_pairs(self) -> List[ModelPair]:
        """Discover available base model + LoRA adapter pairs"""
        pairs = []
        
        # Scan for base models
        base_models_dir = Path("models/base")
        if base_models_dir.exists():
            for base_dir in base_models_dir.iterdir():
                if base_dir.is_dir():
                    base_model_name = base_dir.name
                    base_model_path = str(base_dir)
                    
                    # Calculate base model size
                    base_size_mb = sum(f.stat().st_size for f in base_dir.rglob('*') if f.is_file()) / (1024 * 1024)
                    
                    # Look for corresponding LoRA adapters
                    finetuned_dir = Path("models/finetuned")
                    if finetuned_dir.exists():
                        # Main adapter
                        if (finetuned_dir / "adapter_model.safetensors").exists():
                            adapter_size_mb = (finetuned_dir / "adapter_model.safetensors").stat().st_size / (1024 * 1024)
                            
                            pairs.append(ModelPair(
                                pair_id=f"{base_model_name}_main",
                                base_model_name=base_model_name,
                                base_model_path=base_model_path,
                                adapter_name="main",
                                adapter_path=str(finetuned_dir),
                                base_model_size_mb=base_size_mb,
                                adapter_size_mb=adapter_size_mb
                            ))
                        
                        # Checkpoint adapters
                        for checkpoint_dir in finetuned_dir.glob("checkpoint-*"):
                            if checkpoint_dir.is_dir() and (checkpoint_dir / "adapter_model.safetensors").exists():
                                checkpoint_name = checkpoint_dir.name
                                adapter_size_mb = (checkpoint_dir / "adapter_model.safetensors").stat().st_size / (1024 * 1024)
                                
                                pairs.append(ModelPair(
                                    pair_id=f"{base_model_name}_{checkpoint_name}",
                                    base_model_name=base_model_name,
                                    base_model_path=base_model_path,
                                    adapter_name=checkpoint_name,
                                    adapter_path=str(checkpoint_dir),
                                    base_model_size_mb=base_size_mb,
                                    adapter_size_mb=adapter_size_mb
                                ))
        
        print(f"Discovered {len(pairs)} model pairs")
        return pairs
    
    async def load_model_pair(self, pair: ModelPair, session_id: str) -> Tuple[Any, Any, Any, Any]:
        """Load base model + LoRA adapter pair with KV cache optimization"""
        start_time = time.time()
        
        with self.model_lock:
            try:
                # Load base model (shared across adapters)
                base_model, tokenizer = await self._load_base_model(pair.base_model_name, pair.base_model_path)
                
                # Load LoRA adapter with caching
                finetuned_model, adapter_tokenizer = await self._load_lora_adapter(
                    pair.adapter_path, pair.base_model_name, pair.adapter_name
                )
                
                # Update pair status
                load_time_ms = (time.time() - start_time) * 1000
                pair.status = "loaded"
                pair.load_time_ms = load_time_ms
                pair.memory_usage_mb = pair.base_model_size_mb + pair.adapter_size_mb
                
                # Register active pair
                self.active_pairs[session_id] = pair
                
                # Update dependency graph
                self.dependency_graph.add_dependency(pair.pair_id, f"kv_cache_{session_id}")
                
                print(f"Model pair loaded in {load_time_ms:.1f}ms: {pair.pair_id}")
                
                return base_model, tokenizer, finetuned_model, adapter_tokenizer
                
            except Exception as e:
                pair.status = "error"
                print(f"Failed to load model pair {pair.pair_id}: {e}")
                raise
    
    async def _load_base_model(self, model_name: str, model_path: str) -> Tuple[Any, Any]:
        """Load base model with caching"""
        if model_name in self.base_models:
            print(f"Base model cache hit: {model_name}")
            self.metrics["cache_hits"] += 1
            return self.base_models[model_name]
        
        print(f"Loading base model: {model_name}")
        self.metrics["cache_misses"] += 1
        
        if UNSLOTH_AVAILABLE:
            # Load with Unsloth optimizations
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            
            # Enable inference mode for 2x speedup
            model = FastLanguageModel.for_inference(model)
            
        elif TRANSFORMERS_AVAILABLE:
            # Fallback to standard transformers
            print("Using Transformers fallback for base model")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            
        else:
            raise RuntimeError("Neither Unsloth nor Transformers available for model loading")
        
        # Cache the model
        self.base_models[model_name] = (model, tokenizer)
        
        return model, tokenizer
    
    async def _load_lora_adapter(self, adapter_path: str, base_model_name: str, adapter_name: str) -> Tuple[Any, Any]:
        """Load LoRA adapter with dependency tracking"""
        adapter_id = f"{base_model_name}_{adapter_name}"
        
        if adapter_id in self.lora_adapters:
            metadata = self.lora_adapters[adapter_id]
            if metadata.state == CacheState.ACTIVE:
                print(f"LoRA adapter cache hit: {adapter_id}")
                metadata.access_count += 1
                metadata.last_access_time = time.time()
                self.metrics["cache_hits"] += 1
                
                # Return cached model (simplified - would need actual model reference)
                return None, None  # Placeholder for actual implementation
        
        print(f"Loading LoRA adapter: {adapter_id}")
        self.metrics["cache_misses"] += 1
        
        if UNSLOTH_AVAILABLE:
            # Load with Unsloth optimizations
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            
            # Enable inference mode
            model = FastLanguageModel.for_inference(model)
            
        elif TRANSFORMERS_AVAILABLE:
            # Fallback to standard transformers with PEFT
            print("Using Transformers + PEFT fallback for adapter")
            try:
                from peft import PeftModel, PeftConfig
                
                # Load the base model first
                if base_model_name not in self.base_models:
                    await self._load_base_model(base_model_name, f"models/base/{base_model_name}")
                
                base_model, base_tokenizer = self.base_models[base_model_name]
                
                # Load adapter config and apply to base model
                config = PeftConfig.from_pretrained(adapter_path)
                model = PeftModel.from_pretrained(base_model, adapter_path)
                tokenizer = base_tokenizer
                
            except ImportError:
                print("Warning: PEFT not available, creating placeholder adapter")
                # For demo purposes, return base model (in production, you'd install PEFT)
                if base_model_name not in self.base_models:
                    await self._load_base_model(base_model_name, f"models/base/{base_model_name}")
                model, tokenizer = self.base_models[base_model_name]
                
        else:
            raise RuntimeError("Neither Unsloth nor Transformers available for adapter loading")
        
        # Create metadata
        adapter_size = sum(f.stat().st_size for f in Path(adapter_path).rglob('*') if f.is_file()) / (1024 * 1024)
        metadata = LoRAMetadata(
            adapter_id=adapter_id,
            adapter_path=adapter_path,
            base_model_name=base_model_name,
            size_mb=adapter_size,
            state=CacheState.ACTIVE
        )
        
        self.lora_adapters[adapter_id] = metadata
        
        return model, tokenizer
    
    async def generate_comparison_responses(
        self, 
        session_id: str,
        messages: List[Dict[str, str]], 
        generation_params: Dict = None
    ) -> Dict[str, Any]:
        """Generate responses from both base and fine-tuned models with KV cache optimization"""
        
        if session_id not in self.active_pairs:
            raise ValueError(f"No active model pair for session {session_id}")
        
        pair = self.active_pairs[session_id]
        start_time = time.time()
        
        # Default generation parameters
        if generation_params is None:
            generation_params = {
                "max_new_tokens": 256,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9
            }
        
        try:
            # Load models if needed (with KV cache optimization)
            base_model, tokenizer, finetuned_model, adapter_tokenizer = await self.load_model_pair(pair, session_id)
            
            # Format conversation
            conversation_text = self._format_conversation(messages, tokenizer)
            
            # Generate from both models concurrently
            base_task = asyncio.create_task(self._generate_single_response(
                base_model, tokenizer, conversation_text, generation_params, "base"
            ))
            
            finetuned_task = asyncio.create_task(self._generate_single_response(
                finetuned_model, adapter_tokenizer, conversation_text, generation_params, "finetuned"
            ))
            
            # Wait for both responses
            base_result, finetuned_result = await asyncio.gather(base_task, finetuned_task)
            
            # Update metrics
            total_time_ms = (time.time() - start_time) * 1000
            self.metrics["total_inference_time_ms"] += total_time_ms
            
            # Update pair last inference time
            pair.last_inference_time = time.time()
            
            return {
                "base_response": base_result["text"],
                "finetuned_response": finetuned_result["text"],
                "generation_time_ms": {
                    "base": base_result["time_ms"],
                    "finetuned": finetuned_result["time_ms"],
                    "total": total_time_ms
                },
                "model_info": {
                    "base_model": pair.base_model_name,
                    "adapter_name": pair.adapter_name,
                    "pair_id": pair.pair_id
                },
                "cache_metrics": {
                    "cache_hits": self.metrics["cache_hits"],
                    "cache_misses": self.metrics["cache_misses"]
                }
            }
            
        except Exception as e:
            print(f"Error generating comparison responses: {e}")
            raise
    
    def _format_conversation(self, messages: List[Dict[str, str]], tokenizer) -> str:
        """Format conversation for model input"""
        # Apply chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            return tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback to simple formatting
            formatted_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                formatted_parts.append(f"{role.title()}: {content}")
            
            return "\n".join(formatted_parts) + "\nAssistant: "
    
    async def _generate_single_response(
        self, 
        model, 
        tokenizer, 
        prompt: str, 
        generation_params: Dict, 
        model_type: str
    ) -> Dict[str, Any]:
        """Generate response from a single model"""
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Move to device if using GPU
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_params,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            generation_time_ms = (time.time() - start_time) * 1000
            
            return {
                "text": response,
                "time_ms": generation_time_ms,
                "model_type": model_type
            }
            
        except Exception as e:
            print(f"Error generating {model_type} response: {e}")
            return {
                "text": f"Error generating {model_type} response: {str(e)}",
                "time_ms": (time.time() - start_time) * 1000,
                "model_type": model_type
            }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory usage statistics"""
        stats = {
            "hbm_usage_mb": self.cache_swapper.current_hbm_usage / (1024 * 1024),
            "hbm_limit_mb": self.max_hbm_gb * 1024,
            "active_base_models": len(self.base_models),
            "active_lora_adapters": len([a for a in self.lora_adapters.values() if a.state == CacheState.ACTIVE]),
            "cached_kv_entries": len(self.kv_caches),
            "cache_hit_rate": self.metrics["cache_hits"] / max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"]),
            "total_swap_operations": self.metrics["swap_operations"]
        }
        
        if torch.cuda.is_available():
            stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        return stats
    
    async def cleanup_session(self, session_id: str):
        """Clean up resources for a session"""
        with self.model_lock:
            if session_id in self.active_pairs:
                pair = self.active_pairs[session_id]
                print(f"Cleaning up session {session_id} (pair: {pair.pair_id})")
                
                # Remove dependencies
                self.dependency_graph.remove_dependency(pair.pair_id, f"kv_cache_{session_id}")
                
                # Remove from active pairs
                del self.active_pairs[session_id]
                
                # Clean up KV cache if no dependencies
                kv_cache_id = f"kv_cache_{session_id}"
                if kv_cache_id in self.kv_caches:
                    dependent_loras = self.dependency_graph.get_dependent_loras(kv_cache_id)
                    if not dependent_loras:
                        del self.kv_caches[kv_cache_id]
                        print(f"Cleaned up KV cache: {kv_cache_id}")
    
    async def optimize_memory(self):
        """Periodic memory optimization using FastLibra cost model"""
        current_time = time.time()
        
        # Check if memory optimization is needed
        memory_usage = sum(a.size_mb for a in self.lora_adapters.values() if a.state == CacheState.ACTIVE)
        memory_pressure = memory_usage / (self.max_hbm_gb * 1024)
        
        if memory_pressure > 0.8:  # 80% threshold
            print(f"Memory pressure at {memory_pressure:.1%}, optimizing...")
            
            # Select candidates for eviction
            active_loras = {k: v for k, v in self.lora_adapters.items() if v.state == CacheState.ACTIVE}
            active_kv = {k: v for k, v in self.kv_caches.items() if v.state == CacheState.ACTIVE}
            
            required_memory = memory_usage * 0.2  # Free 20% of memory
            candidates = self.cache_swapper.select_eviction_candidates(
                required_memory, active_loras, active_kv
            )
            
            # Perform evictions
            for item_id in candidates:
                if item_id in self.lora_adapters:
                    self.lora_adapters[item_id].state = CacheState.CACHED
                elif item_id in self.kv_caches:
                    self.kv_caches[item_id].state = CacheState.CACHED
                
                self.metrics["swap_operations"] += 1
            
            print(f"Evicted {len(candidates)} items to free memory")