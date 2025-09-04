"""
Backward Compatibility Wrapper for dual_model_manager.py

This file maintains the original API surface while delegating to the new ModelService.
Existing code that imports from dual_model_manager should continue to work unchanged.
"""

# Import all the original classes and functions from their new locations
from models.entities import ModelPair, LoRAMetadata, KVCacheEntry, CacheState
from services.model_service import (
    ModelService, DependencyGraph, PerformanceDrivenCacheSwapper,
    UNSLOTH_AVAILABLE, TRANSFORMERS_AVAILABLE
)

# Create the original DualModelManager class for backward compatibility
class DualModelManager:
    """
    Backward compatibility wrapper for the original DualModelManager.
    
    This class delegates all method calls to the new ModelService while
    maintaining the exact same API surface area.
    """
    
    def __init__(self, max_hbm_gb: float = 12.0, max_cached_pairs: int = 5):
        # Initialize the new ModelService
        self.model_service = ModelService(max_hbm_gb=max_hbm_gb, max_cached_pairs=max_cached_pairs)
        
        # Maintain direct access to internal components for backward compatibility
        self.max_hbm_gb = max_hbm_gb
        self.max_cached_pairs = max_cached_pairs
        
        # Expose internal structures for backward compatibility
        self.base_models = self.model_service.base_models
        self.lora_adapters = self.model_service.lora_adapters
        self.kv_caches = self.model_service.kv_caches
        self.active_pairs = self.model_service.active_pairs
        self.loaded_models_cache = self.model_service.loaded_models_cache
        self.model_load_lock = self.model_service.model_load_lock
        
        # FastLibra components
        self.dependency_graph = self.model_service.dependency_graph
        self.cache_swapper = self.model_service.cache_swapper
        
        # Synchronization
        self.model_lock = self.model_service.model_lock
        self.cache_lock = self.model_service.cache_lock
        
        # Performance tracking
        self.metrics = self.model_service.metrics
        
        print(f"DualModelManager initialized with {max_hbm_gb}GB HBM limit (modular)")
        print(f"KV cache optimization: FastLibra-inspired architecture")
    
    async def discover_model_pairs(self):
        """Discover available base model + LoRA adapter pairs"""
        return await self.model_service.discover_model_pairs()
    
    async def load_model_pair(self, pair, session_id):
        """Load base model + LoRA adapter pair with KV cache optimization"""
        return await self.model_service.load_model_pair(pair, session_id)
    
    async def generate_comparison_responses(self, session_id, messages, generation_params=None):
        """Generate responses from both base and fine-tuned models with KV cache optimization"""
        return await self.model_service.generate_comparison_responses(session_id, messages, generation_params)
    
    def get_memory_stats(self):
        """Get detailed memory usage statistics"""
        return self.model_service.get_memory_stats()
    
    async def cleanup_session(self, session_id):
        """Clean up resources for a session"""
        return await self.model_service.cleanup_session(session_id)
    
    async def optimize_memory(self):
        """Periodic memory optimization using FastLibra cost model"""
        return await self.model_service.optimize_memory()
    
    def clear_model_cache(self, pair_id=None):
        """Clear loaded models cache to free memory"""
        return self.model_service.clear_model_cache(pair_id)
    
    def get_cached_models_info(self):
        """Get information about currently cached models"""
        return self.model_service.get_cached_models_info()
    
    async def warmup_model_pair(self, pair):
        """Preload and cache a model pair for faster subsequent access"""
        return await self.model_service.warmup_model_pair(pair)
    
    async def warmup_popular_models(self, max_models=3):
        """Warm up the most commonly used model pairs"""
        return await self.model_service.warmup_popular_models(max_models)
    
    # Internal methods that might be called directly
    async def _load_base_model(self, model_name, model_path):
        """Load base model with caching"""
        return await self.model_service._load_base_model(model_name, model_path)
    
    async def _load_lora_adapter(self, adapter_path, base_model_name, adapter_name):
        """Load LoRA adapter with dependency tracking"""
        return await self.model_service._load_lora_adapter(adapter_path, base_model_name, adapter_name)
    
    async def _load_models_for_generation(self, pair):
        """Load models specifically for generation using Unsloth pattern with caching"""
        return await self.model_service._load_models_for_generation(pair)
    
    async def _generate_single_response_simple(self, model, tokenizer, prompt, generation_params, model_type):
        """Generate response from a single model using Unsloth optimized generation"""
        return await self.model_service._generate_single_response_simple(model, tokenizer, prompt, generation_params, model_type)
    
    def _format_conversation_simple(self, messages):
        """Simple conversation formatting for demo purposes"""
        return self.model_service._format_conversation_simple(messages)


# Export all the classes and functions that external code might import
__all__ = [
    'DualModelManager',
    'ModelPair',
    'LoRAMetadata', 
    'KVCacheEntry',
    'CacheState',
    'DependencyGraph',
    'PerformanceDrivenCacheSwapper',
    'UNSLOTH_AVAILABLE',
    'TRANSFORMERS_AVAILABLE'
]