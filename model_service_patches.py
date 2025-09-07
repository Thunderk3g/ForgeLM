
# Add this import at the top of services/model_service.py
try:
    # Import fallback service
    from services.fallback_generation_service import FallbackGenerationService
    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False

# Add this method to ModelService class
async def generate_comparison_responses_with_fallback(
    self, 
    session_id: str,
    messages: List[Dict[str, str]], 
    generation_params: Dict = None
) -> Dict[str, Any]:
    """Generate responses with fallback support for TensorFlow/GPU issues"""
    
    try:
        # Try the normal generation process first
        return await self.generate_comparison_responses(session_id, messages, generation_params)
    
    except Exception as e:
        self.logger.error(f"Normal generation failed: {e}")
        
        if "tensorflow" in str(e).lower() or "dll" in str(e).lower():
            self.logger.info("TensorFlow/DLL issue detected, using fallback service")
            
            if FALLBACK_AVAILABLE:
                fallback_service = FallbackGenerationService()
                return fallback_service.generate_fallback_response(messages)
            else:
                return {
                    "base_response": "Model generation temporarily unavailable due to system configuration issues.",
                    "finetuned_response": "Please check system dependencies and try again.",
                    "generation_time_ms": {"base": 0, "finetuned": 0, "total": 0},
                    "model_info": {"base_model": "error", "adapter_name": "error", "pair_id": "error"},
                    "cache_metrics": {"cache_hits": 0, "cache_misses": 1}
                }
        else:
            # Re-raise non-TensorFlow errors
            raise
