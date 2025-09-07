
"""
Fallback Generation Service

Provides basic text generation capabilities when main model service fails.
"""

from typing import Dict, List, Any
from datetime import datetime
import random

class FallbackGenerationService:
    """Fallback service for when model loading fails"""
    
    def __init__(self):
        self.fallback_responses = {
            "greeting": [
                "Hello! I'm here to help you with your questions.",
                "Hi there! How can I assist you today?",
                "Greetings! What would you like to know?",
                "Hello! I'm ready to help with any questions you have."
            ],
            "question": [
                "That's an interesting question. Let me think about that...",
                "I understand you're asking about this topic. Here's what I can share...",
                "Based on the information available, I can provide some insights...",
                "Let me address your question with what I know..."
            ],
            "general": [
                "I'm here to help with information and answers to your questions.",
                "I can assist with a wide range of topics and inquiries.",
                "Feel free to ask me about anything you'd like to know.",
                "I'm designed to be helpful and informative."
            ]
        }
    
    def generate_fallback_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate fallback responses when models aren't available"""
        
        if not messages:
            category = "greeting"
        else:
            last_message = messages[-1].get("content", "").lower()
            if any(word in last_message for word in ["hello", "hi", "hey", "greetings"]):
                category = "greeting"
            elif "?" in last_message:
                category = "question"
            else:
                category = "general"
        
        base_response = random.choice(self.fallback_responses[category])
        finetuned_response = f"[Fine-tuned] {base_response} (This is a demonstration response as model loading is currently unavailable.)"
        
        return {
            "base_response": base_response,
            "finetuned_response": finetuned_response,
            "generation_time_ms": {"base": 10.0, "finetuned": 12.0, "total": 22.0},
            "model_info": {
                "base_model": "Fallback Service",
                "adapter_name": "demo",
                "pair_id": "fallback_demo",
                "status": "fallback_mode"
            },
            "cache_metrics": {"cache_hits": 0, "cache_misses": 1}
        }
