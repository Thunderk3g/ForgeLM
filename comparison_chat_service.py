"""
Backward Compatibility Wrapper for comparison_chat_service.py

This file maintains the original API surface while delegating to the new modular services.
Existing code that imports from comparison_chat_service should continue to work unchanged.
"""

# Import all the original classes and functions from their new locations
from models.schemas import ChatMessage, ComparisonRequest, ComparisonResponse, SessionCreateRequest, SessionInfo
from models.entities import ChatSession, ModelPair
from services.comparison_service import ComparisonService
from services.chat_service import ChatService
from services.model_service import ModelService

# Create the original ComparisonChatService class for backward compatibility
class ComparisonChatService:
    """
    Backward compatibility wrapper for the original ComparisonChatService.
    
    This class delegates all method calls to the new modular services while
    maintaining the exact same API surface area.
    """
    
    def __init__(self, max_hbm_gb: float = 12.0, max_sessions: int = 20):
        # Initialize the new modular services
        self.model_service = ModelService(max_hbm_gb=max_hbm_gb)
        self.chat_service = ChatService()
        self.comparison_service = ComparisonService(self.model_service, self.chat_service)
        
        # Store original parameters for compatibility
        self.max_hbm_gb = max_hbm_gb
        self.max_sessions = max_sessions
        
        # Maintain reference to the dual model manager for backward compatibility
        self.dual_model_manager = self.model_service
        
        print(f"ComparisonChatService initialized with {max_hbm_gb}GB memory limit (modular)")
        print(f"Maximum concurrent sessions: {max_sessions}")
    
    async def start_background_tasks(self):
        """Start background tasks when event loop is available"""
        return await self.comparison_service.start_background_tasks()
    
    async def get_available_model_pairs(self):
        """Get all available base model + LoRA adapter pairs"""
        return await self.comparison_service.get_available_model_pairs()
    
    async def create_session(self, request):
        """Create a new comparison chat session"""
        return await self.comparison_service.create_session(request)
    
    async def generate_comparison_stream(self, request):
        """Stream comparison responses from both base and fine-tuned models"""
        async for chunk in self.comparison_service.generate_comparison_stream(request):
            yield chunk
    
    async def generate_comparison(self, request):
        """Generate comparison responses from both base and fine-tuned models"""
        return await self.comparison_service.generate_comparison(request)
    
    async def get_session_history(self, session_id):
        """Get complete conversation history for a session"""
        return await self.comparison_service.get_session_history(session_id)
    
    async def list_active_sessions(self):
        """List all active chat sessions"""
        return await self.comparison_service.list_active_sessions()
    
    async def delete_session(self, session_id):
        """Delete a chat session and clean up resources"""
        return await self.comparison_service.delete_session(session_id)
    
    async def get_service_metrics(self):
        """Get comprehensive service performance metrics"""
        return await self.comparison_service.get_service_metrics()


# Create the global service instance exactly as in the original
comparison_chat_service = ComparisonChatService()

# Export all the classes and functions that external code might import
__all__ = [
    'ComparisonChatService',
    'ChatMessage',
    'ComparisonRequest', 
    'ComparisonResponse',
    'SessionCreateRequest',
    'SessionInfo',
    'ChatSession',
    'ModelPair',
    'comparison_chat_service'
]