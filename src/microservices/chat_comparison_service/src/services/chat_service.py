"""
Chat Service 

Handles individual chat sessions and conversation management.
Extracted from comparison_chat_service.py while preserving all functionality.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path

from models.schemas import ChatMessage, SessionCreateRequest, SessionInfo
from models.entities import ChatSession, ModelPair
from utils.logging import get_logger
from utils.performance import performance_monitor, timing_context, timed
from config.settings import settings


class ChatService:
    """Individual chat session management service"""
    
    def __init__(self, model_service=None):
        self.logger = get_logger(__name__)
        self.model_service = model_service
        
        # Session management
        self.active_sessions: Dict[str, ChatSession] = {}
        self.session_history: Dict[str, List[Dict]] = {}  # session_id -> message history
        
        # Performance metrics
        self.metrics = {
            "total_sessions_created": 0,
            "active_sessions_count": 0,
            "total_messages_processed": 0,
            "average_response_time_ms": 0.0,
        }
        
        # Persistence
        self.sessions_file = Path(settings.logs_dir) / "chat_sessions.json"
        self.sessions_file.parent.mkdir(exist_ok=True)
        
        self.logger.info(f"ChatService initialized")
    
    @timed("create_session")
    async def create_session(self, pair_id: str, session_name: Optional[str] = None) -> SessionInfo:
        """Create a new chat session"""
        try:
            # Check session limit
            if len(self.active_sessions) >= settings.max_sessions:
                # Clean up inactive sessions
                await self._cleanup_inactive_sessions()
                
                if len(self.active_sessions) >= settings.max_sessions:
                    raise ValueError(f"Maximum number of sessions ({settings.max_sessions}) reached")
            
            # Create session ID and name
            session_id = str(uuid.uuid4())
            session_name = session_name or f"Chat_{session_id[:8]}"
            
            # Get real model pair from ModelService
            if self.model_service is None:
                raise ValueError("ModelService not available - ChatService not properly initialized")
            
            available_pairs = await self.model_service.discover_model_pairs()
            
            # Find the requested model pair or use the first available one
            model_pair = None
            for pair in available_pairs:
                if pair.pair_id == pair_id:
                    model_pair = pair
                    break
            
            # If specific pair not found, use first available pair
            if model_pair is None and available_pairs:
                model_pair = available_pairs[0]
                self.logger.warning(f"Model pair {pair_id} not found, using {model_pair.pair_id}")
            
            # If no pairs available, create a fallback
            if model_pair is None:
                self.logger.error("No model pairs available, creating fallback")
                model_pair = ModelPair(
                    pair_id=pair_id,
                    base_model_name="fallback_base",
                    base_model_path="models/base/Qwen_Qwen2.5-3B",
                    adapter_name="fallback_adapter", 
                    adapter_path="models/finetuned",
                    base_model_size_mb=1000.0,
                    adapter_size_mb=100.0
                )
            
            session = ChatSession(
                session_id=session_id,
                session_name=session_name,
                pair_id=pair_id,
                model_pair=model_pair
            )
            
            # Store session
            self.active_sessions[session_id] = session
            self.session_history[session_id] = []
            
            # Update metrics
            self.metrics["total_sessions_created"] += 1
            self.metrics["active_sessions_count"] = len(self.active_sessions)
            
            # Save to persistence
            await self._save_sessions()
            
            self.logger.info(f"Created chat session {session_id} with pair {pair_id}")
            
            return SessionInfo(
                session_id=session_id,
                session_name=session_name,
                pair_id=pair_id,
                model_info={
                    "base_model": model_pair.base_model_name,
                    "adapter_name": model_pair.adapter_name
                },
                created_at=session.created_at,
                last_activity=session.last_activity,
                message_count=0,
                status="active"
            )
            
        except Exception as e:
            self.logger.error(f"Error creating chat session: {e}")
            raise
    
    async def add_message(self, session_id: str, message: ChatMessage) -> bool:
        """Add a message to a chat session"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            # Convert to dict format
            message_dict = {
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp or datetime.now().isoformat()
            }
            
            # Add to session
            session.messages.append(message_dict)
            session.last_activity = datetime.now().isoformat()
            
            # Add to history storage
            if session_id not in self.session_history:
                self.session_history[session_id] = []
            
            self.session_history[session_id].append(message_dict)
            
            # Update metrics
            self.metrics["total_messages_processed"] += 1
            
            self.logger.info(f"Added message to session {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding message to session {session_id}: {e}")
            return False
    
    async def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Get complete conversation history for a session"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            messages = self.session_history.get(session_id, [])
            
            return {
                "session_info": session.to_dict(),
                "messages": messages,
                "total_messages": len(messages),
                "last_activity": session.last_activity
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session history: {e}")
            raise
    
    async def list_active_sessions(self) -> List[SessionInfo]:
        """List all active chat sessions"""
        try:
            sessions = []
            
            for session in self.active_sessions.values():
                sessions.append(SessionInfo(
                    session_id=session.session_id,
                    session_name=session.session_name,
                    pair_id=session.pair_id,
                    model_info={
                        "base_model": session.model_pair.base_model_name,
                        "adapter_name": session.model_pair.adapter_name
                    },
                    created_at=session.created_at,
                    last_activity=session.last_activity,
                    message_count=len(session.messages),
                    status=session.status
                ))
            
            return sessions
            
        except Exception as e:
            self.logger.error(f"Error listing sessions: {e}")
            raise
    
    async def delete_session(self, session_id: str) -> Dict[str, str]:
        """Delete a chat session and clean up resources"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            # Remove session data
            del self.active_sessions[session_id]
            if session_id in self.session_history:
                del self.session_history[session_id]
            
            # Update metrics
            self.metrics["active_sessions_count"] = len(self.active_sessions)
            
            # Save sessions
            await self._save_sessions()
            
            self.logger.info(f"Deleted chat session {session_id}")
            
            return {"message": f"Session {session_id} deleted successfully"}
            
        except Exception as e:
            self.logger.error(f"Error deleting session: {e}")
            raise
    
    async def get_session_info(self, session_id: str) -> SessionInfo:
        """Get information about a specific session"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            
            return SessionInfo(
                session_id=session.session_id,
                session_name=session.session_name,
                pair_id=session.pair_id,
                model_info={
                    "base_model": session.model_pair.base_model_name,
                    "adapter_name": session.model_pair.adapter_name
                },
                created_at=session.created_at,
                last_activity=session.last_activity,
                message_count=len(session.messages),
                status=session.status
            )
            
        except Exception as e:
            self.logger.error(f"Error getting session info: {e}")
            raise
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics"""
        return {
            "chat_service_metrics": self.metrics,
            "active_sessions": len(self.active_sessions),
            "total_message_history": sum(len(history) for history in self.session_history.values()),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _cleanup_inactive_sessions(self):
        """Clean up inactive sessions to free resources"""
        current_time = time.time()
        inactive_threshold = settings.inactive_session_threshold  # from config
        
        inactive_sessions = []
        
        for session_id, session in self.active_sessions.items():
            last_activity = datetime.fromisoformat(session.last_activity).timestamp()
            if current_time - last_activity > inactive_threshold:
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            self.logger.info(f"Cleaning up inactive session: {session_id}")
            await self.delete_session(session_id)
    
    async def _save_sessions(self):
        """Save active sessions to persistent storage"""
        try:
            sessions_data = {
                "sessions": {
                    session_id: session.to_dict()
                    for session_id, session in self.active_sessions.items()
                },
                "metrics": self.metrics,
                "last_saved": datetime.now().isoformat()
            }
            
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions_data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to save sessions: {e}")


# Note: Global service instance should be created with model_service dependency
# chat_service = ChatService()  # Commented out - should be initialized in app.py with dependencies