# Comparison Chat Service with FastLibra KV Cache Optimization
# High-performance dual model inference service

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from pydantic import BaseModel
from fastapi import HTTPException

from dual_model_manager import DualModelManager, ModelPair


class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[str] = None


class ComparisonRequest(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    generation_params: Optional[Dict[str, Any]] = None


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


class SessionInfo(BaseModel):
    session_id: str
    session_name: str
    pair_id: str
    model_info: Dict[str, str]
    created_at: str
    last_activity: str
    message_count: int
    status: str  # "active", "inactive", "error"


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


class ComparisonChatService:
    """High-performance chat service with FastLibra KV cache optimization"""
    
    def __init__(self, max_hbm_gb: float = 12.0, max_sessions: int = 20):
        self.dual_model_manager = DualModelManager(max_hbm_gb=max_hbm_gb)
        self.max_sessions = max_sessions
        
        # Session management
        self.active_sessions: Dict[str, ChatSession] = {}
        self.session_history: Dict[str, List[Dict]] = {}  # session_id -> message history
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time_ms": 0.0,
            "active_sessions_count": 0,
            "cache_hit_rate": 0.0
        }
        
        # Persistence
        self.sessions_file = Path("logs/chat_sessions.json")
        self.sessions_file.parent.mkdir(exist_ok=True)
        
        # Start background tasks
        asyncio.create_task(self._periodic_optimization())
        
        print(f"ComparisonChatService initialized with {max_hbm_gb}GB memory limit")
        print(f"Maximum concurrent sessions: {max_sessions}")
    
    async def get_available_model_pairs(self) -> List[Dict[str, Any]]:
        """Get all available base model + LoRA adapter pairs"""
        try:
            pairs = await self.dual_model_manager.discover_model_pairs()
            
            return [
                {
                    "pair_id": pair.pair_id,
                    "base_model_name": pair.base_model_name,
                    "adapter_name": pair.adapter_name,
                    "base_model_size_mb": pair.base_model_size_mb,
                    "adapter_size_mb": pair.adapter_size_mb,
                    "estimated_memory_mb": pair.base_model_size_mb + pair.adapter_size_mb,
                    "status": pair.status,
                    "load_time_ms": pair.load_time_ms
                }
                for pair in pairs
            ]
            
        except Exception as e:
            print(f"Error discovering model pairs: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to discover model pairs: {str(e)}")
    
    async def create_session(self, request: SessionCreateRequest) -> SessionInfo:
        """Create a new comparison chat session"""
        try:
            # Check session limit
            if len(self.active_sessions) >= self.max_sessions:
                # Clean up inactive sessions
                await self._cleanup_inactive_sessions()
                
                if len(self.active_sessions) >= self.max_sessions:
                    raise HTTPException(
                        status_code=429, 
                        detail=f"Maximum number of sessions ({self.max_sessions}) reached"
                    )
            
            # Get available pairs and find requested pair
            available_pairs = await self.dual_model_manager.discover_model_pairs()
            model_pair = next((p for p in available_pairs if p.pair_id == request.pair_id), None)
            
            if not model_pair:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Model pair '{request.pair_id}' not found"
                )
            
            # Create session
            session_id = str(uuid.uuid4())
            session_name = request.session_name or f"Chat_{session_id[:8]}"
            
            session = ChatSession(
                session_id=session_id,
                session_name=session_name,
                pair_id=request.pair_id,
                model_pair=model_pair
            )
            
            # Store session
            self.active_sessions[session_id] = session
            self.session_history[session_id] = []
            
            # Update metrics
            self.metrics["active_sessions_count"] = len(self.active_sessions)
            
            # Save to persistence
            await self._save_sessions()
            
            print(f"Created session {session_id} with pair {request.pair_id}")
            
            return SessionInfo(
                session_id=session_id,
                session_name=session_name,
                pair_id=request.pair_id,
                model_info={
                    "base_model": model_pair.base_model_name,
                    "adapter_name": model_pair.adapter_name
                },
                created_at=session.created_at,
                last_activity=session.last_activity,
                message_count=0,
                status="active"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"Error creating session: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")
    
    async def generate_comparison_stream(self, request: ComparisonRequest):
        """Stream comparison responses from both base and fine-tuned models"""
        start_time = time.time()
        
        try:
            # Update metrics
            self.metrics["total_requests"] += 1
            
            # Validate session
            if request.session_id not in self.active_sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session = self.active_sessions[request.session_id]
            
            # Convert Pydantic messages to dict format
            messages = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp or datetime.now().isoformat()
                }
                for msg in request.messages
            ]
            
            # Yield initial status
            yield {
                "type": "status",
                "data": {"status": "starting", "session_id": request.session_id}
            }
            
            # Start both model generations concurrently with progress updates
            yield {
                "type": "progress",
                "data": {"phase": "generating", "models": ["base", "finetuned"]}
            }
            
            # Generate comparison responses (simplified for streaming demo)
            result = await self.dual_model_manager.generate_comparison_responses(
                session_id=request.session_id,
                messages=messages,
                generation_params=request.generation_params
            )
            
            # Stream base response
            yield {
                "type": "response",
                "data": {
                    "model_type": "base",
                    "content": result["base_response"],
                    "generation_time_ms": result["generation_time_ms"]["base"]
                }
            }
            
            # Stream finetuned response  
            yield {
                "type": "response",
                "data": {
                    "model_type": "finetuned", 
                    "content": result["finetuned_response"],
                    "generation_time_ms": result["generation_time_ms"]["finetuned"]
                }
            }
            
            # Create assistant messages for session history
            base_assistant_msg = {
                "role": "assistant", 
                "content": result["base_response"],
                "timestamp": datetime.now().isoformat(),
                "model_type": "base"
            }
            
            finetuned_assistant_msg = {
                "role": "assistant",
                "content": result["finetuned_response"], 
                "timestamp": datetime.now().isoformat(),
                "model_type": "finetuned"
            }
            
            # Update session history
            session.messages.extend(messages)
            session.messages.extend([base_assistant_msg, finetuned_assistant_msg])
            session.last_activity = datetime.now().isoformat()
            
            # Update session history storage
            if request.session_id not in self.session_history:
                self.session_history[request.session_id] = []
            
            self.session_history[request.session_id].extend(messages)
            self.session_history[request.session_id].extend([base_assistant_msg, finetuned_assistant_msg])
            
            # Update metrics
            total_time_ms = (time.time() - start_time) * 1000
            self.metrics["successful_requests"] += 1
            self.metrics["average_response_time_ms"] = (
                (self.metrics["average_response_time_ms"] * (self.metrics["successful_requests"] - 1) + 
                 total_time_ms) / self.metrics["successful_requests"]
            )
            
            # Final completion status
            yield {
                "type": "complete",
                "data": {
                    "session_id": request.session_id,
                    "total_time_ms": total_time_ms,
                    "model_info": result["model_info"],
                    "cache_metrics": result["cache_metrics"],
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except HTTPException:
            self.metrics["failed_requests"] += 1
            raise
        except Exception as e:
            self.metrics["failed_requests"] += 1
            print(f"Error generating comparison stream: {e}")
            yield {
                "type": "error",
                "data": {"error": str(e), "session_id": request.session_id}
            }

    async def generate_comparison(self, request: ComparisonRequest) -> ComparisonResponse:
        """Generate comparison responses from both base and fine-tuned models"""
        start_time = time.time()
        
        try:
            # Update metrics
            self.metrics["total_requests"] += 1
            
            # Validate session
            if request.session_id not in self.active_sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session = self.active_sessions[request.session_id]
            
            # Convert Pydantic messages to dict format
            messages = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp or datetime.now().isoformat()
                }
                for msg in request.messages
            ]
            
            # Generate comparison responses using FastLibra optimization
            result = await self.dual_model_manager.generate_comparison_responses(
                session_id=request.session_id,
                messages=messages,
                generation_params=request.generation_params
            )
            
            # Create response with assistant messages
            base_assistant_msg = {
                "role": "assistant", 
                "content": result["base_response"],
                "timestamp": datetime.now().isoformat(),
                "model_type": "base"
            }
            
            finetuned_assistant_msg = {
                "role": "assistant",
                "content": result["finetuned_response"], 
                "timestamp": datetime.now().isoformat(),
                "model_type": "finetuned"
            }
            
            # Update session history
            session.messages.extend(messages)
            session.messages.extend([base_assistant_msg, finetuned_assistant_msg])
            session.last_activity = datetime.now().isoformat()
            
            # Update session history storage
            if request.session_id not in self.session_history:
                self.session_history[request.session_id] = []
            
            self.session_history[request.session_id].extend(messages)
            self.session_history[request.session_id].extend([base_assistant_msg, finetuned_assistant_msg])
            
            # Update metrics
            total_time_ms = (time.time() - start_time) * 1000
            self.metrics["successful_requests"] += 1
            self.metrics["average_response_time_ms"] = (
                (self.metrics["average_response_time_ms"] * (self.metrics["successful_requests"] - 1) + 
                 total_time_ms) / self.metrics["successful_requests"]
            )
            
            # Save sessions periodically
            if self.metrics["total_requests"] % 10 == 0:
                await self._save_sessions()
            
            print(f"Generated comparison for session {request.session_id} in {total_time_ms:.1f}ms")
            
            return ComparisonResponse(
                base_response=result["base_response"],
                finetuned_response=result["finetuned_response"],
                generation_time_ms=result["generation_time_ms"],
                model_info=result["model_info"],
                cache_metrics=result["cache_metrics"],
                session_id=request.session_id,
                timestamp=datetime.now().isoformat()
            )
            
        except HTTPException:
            self.metrics["failed_requests"] += 1
            raise
        except Exception as e:
            self.metrics["failed_requests"] += 1
            print(f"Error generating comparison: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate comparison: {str(e)}")
    
    async def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Get complete conversation history for a session"""
        try:
            if session_id not in self.active_sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session = self.active_sessions[session_id]
            messages = self.session_history.get(session_id, [])
            
            return {
                "session_info": session.to_dict(),
                "messages": messages,
                "total_messages": len(messages),
                "last_activity": session.last_activity
            }
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"Error getting session history: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get session history: {str(e)}")
    
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
            print(f"Error listing sessions: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")
    
    async def delete_session(self, session_id: str) -> Dict[str, str]:
        """Delete a chat session and clean up resources"""
        try:
            if session_id not in self.active_sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Clean up model manager resources
            await self.dual_model_manager.cleanup_session(session_id)
            
            # Remove session data
            del self.active_sessions[session_id]
            if session_id in self.session_history:
                del self.session_history[session_id]
            
            # Update metrics
            self.metrics["active_sessions_count"] = len(self.active_sessions)
            
            # Save sessions
            await self._save_sessions()
            
            print(f"Deleted session {session_id}")
            
            return {"message": f"Session {session_id} deleted successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"Error deleting session: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service performance metrics"""
        try:
            # Get memory stats from dual model manager
            memory_stats = self.dual_model_manager.get_memory_stats()
            
            # Combine with service metrics
            return {
                "service_metrics": self.metrics,
                "memory_stats": memory_stats,
                "active_sessions": len(self.active_sessions),
                "total_message_history": sum(len(history) for history in self.session_history.values()),
                "uptime_seconds": time.time() - getattr(self, 'start_time', time.time()),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")
    
    async def _cleanup_inactive_sessions(self):
        """Clean up inactive sessions to free resources"""
        current_time = time.time()
        inactive_threshold = 3600  # 1 hour
        
        inactive_sessions = []
        
        for session_id, session in self.active_sessions.items():
            last_activity = datetime.fromisoformat(session.last_activity).timestamp()
            if current_time - last_activity > inactive_threshold:
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            print(f"Cleaning up inactive session: {session_id}")
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
            print(f"Warning: Failed to save sessions: {e}")
    
    async def _periodic_optimization(self):
        """Background task for periodic memory optimization"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Run memory optimization
                await self.dual_model_manager.optimize_memory()
                
                # Clean up inactive sessions
                await self._cleanup_inactive_sessions()
                
                # Update cache hit rate metric
                memory_stats = self.dual_model_manager.get_memory_stats()
                self.metrics["cache_hit_rate"] = memory_stats.get("cache_hit_rate", 0.0)
                
                # Save sessions
                await self._save_sessions()
                
                print(f"Periodic optimization completed. Active sessions: {len(self.active_sessions)}")
                
            except Exception as e:
                print(f"Warning: Error in periodic optimization: {e}")


# Global service instance
comparison_chat_service = ComparisonChatService()