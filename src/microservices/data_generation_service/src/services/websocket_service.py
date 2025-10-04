"""
WebSocket service for real-time preprocessing updates and previews
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol
from utils.logging import get_logger

logger = get_logger(__name__)


class PreprocessingWebSocketService:
    """WebSocket service for real-time preprocessing updates"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.preprocessing_sessions: Dict[str, Dict[str, Any]] = {}
        self.preview_sessions: Dict[str, Dict[str, Any]] = {}
        
    async def register_connection(self, websocket: WebSocketServerProtocol, session_id: str):
        """Register a new WebSocket connection"""
        self.connections[session_id] = websocket
        logger.info(f"WebSocket connection registered: {session_id}")
        
        # Send welcome message
        await self.send_message(session_id, {
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
    
    async def unregister_connection(self, session_id: str):
        """Unregister a WebSocket connection"""
        if session_id in self.connections:
            del self.connections[session_id]
            logger.info(f"WebSocket connection unregistered: {session_id}")
        
        # Clean up sessions
        if session_id in self.preprocessing_sessions:
            del self.preprocessing_sessions[session_id]
        if session_id in self.preview_sessions:
            del self.preview_sessions[session_id]
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send a message to a specific session"""
        if session_id in self.connections:
            try:
                await self.connections[session_id].send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                await self.unregister_connection(session_id)
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")
    
    async def broadcast_message(self, message: Dict[str, Any], session_ids: Optional[List[str]] = None):
        """Broadcast a message to multiple sessions"""
        target_sessions = session_ids or list(self.connections.keys())
        
        for session_id in target_sessions:
            await self.send_message(session_id, message)
    
    async def start_preprocessing_session(self, session_id: str, config: Dict[str, Any]):
        """Start a preprocessing session with real-time updates"""
        self.preprocessing_sessions[session_id] = {
            "status": "starting",
            "config": config,
            "start_time": datetime.now(),
            "progress": 0,
            "current_step": "initializing",
            "steps_completed": [],
            "errors": []
        }
        
        await self.send_message(session_id, {
            "type": "preprocessing_started",
            "session_id": session_id,
            "config": config,
            "timestamp": datetime.now().isoformat()
        })
    
    async def update_preprocessing_progress(self, session_id: str, progress: int, current_step: str, 
                                          step_data: Optional[Dict[str, Any]] = None):
        """Update preprocessing progress"""
        if session_id in self.preprocessing_sessions:
            session = self.preprocessing_sessions[session_id]
            session["progress"] = progress
            session["current_step"] = current_step
            session["last_update"] = datetime.now()
            
            if step_data and current_step not in session["steps_completed"]:
                session["steps_completed"].append(current_step)
            
            await self.send_message(session_id, {
                "type": "preprocessing_progress",
                "session_id": session_id,
                "progress": progress,
                "current_step": current_step,
                "step_data": step_data,
                "timestamp": datetime.now().isoformat()
            })
    
    async def send_preprocessing_analytics(self, session_id: str, analytics: Dict[str, Any]):
        """Send real-time analytics updates"""
        await self.send_message(session_id, {
            "type": "preprocessing_analytics",
            "session_id": session_id,
            "analytics": analytics,
            "timestamp": datetime.now().isoformat()
        })
    
    async def complete_preprocessing_session(self, session_id: str, result: Dict[str, Any]):
        """Complete a preprocessing session"""
        if session_id in self.preprocessing_sessions:
            session = self.preprocessing_sessions[session_id]
            session["status"] = "completed"
            session["end_time"] = datetime.now()
            session["result"] = result
            
            await self.send_message(session_id, {
                "type": "preprocessing_completed",
                "session_id": session_id,
                "result": result,
                "duration_ms": (session["end_time"] - session["start_time"]).total_seconds() * 1000,
                "timestamp": datetime.now().isoformat()
            })
    
    async def send_preprocessing_error(self, session_id: str, error: str, step: Optional[str] = None):
        """Send preprocessing error"""
        if session_id in self.preprocessing_sessions:
            session = self.preprocessing_sessions[session_id]
            session["status"] = "error"
            session["errors"].append({
                "error": error,
                "step": step,
                "timestamp": datetime.now().isoformat()
            })
            
            await self.send_message(session_id, {
                "type": "preprocessing_error",
                "session_id": session_id,
                "error": error,
                "step": step,
                "timestamp": datetime.now().isoformat()
            })
    
    async def start_preview_session(self, session_id: str, preview_config: Dict[str, Any]):
        """Start a data preview session"""
        self.preview_sessions[session_id] = {
            "status": "generating",
            "config": preview_config,
            "start_time": datetime.now(),
            "samples_processed": 0,
            "total_samples": preview_config.get("sample_size", 10)
        }
        
        await self.send_message(session_id, {
            "type": "preview_started",
            "session_id": session_id,
            "config": preview_config,
            "timestamp": datetime.now().isoformat()
        })
    
    async def send_preview_sample(self, session_id: str, sample_data: Dict[str, Any], sample_index: int):
        """Send a preview sample"""
        if session_id in self.preview_sessions:
            session = self.preview_sessions[session_id]
            session["samples_processed"] = sample_index + 1
            
            await self.send_message(session_id, {
                "type": "preview_sample",
                "session_id": session_id,
                "sample_data": sample_data,
                "sample_index": sample_index,
                "progress": (sample_index + 1) / session["total_samples"] * 100,
                "timestamp": datetime.now().isoformat()
            })
    
    async def complete_preview_session(self, session_id: str, summary: Dict[str, Any]):
        """Complete a preview session"""
        if session_id in self.preview_sessions:
            session = self.preview_sessions[session_id]
            session["status"] = "completed"
            session["end_time"] = datetime.now()
            
            await self.send_message(session_id, {
                "type": "preview_completed",
                "session_id": session_id,
                "summary": summary,
                "samples_processed": session["samples_processed"],
                "timestamp": datetime.now().isoformat()
            })
    
    async def send_embedding_progress(self, session_id: str, progress_data: Dict[str, Any]):
        """Send embedding generation progress"""
        await self.send_message(session_id, {
            "type": "embedding_progress",
            "session_id": session_id,
            "progress_data": progress_data,
            "timestamp": datetime.now().isoformat()
        })
    
    async def send_visualization_update(self, session_id: str, visualization_data: Dict[str, Any]):
        """Send real-time visualization updates"""
        await self.send_message(session_id, {
            "type": "visualization_update",
            "session_id": session_id,
            "visualization_data": visualization_data,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a session"""
        if session_id in self.preprocessing_sessions:
            return self.preprocessing_sessions[session_id]
        elif session_id in self.preview_sessions:
            return self.preview_sessions[session_id]
        return None
    
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions"""
        return {
            "preprocessing": self.preprocessing_sessions,
            "preview": self.preview_sessions,
            "total_connections": len(self.connections)
        }


# Global WebSocket service instance
websocket_service = PreprocessingWebSocketService()


async def handle_websocket_connection(websocket: WebSocketServerProtocol, path: str):
    """Handle WebSocket connection"""
    session_id = str(uuid.uuid4())
    
    try:
        await websocket_service.register_connection(websocket, session_id)
        
        async for message in websocket:
            try:
                data = json.loads(message)
                await handle_websocket_message(session_id, data)
            except json.JSONDecodeError:
                await websocket_service.send_message(session_id, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket_service.send_message(session_id, {
                    "type": "error",
                    "message": str(e)
                })
    
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        await websocket_service.unregister_connection(session_id)


async def handle_websocket_message(session_id: str, data: Dict[str, Any]):
    """Handle incoming WebSocket messages"""
    message_type = data.get("type")
    
    if message_type == "ping":
        await websocket_service.send_message(session_id, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })
    
    elif message_type == "get_session_status":
        status = websocket_service.get_session_status(session_id)
        await websocket_service.send_message(session_id, {
            "type": "session_status",
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
    
    elif message_type == "subscribe_to_updates":
        # Client can subscribe to specific types of updates
        update_types = data.get("update_types", [])
        await websocket_service.send_message(session_id, {
            "type": "subscription_confirmed",
            "update_types": update_types,
            "timestamp": datetime.now().isoformat()
        })
    
    else:
        await websocket_service.send_message(session_id, {
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        })


def get_websocket_service() -> PreprocessingWebSocketService:
    """Get the global WebSocket service instance"""
    return websocket_service