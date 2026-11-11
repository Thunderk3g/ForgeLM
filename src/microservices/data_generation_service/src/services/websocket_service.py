"""
WebSocket service for real-time data preview sessions
"""

import asyncio
import json
from typing import Dict, Any
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol
from utils.logging import get_logger

logger = get_logger(__name__)


class PreprocessingWebSocketService:
    """WebSocket service for real-time preprocessing updates"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketServerProtocol] = {}
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


# Global WebSocket service instance
websocket_service = PreprocessingWebSocketService()


def get_websocket_service() -> PreprocessingWebSocketService:
    """Get the global WebSocket service instance"""
    return websocket_service