"""
Chat Controller

Handles comparison chat endpoints with streaming support.
"""

import json
from typing import Dict, Any, List
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from models.schemas import ComparisonRequest, SessionCreateRequest, ChatMessage
from services.comparison_service import ComparisonService
from utils.logging import get_logger


class ChatController:
    """Controller for comparison chat endpoints"""
    
    def __init__(self, comparison_service: ComparisonService = None):
        self.logger = get_logger(__name__)
        self.comparison_service = comparison_service or ComparisonService()
    
    async def get_model_pairs(self) -> JSONResponse:
        """Get all available base model + LoRA adapter pairs for comparison"""
        try:
            pairs = await self.comparison_service.get_available_model_pairs()
            return JSONResponse(content={
                "status": "success",
                "model_pairs": pairs,
                "total": len(pairs)
            })
        except Exception as e:
            self.logger.error(f"Failed to get model pairs: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get model pairs: {str(e)}")
    
    async def create_chat_session(self, request: SessionCreateRequest) -> JSONResponse:
        """Create a new comparison chat session"""
        try:
            session_info = await self.comparison_service.create_session(request)
            return JSONResponse(content={
                "status": "success",
                "session": session_info.dict()
            })
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")
    
    async def list_chat_sessions(self) -> JSONResponse:
        """List all active comparison chat sessions"""
        try:
            sessions = await self.comparison_service.list_active_sessions()
            return JSONResponse(content={
                "status": "success",
                "sessions": [session.dict() for session in sessions],
                "total": len(sessions)
            })
        except Exception as e:
            self.logger.error(f"Failed to list sessions: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")
    
    async def get_chat_session_history(self, session_id: str) -> JSONResponse:
        """Get conversation history for a specific chat session"""
        try:
            history = await self.comparison_service.get_session_history(session_id)
            return JSONResponse(content={
                "status": "success",
                **history
            })
        except Exception as e:
            self.logger.error(f"Failed to get session history: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get session history: {str(e)}")
    
    async def generate_comparison_response(self, request: ComparisonRequest) -> JSONResponse:
        """Generate comparison responses from both base and fine-tuned models"""
        try:
            response = await self.comparison_service.generate_comparison(request)
            return JSONResponse(content={
                "status": "success",
                "response": response.dict()
            })
        except Exception as e:
            self.logger.error(f"Failed to generate comparison: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate comparison: {str(e)}")
    
    async def generate_comparison_stream(self, request: ComparisonRequest) -> StreamingResponse:
        """Stream comparison responses from both base and fine-tuned models"""
        
        async def generate_stream():
            try:
                async for chunk in self.comparison_service.generate_comparison_stream(request):
                    yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                error_chunk = {
                    "type": "error",
                    "data": {"error": str(e)}
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            }
        )
    
    async def delete_chat_session(self, session_id: str) -> JSONResponse:
        """Delete a comparison chat session and clean up resources"""
        try:
            result = await self.comparison_service.delete_session(session_id)
            return JSONResponse(content={
                "status": "success",
                **result
            })
        except Exception as e:
            self.logger.error(f"Failed to delete session: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")
    
    async def get_chat_service_metrics(self) -> JSONResponse:
        """Get comprehensive chat service performance metrics"""
        try:
            metrics = await self.comparison_service.get_service_metrics()
            return JSONResponse(content={
                "status": "success",
                **metrics
            })
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


# Create global controller instance
chat_controller = ChatController()