"""
Comparison Service

Handles dual model inference and comparison functionality with FastLibra optimization.
Extracted from comparison_chat_service.py while preserving all functionality.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models.schemas import (
    ComparisonRequest, ComparisonResponse, ChatMessage, SessionCreateRequest, SessionInfo,
    EnhancedComparisonRequest, DualModelConfig, GenerationConfig
)
from ..models.entities import ChatSession, ModelPair
from .model_service import ModelService
from .chat_service import ChatService
from ..utils.logging import get_logger
from ..utils.performance import performance_monitor, timing_context, timed
from ..utils.research_logger import research_logger
from ..config.settings import settings


class ComparisonService:
    """High-performance comparison service with FastLibra KV cache optimization"""
    
    def __init__(self, model_service: ModelService = None, chat_service: ChatService = None):
        self.logger = get_logger(__name__)
        
        # Inject dependencies
        self.model_service = model_service or ModelService()
        self.chat_service = chat_service or ChatService(self.model_service)
        
        # Performance metrics
        self.metrics = {
            "total_comparisons": 0,
            "successful_comparisons": 0,
            "failed_comparisons": 0,
            "average_comparison_time_ms": 0.0,
        }
        
        # Start background optimization if configured
        if settings.background_optimization:
            try:
                asyncio.create_task(self._periodic_optimization())
            except RuntimeError:
                # No event loop running, skip background task for now
                self.logger.info("Skipping background optimization task (no event loop)")
        
        self.logger.info(f"ComparisonService initialized")
        self.logger.info(f"Maximum concurrent sessions: {settings.max_sessions}")
    
    async def start_background_tasks(self):
        """Start background tasks when event loop is available"""
        try:
            if settings.background_optimization:
                # Start periodic optimization
                asyncio.create_task(self._periodic_optimization())
                self.logger.info("Background optimization task started")
            
            # Model warmup disabled to save GPU memory - models will be loaded on-demand
            # asyncio.create_task(self._warmup_models())
            self.logger.info("Model warmup disabled - models will be loaded when selected")
            
        except Exception as e:
            self.logger.error(f"Failed to start background tasks: {e}")
    
    async def get_available_model_pairs(self) -> List[Dict[str, Any]]:
        """Get all available base model + LoRA adapter pairs"""
        try:
            pairs = await self.model_service.discover_model_pairs()
            
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
            self.logger.error(f"Error discovering model pairs: {e}")
            raise
    
    @timed("create_comparison_session")
    async def create_session(self, request: SessionCreateRequest) -> SessionInfo:
        """Create a new comparison chat session"""
        try:
            # Get available pairs and validate requested pair
            available_pairs = await self.model_service.discover_model_pairs()
            model_pair = next((p for p in available_pairs if p.pair_id == request.pair_id), None)
            
            if not model_pair:
                raise ValueError(f"Model pair '{request.pair_id}' not found")
            
            # Create session using chat service
            session_info = await self.chat_service.create_session(
                pair_id=request.pair_id,
                session_name=request.session_name
            )
            
            # Load the model pair for this session to make it available for inference
            try:
                await self.model_service.load_model_pair(model_pair, session_info.session_id)
                self.logger.info(f"Model pair {request.pair_id} loaded for session {session_info.session_id}")
            except Exception as model_load_error:
                self.logger.error(f"Failed to load model pair for session {session_info.session_id}: {model_load_error}")
                # Don't fail session creation, but log the error
                # The generate_comparison method will handle the missing models gracefully
            
            self.logger.info(f"Created comparison session {session_info.session_id} with pair {request.pair_id}")
            
            return session_info
            
        except Exception as e:
            self.logger.error(f"Error creating comparison session: {e}")
            raise
    
    @timed("generate_comparison")
    async def generate_comparison(self, request: ComparisonRequest) -> ComparisonResponse:
        """Generate comparison responses from both base and fine-tuned models"""
        start_time = time.time()
        
        try:
            # Update metrics
            self.metrics["total_comparisons"] += 1
            
            # Validate session exists
            session_info = await self.chat_service.get_session_info(request.session_id)
            
            # Convert Pydantic messages to dict format for model service
            messages = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp or datetime.now().isoformat()
                }
                for msg in request.messages
            ]
            
            # Add messages to chat session
            for msg in request.messages:
                await self.chat_service.add_message(request.session_id, msg)
            
            # Generate comparison responses using model service with fallback support
            result = await self.model_service.generate_comparison_responses_with_fallback(
                session_id=request.session_id,
                messages=messages,
                generation_params=request.generation_params,
                custom_system_prompt=request.custom_system_prompt,
                use_agent_orchestrator=request.use_agent_orchestrator
            )
            
            # Create assistant messages for session history
            base_assistant_msg = ChatMessage(
                role="assistant",
                content=result["base_response"],
                timestamp=datetime.now().isoformat()
            )
            
            finetuned_assistant_msg = ChatMessage(
                role="assistant", 
                content=result["finetuned_response"],
                timestamp=datetime.now().isoformat()
            )
            
            # Add responses to session history
            await self.chat_service.add_message(request.session_id, base_assistant_msg)
            await self.chat_service.add_message(request.session_id, finetuned_assistant_msg)
            
            # Update metrics
            total_time_ms = (time.time() - start_time) * 1000
            self.metrics["successful_comparisons"] += 1
            self.metrics["average_comparison_time_ms"] = (
                (self.metrics["average_comparison_time_ms"] * (self.metrics["successful_comparisons"] - 1) + 
                 total_time_ms) / self.metrics["successful_comparisons"]
            )
            
            self.logger.info(f"Generated comparison for session {request.session_id} in {total_time_ms:.1f}ms")
            
            # Create comparison response
            comparison_response = ComparisonResponse(
                base_response=result["base_response"],
                finetuned_response=result["finetuned_response"],
                generation_time_ms=result["generation_time_ms"],
                model_info=result["model_info"],
                cache_metrics=result["cache_metrics"],
                session_id=request.session_id,
                timestamp=datetime.now().isoformat()
            )
            
            # Log for research analysis
            try:
                user_message = request.messages[-1].content if request.messages else ""
                log_id = research_logger.log_comparison(
                    session_id=request.session_id,
                    user_message=user_message,
                    comparison_response=comparison_response,
                    model_pair_info=result["model_info"],
                    generation_params=request.generation_params.dict() if request.generation_params else {}
                )
                self.logger.info(f"Logged comparison for research: {log_id}")
                
                # Enhanced logging for visualization analytics
                self._log_enhanced_comparison_analytics(
                    request.session_id, 
                    user_message, 
                    comparison_response, 
                    result["model_info"]
                )
                
            except Exception as log_error:
                self.logger.warning(f"Failed to log comparison for research: {log_error}")
            
            return comparison_response
            
        except Exception as e:
            self.metrics["failed_comparisons"] += 1
            self.logger.error(f"Error generating comparison: {e}")
            raise
    
    async def generate_comparison_stream(self, request: ComparisonRequest):
        """Stream comparison responses from both base and fine-tuned models"""
        start_time = time.time()
        
        try:
            # Update metrics
            self.metrics["total_comparisons"] += 1
            
            # Validate session exists
            session_info = await self.chat_service.get_session_info(request.session_id)
            
            # Convert Pydantic messages to dict format
            messages = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp or datetime.now().isoformat()
                }
                for msg in request.messages
            ]
            
            # Add messages to chat session
            for msg in request.messages:
                await self.chat_service.add_message(request.session_id, msg)
            
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
            
            # Generate comparison responses with fallback support
            result = await self.model_service.generate_comparison_responses_with_fallback(
                session_id=request.session_id,
                messages=messages,
                generation_params=request.generation_params,
                custom_system_prompt=request.custom_system_prompt,
                use_agent_orchestrator=request.use_agent_orchestrator
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
            base_assistant_msg = ChatMessage(
                role="assistant",
                content=result["base_response"],
                timestamp=datetime.now().isoformat()
            )
            
            finetuned_assistant_msg = ChatMessage(
                role="assistant",
                content=result["finetuned_response"],
                timestamp=datetime.now().isoformat()
            )
            
            # Add responses to session history
            await self.chat_service.add_message(request.session_id, base_assistant_msg)
            await self.chat_service.add_message(request.session_id, finetuned_assistant_msg)
            
            # Update metrics
            total_time_ms = (time.time() - start_time) * 1000
            self.metrics["successful_comparisons"] += 1
            self.metrics["average_comparison_time_ms"] = (
                (self.metrics["average_comparison_time_ms"] * (self.metrics["successful_comparisons"] - 1) + 
                 total_time_ms) / self.metrics["successful_comparisons"]
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
            
        except Exception as e:
            self.metrics["failed_comparisons"] += 1
            self.logger.error(f"Error generating comparison stream: {e}")
            yield {
                "type": "error",
                "data": {"error": str(e), "session_id": request.session_id}
            }
    
    async def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Get complete conversation history for a session"""
        try:
            return await self.chat_service.get_session_history(session_id)
        except Exception as e:
            self.logger.error(f"Error getting session history: {e}")
            raise
    
    async def list_active_sessions(self) -> List[SessionInfo]:
        """List all active chat sessions"""
        try:
            return await self.chat_service.list_active_sessions()
        except Exception as e:
            self.logger.error(f"Error listing sessions: {e}")
            raise
    
    async def delete_session(self, session_id: str) -> Dict[str, str]:
        """Delete a chat session and clean up resources"""
        try:
            # Clean up model manager resources
            await self.model_service.cleanup_session(session_id)
            
            # Delete chat session
            result = await self.chat_service.delete_session(session_id)
            
            self.logger.info(f"Deleted comparison session {session_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error deleting session: {e}")
            raise
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service performance metrics"""
        try:
            # Get memory stats from model service
            memory_stats = self.model_service.get_memory_stats()
            
            # Get chat service metrics
            chat_metrics = self.chat_service.get_service_metrics()
            
            # Combine with comparison metrics
            return {
                "comparison_service_metrics": self.metrics,
                "memory_stats": memory_stats,
                "chat_service_metrics": chat_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            raise
    
    async def _periodic_optimization(self):
        """Background task for periodic memory optimization"""
        while True:
            try:
                await asyncio.sleep(settings.session_cleanup_interval)
                
                # Run memory optimization on model service
                await self.model_service.optimize_memory()
                
                # Update cache hit rate metric
                memory_stats = self.model_service.get_memory_stats()
                
                self.logger.info(f"Periodic optimization completed. Cache hit rate: {memory_stats.get('cache_hit_rate', 0.0):.2%}")
                
            except Exception as e:
                self.logger.warning(f"Error in periodic optimization: {e}")
    
    async def _warmup_models(self):
        """Background task to warm up popular models"""
        try:
            # Wait a bit before starting warmup to let service initialize
            await asyncio.sleep(10)
            
            self.logger.info("Starting model warmup process...")
            results = await self.model_service.warmup_popular_models(max_models=1)
            
            if results:
                successful = sum(1 for _, success in results if success)
                self.logger.info(f"Model warmup completed: {successful}/{len(results)} pairs ready")
            else:
                self.logger.info("No models to warm up or warmup failed")
                
        except Exception as e:
            self.logger.warning(f"Error during model warmup: {e}")
    
    def _log_enhanced_comparison_analytics(self, session_id: str, user_message: str, 
                                           response: 'ComparisonResponse', model_info: Dict[str, Any]):
        """Log enhanced analytics data for visualization"""
        try:
            analytics_data = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "user_message_stats": {
                    "length": len(user_message),
                    "word_count": len(user_message.split()),
                    "has_question": "?" in user_message,
                    "complexity_score": self._calculate_message_complexity(user_message)
                },
                "response_analytics": {
                    "base_response_stats": {
                        "length": len(response.base_response),
                        "word_count": len(response.base_response.split()),
                        "sentence_count": response.base_response.count('.') + response.base_response.count('!') + response.base_response.count('?'),
                    },
                    "finetuned_response_stats": {
                        "length": len(response.finetuned_response),
                        "word_count": len(response.finetuned_response.split()),
                        "sentence_count": response.finetuned_response.count('.') + response.finetuned_response.count('!') + response.finetuned_response.count('?'),
                    },
                    "length_difference": len(response.finetuned_response) - len(response.base_response),
                    "style_similarity": self._calculate_style_similarity(response.base_response, response.finetuned_response)
                },
                "performance_analytics": {
                    "base_time_ms": getattr(response, 'generation_time_ms', {}).get('base', 0) if hasattr(response, 'generation_time_ms') else 0,
                    "finetuned_time_ms": getattr(response, 'generation_time_ms', {}).get('finetuned', 0) if hasattr(response, 'generation_time_ms') else 0,
                    "speedup_factor": self._calculate_speedup_factor(response),
                    "cache_hit_rate": getattr(response, 'cache_metrics', {}).get('hit_rate', 0) if hasattr(response, 'cache_metrics') else 0
                },
                "model_info": model_info
            }
            
            # Store for visualization access
            research_logger.log_resource_metrics(f"comparison_{session_id}", analytics_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to log enhanced comparison analytics: {e}")
    
    def _calculate_message_complexity(self, message: str) -> float:
        """Calculate a simple complexity score for the user message"""
        try:
            # Simple heuristic: longer messages, questions, and technical terms = more complex
            words = message.split()
            word_count = len(words)
            avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
            
            complexity_factors = [
                min(word_count / 20, 1.0),  # Length factor (normalized to 20 words = 1.0)
                min(avg_word_length / 6, 1.0),  # Average word length factor
                0.5 if "?" in message else 0.0,  # Question boost
                0.3 if any(tech_word in message.lower() for tech_word in ["algorithm", "model", "training", "parameter"]) else 0.0
            ]
            
            return sum(complexity_factors) / len(complexity_factors)
            
        except Exception:
            return 0.5  # Default moderate complexity
    
    def _calculate_style_similarity(self, text1: str, text2: str) -> float:
        """Calculate style similarity between two responses"""
        try:
            # Compare sentence structures, punctuation usage, etc.
            text1_sentences = text1.count('.') + text1.count('!') + text1.count('?')
            text2_sentences = text2.count('.') + text2.count('!') + text2.count('?')
            
            text1_words = len(text1.split())
            text2_words = len(text2.split())
            
            # Compare average sentence length
            avg_sent_len1 = text1_words / max(text1_sentences, 1)
            avg_sent_len2 = text2_words / max(text2_sentences, 1)
            
            sentence_similarity = 1 - min(abs(avg_sent_len1 - avg_sent_len2) / max(avg_sent_len1, avg_sent_len2, 1), 1.0)
            
            # Compare punctuation density
            punct_density1 = (text1.count(',') + text1.count(';') + text1.count(':')) / max(text1_words, 1)
            punct_density2 = (text2.count(',') + text2.count(';') + text2.count(':')) / max(text2_words, 1)
            
            punct_similarity = 1 - min(abs(punct_density1 - punct_density2) * 10, 1.0)
            
            return (sentence_similarity + punct_similarity) / 2
            
        except Exception:
            return 0.5  # Default moderate similarity
    
    def _calculate_speedup_factor(self, response: 'ComparisonResponse') -> float:
        """Calculate performance speedup factor"""
        try:
            if not hasattr(response, 'generation_time_ms'):
                return 1.0
            
            times = getattr(response, 'generation_time_ms', {})
            base_time = times.get('base', 0)
            finetuned_time = times.get('finetuned', 0)
            
            if finetuned_time > 0 and base_time > 0:
                return base_time / finetuned_time
            
            return 1.0
            
        except Exception:
            return 1.0
    
    async def get_visual_analytics_summary(self, session_id: str = None, hours_back: int = 24) -> Dict[str, Any]:
        """Get analytics summary optimized for visualization dashboards"""
        try:
            # Get basic comparison stats
            comparison_stats = research_logger.get_model_comparison_stats()
            
            # Get session-specific data if requested
            session_summary = None
            if session_id:
                session_summary = research_logger.get_session_summary(session_id)
            
            # Calculate trending metrics
            trending_metrics = await self._calculate_trending_metrics(hours_back)
            
            return {
                "overall_stats": comparison_stats,
                "session_summary": session_summary,
                "trending_metrics": trending_metrics,
                "service_metrics": await self.get_service_metrics(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get visual analytics summary: {e}")
            return {"error": str(e)}
    
    async def _calculate_trending_metrics(self, hours_back: int) -> Dict[str, Any]:
        """Calculate trending metrics for the specified time period"""
        try:
            # This would analyze comparison data over time to identify trends
            # For now, return placeholder data structure
            return {
                "response_time_trend": "improving",  # "improving", "stable", "degrading"
                "similarity_trend": "stable",
                "usage_trend": "increasing",
                "error_rate": 0.0,
                "avg_session_length": 0.0,
                "popular_query_types": ["questions", "instructions", "conversations"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate trending metrics: {e}")
            return {}
    
    # ============================
    # Enhanced Configuration Methods
    # ============================
    
    async def generate_enhanced_comparison(self, request: EnhancedComparisonRequest) -> ComparisonResponse:
        """Generate comparison with full configuration support"""
        start_time = time.time()
        
        try:
            # Update metrics
            self.metrics["total_comparisons"] += 1
            
            # Validate session exists
            session_info = await self.chat_service.get_session_info(request.session_id)
            
            # Convert Pydantic messages to dict format for model service
            messages = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp or datetime.now().isoformat()
                }
                for msg in request.messages
            ]
            
            # Add messages to chat session
            for msg in request.messages:
                await self.chat_service.add_message(request.session_id, msg)
            
            # Prepare generation parameters with configuration
            generation_params = {}
            debug_info = {}
            
            if request.config:
                # Apply base model configuration
                base_config = request.config.base_model.generation_config
                finetuned_config = request.config.finetuned_model.generation_config
                
                generation_params = {
                    "base_model_params": self._config_to_params(base_config),
                    "finetuned_model_params": self._config_to_params(finetuned_config),
                    "base_system_prompt": request.config.base_model.system_prompt,
                    "finetuned_system_prompt": request.config.finetuned_model.system_prompt,
                    "context_length": {
                        "base": request.config.base_model.context_length,
                        "finetuned": request.config.finetuned_model.context_length
                    }
                }
                
                if request.return_debug_info:
                    debug_info = {
                        "applied_config": request.config.dict(),
                        "parameter_validation": await self._validate_generation_params(generation_params)
                    }
            
            # Generate comparison responses with configuration
            result = await self.model_service.generate_comparison_responses_with_config(
                session_id=request.session_id,
                messages=messages,
                generation_params=generation_params,
                use_agent_orchestrator=request.use_agent_orchestrator
            )
            
            # Calculate generation time
            generation_time = (time.time() - start_time) * 1000
            
            # Create response
            comparison_response = ComparisonResponse(
                base_response=result["base_response"],
                finetuned_response=result["finetuned_response"],
                generation_time_ms=result["generation_time_ms"],
                model_info=result["model_info"],
                cache_metrics=result.get("cache_metrics", {}),
                session_id=request.session_id,
                timestamp=datetime.now().isoformat()
            )
            
            # Add debug info if requested
            if request.return_debug_info:
                comparison_response.debug_info = debug_info
            
            # Update metrics
            self.metrics["successful_comparisons"] += 1
            self.metrics["average_comparison_time_ms"] = (
                (self.metrics["average_comparison_time_ms"] * (self.metrics["successful_comparisons"] - 1) + generation_time)
                / self.metrics["successful_comparisons"]
            )
            
            # Add assistant messages to session history
            base_assistant_msg = ChatMessage(
                role="assistant",
                content=result["base_response"],
                timestamp=datetime.now().isoformat()
            )
            finetuned_assistant_msg = ChatMessage(
                role="assistant", 
                content=result["finetuned_response"],
                timestamp=datetime.now().isoformat()
            )
            
            await self.chat_service.add_message(request.session_id, base_assistant_msg)
            await self.chat_service.add_message(request.session_id, finetuned_assistant_msg)
            
            # Log comparison for research if enabled
            try:
                user_message = request.messages[-1].content if request.messages else ""
                log_id = research_logger.log_enhanced_comparison(
                    session_id=request.session_id,
                    user_message=user_message,
                    comparison_response=comparison_response,
                    model_pair_info=result["model_info"],
                    generation_config=request.config.dict() if request.config else {},
                    debug_info=debug_info
                )
                self.logger.info(f"Logged enhanced comparison for research: {log_id}")
            except Exception as log_error:
                self.logger.warning(f"Failed to log enhanced comparison: {log_error}")
            
            return comparison_response
            
        except Exception as e:
            self.metrics["failed_comparisons"] += 1
            self.logger.error(f"Error generating enhanced comparison: {e}")
            raise
    
    async def generate_enhanced_comparison_stream(self, request: EnhancedComparisonRequest):
        """Stream enhanced comparison responses with configuration support"""
        start_time = time.time()
        
        try:
            # Update metrics
            self.metrics["total_comparisons"] += 1
            
            # Validate session exists
            session_info = await self.chat_service.get_session_info(request.session_id)
            
            # Convert Pydantic messages to dict format
            messages = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp or datetime.now().isoformat()
                }
                for msg in request.messages
            ]
            
            # Add messages to chat session
            for msg in request.messages:
                await self.chat_service.add_message(request.session_id, msg)
            
            # Yield initial status
            yield {
                "type": "status",
                "data": {
                    "status": "starting", 
                    "session_id": request.session_id,
                    "config_applied": request.config is not None
                }
            }
            
            # Prepare generation parameters
            generation_params = {}
            if request.config:
                base_config = request.config.base_model.generation_config
                finetuned_config = request.config.finetuned_model.generation_config
                
                generation_params = {
                    "base_model_params": self._config_to_params(base_config),
                    "finetuned_model_params": self._config_to_params(finetuned_config),
                    "base_system_prompt": request.config.base_model.system_prompt,
                    "finetuned_system_prompt": request.config.finetuned_model.system_prompt
                }
                
                # Yield configuration info
                yield {
                    "type": "config_info",
                    "data": {
                        "base_temperature": base_config.temperature,
                        "finetuned_temperature": finetuned_config.temperature,
                        "base_max_tokens": base_config.max_tokens,
                        "finetuned_max_tokens": finetuned_config.max_tokens
                    }
                }
            
            # Yield generation progress
            yield {
                "type": "progress",
                "data": {"phase": "generating", "models": ["base", "finetuned"]}
            }
            
            # Generate comparison responses
            result = await self.model_service.generate_comparison_responses_with_config(
                session_id=request.session_id,
                messages=messages,
                generation_params=generation_params,
                use_agent_orchestrator=request.use_agent_orchestrator
            )
            
            # Stream base response
            yield {
                "type": "response",
                "data": {
                    "model_type": "base",
                    "content": result["base_response"],
                    "generation_time_ms": result["generation_time_ms"]["base"],
                    "config_used": generation_params.get("base_model_params", {}) if request.config else None
                }
            }
            
            # Stream finetuned response
            yield {
                "type": "response",
                "data": {
                    "model_type": "finetuned",
                    "content": result["finetuned_response"], 
                    "generation_time_ms": result["generation_time_ms"]["finetuned"],
                    "config_used": generation_params.get("finetuned_model_params", {}) if request.config else None
                }
            }
            
            # Create assistant messages
            base_assistant_msg = ChatMessage(
                role="assistant",
                content=result["base_response"],
                timestamp=datetime.now().isoformat()
            )
            finetuned_assistant_msg = ChatMessage(
                role="assistant",
                content=result["finetuned_response"], 
                timestamp=datetime.now().isoformat()
            )
            
            await self.chat_service.add_message(request.session_id, base_assistant_msg)
            await self.chat_service.add_message(request.session_id, finetuned_assistant_msg)
            
            # Yield completion status
            yield {
                "type": "complete",
                "data": {
                    "session_id": request.session_id,
                    "generation_time_ms": (time.time() - start_time) * 1000,
                    "cache_metrics": result.get("cache_metrics", {}),
                    "model_info": result["model_info"]
                }
            }
            
            self.metrics["successful_comparisons"] += 1
            
        except Exception as e:
            self.metrics["failed_comparisons"] += 1
            self.logger.error(f"Error in enhanced comparison stream: {e}")
            yield {
                "type": "error",
                "data": {"error": str(e), "session_id": request.session_id}
            }
    
    def _config_to_params(self, config: GenerationConfig) -> Dict[str, Any]:
        """Convert GenerationConfig to model service parameters"""
        return {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_tokens": config.max_tokens,
            "presence_penalty": config.presence_penalty,
            "frequency_penalty": config.frequency_penalty,
            "repetition_penalty": config.repetition_penalty,
            "do_sample": config.do_sample,
            "early_stopping": config.early_stopping,
            "seed": config.seed,
            "pad_token_id": config.pad_token_id,
            "eos_token_id": config.eos_token_id
        }
    
    async def _validate_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generation parameters and return validation results"""
        validation_results = {
            "base_model": {"valid": True, "warnings": [], "errors": []},
            "finetuned_model": {"valid": True, "warnings": [], "errors": []}
        }
        
        for model_type in ["base_model_params", "finetuned_model_params"]:
            if model_type in params:
                model_params = params[model_type]
                model_key = "base_model" if "base" in model_type else "finetuned_model"
                
                # Validate temperature
                if model_params.get("temperature", 0.7) > 1.5:
                    validation_results[model_key]["warnings"].append(
                        f"High temperature ({model_params['temperature']}) may produce unpredictable results"
                    )
                
                # Validate max_tokens
                if model_params.get("max_tokens", 1024) > 2048:
                    validation_results[model_key]["warnings"].append(
                        f"High max_tokens ({model_params['max_tokens']}) may impact performance"
                    )
                
                # Check for invalid combinations
                if not model_params.get("do_sample", True) and model_params.get("temperature", 0.7) > 1.0:
                    validation_results[model_key]["warnings"].append(
                        "High temperature with do_sample=False may not have expected effect"
                    )
        
        return validation_results


# Note: Global service instance should be created with proper dependencies
# comparison_service = ComparisonService()  # Commented out - should be initialized in app.py with dependencies