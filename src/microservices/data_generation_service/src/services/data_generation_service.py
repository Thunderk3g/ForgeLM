"""
Data Generation Service

Provides automated dataset generation capabilities using available models and templates.
Supports multi-format generation (Alpaca, ChatML, ShareGPT, Qwen) with quality control.
"""

import json
import uuid
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Data generation service is torch-independent 
# It generates synthetic data using templates and external APIs

from .template_service import TemplateService

# Import shared utilities from parent directory  
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from utils.research_logger import research_logger
from utils.logging import get_logger


class DataGenerationError(Exception):
    """Custom exception for data generation errors"""
    def __init__(self, message: str, error_type: str = "generation_error", details: Optional[Dict] = None):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        super().__init__(message)


class GenerationConfig:
    """Configuration for data generation tasks"""
    
    def __init__(self, **kwargs):
        self.task_type = kwargs.get("task_type", "instruction_tuning")  # instruction_tuning, conversation, domain_specific
        self.template_format = kwargs.get("template_format", "alpaca")  # alpaca, chatml, sharegpt, qwen
        self.num_samples = kwargs.get("num_samples", 100)
        self.model_pair_id = kwargs.get("model_pair_id", None)
        self.domain = kwargs.get("domain", "general")
        self.difficulty_level = kwargs.get("difficulty_level", "mixed")  # easy, medium, hard, mixed
        self.generation_strategy = kwargs.get("generation_strategy", "self_supervised")  # self_supervised, iterative_refinement, multi_model_consensus
        self.quality_threshold = kwargs.get("quality_threshold", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 512)
        self.temperature = kwargs.get("temperature", 0.8)
        self.diversity_factor = kwargs.get("diversity_factor", 0.5)
        
        # Seed prompts for generation
        self.seed_prompts = kwargs.get("seed_prompts", [])
        if not self.seed_prompts:
            self.seed_prompts = self._get_default_seed_prompts()
    
    def _get_default_seed_prompts(self) -> List[str]:
        """Get default seed prompts based on task type"""
        if self.task_type == "instruction_tuning":
            return [
                "Explain the concept of",
                "How do you",
                "What is the difference between",
                "Describe the process of",
                "Compare and contrast",
                "Analyze the importance of",
                "Summarize the key points about",
                "Provide step-by-step instructions for",
                "Evaluate the pros and cons of",
                "Define and give examples of"
            ]
        elif self.task_type == "conversation":
            return [
                "Hi, I need help with",
                "Can you assist me in",
                "I'm looking for advice on",
                "Could you explain",
                "I'm curious about",
                "Help me understand",
                "I want to learn about",
                "Can you guide me through",
                "I'm struggling with",
                "Tell me more about"
            ]
        elif self.task_type == "domain_specific":
            if self.domain == "code":
                return [
                    "Write a function to",
                    "Debug this code:",
                    "Optimize the following algorithm:",
                    "Explain this programming concept:",
                    "Refactor this code to",
                    "Create a class that",
                    "Implement a solution for",
                    "Review this code and suggest improvements:",
                    "Write unit tests for",
                    "Convert this code from"
                ]
            elif self.domain == "math":
                return [
                    "Solve this equation:",
                    "Prove that",
                    "Calculate the",
                    "Derive the formula for",
                    "Find the solution to",
                    "Explain the mathematical concept of",
                    "Simplify the expression",
                    "Graph the function",
                    "Find the derivative of",
                    "Integrate the following"
                ]
        
        return ["Explain", "Describe", "Analyze", "Compare", "Summarize"]


class DataGenerationService:
    """Service for automated training data generation"""
    
    def __init__(self, 
                 template_service: Optional[TemplateService] = None):
        self.logger = get_logger(__name__)
        # Data generation service focuses on template-based and API-based generation
        # No local model dependencies
        self.template_service = template_service
        
        # Generation status tracking
        self.active_generations = {}
        
        # Output directory
        self.output_dir = Path("data/generated")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("DataGenerationService initialized")
    
    async def start_generation(self, config: GenerationConfig) -> str:
        """Start a new data generation task"""
        task_id = str(uuid.uuid4())
        
        try:
            # Validate configuration
            self._validate_config(config)
            
            # Initialize generation status
            self.active_generations[task_id] = {
                "status": "initializing",
                "config": config,
                "progress": 0,
                "samples_generated": 0,
                "samples_validated": 0,
                "errors": [],
                "started_at": datetime.now().isoformat()
            }
            
            # Start generation in background
            asyncio.create_task(self._run_generation(task_id, config))
            
            self.logger.info(f"Started data generation task {task_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to start generation: {e}")
            raise DataGenerationError(
                f"Failed to start data generation: {e}",
                error_type="initialization_failed",
                details={"config": config.__dict__ if hasattr(config, '__dict__') else str(config)}
            )
    
    async def get_generation_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a data generation task"""
        if task_id not in self.active_generations:
            raise DataGenerationError(
                f"Generation task {task_id} not found",
                error_type="task_not_found",
                details={"task_id": task_id}
            )
        
        return self.active_generations[task_id]
    
    async def cancel_generation(self, task_id: str) -> bool:
        """Cancel an active generation task"""
        if task_id not in self.active_generations:
            return False
        
        self.active_generations[task_id]["status"] = "cancelled"
        self.logger.info(f"Cancelled generation task {task_id}")
        return True
    
    async def list_generated_datasets(self) -> List[Dict[str, Any]]:
        """List all generated datasets"""
        datasets = []
        
        try:
            for dataset_file in self.output_dir.glob("*.json"):
                try:
                    with open(dataset_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f).get('metadata', {})
                    
                    stat = dataset_file.stat()
                    datasets.append({
                        "filename": dataset_file.name,
                        "path": str(dataset_file),
                        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "sample_count": metadata.get("sample_count", 0),
                        "template_format": metadata.get("template_format", "unknown"),
                        "task_type": metadata.get("task_type", "unknown"),
                        "quality_score": metadata.get("avg_quality_score", 0)
                    })
                except Exception as e:
                    self.logger.warning(f"Error reading dataset metadata {dataset_file}: {e}")
                    continue
            
            # Sort by creation time, newest first
            datasets.sort(key=lambda x: x["created_at"], reverse=True)
            return datasets
            
        except Exception as e:
            self.logger.error(f"Error listing generated datasets: {e}")
            raise DataGenerationError(
                "Failed to list generated datasets",
                error_type="listing_failed",
                details={"error": str(e)}
            )
    
    async def _run_generation(self, task_id: str, config: GenerationConfig):
        """Run the actual data generation process"""
        try:
            self.active_generations[task_id]["status"] = "running"
            
            # Generate samples based on strategy
            if config.generation_strategy == "self_supervised":
                samples = await self._generate_self_supervised(task_id, config)
            elif config.generation_strategy == "iterative_refinement":
                samples = await self._generate_iterative_refinement(task_id, config)
            elif config.generation_strategy == "multi_model_consensus":
                samples = await self._generate_multi_model_consensus(task_id, config)
            else:
                raise DataGenerationError(
                    f"Unknown generation strategy: {config.generation_strategy}",
                    error_type="invalid_strategy"
                )
            
            # Validate generated samples
            validated_samples = await self._validate_samples(task_id, samples, config)
            
            # Format samples according to template
            formatted_samples = await self._format_samples(task_id, validated_samples, config)
            
            # Save dataset
            dataset_path = await self._save_dataset(task_id, formatted_samples, config)
            
            # Update final status
            self.active_generations[task_id].update({
                "status": "completed",
                "progress": 100,
                "dataset_path": str(dataset_path),
                "completed_at": datetime.now().isoformat()
            })
            
            self.logger.info(f"Completed data generation task {task_id}: {len(formatted_samples)} samples")
            
        except Exception as e:
            self.logger.error(f"Generation task {task_id} failed: {e}")
            self.active_generations[task_id].update({
                "status": "error",
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            })
    
    async def _generate_self_supervised(self, task_id: str, config: GenerationConfig) -> List[Dict[str, Any]]:
        """Generate samples using self-supervised approach with base model"""
        samples = []
        
        try:
            # Data generation service works with external APIs instead of local models
            self.logger.info("Using template-based generation without local models")
            model_pairs = []  # No local model pairs needed
            if not model_pairs:
                raise DataGenerationError("No model pairs available", error_type="no_models")
            
            # Use specified model pair or select the best available
            if config.model_pair_id:
                model_pair = next((p for p in model_pairs if p.pair_id == config.model_pair_id), None)
                if not model_pair:
                    raise DataGenerationError(f"Model pair {config.model_pair_id} not found")
            else:
                model_pair = model_pairs[0]  # Use first available
            
            batch_size = min(10, config.num_samples)  # Process in batches
            for batch_start in range(0, config.num_samples, batch_size):
                batch_end = min(batch_start + batch_size, config.num_samples)
                batch_samples = []
                
                for i in range(batch_start, batch_end):
                    # Update progress
                    progress = int((i / config.num_samples) * 100)
                    self.active_generations[task_id]["progress"] = progress
                    
                    # Select seed prompt with some randomization
                    seed_prompt = np.random.choice(config.seed_prompts)
                    
                    # Generate instruction-response pair
                    sample = await self._generate_sample(seed_prompt, model_pair, config)
                    if sample:
                        batch_samples.append(sample)
                        self.active_generations[task_id]["samples_generated"] += 1
                
                samples.extend(batch_samples)
                
                # Small delay between batches to avoid overwhelming the system
                await asyncio.sleep(1)
            
            return samples
            
        except Exception as e:
            raise DataGenerationError(
                f"Self-supervised generation failed: {e}",
                error_type="generation_failed",
                details={"samples_generated": len(samples)}
            )
    
    async def _generate_sample(self, seed_prompt: str, model_pair, config: GenerationConfig) -> Optional[Dict[str, Any]]:
        """Generate a single training sample"""
        try:
            # Create instruction from seed prompt
            instruction = await self._create_instruction(seed_prompt, config)
            
            # Generate response using model
            response = await self._generate_response(instruction, model_pair, config)
            
            if not response or len(response.strip()) < 10:
                return None
            
            # Create sample structure
            sample = {
                "instruction": instruction,
                "response": response,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "seed_prompt": seed_prompt,
                    "model_pair_id": model_pair.pair_id,
                    "template_format": config.template_format,
                    "domain": config.domain,
                    "generation_strategy": config.generation_strategy
                }
            }
            
            return sample
            
        except Exception as e:
            self.logger.warning(f"Failed to generate sample from seed '{seed_prompt}': {e}")
            return None
    
    async def _create_instruction(self, seed_prompt: str, config: GenerationConfig) -> str:
        """Create instruction from seed prompt"""
        # Add domain-specific context and variety
        if config.domain == "code":
            topics = ["Python", "JavaScript", "algorithm", "data structure", "API", "database", "web development"]
        elif config.domain == "math":
            topics = ["algebra", "calculus", "statistics", "geometry", "probability", "linear algebra"]
        elif config.domain == "science":
            topics = ["physics", "chemistry", "biology", "environmental science", "astronomy"]
        else:
            topics = ["business", "technology", "health", "education", "communication", "problem-solving"]
        
        topic = np.random.choice(topics)
        
        # Create instruction with variety
        instruction_templates = [
            f"{seed_prompt} {topic}",
            f"{seed_prompt} {topic} in detail",
            f"{seed_prompt} {topic} and provide examples",
            f"{seed_prompt} {topic} for a beginner",
            f"{seed_prompt} practical applications of {topic}"
        ]
        
        return np.random.choice(instruction_templates)
    
    async def _generate_response(self, instruction: str, model_pair, config: GenerationConfig) -> str:
        """Generate response using model service"""
        try:
            # Create messages for generation
            messages = [{"role": "user", "content": instruction}]
            
            # Generate using external API calls instead of local models
            session_id = f"generation_{uuid.uuid4()}"
            self.logger.info(f"Would generate response for session {session_id} using external API")
            # TODO: Implement external API calls (OpenAI, Anthropic, etc.) for response generation
            comparison_response = {
                "responses": [{"model": "external-api", "response": f"Generated response for: {instruction}"}]
            }
            
            # Use fine-tuned response if available, otherwise base response
            if comparison_response.get("finetuned_response"):
                return comparison_response["finetuned_response"]
            else:
                return comparison_response.get("base_response", "")
                
        except Exception as e:
            self.logger.error(f"Response generation failed for instruction '{instruction[:50]}...': {e}")
            return ""
    
    async def _generate_iterative_refinement(self, task_id: str, config: GenerationConfig) -> List[Dict[str, Any]]:
        """Generate samples using iterative refinement approach"""
        # For now, fallback to self-supervised with quality filtering
        # This could be enhanced to do multiple rounds of generation and refinement
        return await self._generate_self_supervised(task_id, config)
    
    async def _generate_multi_model_consensus(self, task_id: str, config: GenerationConfig) -> List[Dict[str, Any]]:
        """Generate samples using multi-model consensus approach"""
        # For now, fallback to self-supervised
        # This could be enhanced to use multiple models and select best responses
        return await self._generate_self_supervised(task_id, config)
    
    async def _validate_samples(self, task_id: str, samples: List[Dict[str, Any]], config: GenerationConfig) -> List[Dict[str, Any]]:
        """Validate generated samples for quality"""
        validated_samples = []
        
        for sample in samples:
            try:
                quality_score = self._calculate_quality_score(sample)
                
                if quality_score >= config.quality_threshold:
                    sample["metadata"]["quality_score"] = quality_score
                    validated_samples.append(sample)
                    self.active_generations[task_id]["samples_validated"] += 1
                
            except Exception as e:
                self.logger.warning(f"Sample validation failed: {e}")
                continue
        
        return validated_samples
    
    def _calculate_quality_score(self, sample: Dict[str, Any]) -> float:
        """Calculate quality score for a sample"""
        score = 0.0
        
        try:
            instruction = sample.get("instruction", "")
            response = sample.get("response", "")
            
            # Basic quality checks
            if len(instruction) > 10 and len(response) > 20:
                score += 0.3
            
            # Length appropriateness (not too short, not too long)
            if 50 <= len(response) <= 2000:
                score += 0.2
            
            # Contains complete sentences
            if response.count('.') >= 1 or response.count('!') >= 1 or response.count('?') >= 1:
                score += 0.2
            
            # No repetition (simple check)
            words = response.lower().split()
            if len(set(words)) / max(len(words), 1) > 0.5:  # At least 50% unique words
                score += 0.2
            
            # Instruction-response relevance (basic keyword overlap)
            instruction_words = set(instruction.lower().split())
            response_words = set(response.lower().split())
            overlap = len(instruction_words.intersection(response_words))
            if overlap > 0:
                score += 0.1
            
        except Exception:
            score = 0.5  # Neutral score if calculation fails
        
        return min(score, 1.0)
    
    async def _format_samples(self, task_id: str, samples: List[Dict[str, Any]], config: GenerationConfig) -> List[Dict[str, Any]]:
        """Format samples according to specified template"""
        if not self.template_service:
            # Simple formatting without template service
            return samples
        
        try:
            formatted_samples = []
            
            for sample in samples:
                # Format using template service
                formatted_sample = await self.template_service.format_sample(
                    sample, 
                    template_name=config.template_format
                )
                formatted_samples.append(formatted_sample)
            
            return formatted_samples
            
        except Exception as e:
            self.logger.warning(f"Template formatting failed: {e}")
            return samples  # Return unformatted samples
    
    async def _save_dataset(self, task_id: str, samples: List[Dict[str, Any]], config: GenerationConfig) -> Path:
        """Save generated dataset to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{config.task_type}_{config.template_format}_{timestamp}.json"
            dataset_path = self.output_dir / filename
            
            # Create dataset with metadata
            dataset = {
                "metadata": {
                    "task_id": task_id,
                    "generated_at": datetime.now().isoformat(),
                    "sample_count": len(samples),
                    "template_format": config.template_format,
                    "task_type": config.task_type,
                    "domain": config.domain,
                    "generation_strategy": config.generation_strategy,
                    "quality_threshold": config.quality_threshold,
                    "avg_quality_score": np.mean([s.get("metadata", {}).get("quality_score", 0) for s in samples]),
                    "config": {
                        "num_samples_requested": config.num_samples,
                        "model_pair_id": config.model_pair_id,
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens
                    }
                },
                "samples": samples
            }
            
            # Save to file
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            # Log to research logger
            research_logger.log_dataset_generation(
                task_id=task_id,
                dataset_path=str(dataset_path),
                sample_count=len(samples),
                config=config.__dict__ if hasattr(config, '__dict__') else str(config)
            )
            
            return dataset_path
            
        except Exception as e:
            raise DataGenerationError(
                f"Failed to save dataset: {e}",
                error_type="save_failed",
                details={"sample_count": len(samples)}
            )
    
    def _validate_config(self, config: GenerationConfig):
        """Validate generation configuration"""
        if not isinstance(config, GenerationConfig):
            raise DataGenerationError("Invalid configuration type", error_type="invalid_config")
        
        if config.num_samples < 1 or config.num_samples > 10000:
            raise DataGenerationError("Number of samples must be between 1 and 10000", error_type="invalid_config")
        
        if config.quality_threshold < 0 or config.quality_threshold > 1:
            raise DataGenerationError("Quality threshold must be between 0 and 1", error_type="invalid_config")
        
        valid_task_types = ["instruction_tuning", "conversation", "domain_specific"]
        if config.task_type not in valid_task_types:
            raise DataGenerationError(f"Invalid task type. Must be one of: {valid_task_types}", error_type="invalid_config")
        
        valid_templates = ["alpaca", "chatml", "sharegpt", "qwen"]
        if config.template_format not in valid_templates:
            raise DataGenerationError(f"Invalid template format. Must be one of: {valid_templates}", error_type="invalid_config")
    
    def get_active_job_count(self) -> int:
        """Get the count of active generation jobs"""
        return len([
            job for job in self.active_generations.values() 
            if job.get("status") in ["initializing", "running"]
        ])