"""
Template Service for Dataset Standardization

Provides template management and dataset formatting capabilities
inspired by LLaMA-Factory's template system.
"""

import json
import asyncio
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from models.templates import Template, get_template, list_available_templates
from models.schemas import FineTuneRequest
from utils.logging import get_logger


class TemplateService:
    """Service for managing dataset templates and formatting"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cached_templates: Dict[str, Template] = {}
        
    async def get_template(self, template_name: str) -> Template:
        """Get template by name with caching"""
        if template_name not in self.cached_templates:
            try:
                template = get_template(template_name)
                self.cached_templates[template_name] = template
                self.logger.info(f"Loaded template: {template_name}")
            except ValueError as e:
                self.logger.error(f"Failed to load template {template_name}: {e}")
                # Fallback to alpaca template
                template = get_template("alpaca")
                self.cached_templates[template_name] = template
                self.logger.warning(f"Using Alpaca template as fallback for {template_name}")
        
        return self.cached_templates[template_name]
    
    async def list_templates(self) -> List[str]:
        """List available templates"""
        return list_available_templates()
    
    async def format_dataset(self, dataset_path: str, template_name: str,
                           system_message: Optional[str] = None) -> List[Dict[str, Any]]:
        """Format dataset using specified template"""
        try:
            template = await self.get_template(template_name)
            
            # Load dataset
            dataset = await self._load_dataset(dataset_path)
            
            formatted_data = []
            for item in dataset:
                # Convert item to standard conversation format
                messages = self._standardize_item(item)
                
                # Format using template
                formatted_text = template.format_conversation(messages, system_message)
                
                formatted_data.append({
                    "text": formatted_text,
                    "template": template_name,
                    "original": item
                })
            
            self.logger.info(f"Formatted {len(formatted_data)} items using {template_name} template")
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error formatting dataset: {e}")
            raise
    
    async def _load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load dataset from file"""
        dataset_file = Path(dataset_path)
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        if dataset_file.suffix == '.json':
            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        elif dataset_file.suffix == '.jsonl':
            data = []
            with open(dataset_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_file.suffix}")
    
    def _standardize_item(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert dataset item to standard conversation format"""
        
        # Handle Alpaca format
        if 'instruction' in item:
            messages = [{"role": "user", "content": item['instruction']}]
            
            if 'input' in item and item['input'].strip():
                messages[0]['content'] += f"\n\nInput: {item['input']}"
            
            if 'output' in item:
                messages.append({"role": "assistant", "content": item['output']})
            
            return messages
        
        # Handle ShareGPT/OpenAI format
        elif 'conversations' in item:
            return item['conversations']
        elif 'messages' in item:
            return item['messages']
        
        # Handle simple question-answer format
        elif 'question' in item and 'answer' in item:
            return [
                {"role": "user", "content": item['question']},
                {"role": "assistant", "content": item['answer']}
            ]
        
        # Handle custom format
        elif 'prompt' in item and 'response' in item:
            return [
                {"role": "user", "content": item['prompt']},
                {"role": "assistant", "content": item['response']}
            ]
        
        # Fallback: treat as single user message
        elif 'text' in item:
            return [{"role": "user", "content": item['text']}]
        
        else:
            # Last resort: convert entire item to string
            return [{"role": "user", "content": str(item)}]
    
    async def validate_template_compatibility(self, template_name: str, 
                                            dataset_path: str) -> Dict[str, Any]:
        """Validate template compatibility with dataset"""
        try:
            template = await self.get_template(template_name)
            dataset = await self._load_dataset(dataset_path)
            
            # Sample validation
            sample_size = min(10, len(dataset))
            sample_items = dataset[:sample_size]
            
            validation_results = {
                "compatible": True,
                "issues": [],
                "sample_formats": []
            }
            
            for i, item in enumerate(sample_items):
                try:
                    messages = self._standardize_item(item)
                    formatted = template.format_conversation(messages)
                    
                    validation_results["sample_formats"].append({
                        "item_index": i,
                        "original_format": list(item.keys()),
                        "standardized_messages": len(messages),
                        "formatted_length": len(formatted)
                    })
                    
                except Exception as e:
                    validation_results["issues"].append({
                        "item_index": i,
                        "error": str(e)
                    })
            
            if validation_results["issues"]:
                validation_results["compatible"] = False
            
            return validation_results
            
        except Exception as e:
            return {
                "compatible": False,
                "error": str(e),
                "issues": [],
                "sample_formats": []
            }
    
    async def auto_detect_template(self, dataset_path: str) -> str:
        """Auto-detect best template for dataset"""
        try:
            dataset = await self._load_dataset(dataset_path)
            
            if not dataset:
                return "alpaca"  # Default fallback
            
            sample = dataset[0]
            
            # Check for Alpaca format
            if 'instruction' in sample and ('output' in sample or 'response' in sample):
                return "alpaca"
            
            # Check for ChatML/OpenAI format
            if 'messages' in sample or 'conversations' in sample:
                return "chatml"
            
            # Check for Q&A format
            if 'question' in sample and 'answer' in sample:
                return "vicuna"
            
            # Default to alpaca
            return "alpaca"
            
        except Exception as e:
            self.logger.warning(f"Auto-detection failed: {e}, using alpaca template")
            return "alpaca"


# Global template service instance
template_service = TemplateService()