"""
Model configuration utilities for loading and managing model pairs
"""

import yaml
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from utils.logging import get_logger


class ModelConfig:
    """Manages model pair configurations from YAML"""
    
    def __init__(self, config_path: str = "config/model_pairs.yaml"):
        self.config_path = config_path
        self.logger = get_logger(__name__)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"Loaded model config from {self.config_path}")
                return config
            else:
                self.logger.warning(f"Model config file not found: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading model config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when config file is not available"""
        return {
            "model_pairs": [],
            "naming_patterns": {
                "base_suffixes": ["-base", "_base"],
                "instruct_suffixes": ["-instruct", "_instruct", "-sft", "_sft"]
            },
            "loading": {
                "prefer_unsloth": True,
                "fallback_to_transformers": True,
                "enable_4bit": True,
                "max_seq_length": 2048,
                "device_map": "auto"
            }
        }
    
    def get_configured_pairs(self) -> List[Dict[str, Any]]:
        """Get manually configured model pairs"""
        return self.config.get("model_pairs", [])
    
    def get_naming_patterns(self) -> Dict[str, List[str]]:
        """Get naming patterns for intelligent model pairing"""
        return self.config.get("naming_patterns", {})
    
    def get_loading_config(self) -> Dict[str, Any]:
        """Get model loading configuration"""
        return self.config.get("loading", {})
    
    def is_auto_discovery_enabled(self) -> bool:
        """Check if auto-discovery is enabled"""
        pairs = self.get_configured_pairs()
        for pair in pairs:
            if pair.get("auto_discover", False) and pair.get("enabled", True):
                return True
        return False
    
    def get_base_suffixes(self) -> List[str]:
        """Get base model naming suffixes"""
        patterns = self.get_naming_patterns()
        return patterns.get("base_suffixes", ["-base", "_base"])
    
    def get_instruct_suffixes(self) -> List[str]:
        """Get instruct model naming suffixes"""  
        patterns = self.get_naming_patterns()
        return patterns.get("instruct_suffixes", ["-instruct", "_instruct"])
    
    def find_pair_for_models(self, base_path: str, instruct_path: str) -> Optional[Dict[str, Any]]:
        """Find configured pair for given model paths"""
        pairs = self.get_configured_pairs()
        
        for pair in pairs:
            if not pair.get("enabled", True):
                continue
                
            base_config = pair.get("base_model", {})
            instruct_config = pair.get("finetuned_model", {})
            
            if (base_config.get("path") == base_path and 
                instruct_config.get("path") == instruct_path):
                return pair
        
        return None
    
    def get_qwen_pair_config(self) -> Optional[Dict[str, Any]]:
        """Get Qwen-specific pair configuration"""
        pairs = self.get_configured_pairs()
        
        for pair in pairs:
            if ("qwen" in pair.get("pair_id", "").lower() and 
                pair.get("enabled", True)):
                return pair
        
        return None


# Global model config instance
model_config = ModelConfig()
