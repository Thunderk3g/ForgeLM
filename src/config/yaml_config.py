"""
YAML Configuration Management System

Inspired by LLaMA-Factory's configuration system, provides
reproducible, version-controlled training configurations.
"""

import yaml
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from models.schemas import FineTuneRequest
from models.peft_methods import PEFTMethod, PEFTConfig
from utils.logging import get_logger


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    path: str
    format: str = "alpaca"  # alpaca, sharegpt, openai
    template: str = "alpaca"
    system_message: Optional[str] = None
    max_samples: Optional[int] = None
    test_size: float = 0.1


@dataclass  
class ModelConfig:
    """Model configuration"""
    model_name: str
    model_path: str
    quantization: Optional[str] = None  # 4bit, 8bit, none
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    model_revision: str = "main"


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic training parameters
    output_dir: str
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Scheduling
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    evaluation_strategy: str = "steps"
    
    # Advanced options
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    report_to: List[str] = None
    
    # Sequence length
    max_seq_length: int = 512
    
    def __post_init__(self):
        if self.report_to is None:
            self.report_to = []


@dataclass
class YAMLConfig:
    """Complete YAML configuration"""
    # Required components (no defaults)
    config_name: str
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    peft: PEFTConfig
    
    # Optional metadata (with defaults)
    description: str = ""
    version: str = "1.0"
    created_at: str = ""
    author: str = ""
    
    # Environment
    environment: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.environment is None:
            self.environment = {}


class YAMLConfigManager:
    """Manager for YAML configuration files"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = get_logger(__name__)
        
        # Create subdirectories
        (self.config_dir / "training").mkdir(exist_ok=True)
        (self.config_dir / "models").mkdir(exist_ok=True)
        (self.config_dir / "templates").mkdir(exist_ok=True)
    
    def save_config(self, config: YAMLConfig, filename: str = None) -> str:
        """Save configuration to YAML file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{config.config_name}_{timestamp}.yaml"
            
            config_path = self.config_dir / "training" / filename
            
            # Convert to dict and save
            config_dict = asdict(config)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2, allow_unicode=True)
            
            self.logger.info(f"Configuration saved to: {config_path}")
            return str(config_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def load_config(self, config_path: str) -> YAMLConfig:
        """Load configuration from YAML file"""
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                # Try relative to config directory
                config_file = self.config_dir / "training" / config_path
                
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            # Reconstruct dataclass objects
            config = self._dict_to_config(config_dict)
            
            self.logger.info(f"Configuration loaded from: {config_file}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> YAMLConfig:
        """Convert dictionary to YAMLConfig object"""
        
        # Extract components
        model_config = ModelConfig(**config_dict.get('model', {}))
        dataset_config = DatasetConfig(**config_dict.get('dataset', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        # Handle PEFT config
        peft_dict = config_dict.get('peft', {})
        if 'method' in peft_dict and isinstance(peft_dict['method'], str):
            peft_dict['method'] = PEFTMethod(peft_dict['method'])
        peft_config = PEFTConfig(**peft_dict)
        
        # Create main config
        yaml_config = YAMLConfig(
            config_name=config_dict.get('config_name', 'unnamed'),
            description=config_dict.get('description', ''),
            version=config_dict.get('version', '1.0'),
            created_at=config_dict.get('created_at', ''),
            author=config_dict.get('author', ''),
            model=model_config,
            dataset=dataset_config,
            training=training_config,
            peft=peft_config,
            environment=config_dict.get('environment', {})
        )
        
        return yaml_config
    
    def create_config_from_request(self, request: FineTuneRequest, config_name: str) -> YAMLConfig:
        """Create YAML config from FineTuneRequest"""
        
        # Model config
        model_config = ModelConfig(
            model_name=Path(request.base_model_dir).name,
            model_path=request.base_model_dir
        )
        
        # Dataset config  
        dataset_config = DatasetConfig(
            path=request.dataset_dir,
            template=request.template_name,
            system_message=request.system_message
        )
        
        # Training config
        training_config = TrainingConfig(
            output_dir=request.output_dir,
            num_train_epochs=request.epochs,
            per_device_train_batch_size=request.batch_size,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            learning_rate=request.learning_rate,
            max_seq_length=request.max_seq_length
        )
        
        # PEFT config
        peft_config = PEFTConfig(
            method=PEFTMethod(request.peft_method),
            rank=request.peft_rank,
            alpha=request.peft_alpha,
            dropout=request.peft_dropout,
            target_modules=request.target_modules
        )
        
        # Create complete config
        yaml_config = YAMLConfig(
            config_name=config_name,
            description=f"Auto-generated from API request",
            model=model_config,
            dataset=dataset_config,
            training=training_config,
            peft=peft_config
        )
        
        return yaml_config
    
    def list_configs(self) -> List[Dict[str, Any]]:
        """List available configuration files"""
        configs = []
        training_dir = self.config_dir / "training"
        
        if training_dir.exists():
            for config_file in training_dir.glob("*.yaml"):
                try:
                    # Load basic metadata
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_dict = yaml.safe_load(f)
                    
                    configs.append({
                        "filename": config_file.name,
                        "path": str(config_file),
                        "config_name": config_dict.get('config_name', config_file.stem),
                        "description": config_dict.get('description', ''),
                        "version": config_dict.get('version', ''),
                        "created_at": config_dict.get('created_at', ''),
                        "author": config_dict.get('author', ''),
                        "modified_at": datetime.fromtimestamp(config_file.stat().st_mtime).isoformat()
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error reading config {config_file}: {e}")
        
        return sorted(configs, key=lambda x: x['modified_at'], reverse=True)
    
    def validate_config(self, config: YAMLConfig) -> Dict[str, Any]:
        """Validate configuration"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check model path exists
            if not Path(config.model.model_path).exists():
                validation_result["errors"].append(f"Model path does not exist: {config.model.model_path}")
            
            # Check dataset path exists
            if not Path(config.dataset.path).exists():
                validation_result["errors"].append(f"Dataset path does not exist: {config.dataset.path}")
            
            # Check output directory is writable
            output_dir = Path(config.training.output_dir)
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                validation_result["errors"].append(f"Cannot create output directory: {e}")
            
            # Validate PEFT parameters
            if config.peft.rank <= 0:
                validation_result["errors"].append("PEFT rank must be positive")
            
            if config.peft.alpha <= 0:
                validation_result["errors"].append("PEFT alpha must be positive")
            
            # Validate training parameters
            if config.training.learning_rate <= 0:
                validation_result["warnings"].append("Very low learning rate detected")
            
            if config.training.per_device_train_batch_size < 1:
                validation_result["errors"].append("Batch size must be at least 1")
            
            validation_result["valid"] = len(validation_result["errors"]) == 0
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {e}")
        
        return validation_result
    
    def create_default_configs(self):
        """Create default configuration templates"""
        
        # Default Alpaca config
        default_alpaca = YAMLConfig(
            config_name="alpaca_default",
            description="Default configuration for Alpaca-style instruction tuning",
            model=ModelConfig(
                model_name="Qwen2.5-3B",
                model_path="models/base/Qwen_Qwen2.5-3B"
            ),
            dataset=DatasetConfig(
                path="data/prepared",
                format="alpaca",
                template="alpaca"
            ),
            training=TrainingConfig(
                output_dir="models/fine_tuned",
                num_train_epochs=3,
                per_device_train_batch_size=4,
                learning_rate=5e-5
            ),
            peft=PEFTConfig(
                method=PEFTMethod.QLORA,
                rank=16,
                alpha=32
            )
        )
        
        self.save_config(default_alpaca, "alpaca_default.yaml")
        
        # DoRA config
        dora_config = YAMLConfig(
            config_name="dora_advanced",
            description="Advanced configuration using DoRA method",
            model=ModelConfig(
                model_name="Qwen2.5-3B",
                model_path="models/base/Qwen_Qwen2.5-3B"
            ),
            dataset=DatasetConfig(
                path="data/prepared",
                template="qwen"
            ),
            training=TrainingConfig(
                output_dir="models/fine_tuned_dora",
                num_train_epochs=2,
                learning_rate=1e-4
            ),
            peft=PEFTConfig(
                method=PEFTMethod.DORA,
                rank=8,
                alpha=16,
                use_dora=True
            )
        )
        
        self.save_config(dora_config, "dora_advanced.yaml")


# Global config manager
yaml_config_manager = YAMLConfigManager()