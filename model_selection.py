import os
import torch
import json
from typing import Dict, Optional, List
from pathlib import Path
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel


class SelectModelRequest(BaseModel):
    model_id: str
    output_dir: str = "/app/models/base"
    quantize: Optional[str] = "none"  # none, 4bit, 8bit


def authenticate_huggingface() -> None:
    """Authenticate with Hugging Face using token from environment."""
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is required")
    login(token=token)


def scan_available_models(models_dir: str = "/app/models/base") -> Dict[str, any]:
    """
    Scan the models directory for available models.
    
    Args:
        models_dir: Directory to scan for models
        
    Returns:
        Dictionary with available models and their metadata
    """
    try:
        models_path = Path(models_dir)
        available_models = []
        
        if not models_path.exists():
            models_path.mkdir(parents=True, exist_ok=True)
            
        # Scan for model directories
        for model_dir in models_path.iterdir():
            if model_dir.is_dir():
                # Check if it's a valid model directory (has config.json or similar files)
                config_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
                has_model_files = any((model_dir / file).exists() for file in config_files)
                
                if has_model_files:
                    # Extract model metadata
                    # Convert path to Unix format for Docker compatibility
                    unix_path = str(model_dir).replace("\\", "/")
                    # If running locally, convert to Docker path format
                    if not unix_path.startswith("/app/"):
                        unix_path = unix_path.replace("models/", "/app/models/")
                    
                    model_info = {
                        "model_id": model_dir.name.replace("_", "/"),
                        "local_name": model_dir.name,
                        "path": unix_path,
                        "status": "available",
                        "size_mb": get_directory_size(model_dir),
                        "files": [str(f.name) for f in model_dir.glob("*")][:10],  # First 10 files for info
                        "has_tokenizer": (model_dir / "tokenizer.json").exists() or (model_dir / "tokenizer_config.json").exists()
                    }
                    
                    # Try to read model config for additional info
                    config_path = model_dir / "config.json"
                    if config_path.exists():
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                                model_info["architecture"] = config.get("architectures", ["unknown"])[0]
                                model_info["model_type"] = config.get("model_type", "unknown")
                        except:
                            pass
                            
                    available_models.append(model_info)
        
        return {
            "status": "success",
            "models": available_models,
            "models_directory": str(models_path),
            "total_models": len(available_models)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to scan models: {str(e)}",
            "models": []
        }


def get_directory_size(path: Path) -> float:
    """Get the size of a directory in MB."""
    try:
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return round(total_size / (1024 * 1024), 2)  # Convert to MB
    except:
        return 0.0


def model_exists(model_id: str, models_dir: str = "/app/models/base") -> bool:
    """
    Check if a model already exists locally.
    
    Args:
        model_id: The model ID (e.g., 'unsloth/llama-3.1-8b-unsloth-bnb-4bit')
        models_dir: Directory where models are stored
        
    Returns:
        True if model exists, False otherwise
    """
    models_path = Path(models_dir)
    model_local_name = model_id.replace("/", "_")
    model_path = models_path / model_local_name
    
    if not model_path.exists():
        return False
        
    # Check if it has essential model files
    config_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
    return any((model_path / file).exists() for file in config_files)


def select_model(request: SelectModelRequest) -> Dict[str, str]:
    """
    Download and save a base model from Hugging Face Hub, or use existing one if available.
    
    Args:
        request: SelectModelRequest containing model_id, output_dir, and quantization options
        
    Returns:
        Dictionary with status, message, and model_path
    """
    try:
        # Check if model already exists
        if model_exists(request.model_id, request.output_dir):
            model_local_name = request.model_id.replace("/", "_")
            existing_model_path = Path(request.output_dir) / model_local_name
            
            return {
                "status": "success",
                "message": f"Model {request.model_id} is already available locally",
                "model_path": str(existing_model_path),
                "already_exists": True
            }
        
        authenticate_huggingface()
        
        # Create output directory
        output_path = Path(request.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_path = output_path / request.model_id.replace("/", "_")
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Check if Unsloth supports this model
        unsloth_supported_models = [
            "unsloth/llama-3.1-8b-unsloth-bnb-4bit",
            "unsloth/llama-3.1-70b-unsloth-bnb-4bit", 
            "unsloth/mistral-7b-unsloth-bnb-4bit",
            "unsloth/codellama-34b-unsloth-bnb-4bit"
        ]
        
        if request.model_id in unsloth_supported_models or "unsloth" in request.model_id:
            # Use Unsloth for supported models
            try:
                from unsloth import FastLanguageModel
                
                max_seq_length = 2048
                dtype = None  # Auto detection
                load_in_4bit = request.quantize == "4bit"
                
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=request.model_id,
                    max_seq_length=max_seq_length,
                    dtype=dtype,
                    load_in_4bit=load_in_4bit,
                )
                
                # Save model and tokenizer
                model.save_pretrained(str(model_path))
                tokenizer.save_pretrained(str(model_path))
                
            except ImportError:
                # Fall back to transformers if Unsloth not available
                use_standard_loading = True
            except Exception as e:
                # Fall back to transformers if Unsloth fails
                use_standard_loading = True
        else:
            use_standard_loading = True
            
        if 'use_standard_loading' in locals() and use_standard_loading:
            # Standard transformers loading with optimizations
            print(f"Loading {request.model_id} with standard transformers...")
            
            load_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            if request.quantize == "4bit":
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            elif request.quantize == "8bit":
                load_kwargs["load_in_8bit"] = True
                
            if torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"
                load_kwargs["torch_dtype"] = torch.float16
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(request.model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                request.model_id,
                **load_kwargs
            )
            
            # Validate it's a causal LM
            if not hasattr(model, 'lm_head'):
                raise ValueError(f"Model {request.model_id} is not a causal language model")
            
            # Save model and tokenizer
            model.save_pretrained(str(model_path))
            tokenizer.save_pretrained(str(model_path))
        
        return {
            "status": "success",
            "message": f"Model {request.model_id} downloaded and saved successfully",
            "model_path": str(model_path),
            "already_exists": False
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Failed to download model: {str(e)}",
            "model_path": ""
        }