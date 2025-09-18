"""
Hardware Detection and Adaptive Configuration System

Dynamically detects system capabilities and adapts training configurations
for optimal performance across different hardware profiles.
"""

import os
import psutil
import platform
import subprocess
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import torch

from utils.logging import get_logger

try:
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    def is_bfloat16_supported():
        return False

logger = get_logger(__name__)


@dataclass
class HardwareProfile:
    """Hardware profile with detected capabilities"""
    # GPU Information
    gpu_count: int
    gpu_memory_gb: float
    gpu_name: str
    gpu_compute_capability: Optional[Tuple[int, int]]
    gpu_driver_version: str
    cuda_available: bool
    
    # CPU Information  
    cpu_cores: int
    cpu_threads: int
    cpu_freq_max: float
    cpu_architecture: str
    
    # Memory Information
    system_memory_gb: float
    available_memory_gb: float
    container_memory_gb: Optional[float]
    
    # Storage Information
    storage_type: str  # SSD, HDD, NVMe
    available_storage_gb: float
    
    # System Information
    os_type: str
    python_version: str
    is_container: bool
    is_kubernetes: bool


class HardwareDetector:
    """Detects and profiles system hardware capabilities"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def detect_full_profile(self) -> HardwareProfile:
        """Detect complete hardware profile"""
        try:
            return HardwareProfile(
                # GPU Detection
                gpu_count=self._detect_gpu_count(),
                gpu_memory_gb=self._detect_gpu_memory(),
                gpu_name=self._detect_gpu_name(),
                gpu_compute_capability=self._detect_compute_capability(),
                gpu_driver_version=self._detect_gpu_driver(),
                cuda_available=torch.cuda.is_available(),
                
                # CPU Detection
                cpu_cores=self._detect_cpu_cores(),
                cpu_threads=self._detect_cpu_threads(),
                cpu_freq_max=self._detect_cpu_frequency(),
                cpu_architecture=self._detect_cpu_architecture(),
                
                # Memory Detection
                system_memory_gb=self._detect_system_memory(),
                available_memory_gb=self._detect_available_memory(),
                container_memory_gb=self._detect_container_memory(),
                
                # Storage Detection
                storage_type=self._detect_storage_type(),
                available_storage_gb=self._detect_available_storage(),
                
                # System Detection
                os_type=self._detect_os_type(),
                python_version=self._detect_python_version(),
                is_container=self._detect_container_environment(),
                is_kubernetes=self._detect_kubernetes_environment(),
            )
        except Exception as e:
            self.logger.error(f"Hardware detection failed: {e}")
            # Return minimal safe profile
            return self._get_safe_fallback_profile()
    
    def _detect_gpu_count(self) -> int:
        """Detect number of available GPUs"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.device_count()
            return 0
        except Exception:
            return 0
    
    def _detect_gpu_memory(self) -> float:
        """Detect total GPU memory in GB"""
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                # Get memory of primary GPU
                props = torch.cuda.get_device_properties(0)
                return props.total_memory / (1024**3)  # Convert bytes to GB
            return 0.0
        except Exception:
            return 0.0
    
    def _detect_gpu_name(self) -> str:
        """Detect GPU name/model"""
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                props = torch.cuda.get_device_properties(0)
                return props.name
            return "No GPU"
        except Exception:
            return "Unknown GPU"
    
    def _detect_compute_capability(self) -> Optional[Tuple[int, int]]:
        """Detect GPU compute capability"""
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                props = torch.cuda.get_device_properties(0)
                return (props.major, props.minor)
            return None
        except Exception:
            return None
    
    def _detect_gpu_driver(self) -> str:
        """Detect GPU driver version"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
            return "Unknown"
        except Exception:
            return "Unknown"
    
    def _detect_cpu_cores(self) -> int:
        """Detect number of physical CPU cores"""
        return psutil.cpu_count(logical=False) or 1
    
    def _detect_cpu_threads(self) -> int:
        """Detect number of logical CPU threads"""
        return psutil.cpu_count(logical=True) or 1
    
    def _detect_cpu_frequency(self) -> float:
        """Detect maximum CPU frequency in GHz"""
        try:
            freq_info = psutil.cpu_freq()
            if freq_info:
                return freq_info.max / 1000.0  # Convert MHz to GHz
            return 2.0  # Reasonable default
        except Exception:
            return 2.0
    
    def _detect_cpu_architecture(self) -> str:
        """Detect CPU architecture"""
        return platform.machine()
    
    def _detect_system_memory(self) -> float:
        """Detect total system memory in GB"""
        try:
            memory = psutil.virtual_memory()
            return memory.total / (1024**3)  # Convert bytes to GB
        except Exception:
            return 8.0  # Safe fallback
    
    def _detect_available_memory(self) -> float:
        """Detect currently available memory in GB"""
        try:
            memory = psutil.virtual_memory()
            return memory.available / (1024**3)  # Convert bytes to GB
        except Exception:
            return 4.0  # Safe fallback
    
    def _detect_container_memory(self) -> Optional[float]:
        """Detect container memory limit if running in container"""
        try:
            # Check cgroup memory limit
            cgroup_paths = [
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                "/sys/fs/cgroup/memory.max",
                "/proc/meminfo"
            ]
            
            for path in cgroup_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        content = f.read().strip()
                        if path.endswith("memory.limit_in_bytes"):
                            limit = int(content)
                            # Convert to GB, ignore if it's the system max
                            if limit < (2**63 - 1):  # Not system max
                                return limit / (1024**3)
                        elif path.endswith("memory.max"):
                            if content != "max":
                                limit = int(content)
                                return limit / (1024**3)
            
            # Check environment variables
            container_memory = os.environ.get("CONTAINER_MEMORY_LIMIT_GB")
            if container_memory:
                return float(container_memory)
                
            return None
        except Exception:
            return None
    
    def _detect_storage_type(self) -> str:
        """Detect primary storage type"""
        try:
            # Try to detect if we're on SSD/NVMe
            if os.path.exists("/sys/block"):
                for device in os.listdir("/sys/block"):
                    if device.startswith(("sd", "nvme", "vd")):
                        queue_path = f"/sys/block/{device}/queue/rotational"
                        if os.path.exists(queue_path):
                            with open(queue_path, 'r') as f:
                                rotational = f.read().strip()
                                if rotational == "0":
                                    if device.startswith("nvme"):
                                        return "NVMe"
                                    return "SSD"
                                else:
                                    return "HDD"
            return "Unknown"
        except Exception:
            return "Unknown"
    
    def _detect_available_storage(self) -> float:
        """Detect available storage space in GB"""
        try:
            # Check current working directory space
            statvfs = os.statvfs('.')
            available_bytes = statvfs.f_frsize * statvfs.f_bavail
            return available_bytes / (1024**3)  # Convert to GB
        except Exception:
            return 10.0  # Safe fallback
    
    def _detect_os_type(self) -> str:
        """Detect operating system"""
        return platform.system()
    
    def _detect_python_version(self) -> str:
        """Detect Python version"""
        return platform.python_version()
    
    def _detect_container_environment(self) -> bool:
        """Detect if running in a container"""
        # Check for common container indicators
        container_indicators = [
            os.path.exists("/.dockerenv"),
            os.path.exists("/run/secrets/kubernetes.io"),
            os.environ.get("container") is not None,
            os.environ.get("KUBERNETES_SERVICE_HOST") is not None,
        ]
        return any(container_indicators)
    
    def _detect_kubernetes_environment(self) -> bool:
        """Detect if running in Kubernetes"""
        return os.environ.get("KUBERNETES_SERVICE_HOST") is not None
    
    def _get_safe_fallback_profile(self) -> HardwareProfile:
        """Return a minimal safe hardware profile for error cases"""
        return HardwareProfile(
            gpu_count=0,
            gpu_memory_gb=0.0,
            gpu_name="Unknown",
            gpu_compute_capability=None,
            gpu_driver_version="Unknown",
            cuda_available=False,
            cpu_cores=2,
            cpu_threads=2,
            cpu_freq_max=2.0,
            cpu_architecture="x86_64",
            system_memory_gb=4.0,
            available_memory_gb=2.0,
            container_memory_gb=None,
            storage_type="Unknown",
            available_storage_gb=10.0,
            os_type="Linux",
            python_version="3.11",
            is_container=True,
            is_kubernetes=False
        )


class AdaptiveTrainingConfig:
    """Generates adaptive training configurations based on hardware profile"""
    
    def __init__(self, hardware_profile: HardwareProfile):
        self.profile = hardware_profile
        self.logger = get_logger(__name__)
    
    def generate_optimal_config(self, base_request) -> Dict[str, Any]:
        """Generate optimal training configuration for detected hardware"""
        
        # Determine memory constraint (use container limit if available)
        effective_memory_gb = (
            self.profile.container_memory_gb or 
            min(self.profile.available_memory_gb, self.profile.system_memory_gb * 0.8)
        )
        
        # Extract model name from request for precision optimization
        model_name = getattr(base_request, 'model_name', None) or getattr(base_request, 'base_model', None)
        
        # Base configuration with model-aware optimizations
        config = {
            "hardware_profile": "auto_detected",
            "effective_memory_gb": effective_memory_gb,
            "gpu_optimizations": self._get_gpu_optimizations(model_name),
            "training_args": self._get_adaptive_training_args(base_request, effective_memory_gb),
            "model_loading": self._get_model_loading_config(effective_memory_gb),
            "lora_config": self._get_adaptive_lora_config(effective_memory_gb),
            "system_optimizations": self._get_system_optimizations(),
            "precision_info": self._get_precision_info(model_name),
        }
        
        self.logger.info(f"Generated adaptive config for {self.profile.gpu_name} with {effective_memory_gb:.1f}GB memory")
        self.logger.info(f"Model precision: {config['gpu_optimizations']['torch_dtype']} (optimized for {model_name or 'generic'})")
        return config
    
    def _get_gpu_optimizations(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get GPU-specific optimizations with model-aware precision selection"""
        if not self.profile.cuda_available or self.profile.gpu_count == 0:
            return {
                "enabled": False,
                "device_map": "cpu",
                "torch_dtype": "float32",
                "load_in_4bit": False,
                "load_in_8bit": False,
            }
        
        # Detect optimal precision for model and hardware combination
        optimal_dtype = self._detect_optimal_precision(model_name)
        
        # GPU memory-based optimizations
        gpu_memory = self.profile.gpu_memory_gb
        
        if gpu_memory <= 4:  # Low VRAM GPUs (GTX 1650, RTX 3050 4GB, etc.)
            return {
                "enabled": True,
                "device_map": "auto",
                "torch_dtype": optimal_dtype if optimal_dtype != "bfloat16" else "float16",  # Fallback for older GPUs
                "load_in_4bit": True,
                "load_in_8bit": False,
                "flash_attention": False,  # May not be supported on older cards
                "gradient_checkpointing": True,
            }
        elif gpu_memory <= 8:  # Mid-range GPUs (RTX 4060, RTX 3070, etc.)
            return {
                "enabled": True,
                "device_map": "auto",
                "torch_dtype": optimal_dtype,  # Use model-optimized precision
                "load_in_4bit": True,
                "load_in_8bit": False,
                "flash_attention": True,
                "gradient_checkpointing": True,
            }
        elif gpu_memory <= 16:  # High-end consumer GPUs (RTX 4080, RTX 3080 Ti, etc.)
            return {
                "enabled": True,
                "device_map": "auto", 
                "torch_dtype": optimal_dtype,  # Use model-optimized precision
                "load_in_4bit": False,
                "load_in_8bit": True,
                "flash_attention": True,
                "gradient_checkpointing": False,
            }
        else:  # Enthusiast/Professional GPUs (RTX 4090, A100, etc.)
            return {
                "enabled": True,
                "device_map": "balanced",
                "torch_dtype": optimal_dtype,  # Use model-optimized precision
                "load_in_4bit": False,
                "load_in_8bit": False,
                "flash_attention": True,
                "gradient_checkpointing": False,
            }
    
    def _get_adaptive_training_args(self, base_request, memory_gb: float) -> Dict[str, Any]:
        """Get adaptive training arguments based on hardware"""
        
        # Base configuration
        args = {
            "dataloader_num_workers": min(self.profile.cpu_cores - 1, 4),
            "dataloader_pin_memory": self.profile.system_memory_gb > 16,
        }
        
        # Memory-based batch size and gradient accumulation
        if memory_gb <= 4:  # Very constrained
            args.update({
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": max(base_request.gradient_accumulation_steps, 32),
                "eval_strategy": "no",
                "save_steps": 25,
                "logging_steps": 5,
                "save_total_limit": 1,
                "report_to": [],
            })
        elif memory_gb <= 8:  # Constrained
            args.update({
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": max(base_request.gradient_accumulation_steps, 16),
                "eval_strategy": "no",
                "save_steps": 50,
                "logging_steps": 10,
                "save_total_limit": 2,
                "report_to": [],
            })
        elif memory_gb <= 16:  # Moderate
            args.update({
                "per_device_train_batch_size": min(base_request.batch_size, 2),
                "gradient_accumulation_steps": max(base_request.gradient_accumulation_steps, 8),
                "eval_strategy": "steps" if hasattr(base_request, 'eval_steps') else "no",
                "save_steps": 100,
                "logging_steps": 10,
                "save_total_limit": 3,
            })
        else:  # High memory
            args.update({
                "per_device_train_batch_size": base_request.batch_size,
                "gradient_accumulation_steps": base_request.gradient_accumulation_steps,
                "eval_strategy": "steps",
                "save_steps": 100,
                "logging_steps": 10,
                "save_total_limit": 3,
            })
        
        return args
    
    def _get_model_loading_config(self, memory_gb: float) -> Dict[str, Any]:
        """Get model loading configuration"""
        config = {
            "trust_remote_code": True,
        }
        
        # Sequence length based on memory
        if memory_gb <= 4:
            config["max_seq_length"] = 512
        elif memory_gb <= 8:
            config["max_seq_length"] = 1024  
        elif memory_gb <= 16:
            config["max_seq_length"] = 2048
        else:
            config["max_seq_length"] = 4096
            
        return config
    
    def _get_adaptive_lora_config(self, memory_gb: float) -> Dict[str, Any]:
        """Get LoRA configuration based on available memory"""
        
        if memory_gb <= 4:  # Very constrained - minimal LoRA
            return {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": [
                    "q_proj", "v_proj",  # Minimal attention modules
                    "embed_tokens", "lm_head"  # Essential for Qwen2.5
                ]
            }
        elif memory_gb <= 8:  # Constrained - standard LoRA
            return {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",  # All attention
                    "embed_tokens", "lm_head"  # Essential for Qwen2.5
                ]
            }
        elif memory_gb <= 16:  # Moderate - full LoRA
            return {
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                    "gate_proj", "up_proj", "down_proj",      # MLP
                    "embed_tokens", "lm_head"  # Essential for Qwen2.5
                ]
            }
        else:  # High memory - high-rank LoRA
            return {
                "r": 64,
                "lora_alpha": 128,
                "lora_dropout": 0.05,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                    "gate_proj", "up_proj", "down_proj",      # MLP  
                    "embed_tokens", "lm_head"  # Essential for Qwen2.5
                ]
            }
    
    def _get_system_optimizations(self) -> Dict[str, Any]:
        """Get system-level optimizations"""
        optimizations = {}
        
        # CPU optimizations
        if self.profile.cpu_cores >= 8:
            optimizations["OMP_NUM_THREADS"] = str(min(self.profile.cpu_cores, 8))
        else:
            optimizations["OMP_NUM_THREADS"] = "1"  # Avoid contention on lower-end CPUs
        
        optimizations["MKL_NUM_THREADS"] = optimizations["OMP_NUM_THREADS"]
        
        # Memory optimizations
        if self.profile.system_memory_gb <= 8:
            optimizations["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"
        else:
            optimizations["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024,expandable_segments:True"
        
        # Storage optimizations
        if self.profile.storage_type in ["SSD", "NVMe"]:
            optimizations["dataset_cache_strategy"] = "memory_mapped"
        else:
            optimizations["dataset_cache_strategy"] = "minimal"
        
        return optimizations
    
    def _detect_optimal_precision(self, model_name: Optional[str] = None) -> str:
        """Detect optimal precision for model and hardware combination"""
        
        # Model-specific precision preferences
        model_precision_preferences = {
            "qwen": "bfloat16",  # Qwen models work best with bfloat16 in Unsloth
            "llama": "bfloat16",  # LLaMA models prefer bfloat16
            "mistral": "bfloat16",  # Mistral models prefer bfloat16
            "phi": "float16",  # Phi models work better with float16
            "gemma": "bfloat16",  # Gemma models prefer bfloat16
            "codellama": "bfloat16",  # CodeLLaMA prefers bfloat16
        }
        
        # Check hardware bfloat16 support
        hardware_supports_bf16 = self._check_bfloat16_support()
        
        # Determine model family
        model_family = "generic"
        if model_name:
            model_name_lower = model_name.lower()
            for family in model_precision_preferences.keys():
                if family in model_name_lower:
                    model_family = family
                    break
        
        # Get model preference
        preferred_precision = model_precision_preferences.get(model_family, "float16")
        
        # Return optimal precision based on model preference and hardware support
        if preferred_precision == "bfloat16" and hardware_supports_bf16:
            return "bfloat16"
        elif preferred_precision == "bfloat16" and not hardware_supports_bf16:
            self.logger.warning(f"Model {model_name} prefers bfloat16 but hardware doesn't support it, falling back to float16")
            return "float16"
        else:
            return preferred_precision
    
    def _check_bfloat16_support(self) -> bool:
        """Check if hardware supports bfloat16 precision"""
        if not self.profile.cuda_available:
            return False
            
        # Use Unsloth detection if available
        if UNSLOTH_AVAILABLE:
            try:
                return is_bfloat16_supported()
            except Exception:
                pass
        
        # Fallback: check compute capability and GPU generation
        if self.profile.gpu_compute_capability:
            major, minor = self.profile.gpu_compute_capability
            # Ampere (8.x) and newer support bfloat16, some Turing (7.5) cards do too
            if major >= 8:  # Ampere, Ada Lovelace, Hopper
                return True
            elif major == 7 and minor >= 5:  # Some Turing cards (RTX 20x0, GTX 16x0 Super)
                return True
        
        # Check by GPU name patterns for known bfloat16 support
        gpu_name = self.profile.gpu_name.lower()
        bf16_supported_patterns = [
            "rtx 40",  # RTX 4000 series
            "rtx 30",  # RTX 3000 series
            "rtx 20",  # RTX 2000 series (most)
            "a100", "a40", "a30", "a10",  # Professional Ampere
            "v100",  # Some configurations
            "t4",  # Turing T4
        ]
        
        for pattern in bf16_supported_patterns:
            if pattern in gpu_name:
                return True
                
        return False
    
    def _get_precision_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get precision information and recommendations"""
        optimal_precision = self._detect_optimal_precision(model_name)
        hardware_support = {
            "bfloat16": self._check_bfloat16_support(),
            "float16": self.profile.cuda_available,  # All CUDA GPUs support float16
            "float32": True,  # Always supported
        }
        
        return {
            "optimal_precision": optimal_precision,
            "hardware_support": hardware_support,
            "model_family": self._get_model_family(model_name),
            "fallback_precision": "float16" if hardware_support["float16"] else "float32",
        }
    
    def _get_model_family(self, model_name: Optional[str] = None) -> str:
        """Identify model family from model name"""
        if not model_name:
            return "generic"
            
        model_name_lower = model_name.lower()
        model_families = {
            "qwen": ["qwen", "qwen2", "qwen2.5", "qwen3"],
            "llama": ["llama", "llama2", "llama-2", "llama3", "llama-3"],
            "mistral": ["mistral", "mixtral"],
            "phi": ["phi", "phi-2", "phi-3"],
            "gemma": ["gemma"],
            "codellama": ["codellama", "code-llama"],
        }
        
        for family, patterns in model_families.items():
            for pattern in patterns:
                if pattern in model_name_lower:
                    return family
                    
        return "generic"
    
    def get_hardware_summary(self) -> Dict[str, Any]:
        """Get human-readable hardware summary"""
        return {
            "gpu": f"{self.profile.gpu_name} ({self.profile.gpu_memory_gb:.1f}GB)" if self.profile.cuda_available else "CPU Only",
            "memory": f"{self.profile.system_memory_gb:.1f}GB System",
            "container_limit": f"{self.profile.container_memory_gb:.1f}GB" if self.profile.container_memory_gb else "No Limit",
            "cpu": f"{self.profile.cpu_cores}C/{self.profile.cpu_threads}T @ {self.profile.cpu_freq_max:.1f}GHz",
            "storage": f"{self.profile.storage_type} ({self.profile.available_storage_gb:.1f}GB available)",
            "environment": "Container" if self.profile.is_container else "Native",
            "precision_support": {
                "bfloat16": self._check_bfloat16_support(),
                "float16": self.profile.cuda_available,
            }
        }


# Global hardware detector instance
hardware_detector = HardwareDetector()