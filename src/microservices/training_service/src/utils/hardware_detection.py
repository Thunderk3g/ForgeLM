"""
Hardware Detection and Adaptive Configuration System

Dynamically detects system capabilities and adapts training configurations
for optimal performance across different hardware profiles.

Vendor awareness
----------------
The VRAM tiers in ``AdaptiveTrainingConfig`` were tuned on specific NVIDIA
cards (see FINAL_CONFIG_RTX3050.md and RTX_3050_4GB_SETUP.md at the repository
root). They are kept exactly as they were for NVIDIA CUDA: a test compares
every setting against the version of this file before vendor awareness was
added. What changed is what happens on everything else:

    rocm   AMD GPU. PyTorch reports it through ``torch.cuda`` with
           ``torch.version.hip`` set, so it used to be configured as an NVIDIA
           card, including bitsandbytes 4-bit loading, which is reported to
           be unreliable there. It keeps the tiers but defaults to LoRA on
           16-bit weights when the model fits.
    mps    Apple Silicon. Used to be configured as a CPU. Memory is shared
           with the operating system, so a fraction of system RAM is budgeted.
    xpu    Intel GPU (``torch.xpu``). Used to be configured as a CPU.
    none   No accelerator: the original CPU fallback, unchanged.

None of the non-NVIDIA paths has been run on real AMD, Intel or Apple
hardware. They are exercised with simulated torch in
tests/factory/test_training_hardware.py.
"""

import os
import re
import psutil
import platform
import subprocess
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import torch

from utils.logging import get_logger

try:
    # The bridge owns the device-to-dtype rule (shared with torch_device_and_dtype)
    # and the memory estimate. It is optional here: detection has to keep working
    # if the bridge cannot be imported, so every use below has a fallback.
    from utils import factory_bridge as _bridge
except Exception:
    _bridge = None

try:
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    def is_bfloat16_supported():
        return False

logger = get_logger(__name__)

# Environment override for AMD ROCm. By default a ROCm GPU does not use
# bitsandbytes 4-bit / 8-bit loading when the model fits as 16-bit weights,
# because quantised loading is reported to produce NaN losses there. Someone
# who has verified that quantised loading works on their card can set
#     FORGELM_ROCM_ALLOW_4BIT=1      (also accepted: true, yes, on)
# to get the NVIDIA quantised-loading tiers on ROCm unchanged.
ROCM_ALLOW_4BIT_ENV = "FORGELM_ROCM_ALLOW_4BIT"

# Apple Silicon has no separate VRAM: the GPU uses system RAM, which the
# operating system and every other program also need. This fraction of total
# RAM is budgeted for training (forgelm.generative uses the same 65%). Set
#     FORGELM_MPS_MEMORY_FRACTION=0.5   (any value above 0 and at most 1)
# to budget a different share.
MPS_MEMORY_FRACTION_ENV = "FORGELM_MPS_MEMORY_FRACTION"
DEFAULT_MPS_MEMORY_FRACTION = 0.65

# Accelerators that have their own configuration path. "cuda" (NVIDIA) and
# "none" (CPU) are deliberately not in this set: they take the original code.
_VENDOR_ACCELERATORS = ("rocm", "mps", "xpu")


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

    # Accelerator runtime. These fields come last and have defaults so that
    # code which builds a profile from the original fields keeps working; such
    # a profile is read as NVIDIA CUDA when cuda_available is set and as a CPU
    # otherwise, which is what the original fields meant.
    accelerator: str = ""  # "cuda" | "rocm" | "mps" | "xpu" | "none"; "" means derive it
    # Memory to plan training against: VRAM for cuda, rocm and xpu; the
    # budgeted share of unified memory for mps; 0 without an accelerator.
    accelerator_memory_gb: float = 0.0
    accelerator_name: str = ""
    # What torch.cuda.is_bf16_supported() answered. Only asked on ROCm, where
    # the NVIDIA compute-capability and card-name rules below say nothing.
    bf16_supported: Optional[bool] = None
    # Things the detector had to assume, for example the unified-memory share.
    accelerator_notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.accelerator:
            self.accelerator = "cuda" if self.cuda_available else "none"
        if self.accelerator != "none":
            if not self.accelerator_memory_gb:
                self.accelerator_memory_gb = self.gpu_memory_gb
            if not self.accelerator_name:
                self.accelerator_name = self.gpu_name


class HardwareDetector:
    """Detects and profiles system hardware capabilities"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        # The most recent profile, so that callers that only need a summary
        # (current_hardware_warnings) do not run the whole detection again.
        self.last_profile: Optional[HardwareProfile] = None
    
    def detect_full_profile(self) -> HardwareProfile:
        """Detect complete hardware profile"""
        try:
            accelerator = self._detect_accelerator()

            if accelerator in _VENDOR_ACCELERATORS:
                # nvidia-smi does not exist on these machines; its absence is not worth a warning.
                self.logger.info(f"Accelerator runtime: {accelerator} (nvidia-smi status check skipped)")
            else:
                # Log nvidia-smi for GPU status
                self._log_nvidia_smi_status()

            profile = HardwareProfile(
                # GPU Detection
                gpu_count=self._detect_gpu_count(),
                gpu_memory_gb=self._detect_gpu_memory(),
                gpu_name=self._detect_gpu_name(),
                gpu_compute_capability=self._detect_compute_capability(),
                # The driver query is an nvidia-smi call: there is nothing to ask on AMD, Apple or Intel.
                gpu_driver_version="Unknown" if accelerator in _VENDOR_ACCELERATORS else self._detect_gpu_driver(),
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

                # Accelerator runtime
                accelerator=accelerator,
                bf16_supported=self._detect_rocm_bf16_support() if accelerator == "rocm" else None,
            )
            if accelerator in ("mps", "xpu"):
                self._describe_native_accelerator(profile)
            self.last_profile = profile
            return profile
        except Exception as e:
            self.logger.error(f"Hardware detection failed: {e}")
            # Return minimal safe profile
            self.last_profile = self._get_safe_fallback_profile()
            return self.last_profile

    # ---- accelerator runtime ------------------------------------------------
    # Every probe below reads torch attributes that do not exist on older
    # releases (torch.xpu, torch.backends.mps, torch.version.hip), so each one
    # goes through getattr and a try block and answers "not there" on failure.

    def _detect_accelerator(self) -> str:
        """Which accelerator runtime torch can use: cuda, rocm, mps, xpu or none.

        The order (CUDA API, then XPU, then MPS) is the order
        factory_bridge.torch_device_and_dtype uses, so the two always agree
        about which device a model is loaded on.
        """
        try:
            if torch.cuda.is_available():
                # AMD's ROCm build answers through torch.cuda; torch.version.hip is how it differs.
                hip = getattr(getattr(torch, "version", None), "hip", None)
                return "rocm" if hip else "cuda"
        except Exception:
            pass
        try:
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                return "xpu"
        except Exception:
            pass
        try:
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
        except Exception:
            pass
        return "none"

    def _detect_rocm_bf16_support(self) -> Optional[bool]:
        """torch's own answer about bfloat16 on a ROCm GPU, or None if it cannot say."""
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            return None

    def _mps_memory_fraction(self) -> float:
        """Share of system RAM budgeted for training on Apple Silicon."""
        raw = os.environ.get(MPS_MEMORY_FRACTION_ENV)
        if raw:
            try:
                value = float(raw)
                if 0.0 < value <= 1.0:
                    return value
            except ValueError:
                pass
            self.logger.warning(
                f"{MPS_MEMORY_FRACTION_ENV}={raw!r} is not a number above 0 and at most 1; "
                f"using {DEFAULT_MPS_MEMORY_FRACTION}"
            )
        return DEFAULT_MPS_MEMORY_FRACTION

    def _detect_apple_chip_name(self) -> str:
        """Chip name on macOS (for example "Apple M2"), or a generic name."""
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            name = result.stdout.strip() if result.returncode == 0 else ""
            if name:
                return name
        except Exception:
            pass
        return "Apple Silicon"

    def _detect_xpu_memory(self) -> float:
        """Total memory of the first Intel XPU device in GB, or 0.0 when torch does not report it."""
        xpu = getattr(torch, "xpu", None)
        try:
            return xpu.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            pass
        try:
            # mem_get_info returns (free, total) in bytes.
            return xpu.mem_get_info(0)[1] / (1024**3)
        except Exception:
            return 0.0

    def _describe_native_accelerator(self, profile: HardwareProfile) -> None:
        """Fill in name and memory for an Apple MPS or Intel XPU device.

        The original gpu_* fields are filled as well as the accelerator_*
        ones, because the rest of the service reports gpu_name and
        gpu_memory_gb. cuda_available stays False: nothing on these machines
        can be reached through torch.cuda.
        """
        if profile.accelerator == "mps":
            fraction = self._mps_memory_fraction()
            name = self._detect_apple_chip_name()
            memory_gb = profile.system_memory_gb * fraction
            count = 1
            profile.accelerator_notes.append(
                f"Apple MPS shares system memory with the operating system: budgeting {fraction:.0%} of "
                f"{profile.system_memory_gb:.1f}GB ({memory_gb:.1f}GB) for training. "
                f"Set {MPS_MEMORY_FRACTION_ENV} to budget a different share."
            )
        else:
            xpu = getattr(torch, "xpu", None)
            try:
                count = int(xpu.device_count()) or 1
            except Exception:
                count = 1
            try:
                name = str(xpu.get_device_name(0))
            except Exception:
                name = "Intel XPU device"
            memory_gb = self._detect_xpu_memory()
            if memory_gb <= 0:
                profile.accelerator_notes.append(
                    "Intel XPU: this PyTorch build does not report the device's memory, so the most "
                    "conservative (smallest-memory) training settings are used."
                )
        profile.accelerator_name = name
        profile.accelerator_memory_gb = memory_gb
        profile.gpu_count = count
        profile.gpu_name = name
        profile.gpu_memory_gb = memory_gb

    def _log_nvidia_smi_status(self):
        """Log current nvidia-smi status for debugging GPU sharing issues"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=compute_mode,memory.usage', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            compute_mode, mem_usage = parts[0].strip(), parts[1].strip()
                            self.logger.info(f"GPU compute mode: {compute_mode}, Memory usage: {mem_usage}")
                            if 'MPS' in compute_mode or 'EXCLUSIVE_PROCESS' in compute_mode:
                                self.logger.info(f"MPS/Exclusive mode detected - good for sharing")
                            else:
                                self.logger.warning(f"GPU in {compute_mode} mode - may cause sharing issues")
            else:
                self.logger.warning("nvidia-smi query failed - GPU status unknown")
        except Exception as e:
            self.logger.warning(f"Error querying nvidia-smi: {e}")
    
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
        # Things the operator should know about this hardware. Always empty on
        # NVIDIA CUDA and on a CPU. Rebuilt whenever GPU settings are generated:
        # the hardware-level warnings plus those about the model just asked for.
        self.warnings: List[str] = self._hardware_warnings()

    # ---- vendor awareness ---------------------------------------------------
    # Helpers used only by the AMD ROCm, Apple MPS and Intel XPU paths. The
    # NVIDIA CUDA and CPU paths further down do not call them, apart from
    # _accelerator() and _tier_memory_gb(), which return for those machines
    # exactly what the original code read from the profile.

    def _accelerator(self) -> str:
        """The profile's accelerator runtime: cuda, rocm, mps, xpu or none."""
        accelerator = getattr(self.profile, "accelerator", None)
        if accelerator:
            return accelerator
        # A profile object without the field: read it the way the original code did.
        return "cuda" if self.profile.cuda_available else "none"

    def _tier_memory_gb(self) -> float:
        """The memory figure the VRAM tiers are applied to.

        For Apple MPS and Intel XPU this is the accelerator's memory. For
        everything else it is gpu_memory_gb, as it always was.
        """
        if self._accelerator() in ("mps", "xpu"):
            return getattr(self.profile, "accelerator_memory_gb", 0.0) or self.profile.gpu_memory_gb
        return self.profile.gpu_memory_gb

    def _native_dtype(self) -> str:
        """Weight dtype on Apple MPS or Intel XPU, by the rule the model loader uses.

        factory_bridge.torch_device_and_dtype loads the model, so its rule is
        the one to report. Without the bridge the answer is float32, which
        every device supports.
        """
        if _bridge is not None and hasattr(_bridge, "dtype_name_for"):
            return _bridge.dtype_name_for(self._accelerator())
        return "float32"

    def _supports_float16(self) -> bool:
        """Whether float16 weights can be used on this hardware."""
        if self._accelerator() in ("mps", "xpu"):
            return True
        return self.profile.cuda_available  # All CUDA GPUs support float16

    def _rocm_quantisation_verified(self) -> bool:
        """True when the operator has set FORGELM_ROCM_ALLOW_4BIT (see ROCM_ALLOW_4BIT_ENV)."""
        return os.environ.get(ROCM_ALLOW_4BIT_ENV, "").strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _params_billion_from_name(model_name: Optional[str]) -> Optional[float]:
        """Parameter count in billions, read from a model name, or None.

        "Qwen2.5-0.5B-Instruct" gives 0.5, "SmolLM2-135M" gives 0.135. A number
        counts only when it stands alone before the B or M, so "4bit", "4k"
        and the "7B" in "8x7B" (a mixture of eight experts, not a 7B model)
        are not read as sizes.
        """
        if not model_name:
            return None
        match = re.search(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)([bm])(?![a-z0-9])", str(model_name).lower())
        if not match:
            return None
        value = float(match.group(1))
        return value if match.group(2) == "b" else value / 1000.0

    @staticmethod
    def _tier_batch_and_seq(memory_gb: float) -> Tuple[int, int]:
        """Largest batch size and sequence length _get_adaptive_training_args allows in this tier."""
        if memory_gb <= 8:
            return 1, 1024
        if memory_gb <= 12:
            return 2, 1536
        return 4, 2048

    def _lora_16bit_memory_gb(self, params_billion: float, batch_size: int, seq_len: int) -> float:
        """Estimated peak memory for LoRA on 16-bit weights, in GB."""
        if _bridge is not None and hasattr(_bridge, "estimate_memory_gb"):
            estimate = _bridge.estimate_memory_gb(params_billion, bits=16, batch_size=batch_size, seq_len=seq_len)
            if isinstance(estimate, dict) and "total_gb" in estimate:
                return float(estimate["total_gb"])
        # Without forgelm: 16-bit weights are 2 GB per billion parameters; allow
        # half as much again for activations and adapter gradients, plus the
        # runtime's own overhead. Deliberately on the high side.
        return params_billion * 2.0 * 1.5 + 0.8

    def _hardware_warnings(self) -> List[str]:
        """Warnings that follow from the hardware alone, whatever model is trained."""
        accelerator = self._accelerator()
        if accelerator not in _VENDOR_ACCELERATORS:
            return []
        warnings = list(getattr(self.profile, "accelerator_notes", None) or [])
        if accelerator == "rocm":
            if self._rocm_quantisation_verified():
                warnings.append(
                    f"AMD ROCm: {ROCM_ALLOW_4BIT_ENV} is set, so the NVIDIA quantised-loading tiers "
                    "(bitsandbytes 4-bit and 8-bit) are used as they are. This service has not verified "
                    "that quantised loading works on this card."
                )
            else:
                warnings.append(
                    "AMD ROCm: bitsandbytes 4-bit and 8-bit loading is reported to be unreliable on ROCm "
                    "(NaN losses), so a model that fits is trained as LoRA on 16-bit weights and only a model "
                    f"too large for that keeps quantised loading. Set {ROCM_ALLOW_4BIT_ENV}=1 once quantised "
                    "loading is verified on this card."
                )
            warnings.append(
                "AMD ROCm: flash attention is turned off (it is unavailable on most Radeon cards); "
                "expect standard attention speed."
            )
        elif accelerator == "mps":
            warnings.append(
                "Apple MPS: bitsandbytes 4-bit and 8-bit loading is not used on MPS, so the model is loaded "
                f"as {self._native_dtype()} weights. A model needs about four times the memory it would as "
                "4-bit weights on an NVIDIA card."
            )
        elif accelerator == "xpu":
            warnings.append(
                "Intel XPU: bitsandbytes 4-bit and 8-bit loading is not used on XPU (support there is recent "
                f"and unverified by this service), so the model is loaded as {self._native_dtype()} weights."
            )
            warnings.append(
                "Intel XPU: Arc A-series GPUs have no float64 support; keep gradient scaling (GradScaler) off."
            )
        return warnings

    def _remember_warnings(self, model_warnings: List[str]) -> None:
        """Make self.warnings the hardware warnings plus those about the latest model."""
        warnings = self._hardware_warnings()
        warnings.extend(w for w in model_warnings if w not in warnings)
        self.warnings = warnings

    def _native_accelerator_optimizations(self, model_name: Optional[str] = None,
                                          params_billion: Optional[float] = None) -> Dict[str, Any]:
        """GPU settings for Apple MPS and Intel XPU.

        These devices used to get the CPU fallback. They are real accelerators
        with three differences from the NVIDIA tiers: no bitsandbytes 4-bit or
        8-bit loading (weights stay at 16 bits), no flash attention, and
        gradient checkpointing always on, because 16-bit weights leave less
        room for activations than the 4-bit weights the tiers assume.
        """
        accelerator = self._accelerator()
        memory_gb = self._tier_memory_gb()
        dtype = self._native_dtype()
        warnings: List[str] = []

        # There is no quantised fallback here, so a model that does not fit can
        # only be reported. Checked only when the model's size is known.
        params = params_billion or self._params_billion_from_name(model_name)
        if params:
            batch, seq = self._tier_batch_and_seq(memory_gb)
            needed = self._lora_16bit_memory_gb(params, batch, seq)
            budget = memory_gb * 0.85
            if needed > budget:
                label = "Apple MPS" if accelerator == "mps" else "Intel XPU"
                warnings.append(
                    f"{label}: {model_name or 'the model'} (about {params:g}B parameters) needs an estimated "
                    f"{needed:.1f}GB as 16-bit weights at batch {batch} and length {seq}, but {budget:.1f}GB of "
                    f"{memory_gb:.1f}GB is usable. Expect an out-of-memory failure; choose a smaller model."
                )

        self._remember_warnings(warnings)
        return {
            "enabled": True,
            # The whole model goes on the one device. "auto" would ask accelerate
            # to plan across CUDA devices, which do not exist here.
            "device_map": accelerator,
            "torch_dtype": dtype,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "flash_attention": False,
            "gradient_checkpointing": True,
            "accelerator": accelerator,
            "device": accelerator,
            "load_bits": 32 if dtype == "float32" else 16,
            "warnings": warnings,
        }

    def _adjust_for_rocm(self, opts: Dict[str, Any], model_name: Optional[str] = None,
                         params_billion: Optional[float] = None) -> Dict[str, Any]:
        """Turn the NVIDIA tier settings into settings for an AMD ROCm GPU.

        The tier logic stays. What changes: flash attention is off, and
        quantised loading is avoided when the model fits as 16-bit weights.
        When it does not fit, or its size is unknown, quantised loading stays
        and a warning says that it is the fragile path on this vendor.
        """
        opts = dict(opts)
        opts["flash_attention"] = False
        memory_gb = self._tier_memory_gb()
        budget = memory_gb * 0.85  # the headroom forgelm.generative.select_backend uses for AMD
        load_bits = 4 if opts["load_in_4bit"] else 8 if opts["load_in_8bit"] else 16
        warnings: List[str] = []

        params = params_billion or self._params_billion_from_name(model_name)
        needed = None
        if params:
            batch, seq = self._tier_batch_and_seq(memory_gb)
            needed = self._lora_16bit_memory_gb(params, batch, seq)
        name = model_name or "the model"

        if load_bits == 16:
            # The tier already loads unquantised. Nothing to change, but say so if it will not fit.
            if needed is not None and needed > budget:
                warnings.append(
                    f"AMD ROCm: {name} (about {params:g}B parameters) needs an estimated {needed:.1f}GB as "
                    f"16-bit weights, but {budget:.1f}GB of {memory_gb:.1f}GB is usable. The only smaller "
                    "option is bitsandbytes quantised loading, which is reported to be unreliable on ROCm."
                )
        elif self._rocm_quantisation_verified():
            pass  # the operator has verified quantised loading: keep the tier as tuned
        elif needed is not None and needed <= budget:
            opts["load_in_4bit"] = False
            opts["load_in_8bit"] = False
            # 16-bit weights take the room the tier assumed quantisation would free.
            opts["gradient_checkpointing"] = True
            load_bits = 16
        elif needed is None:
            warnings.append(
                f"AMD ROCm: the size of {name} could not be read from its name, so it is not known whether "
                f"it fits as 16-bit weights in {memory_gb:.1f}GB. Keeping {load_bits}-bit bitsandbytes "
                "loading, which is reported to be unreliable on ROCm (NaN losses): run a short test and "
                f"watch the loss before a long run, or set {ROCM_ALLOW_4BIT_ENV}=1 once it is verified."
            )
        else:
            warnings.append(
                f"AMD ROCm: {name} (about {params:g}B parameters) needs an estimated {needed:.1f}GB as "
                f"16-bit weights, but {budget:.1f}GB of {memory_gb:.1f}GB is usable. Keeping {load_bits}-bit "
                "bitsandbytes loading, which is reported to be unreliable on ROCm (NaN losses): run a short "
                f"test and watch the loss before a long run, or set {ROCM_ALLOW_4BIT_ENV}=1 once it is verified."
            )

        opts["accelerator"] = "rocm"
        opts["device"] = "cuda"  # ROCm builds of PyTorch name the GPU "cuda"
        opts["load_bits"] = load_bits
        opts["warnings"] = warnings
        self._remember_warnings(warnings)
        return opts

    def _generate_accelerator_config(self, base_request) -> Dict[str, Any]:
        """generate_optimal_config for AMD ROCm, Apple MPS and Intel XPU.

        Same keys and the same tier thresholds as the NVIDIA path, applied to
        the accelerator's memory, plus "accelerator" and "warnings".
        """
        accelerator = self._accelerator()
        memory_gb = self._tier_memory_gb()

        if accelerator == "mps":
            # accelerator_memory_gb is already the share of unified memory set
            # aside for training, so it is not discounted a second time.
            device_effective = memory_gb
        elif memory_gb <= 8:
            device_effective = memory_gb * 0.70
        elif memory_gb <= 12:
            device_effective = memory_gb * 0.80
        else:
            device_effective = memory_gb * 0.85

        container_effective = self.profile.container_memory_gb or min(self.profile.available_memory_gb, self.profile.system_memory_gb * 0.8)

        if accelerator == "mps":
            # Unified memory is the same pool as system RAM, so what is free
            # there limits training at every size.
            effective_memory_gb = min(device_effective, container_effective)
        elif memory_gb <= 8:
            effective_memory_gb = device_effective  # device memory is the bottleneck
        else:
            effective_memory_gb = min(device_effective, container_effective)

        model_name = getattr(base_request, 'model_name', None) or getattr(base_request, 'base_model', None) or getattr(base_request, 'base_model_dir', None)
        params_billion = getattr(base_request, 'model_params_billion', None)

        config = {
            "hardware_profile": "auto_detected",
            "effective_memory_gb": effective_memory_gb,
            "gpu_optimizations": self._get_gpu_optimizations(model_name, params_billion=params_billion),
            "training_args": self._get_adaptive_training_args(base_request, effective_memory_gb),
            "model_loading": self._get_model_loading_config(effective_memory_gb),
            "lora_config": self._get_adaptive_lora_config(effective_memory_gb, model_name),
            "system_optimizations": self._get_system_optimizations(),
            "precision_info": self._get_precision_info(model_name),
            "accelerator": accelerator,
            "warnings": list(self.warnings),
        }

        self.logger.info(f"Generated adaptive config for {self.profile.gpu_name} ({accelerator}, {memory_gb:.1f}GB) with {effective_memory_gb:.1f}GB effective memory")
        self.logger.info(f"Model precision: {config['gpu_optimizations']['torch_dtype']} (optimized for {model_name or 'generic'})")
        for warning in config["warnings"]:
            self.logger.warning(warning)
        return config

    def generate_optimal_config(self, base_request) -> Dict[str, Any]:
        """Generate optimal training configuration for detected hardware"""

        # AMD ROCm, Apple MPS and Intel XPU have their own path. NVIDIA CUDA and
        # CPU-only machines continue below, through the original code.
        if self._accelerator() in _VENDOR_ACCELERATORS:
            return self._generate_accelerator_config(base_request)
        
        # Enhanced GPU memory calculation for RTX 4060 8GB optimization
        if self.profile.gpu_memory_gb <= 8:
            # RTX 4060 8GB specific optimizations based on Unsloth 2025 benchmarks
            gpu_effective = self.profile.gpu_memory_gb * 0.70  # More conservative for stability
            self.logger.info(f"RTX 4060 8GB detected - applying Unsloth 2025 memory optimizations")
            self.logger.info(f"GPU memory allocation: {gpu_effective:.1f}GB / {self.profile.gpu_memory_gb}GB (70% for QLoRA-All)")
        elif self.profile.gpu_memory_gb <= 12:
            gpu_effective = self.profile.gpu_memory_gb * 0.80  # 80% for mid-range GPUs
        else:
            gpu_effective = self.profile.gpu_memory_gb * 0.85  # 85% for high-end GPUs
            
        # Use actual GPU memory as primary constraint for GPU-bound workloads
        gpu_effective = gpu_effective if self.profile.cuda_available else 0
        container_effective = self.profile.container_memory_gb or min(self.profile.available_memory_gb, self.profile.system_memory_gb * 0.8)
        
        # For RTX 4060 8GB, prioritize GPU memory as the limiting factor
        if self.profile.gpu_memory_gb <= 8:
            effective_memory_gb = gpu_effective  # GPU memory is the bottleneck
            self.logger.warning(f"GPU-optimized config: using {effective_memory_gb:.1f}GB GPU memory as primary constraint")
        else:
            effective_memory_gb = min(gpu_effective, container_effective)
        
        # Extract model name from request for precision optimization
        model_name = getattr(base_request, 'model_name', None) or getattr(base_request, 'base_model', None) or getattr(base_request, 'base_model_dir', None)
        
        # Base configuration with model-aware optimizations
        config = {
            "hardware_profile": "auto_detected",
            "effective_memory_gb": effective_memory_gb,
            "gpu_optimizations": self._get_gpu_optimizations(model_name),
            "training_args": self._get_adaptive_training_args(base_request, effective_memory_gb),
            "model_loading": self._get_model_loading_config(effective_memory_gb),
            "lora_config": self._get_adaptive_lora_config(effective_memory_gb, model_name),
            "system_optimizations": self._get_system_optimizations(),
            "precision_info": self._get_precision_info(model_name),
        }
        
        self.logger.info(f"Generated GPU-aware adaptive config for {self.profile.gpu_name} ({self.profile.gpu_memory_gb}GB GPU) with {effective_memory_gb:.1f}GB effective memory (capped by GPU)")
        self.logger.info(f"Model precision: {config['gpu_optimizations']['torch_dtype']} (optimized for {model_name or 'generic'})")
        return config

    def generate_enhanced_adaptive_config(
        self, 
        request, 
        training_type: str = "instruction"
    ) -> Dict[str, Any]:
        """Generate enhanced adaptive configuration with training type support"""
        
        # Get base config
        base_config = self.generate_optimal_config(request)
        
        # Add training type specific enhancements
        training_config = self._get_training_type_config(training_type)
        chat_config = self._get_chat_configuration(training_type)
        grpo_config = self._get_grpo_configuration(training_type)
        
        # Enhanced configuration structure
        enhanced_config = {
            **base_config,
            "training_type": training_type,
            "training_type_config": training_config,
            "chat_config": chat_config,
            "grpo_config": grpo_config,
            "advanced_optimizations": self._get_advanced_optimizations(training_type),
            "unsloth_enhancements": self._get_unsloth_enhancements(training_type),
        }
        
        return enhanced_config
    
    def _get_training_type_config(self, training_type: str) -> Dict[str, Any]:
        """Get training type specific configuration"""
        configs = {
            "instruction": {
                "description": "Standard instruction tuning",
                "batch_size_multiplier": 1.0,
                "gradient_accumulation_multiplier": 1.0,
                "learning_rate_multiplier": 1.0,
                "max_seq_length": 2048,
                "scheduler_type": "linear",
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
            },
            "chat": {
                "description": "Chat completion training",
                "batch_size_multiplier": 0.5,
                "gradient_accumulation_multiplier": 2.0,
                "learning_rate_multiplier": 0.8,
                "max_seq_length": 4096,
                "scheduler_type": "cosine",
                "weight_decay": 0.005,
                "warmup_ratio": 0.15,
            },
            "reasoning": {
                "description": "GRPO reasoning training",
                "batch_size_multiplier": 0.25,
                "gradient_accumulation_multiplier": 4.0,
                "learning_rate_multiplier": 0.5,
                "max_seq_length": 8192,
                "scheduler_type": "cosine",
                "weight_decay": 0.001,
                "warmup_ratio": 0.2,
            }
        }
        
        return configs.get(training_type, configs["instruction"])
    
    def _get_chat_configuration(self, training_type: str) -> Dict[str, Any]:
        """Get chat-specific configuration"""
        if training_type not in ["chat", "reasoning"]:
            return {}
        
        return {
            "template_type": "chatml",
            "system_message": "You are a helpful assistant.",
            "max_turns": 10,
            "truncate_method": "keep_recent",
            "assistant_only_loss": True,
            "add_special_tokens": False,
            "enable_packing": True,
            "use_unsloth_template": UNSLOTH_AVAILABLE,
        }
    
    def _get_grpo_configuration(self, training_type: str) -> Dict[str, Any]:
        """Get GRPO-specific configuration"""
        if training_type != "reasoning":
            return {}
        
        return {
            "enable_reasoning": True,
            "process_supervision": True,
            "reasoning_chain_format": "step_by_step",
            "reward_threshold": 0.7,
            "step_evaluation": True,
            "chain_of_thought": True,
            "process_reward_weight": 0.3,
            "outcome_reward_weight": 0.7,
        }
    
    def _get_advanced_optimizations(self, training_type: str) -> Dict[str, Any]:
        """Get advanced optimizations based on training type"""
        base_optimizations = {
            "use_flash_attention": True,
            "gradient_checkpointing_ratio": 0.5,
            "memory_efficient_attention": True,
            "compile_model": False,  # Can cause issues with some models
        }
        
        if training_type == "chat":
            base_optimizations.update({
                "sequence_packing": True,
                "dynamic_padding": True,
                "attention_dropout": 0.0,
            })
        elif training_type == "reasoning":
            base_optimizations.update({
                "sequence_packing": False,  # Don't pack reasoning chains
                "dynamic_padding": False,
                "attention_dropout": 0.0,
                "reasoning_memory_optimization": True,
            })

        if self._accelerator() in _VENDOR_ACCELERATORS:
            # Agrees with gpu_optimizations["flash_attention"], which is off on these devices.
            base_optimizations["use_flash_attention"] = False
        
        return base_optimizations
    
    def _get_unsloth_enhancements(self, training_type: str) -> Dict[str, Any]:
        """Get Unsloth-specific enhancements"""
        if not UNSLOTH_AVAILABLE:
            return {}
        
        base_enhancements = {
            "use_unsloth_optimizer": True,
            "unsloth_attention_implementation": "flash_attention_2",
            "gradient_checkpointing_method": "unsloth",
            "memory_optimization_level": "aggressive",
        }
        
        if training_type == "chat":
            base_enhancements.update({
                "unsloth_chat_template": True,
                "conversation_packing": True,
                "multi_turn_optimization": True,
            })
        elif training_type == "reasoning":
            base_enhancements.update({
                "reasoning_chain_optimization": True,
                "process_supervision_support": True,
                "step_wise_gradients": True,
            })
        
        return base_enhancements
    
    def _get_gpu_optimizations(self, model_name: Optional[str] = None,
                               params_billion: Optional[float] = None) -> Dict[str, Any]:
        """Get GPU-specific optimizations with model-aware precision selection

        NVIDIA CUDA and CPU-only machines get the tuned tiers below as they
        are. Apple MPS and Intel XPU get their own settings, and AMD ROCm gets
        the tiers with quantised loading reconsidered. params_billion is the
        model's size when the caller knows it; otherwise it is read from the
        model name where possible. It matters only on the non-NVIDIA paths.
        """
        accelerator = self._accelerator()
        if accelerator in ("mps", "xpu"):
            return self._native_accelerator_optimizations(model_name, params_billion)
        opts = self._cuda_tier_optimizations(model_name)
        if accelerator == "rocm" and opts["enabled"]:
            opts = self._adjust_for_rocm(opts, model_name, params_billion)
        return opts

    def _cuda_tier_optimizations(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """The VRAM tiers tuned on NVIDIA cards, and the CPU fallback (original code)"""
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
        
        # Enhanced RTX 4060 8GB optimization with Unsloth 2025 techniques
        gpu_memory = self._tier_memory_gb()  # gpu_memory_gb, except on Apple MPS / Intel XPU
        accelerator = self._accelerator()
        vendor_path = accelerator in _VENDOR_ACCELERATORS  # log lines naming NVIDIA cards would mislead there
        if gpu_memory <= 8:
            if not vendor_path:
                self.logger.info(f"Applying RTX 4060 8GB Unsloth 2025 optimizations")
            # Ultra-conservative settings for maximum stability
            per_device_batch = 1  # Single batch for maximum memory safety
            grad_accum = max(base_request.gradient_accumulation_steps, 24)  # Larger accumulation for QLoRA-All
            max_seq = 1024  # Optimal sequence length for 8GB with QLoRA-All
            eval_strategy = "no"  # Disable evaluation to maximize training VRAM
            
            # Log specific RTX 4060 optimizations
            if not vendor_path:
                self.logger.info(f"RTX 4060 8GB config: batch=1, grad_accum={grad_accum}, seq_len={max_seq}")
                self.logger.info(f"Effective batch size: {per_device_batch * grad_accum} (optimized for QLoRA-All)")
            else:
                self.logger.info(f"{accelerator} 8GB-or-less tier: batch=1, grad_accum={grad_accum}, seq_len={max_seq}")
        elif gpu_memory <= 12:
            # RTX 3060 12GB / RTX 4060 Ti optimizations
            per_device_batch = min(base_request.batch_size, 2)
            grad_accum = max(base_request.gradient_accumulation_steps, 12)
            max_seq = 1536  # Slightly longer sequences for 12GB
            eval_strategy = "steps" if hasattr(base_request, 'eval_steps') else "no"
        else:
            # High-end GPU configurations
            per_device_batch = min(base_request.batch_size, 4)
            grad_accum = max(base_request.gradient_accumulation_steps, 8)
            max_seq = 2048
            eval_strategy = "steps" if hasattr(base_request, 'eval_steps') else "no"
        
        # Base configuration
        args = {
            "dataloader_num_workers": 0,  # FIXED: Force single worker to prevent semaphore leaks
            "dataloader_pin_memory": self.profile.system_memory_gb > 16 and gpu_memory > 8,  # FIXED: Disable for low GPU memory
            "per_device_train_batch_size": per_device_batch,
            "gradient_accumulation_steps": grad_accum,
            "max_seq_length": max_seq,  # FIXED: Enforce GPU-safe sequence length
            "eval_strategy": eval_strategy,
            "save_steps": 50 if gpu_memory <= 8 else 100,  # FIXED: More frequent saves for low memory recovery
            "logging_steps": 5 if gpu_memory <= 8 else 10,  # FIXED: Frequent logging for monitoring hangs
            "save_total_limit": 2 if gpu_memory <= 8 else 3,  # FIXED: Fewer checkpoints for low memory
            "report_to": [],  # FIXED: Disable external logging to save resources
        }

        if accelerator in ("mps", "xpu"):
            # Page-locked host memory only speeds up copies to a CUDA device.
            args["dataloader_pin_memory"] = False
        
        self.logger.info(f"GPU-aware training args: batch={per_device_batch}, accum={grad_accum}, effective_batch={per_device_batch*grad_accum}, seq={max_seq}, eval={eval_strategy}")
        
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
    
    def _get_unsloth_2025_target_modules(self, memory_gb: float, model_name: Optional[str] = None) -> List[str]:
        """Get Unsloth 2025-optimized target modules based on latest research and model architecture"""
        # Based on Unsloth 2025 official recommendations and QLoRA-All approach
        model_family = self._get_model_family(model_name)
        
        if memory_gb <= 4:
            # Ultra-low memory configuration - focus on attention
            return ["q_proj", "v_proj", "gate_proj"]  # Minimal but effective
        elif memory_gb <= 8:
            # RTX 4060 8GB optimized - QLoRA-All approach for best performance
            if model_family in ["qwen", "llama", "mistral", "gemma"]:
                # 2025 Unsloth recommendation: Apply LoRA to all FFN/MLP and Attention layers
                return [
                    # Attention layers (critical for all models)
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    # MLP/FFN layers (essential for QLoRA-All performance)
                    "gate_proj", "up_proj", "down_proj"
                    # Note: embed_tokens and lm_head excluded for 8GB VRAM efficiency
                ]
            else:
                # Generic transformer architecture
                return [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "fc1", "fc2"  # Standard MLP naming
                ]
        elif memory_gb <= 16:
            # High memory - full QLoRA-All with embedding layers
            return [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj",      # MLP
                "embed_tokens"  # Embedding (excluding lm_head for stability)
            ]
        else:
            # Maximum configuration for 24GB+ VRAM
            return [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj",      # MLP
                "embed_tokens", "lm_head"                 # Full embedding layers
            ]

    def _get_adaptive_lora_config(self, memory_gb: float, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get LoRA configuration based on available memory using Unsloth 2025 optimizations"""
        
        # Get model-specific target modules using 2025 recommendations
        target_modules = self._get_unsloth_2025_target_modules(memory_gb, model_name)
        model_family = self._get_model_family(model_name)
        
        if memory_gb <= 4:  # Ultra-constrained - efficient minimal LoRA
            return {
                "r": 8,
                "lora_alpha": 16,  # 2x ratio maintained
                "lora_dropout": 0.0,  # CRITICAL: Must be 0.0 for Unsloth kernel optimizations
                "target_modules": target_modules,
                "use_rslora": True,  # 2025: Rank-Stabilized LoRA for better stability
                "use_dora": False,  # Disabled for memory constraints
            }
        elif memory_gb <= 8:  # RTX 4060 8GB - 2025 optimized configuration
            # Enhanced configuration based on Unsloth 2025 QLoRA-All research
            base_config = {
                "r": 16,  # Optimal for 8GB VRAM with QLoRA-All
                "lora_alpha": 32,  # Maintains 2x ratio for stability
                "lora_dropout": 0.0,  # CRITICAL: 0.0 for maximum Unsloth acceleration
                "target_modules": target_modules,
                "use_rslora": True,  # 2025: Improved training stability
                "use_dora": False,  # Memory-intensive, disabled for 8GB
                "fan_in_fan_out": False,  # Standard setting for most models
                "bias": "none",  # No bias for efficiency
            }
            
            # Model-specific optimizations for 2025
            if model_family in ["qwen", "llama"]:
                base_config.update({
                    "r": 20,  # Slightly higher rank for these model families
                    "lora_alpha": 40,  # Maintain 2x ratio
                })
            elif model_family == "gemma":
                base_config.update({
                    "r": 14,  # Lower rank for Gemma efficiency
                    "lora_alpha": 28,
                })
            
            return base_config
            
        elif memory_gb <= 16:  # High memory - advanced LoRA features
            return {
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.0,  # Still 0.0 for Unsloth optimization
                "target_modules": target_modules,
                "use_rslora": True,  # Rank-Stabilized LoRA
                "use_dora": True,   # 2025: DoRA for better accuracy (when memory allows)
                "fan_in_fan_out": False,
                "bias": "none",
            }
        else:  # Maximum memory - full feature set
            return {
                "r": 64,
                "lora_alpha": 128,
                "lora_dropout": 0.0,  # Unsloth optimization requirement
                "target_modules": target_modules,
                "use_rslora": True,  # All advanced features enabled
                "use_dora": True,
                "fan_in_fan_out": False,
                "bias": "none",
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

        if self._accelerator() in ("mps", "xpu"):
            # The model-family preferences below were established with Unsloth on
            # NVIDIA cards. Here the dtype the model loader uses is the answer.
            return self._native_dtype()
        
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
        accelerator = self._accelerator()
        if accelerator in ("mps", "xpu"):
            return self._native_dtype() == "bfloat16"
        if accelerator == "rocm" and getattr(self.profile, "bf16_supported", None) is not None:
            # torch's own answer, recorded at detection. The rules below know only NVIDIA cards.
            return bool(self.profile.bf16_supported)

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
            "float16": self._supports_float16(),  # All CUDA GPUs support float16; so do MPS and XPU
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
        """Get human-readable hardware summary

        "accelerator" and "warnings" are present on every machine; on NVIDIA
        CUDA and on a CPU the other keys read as they always did and
        "warnings" is empty.
        """
        accelerator = self._accelerator()
        if accelerator == "mps":
            name = getattr(self.profile, "accelerator_name", "") or self.profile.gpu_name
            gpu_text = (f"{name} (Apple MPS, {self._tier_memory_gb():.1f}GB budgeted of "
                        f"{self.profile.system_memory_gb:.1f}GB unified memory)")
        elif accelerator == "xpu":
            name = getattr(self.profile, "accelerator_name", "") or self.profile.gpu_name
            gpu_text = f"{name} (Intel XPU, {self._tier_memory_gb():.1f}GB)"
        elif accelerator == "rocm":
            gpu_text = f"{self.profile.gpu_name} (AMD ROCm, {self.profile.gpu_memory_gb:.1f}GB)"
        else:
            gpu_text = f"{self.profile.gpu_name} ({self.profile.gpu_memory_gb:.1f}GB)" if self.profile.cuda_available else "CPU Only"
        return {
            "gpu": gpu_text,
            "memory": f"{self.profile.system_memory_gb:.1f}GB System",
            "container_limit": f"{self.profile.container_memory_gb:.1f}GB" if self.profile.container_memory_gb else "No Limit",
            "cpu": f"{self.profile.cpu_cores}C/{self.profile.cpu_threads}T @ {self.profile.cpu_freq_max:.1f}GHz",
            "storage": f"{self.profile.storage_type} ({self.profile.available_storage_gb:.1f}GB available)",
            "environment": "Container" if self.profile.is_container else "Native",
            "precision_support": {
                "bfloat16": self._check_bfloat16_support(),
                "float16": self._supports_float16(),
            },
            "accelerator": accelerator,
            "warnings": list(self.warnings),
        }


# Global hardware detector instance
hardware_detector = HardwareDetector()


def current_hardware_warnings() -> List[str]:
    """Vendor warnings for the hardware this process runs on.

    Cheap enough to call per request: it reuses the profile the detector last
    produced and only runs detection if there has not been one yet.
    """
    profile = hardware_detector.last_profile or hardware_detector.detect_full_profile()
    return AdaptiveTrainingConfig(profile).get_hardware_summary()["warnings"]