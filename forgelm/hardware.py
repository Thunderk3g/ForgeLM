"""Hardware and software capability probing.

Design notes
------------
* Works without torch. When torch is importable it is used to confirm what the
  operating system reports (a GPU that the driver sees but torch cannot use is
  not a training device), but it is never required.
* Reports the *vendor runtime* rather than a single ``cuda_available`` flag:
  NVIDIA CUDA, AMD ROCm (which also answers ``torch.cuda.is_available()``),
  Apple MPS/MLX, Intel XPU. The planner needs that distinction because the
  usable training stack differs for each.
* Memory is the binding constraint for nearly every decision, so the probe
  honours container (cgroup) limits rather than reporting host RAM.
* The parsers are pure functions over text so they can be tested without the
  hardware.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

GIB = 1024**3

# CPU instruction-set flags that change which inference path is fast or safe.
# avx512_vnni / avx_vnni decide whether int8 ONNX is both fast and free of the
# u8s8 saturation problem documented by ONNX Runtime.
_INTERESTING_FLAGS = (
    "avx", "avx2", "avx512f", "avx512bw", "avx512_vnni", "avx_vnni",
    "amx_int8", "amx_bf16", "fma", "f16c", "neon", "asimd", "asimddp", "sve", "i8mm",
)

# Packages whose presence changes what the planner may recommend.
# key = name used throughout forgelm, value = distribution name on PyPI.
KNOWN_PACKAGES: Dict[str, str] = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "trl": "trl",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "unsloth": "unsloth",
    "mlx": "mlx",
    "mlx_lm": "mlx-lm",
    "sentence_transformers": "sentence-transformers",
    "model2vec": "model2vec",
    "setfit": "setfit",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "onnxruntime_gpu": "onnxruntime-gpu",
    "openvino": "openvino",
    "sklearn": "scikit-learn",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "interpret": "interpret-core",
    "tabicl": "tabicl",
    "tabpfn": "tabpfn",
    "llama_cpp": "llama-cpp-python",
    "vllm": "vllm",
}


@dataclass
class Accelerator:
    """One usable compute device other than the CPU."""

    runtime: str  # "cuda" | "rocm" | "mps" | "xpu"
    vendor: str  # "nvidia" | "amd" | "apple" | "intel"
    name: str
    memory_gb: float  # dedicated VRAM, or unified memory on Apple Silicon
    count: int = 1
    unified_memory: bool = False
    # False when the OS reports the device but the installed torch cannot use it.
    torch_usable: Optional[bool] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class HardwareProfile:
    os: str
    arch: str
    python: str
    cpu_model: str
    physical_cores: int
    logical_cores: int
    cpu_flags: List[str]
    ram_total_gb: float
    ram_available_gb: float
    in_container: bool
    accelerators: List[Accelerator] = field(default_factory=list)
    packages: Dict[str, str] = field(default_factory=dict)
    onnx_providers: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # ---- derived views -------------------------------------------------
    @property
    def primary(self) -> Optional[Accelerator]:
        """The accelerator to train on: usable by torch (or MLX), most memory."""
        usable = [a for a in self.accelerators if a.torch_usable is not False]
        return max(usable, key=lambda a: a.memory_gb * a.count, default=None)

    @property
    def device_class(self) -> str:
        """One of: cpu, nvidia, amd, apple, intel_gpu."""
        p = self.primary
        if p is None:
            return "cpu"
        return {"nvidia": "nvidia", "amd": "amd", "apple": "apple", "intel": "intel_gpu"}[p.vendor]

    @property
    def accelerator_memory_gb(self) -> float:
        p = self.primary
        return p.memory_gb if p else 0.0

    @property
    def has_vnni(self) -> bool:
        return any(f in self.cpu_flags for f in ("avx512_vnni", "avx_vnni", "amx_int8", "asimddp", "i8mm"))

    @property
    def is_arm(self) -> bool:
        return self.arch.lower() in ("arm64", "aarch64")

    def has(self, package: str) -> bool:
        return package in self.packages

    def to_dict(self) -> dict:
        d = asdict(self)
        d["device_class"] = self.device_class
        d["has_vnni"] = self.has_vnni
        return d

    def summary(self) -> str:
        lines = [
            f"OS / arch      : {self.os} / {self.arch}  (python {self.python})",
            f"CPU            : {self.cpu_model or 'unknown'}",
            f"Cores          : {self.physical_cores} physical / {self.logical_cores} logical",
            f"CPU features   : {', '.join(self.cpu_flags) or 'none detected'}",
            f"RAM            : {self.ram_total_gb:.1f} GiB total, {self.ram_available_gb:.1f} GiB available"
            + ("  (container limit applied)" if self.in_container else ""),
        ]
        if self.accelerators:
            for a in self.accelerators:
                usable = {True: "usable by torch", False: "NOT usable by installed torch", None: "torch not checked"}[
                    a.torch_usable
                ]
                mem = f"{a.memory_gb:.1f} GiB" + (" unified" if a.unified_memory else "")
                lines.append(f"Accelerator    : {a.count} x {a.name} [{a.runtime}] {mem} - {usable}")
                lines.extend(f"                 note: {n}" for n in a.notes)
        else:
            lines.append("Accelerator    : none")
        lines.append(f"Device class   : {self.device_class}")
        if self.onnx_providers:
            lines.append(f"ONNX providers : {', '.join(self.onnx_providers)}")
        lines.extend(f"Note           : {n}" for n in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Pure parsers (unit-tested without hardware)
# --------------------------------------------------------------------------

def parse_cpu_flags(cpuinfo_text: str) -> List[str]:
    """Extract the instruction-set flags we care about from /proc/cpuinfo."""
    for line in cpuinfo_text.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() in ("flags", "features"):
            present = set(value.split())
            return [f for f in _INTERESTING_FLAGS if f in present]
    return []


def parse_cpu_model(cpuinfo_text: str) -> str:
    for line in cpuinfo_text.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() in ("model name", "hardware", "cpu model"):
            return value.strip()
    return ""


def parse_nvidia_smi(csv_text: str) -> List[Accelerator]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits``.

    Identical cards are merged into one entry with ``count`` set, because the
    planner reasons about per-device memory and device count separately.
    """
    by_key: Dict[tuple, Accelerator] = {}
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            mem_gb = float(parts[1]) / 1024.0  # MiB -> GiB
        except ValueError:
            continue
        key = (parts[0], round(mem_gb, 1))
        if key in by_key:
            by_key[key].count += 1
        else:
            by_key[key] = Accelerator("cuda", "nvidia", parts[0], round(mem_gb, 2))
    return list(by_key.values())


def parse_cgroup_limit(text: str) -> Optional[int]:
    """cgroup v1/v2 memory limit in bytes, or None when unlimited."""
    text = text.strip()
    if not text or text == "max":
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    # cgroup v1 reports a huge sentinel when no limit is set.
    return None if value >= 1 << 60 else value


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------

def _read(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def _run(cmd: Sequence[str], timeout: float = 5.0) -> str:
    if not shutil.which(cmd[0]):
        return ""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout if out.returncode == 0 else ""


def installed_packages() -> Dict[str, str]:
    """Versions of the packages forgelm can use, found without importing them."""
    found: Dict[str, str] = {}
    for key, dist in KNOWN_PACKAGES.items():
        try:
            found[key] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return found


def _memory(reader: Callable[[str], str]) -> tuple[float, float, bool]:
    """(total_gb, available_gb, limited_by_container)."""
    total = available = 0.0
    try:
        import psutil  # optional

        vm = psutil.virtual_memory()
        total, available = vm.total / GIB, vm.available / GIB
    except Exception:
        meminfo = {}
        for line in reader("/proc/meminfo").splitlines():
            k, _, v = line.partition(":")
            if v.strip():
                meminfo[k] = v.split()[0]
        if "MemTotal" in meminfo:
            total = int(meminfo["MemTotal"]) * 1024 / GIB
            available = int(meminfo.get("MemAvailable", meminfo["MemTotal"])) * 1024 / GIB
        elif hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / GIB
            available = total

    limit = parse_cgroup_limit(reader("/sys/fs/cgroup/memory.max"))
    usage_text = reader("/sys/fs/cgroup/memory.current")
    if limit is None:
        limit = parse_cgroup_limit(reader("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        usage_text = reader("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    limited = False
    if limit is not None and (total == 0 or limit / GIB < total):
        limited = True
        total = limit / GIB
        try:
            available = min(available or total, max(0.0, (limit - int(usage_text.strip())) / GIB))
        except ValueError:
            available = min(available or total, total)
    return round(total, 2), round(available, 2), limited


def _sysfs_gpus(root: str = "/sys/class/drm") -> List[Accelerator]:
    """AMD and Intel discrete GPUs as the kernel reports them (Linux only)."""
    found: List[Accelerator] = []
    base = Path(root)
    if not base.is_dir():
        return found
    for card in sorted(base.glob("card[0-9]")):
        vendor = _read(str(card / "device" / "vendor")).strip().lower()
        vram = _read(str(card / "device" / "mem_info_vram_total")).strip()
        # Integrated graphics report a few hundred MiB of "VRAM"; they are not training devices.
        if vendor == "0x1002" and vram.isdigit() and int(vram) >= 2 * GIB:
            found.append(Accelerator("rocm", "amd", f"AMD GPU ({card.name})", round(int(vram) / GIB, 2)))
    return found


def _torch_view() -> Optional[dict]:
    """What the installed torch can actually use. None when torch is absent."""
    if importlib.util.find_spec("torch") is None:
        return None
    try:
        import torch
    except Exception as exc:  # broken install
        return {"error": repr(exc)}
    view: dict = {"version": torch.__version__, "cuda": False, "hip": None, "mps": False, "xpu": False, "devices": []}
    try:
        view["hip"] = getattr(torch.version, "hip", None)
        if torch.cuda.is_available():
            view["cuda"] = True
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                view["devices"].append((props.name, props.total_memory / GIB))
    except Exception as exc:
        view["cuda_error"] = repr(exc)
    try:
        view["mps"] = bool(torch.backends.mps.is_available())
    except Exception:
        pass
    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            view["xpu"] = True
            view["xpu_devices"] = [
                (xpu.get_device_name(i), xpu.get_device_properties(i).total_memory / GIB)
                for i in range(xpu.device_count())
            ]
    except Exception:
        pass
    return view


def probe(use_torch: bool = True) -> HardwareProfile:
    """Detect the compute this process can use.

    ``use_torch=False`` skips importing torch (faster; reports what the OS sees).
    """
    system = platform.system()
    arch = platform.machine()
    cpuinfo = _read("/proc/cpuinfo")
    cpu_model = parse_cpu_model(cpuinfo) or platform.processor()
    flags = parse_cpu_flags(cpuinfo)
    notes: List[str] = []

    if system == "Darwin":
        cpu_model = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or cpu_model
        feats = _run(["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"]).lower().split()
        flags = [f for f in _INTERESTING_FLAGS if f in feats or f.replace("_", "") in feats]
        if arch == "arm64":
            flags = sorted(set(flags) | {"neon", "asimddp"})

    logical = os.cpu_count() or 1
    try:
        usable = len(os.sched_getaffinity(0))  # honours cpusets / container CPU limits
        logical = min(logical, usable)
    except AttributeError:
        pass
    physical = logical
    try:
        import psutil

        physical = min(psutil.cpu_count(logical=False) or logical, logical)
    except Exception:
        pass

    total_gb, avail_gb, limited = _memory(_read)
    in_container = limited or Path("/.dockerenv").exists() or bool(os.environ.get("KUBERNETES_SERVICE_HOST"))

    accelerators: List[Accelerator] = []
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if smi:
        accelerators.extend(parse_nvidia_smi(smi))
    if system == "Linux":
        accelerators.extend(_sysfs_gpus())
    if system == "Darwin" and arch == "arm64":
        accelerators.append(
            Accelerator(
                "mps", "apple", cpu_model or "Apple Silicon", total_gb, unified_memory=True,
                notes=["memory is shared with the OS; budget roughly 60-70% of it for a model"],
            )
        )

    packages = installed_packages()
    tv = _torch_view() if use_torch else None
    if tv is not None and "error" not in tv:
        if tv["cuda"]:
            vendor, runtime = ("amd", "rocm") if tv["hip"] else ("nvidia", "cuda")
            matching = [a for a in accelerators if a.vendor == vendor]
            if matching:
                for a in matching:
                    a.torch_usable = True
            else:  # torch sees a device the OS tools did not report
                for name, mem in tv["devices"]:
                    accelerators.append(Accelerator(runtime, vendor, name, round(mem, 2), torch_usable=True))
        for a in accelerators:
            if a.vendor in ("nvidia", "amd") and a.torch_usable is None:
                a.torch_usable = False
                want = "ROCm" if a.vendor == "amd" else "CUDA"
                a.notes.append(f"installed torch {tv['version']} has no {want} support; install the {want} build")
            if a.vendor == "apple":
                a.torch_usable = tv["mps"] or "mlx" in packages
        if tv["xpu"]:
            for name, mem in tv.get("xpu_devices", []):
                accelerators.append(Accelerator("xpu", "intel", name, round(mem, 2), torch_usable=True))
    elif tv is not None:
        notes.append(f"torch is installed but failed to import: {tv['error']}")
    else:
        for a in accelerators:
            if a.vendor == "apple" and "mlx" in packages:
                a.torch_usable = True

    providers: List[str] = []
    if importlib.util.find_spec("onnxruntime") is not None:
        try:
            import onnxruntime

            providers = list(onnxruntime.get_available_providers())
        except Exception:
            pass

    return HardwareProfile(
        os=system,
        arch=arch,
        python=".".join(map(str, sys.version_info[:3])),
        cpu_model=cpu_model,
        physical_cores=physical,
        logical_cores=logical,
        cpu_flags=flags,
        ram_total_gb=total_gb,
        ram_available_gb=avail_gb,
        in_container=in_container,
        accelerators=accelerators,
        packages=packages,
        onnx_providers=providers,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Reference profiles: used for "plan for hardware I do not have here"
# and by the test-suite. Values are representative, not measured.
# --------------------------------------------------------------------------

def reference_profile(name: str) -> HardwareProfile:
    base = dict(os="Linux", arch="x86_64", python="3.12.0", cpu_model="reference", physical_cores=8,
                logical_cores=16, cpu_flags=["avx", "avx2", "fma", "f16c"], ram_total_gb=32.0,
                ram_available_gb=28.0, in_container=False)
    full_stack = {"torch": "ref", "transformers": "ref", "peft": "ref", "trl": "ref", "sklearn": "ref",
                  "sentence_transformers": "ref", "onnxruntime": "ref"}
    table = {
        "cpu-laptop": dict(base, physical_cores=4, logical_cores=8, ram_total_gb=16.0, ram_available_gb=11.0,
                           packages=dict(full_stack)),
        "cpu-server": dict(base, physical_cores=32, logical_cores=64, ram_total_gb=128.0, ram_available_gb=120.0,
                           cpu_flags=["avx", "avx2", "avx512f", "avx512bw", "avx512_vnni", "fma", "f16c"],
                           packages=dict(full_stack)),
        "nvidia-4gb": dict(base, accelerators=[Accelerator("cuda", "nvidia", "RTX 3050 4GB", 4.0, torch_usable=True)],
                           packages=dict(full_stack, bitsandbytes="ref")),
        "nvidia-24gb": dict(base, ram_total_gb=64.0, ram_available_gb=56.0,
                            accelerators=[Accelerator("cuda", "nvidia", "RTX 4090", 24.0, torch_usable=True)],
                            packages=dict(full_stack, bitsandbytes="ref", unsloth="ref")),
        "nvidia-2x80gb": dict(base, ram_total_gb=512.0, ram_available_gb=480.0,
                              accelerators=[Accelerator("cuda", "nvidia", "H100", 80.0, count=2, torch_usable=True)],
                              packages=dict(full_stack, bitsandbytes="ref")),
        "amd-16gb": dict(base, accelerators=[Accelerator("rocm", "amd", "Radeon RX 7800 XT", 16.0, torch_usable=True)],
                         packages=dict(full_stack)),
        "apple-16gb": dict(base, os="Darwin", arch="arm64", cpu_model="Apple M2", physical_cores=8, logical_cores=8,
                           cpu_flags=["neon", "asimddp"], ram_total_gb=16.0, ram_available_gb=10.0,
                           accelerators=[Accelerator("mps", "apple", "Apple M2", 16.0, unified_memory=True,
                                                     torch_usable=True)],
                           packages=dict(full_stack, mlx="ref", mlx_lm="ref")),
        "apple-64gb": dict(base, os="Darwin", arch="arm64", cpu_model="Apple M3 Max", physical_cores=14,
                           logical_cores=14, cpu_flags=["neon", "asimddp"], ram_total_gb=64.0, ram_available_gb=50.0,
                           accelerators=[Accelerator("mps", "apple", "Apple M3 Max", 64.0, unified_memory=True,
                                                     torch_usable=True)],
                           packages=dict(full_stack, mlx="ref", mlx_lm="ref")),
        "intel-arc-16gb": dict(base, accelerators=[Accelerator("xpu", "intel", "Arc A770", 16.0, torch_usable=True)],
                               packages=dict(full_stack, openvino="ref")),
    }
    if name not in table:
        raise KeyError(f"unknown reference profile {name!r}; choose from {sorted(table)}")
    return HardwareProfile(**table[name])


REFERENCE_PROFILES = (
    "cpu-laptop", "cpu-server", "nvidia-4gb", "nvidia-24gb", "nvidia-2x80gb",
    "amd-16gb", "apple-16gb", "apple-64gb", "intel-arc-16gb",
)
