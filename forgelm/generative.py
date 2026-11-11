"""Hardware-aware choices for fine-tuning generative language models.

Three decisions the previous ``hardware_detection.py`` made from hard-coded
NVIDIA VRAM tiers are made here from the probed hardware and the installed
software instead:

    select_backend    which training stack can actually run on this machine
    estimate_memory   whether a given model fits, before downloading it
    lora_config       how to express a PEFT method in Hugging Face ``peft``

Nothing here imports torch. The functions return plain data so the training
service, the CLI and the tests all use the same logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from .hardware import HardwareProfile

# Share of unified memory budgeted for the model on Apple Silicon.
MPS_MEMORY_FRACTION = 0.65


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------

def estimate_memory_gb(params_billion: float, bits: int = 4, batch_size: int = 2, seq_len: int = 1024,
                       gradient_checkpointing: bool = True, full_finetune: bool = False) -> Dict[str, float]:
    """Rough peak memory for (Q)LoRA or full fine-tuning.

    A heuristic, not a measurement. It combines the weights-at-rest rule of
    thumb (parameters x bits / 8, plus ~15% for layers that stay unquantised)
    with an activation term that scales with tokens per step and with the
    square root of model size. It lands inside the range of published minimum
    requirements for 3B-70B models (see tests), and those published figures
    themselves disagree by up to 2x. Always confirm with a few real steps at
    the intended batch size and sequence length before committing a long run.

    Raises ``ValueError`` for inputs the formula has no meaning for.
    """
    # Checked up front because the formula takes a square root of the model
    # size: a negative size gives a complex number and the caller would only
    # see an unrelated TypeError from round(). ``not (x > 0)`` also rejects NaN.
    if not (params_billion > 0):
        raise ValueError(f"params_billion must be greater than 0, got {params_billion!r}")
    if bits not in (4, 8, 16, 32):
        raise ValueError(f"bits must be one of 4, 8, 16 or 32, got {bits!r}")
    if not (batch_size >= 1):
        raise ValueError(f"batch_size must be at least 1, got {batch_size!r}")
    if not (seq_len >= 1):
        raise ValueError(f"seq_len must be at least 1, got {seq_len!r}")
    weights =params_billion * bits / 8.0 * 1.15
    tokens = batch_size * seq_len
    activations = 0.45 * (params_billion ** 0.5) * (tokens / 2048.0)
    if not gradient_checkpointing:
        activations *= 2.5
    optimizer = 0.0
    if full_finetune:
        # gradients + AdamW moments at the training precision (assume 16-bit weights, 32-bit moments)
        weights = params_billion * 2.0
        optimizer = params_billion * (2.0 + 8.0)
    overhead = 0.8  # runtime context, allocator fragmentation
    total = weights + activations + optimizer + overhead
    return {"weights_gb": round(weights, 2), "activations_gb": round(activations, 2),
            "optimizer_gb": round(optimizer, 2), "overhead_gb": overhead, "total_gb": round(total, 2)}


def largest_model_that_fits(memory_gb: float, bits: int = 4, batch_size: int = 2, seq_len: int = 1024,
                            headroom: float = 0.85) -> float:
    """Largest parameter count (billions) whose estimate fits in ``memory_gb``."""
    lo, hi = 0.0, 200.0
    budget = memory_gb * headroom
    for _ in range(40):
        mid = (lo + hi) / 2  # always above 0, so the estimate's input check never trips here
        if estimate_memory_gb(mid, bits, batch_size, seq_len)["total_gb"] <= budget:
            lo = mid
        else:
            hi = mid
    return round(lo, 1)


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------

@dataclass
class BackendChoice:
    backend: str  # "unsloth" | "peft_trl" | "mlx_lm" | "none"
    device: str  # "cuda" | "rocm" | "mps" | "xpu" | "cpu"
    load_bits: int  # 4, 8 or 16
    precision: str  # "bf16" | "fp16" | "fp32"
    feasible: bool
    memory: Dict[str, float]
    available_gb: float
    serve_format: str
    serve_runtime: str
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    install_hint: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def select_backend(hw: HardwareProfile, params_billion: float, batch_size: int = 2, seq_len: int = 1024
                   ) -> BackendChoice:
    """Pick the training stack, load precision and serving path for this machine.

    The rules follow the hardware matrix in docs/RESEARCH.md section 4.
    Where support exists but is recent or reported fragile (4-bit loading on
    AMD, Apple and Intel), the default avoids it and the warning says so.
    """
    cls = hw.device_class
    reasons: List[str] = []
    warnings: List[str] = []
    install: List[str] = []
    acc = hw.primary

    if cls == "nvidia":
        device, avail = "cuda", acc.memory_gb
        bits, precision = 4, "bf16"
        if hw.has("unsloth"):
            backend = "unsloth"
            reasons.append("Unsloth is installed and is the fastest single-GPU path on NVIDIA")
        else:
            backend = "peft_trl"
            reasons.append("PEFT + TRL: the portable reference stack")
            if not hw.has("unsloth"):
                reasons.append("install unsloth for lower VRAM use and faster steps on a single NVIDIA GPU")
        if not hw.has("bitsandbytes"):
            install.append("bitsandbytes")
        if estimate_memory_gb(params_billion, 16, batch_size, seq_len)["total_gb"] <= avail * 0.85:
            bits = 16
            reasons.append("model fits unquantised; LoRA on 16-bit weights avoids quantisation error")
        serve_format, serve_runtime = "GGUF Q4_K_M, or AWQ/GPTQ W4A16 via llm-compressor", "llama.cpp, or vLLM"
        if acc.count > 1:
            reasons.append(f"{acc.count} GPUs: use accelerate / FSDP for models that exceed one device")
    elif cls == "amd":
        device, avail, backend = "rocm", acc.memory_gb, "peft_trl"
        bits, precision = 16, "bf16"
        reasons.append("PEFT + TRL on ROCm PyTorch (Linux)")
        warnings.append("4-bit (bitsandbytes) on AMD is reported to produce NaNs on releases up to 0.49.2; "
                        "run a short smoke test before relying on QLoRA")
        warnings.append("flash-attention is unavailable on most Radeon cards; expect standard attention speed")
        if estimate_memory_gb(params_billion, 16, batch_size, seq_len)["total_gb"] > avail * 0.85:
            bits = 4
            warnings.append("model does not fit at 16-bit; falling back to 4-bit, which is the fragile path here")
        else:
            reasons.append("model fits unquantised: LoRA on 16-bit weights avoids the 4-bit path entirely")
        serve_format, serve_runtime = "GGUF Q4_K_M", "llama.cpp (HIP or Vulkan) or Ollama"
    elif cls == "apple":
        device, avail = "mps", acc.memory_gb * MPS_MEMORY_FRACTION  # unified memory is shared with the OS
        bits, precision = 4, "fp16"
        if hw.has("mlx_lm"):
            backend = "mlx_lm"
            reasons.append("mlx-lm is the native Apple Silicon trainer (LoRA, DoRA, QLoRA on a quantised base)")
        else:
            backend = "peft_trl"
            bits, precision = 16, "fp32"
            install.append("mlx-lm")
            warnings.append("PyTorch MPS works for small models but lacks mature 4-bit loading; "
                            "install mlx-lm for LLM fine-tuning on a Mac")
        reasons.append(f"budgeting {MPS_MEMORY_FRACTION:.0%} of unified memory for the model")
        serve_format, serve_runtime = "MLX 4-bit, or GGUF Q4_K_M", "mlx-lm, or llama.cpp (Metal) / Ollama"
    elif cls == "intel_gpu":
        device, avail, backend = "xpu", acc.memory_gb, "peft_trl"
        bits, precision = 16, "bf16"
        reasons.append("PEFT + TRL on PyTorch's native XPU backend (IPEX is retired)")
        warnings.append("Arc A-series has no float64; disable GradScaler")
        if estimate_memory_gb(params_billion, 16, batch_size, seq_len)["total_gb"] > avail * 0.85:
            bits = 4
            warnings.append("4-bit loading on XPU is recent; smoke-test before a long run")
        serve_format, serve_runtime = "OpenVINO int4, or GGUF Q4_K_M", "OpenVINO GenAI, or llama.cpp (SYCL/Vulkan)"
    else:
        device, avail, backend = "cpu", hw.ram_available_gb, "peft_trl"
        bits, precision = 16, "fp32"
        reasons.append("no usable accelerator: CPU LoRA is practical only for models of about 1B parameters or less")
        warnings.append("for larger generative models, train on rented or shared GPU and deploy the GGUF here; "
                        "for classification, prefer a distilled encoder (see `forgelm plan`)")
        serve_format, serve_runtime = "GGUF Q4_K_M (Q5_K_M/Q6_K if RAM allows)", "llama.cpp"

    memory = estimate_memory_gb(params_billion, bits, batch_size, seq_len)
    if cls == "cpu":
        # fp32 weights on CPU: 4 bytes per parameter
        memory = estimate_memory_gb(params_billion, 32, batch_size, seq_len)
    feasible = memory["total_gb"] <= avail * 0.9 and backend != "none"
    if cls == "cpu" and params_billion > 1.5:
        feasible = False
    if not feasible:
        fit = largest_model_that_fits(avail, 32 if cls == "cpu" else 4, batch_size, seq_len)
        fit = min(fit, 1.5) if cls == "cpu" else fit
        warnings.append(f"estimated {memory['total_gb']} GiB needed, {avail:.1f} GiB usable; "
                        f"largest model that fits here is about {fit}B parameters at this batch and length")
    for pkg in ("torch", "transformers", "peft", "trl"):
        if backend in ("peft_trl", "unsloth") and not hw.has(pkg):
            install.append(pkg)
    return BackendChoice(backend, device, bits, precision, feasible, memory, round(avail, 2), serve_format,
                         serve_runtime, reasons, warnings, sorted(set(install)))


def capacity(hw: HardwareProfile, batch_size: int = 2, seq_len: int = 1024) -> Dict[str, object]:
    """The largest generative model this machine can fine-tune, by estimate."""
    probe_choice = select_backend(hw, 0.5, batch_size, seq_len)
    if hw.device_class == "cpu":
        fit = min(largest_model_that_fits(probe_choice.available_gb, 32, batch_size, seq_len), 1.5)
        basis = "fp32 LoRA on CPU; capped at 1.5B because larger models train too slowly to be practical"
    else:
        fit = largest_model_that_fits(probe_choice.available_gb, 4, batch_size, seq_len)
        basis = "4-bit QLoRA" if hw.device_class in ("nvidia", "apple") else \
            "4-bit QLoRA (smoke-test first on this vendor); 16-bit LoRA fits about " \
            f"{largest_model_that_fits(probe_choice.available_gb, 16, batch_size, seq_len)}B"
    return {"backend": probe_choice.backend, "device": probe_choice.device,
            "usable_memory_gb": probe_choice.available_gb, "max_params_billion": fit, "basis": basis,
            "install_hint": probe_choice.install_hint}


# --------------------------------------------------------------------------
# PEFT
# --------------------------------------------------------------------------

# How each method name maps onto Hugging Face ``peft``. The training service once
# re-implemented DoRA, OFT and PiSSA as custom layers that nothing called; ``peft``
# ships maintained versions of all three.
#
# Every key here is offered to API callers, so each one must build a working
# ``peft`` config from the arguments of ``lora_config`` alone (the tests check
# every key against the installed ``peft``). LoftQ is left out for that reason:
# ``peft`` rejects it without a ``loftq_config``, and it also needs the
# unquantised base model, which a fine-tune request does not carry.
PEFT_METHODS: Dict[str, dict] = {
    "lora": {"config": "LoraConfig", "kwargs": {}, "tier": "default"},
    "qlora": {"config": "LoraConfig", "kwargs": {}, "tier": "default",
              "note": "QLoRA is LoRA on a 4-bit base model; quantisation is set at load time, not here"},
    "rslora": {"config": "LoraConfig", "kwargs": {"use_rslora": True}, "tier": "default"},
    "dora": {"config": "LoraConfig", "kwargs": {"use_dora": True}, "tier": "advanced",
             "note": "measurably more VRAM and time than LoRA; merge the adapter for inference"},
    "pissa": {"config": "LoraConfig", "kwargs": {"init_lora_weights": "pissa_niter_16"}, "tier": "advanced",
              "note": "SVD initialisation; use a lower learning rate than plain LoRA"},
    "olora": {"config": "LoraConfig", "kwargs": {"init_lora_weights": "olora"}, "tier": "advanced"},
    "oft": {"config": "OFTConfig", "kwargs": {}, "tier": "niche"},
}

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def lora_config(method: str = "lora", rank: int = 16, alpha: Optional[int] = None, dropout: float = 0.0,
                target_modules: Optional[List[str]] = None, task_type: str = "CAUSAL_LM"
                ) -> Tuple[str, dict, List[str]]:
    """Return ``(config_class_name, kwargs, notes)`` for Hugging Face ``peft``.

    The rank and alpha that are asked for are the rank and alpha that come
    back: by default nothing here changes how the adapter is scaled.

    Rank-stabilised scaling (rsLoRA) is opt-in through the ``"rslora"``
    method. It is not a free switch. It changes the adapter scale
    from alpha/rank to alpha/sqrt(rank): with alpha = 2 * rank, rank 64 goes
    from a scale of 2 to a scale of 16, an 8 times larger effective update at
    the same learning rate. The PEFT project's benchmark shows rsLoRA ahead of
    plain LoRA, but that is one untuned run per method and the gain may simply
    be the larger step, so re-tune the learning rate whenever it is enabled.
    The returned notes say so each time ``use_rslora`` is set.
    """
    key = method.lower()
    if key not in PEFT_METHODS:
        raise ValueError(f"unsupported PEFT method {method!r}; choose from {sorted(PEFT_METHODS)}")
    spec = PEFT_METHODS[key]
    notes = [spec["note"]] if "note" in spec else []
    modules = list(target_modules) if target_modules else list(DEFAULT_TARGET_MODULES)
    if spec["config"] == "OFTConfig":
        # peft's OFT takes either a rank or a block size, never both.
        return "OFTConfig", {"r": rank, "oft_block_size": 0, "target_modules": modules,
                             "module_dropout": dropout, "task_type": task_type}, notes
    kwargs = {"r": rank, "lora_alpha": alpha if alpha is not None else 2 * rank, "lora_dropout": dropout,
              "target_modules": modules, "bias": "none", "task_type": task_type}
    kwargs.update(spec["kwargs"])
    if kwargs.get("use_rslora"):
        # The same rank and alpha now mean a different effective step size.
        notes.append("rank-stabilised scaling (use_rslora) is on: the adapter scale is alpha/sqrt(rank) instead "
                     "of alpha/rank, so re-tune the learning rate")
    return "LoraConfig", kwargs, notes


def build_peft_config(method: str = "lora", **kwargs):
    """Instantiate the ``peft`` config object (imports ``peft``)."""
    import peft

    name, cfg, _ = lora_config(method, **kwargs)
    return getattr(peft, name)(**cfg)
