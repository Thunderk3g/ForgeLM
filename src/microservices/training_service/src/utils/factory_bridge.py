"""Bridge between the training service and the hardware-agnostic ``forgelm`` core.

The service used to decide everything from NVIDIA VRAM tiers and a hard-coded
LoRA config. Those decisions now live in ``forgelm`` (see ``forgelm/generative.py``
and ``forgelm/hardware.py``). This module is the single place the service
touches that package, and every function degrades to the previous behaviour
when ``forgelm`` is not installed, so an existing deployment keeps working.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # optional dependency: installed by the training image, absent in old ones
    from forgelm import generative as _generative
    from forgelm.hardware import probe as _probe

    FORGELM_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without the package
    _generative = None
    _probe = None
    FORGELM_AVAILABLE = False

# Methods the previous code accepted in the API schema.
_LEGACY_LORA = {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM"}


def resolve_peft_config(request) -> Optional[Any]:
    """Build the Hugging Face ``peft`` config for a fine-tune request.

    Honours ``peft_method``, ``peft_rank``, ``peft_alpha``, ``peft_dropout`` and
    ``target_modules`` from the request. Previously the Transformers path
    applied a fixed r=16 LoRA for "lora"/"qlora" and applied *no adapter at
    all* for "dora", "pissa" or "oft", which silently turned those requests
    into full fine-tuning.

    Returns ``None`` only when ``peft`` itself is unavailable.
    """
    try:
        import peft
    except ImportError:
        return None

    method = (getattr(request, "peft_method", None) or "lora").lower()
    if not FORGELM_AVAILABLE:
        return peft.LoraConfig(**_LEGACY_LORA)

    if method not in _generative.PEFT_METHODS:
        raise ValueError(f"Unsupported peft_method {method!r}. Supported: {sorted(_generative.PEFT_METHODS)}")
    name, kwargs, _notes = _generative.lora_config(
        method,
        rank=int(getattr(request, "peft_rank", 16) or 16),
        alpha=getattr(request, "peft_alpha", None),
        dropout=float(getattr(request, "peft_dropout", 0.0) or 0.0),
        target_modules=getattr(request, "target_modules", None),
    )
    return getattr(peft, name)(**kwargs)


def dtype_name_for(accelerator: str, bf16_supported: bool = False) -> str:
    """Name of the torch dtype a causal LM is loaded with on an accelerator.

    ``accelerator`` is "cuda", "rocm", "xpu", "mps" or anything else for a CPU.
    This is the one statement of the rule: ``torch_device_and_dtype`` applies
    it to the machine it runs on, and ``hardware_detection`` uses it to say
    which precision a profile will train in, so the reported precision and the
    loaded precision cannot drift apart. ``forgelm.dryrun`` keeps a copy for
    models it loads itself; a test checks that the two agree.
    """
    if accelerator in ("cuda", "rocm"):  # ROCm goes through torch.cuda
        return "bfloat16" if bf16_supported else "float16"
    if accelerator == "xpu":
        return "bfloat16"
    if accelerator == "mps":
        return "float16"
    return "float32"


def torch_device_and_dtype():
    """(device string, torch dtype) for loading a causal LM on this machine.

    Covers NVIDIA CUDA, AMD ROCm (which reports as cuda), Intel XPU, Apple MPS
    and CPU. The previous code only distinguished CUDA from CPU, so Apple and
    Intel accelerators were loaded as CPU float32.
    """
    import torch

    if torch.cuda.is_available():
        bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        return "cuda", getattr(torch, dtype_name_for("cuda", bf16))
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu", getattr(torch, dtype_name_for("xpu"))
    if torch.backends.mps.is_available():
        return "mps", getattr(torch, dtype_name_for("mps"))
    return "cpu", getattr(torch, dtype_name_for("cpu"))


def estimate_memory_gb(params_billion: float, bits: int = 16, batch_size: int = 2,
                       seq_len: int = 1024) -> Optional[Dict[str, float]]:
    """``forgelm.generative.estimate_memory_gb`` for the service, or ``None`` without forgelm.

    ``hardware_detection`` uses it to decide whether a model fits as 16-bit
    weights on an AMD GPU. It falls back to a coarser rule of its own on
    ``None``, so an image without forgelm still gets an answer.
    """
    if not FORGELM_AVAILABLE:
        return None
    return _generative.estimate_memory_gb(params_billion, bits, batch_size, seq_len)


def training_dry_run(model, tokenizer_or_vocab_size=None, batch_size: int = 1, seq_len: int = 1024, steps: int = 3,
                     device: Optional[str] = None, lora: Optional[Dict[str, Any]] = None,
                     available_gb: Optional[float] = None, params_billion: Optional[float] = None,
                     restore: bool = True) -> Dict[str, Any]:
    """Measure what a few real training steps cost on a loaded model.

    The memory figures the service plans with are estimates. This runs
    ``steps`` real optimisation steps on synthetic token ids, at the batch
    size and sequence length the job will use, and returns the estimate, the
    measurement and the verdict together (see ``forgelm.dryrun``):

        {"available": True, "estimate": {...}, "measurement": {...},
         "available_gb": ..., "measured_fits": True, "verdict": "fits", ...}

    ``model`` is the loaded model, with its adapter already attached if there
    is one, or a Hugging Face id to load for the measurement only. With
    ``restore`` (the default here) the trainable weights are put back
    afterwards, so the model the job goes on to train is the model it loaded.

    Returns ``{"available": False, "reason": ...}`` when forgelm, or the
    ``forgelm.dryrun`` module, is not installed in this image.
    """
    if not FORGELM_AVAILABLE:
        return {"available": False, "reason": "forgelm package not installed in this image"}
    try:
        from forgelm import dryrun
    except Exception as exc:  # an older forgelm without the module
        return {"available": False, "reason": f"forgelm.dryrun is not available: {exc}"}

    measurement = dryrun.measure_training_memory(model, tokenizer_or_vocab_size, batch_size, seq_len, steps=steps,
                                                 device=device, lora=lora, restore=restore)
    report = dryrun.compare_with_estimate(measurement, available_gb=available_gb, params_billion=params_billion)
    return {"available": True, **report}


def _detector_warnings() -> List[str]:
    """Vendor warnings from the service's own hardware detector, or [] if it cannot be asked.

    Imported here, not at the top: ``hardware_detection`` imports this module,
    and it needs torch, which the bridge must not require just to be imported.
    """
    try:
        from utils import hardware_detection

        warnings = hardware_detection.current_hardware_warnings()
    except Exception:
        return []
    if not isinstance(warnings, (list, tuple)):
        return []
    return [str(w) for w in warnings]


def backend_advice(params_billion: Optional[float] = None) -> Dict[str, Any]:
    """Hardware class, usable backends and fine-tuning capacity for the API."""
    if not FORGELM_AVAILABLE:
        return {"available": False, "reason": "forgelm package not installed in this image"}
    hw = _probe()
    out: Dict[str, Any] = {
        "available": True,
        "device_class": hw.device_class,
        "accelerators": [{"runtime": a.runtime, "vendor": a.vendor, "name": a.name, "memory_gb": a.memory_gb,
                          "count": a.count, "torch_usable": a.torch_usable, "notes": a.notes}
                         for a in hw.accelerators],
        "cpu_features": hw.cpu_flags,
        "capacity": _generative.capacity(hw),
        "peft_methods": {k: {"tier": v["tier"], "note": v.get("note", "")}
                         for k, v in _generative.PEFT_METHODS.items()},
        # What AdaptiveTrainingConfig will do differently on this vendor (empty on NVIDIA and on a CPU).
        "detector_warnings": _detector_warnings(),
    }
    if params_billion:
        out["selection"] = _generative.select_backend(hw, params_billion).to_dict()
    return out
