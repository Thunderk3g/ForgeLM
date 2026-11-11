"""Bridge between the training service and the hardware-agnostic ``forgelm`` core.

The service used to decide everything from NVIDIA VRAM tiers and a hard-coded
LoRA config. Those decisions now live in ``forgelm`` (see ``forgelm/generative.py``
and ``forgelm/hardware.py``). This module is the single place the service
touches that package, and every function degrades to the previous behaviour
when ``forgelm`` is not installed, so an existing deployment keeps working.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

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


def torch_device_and_dtype():
    """(device string, torch dtype) for loading a causal LM on this machine.

    Covers NVIDIA CUDA, AMD ROCm (which reports as cuda), Intel XPU, Apple MPS
    and CPU. The previous code only distinguished CUDA from CPU, so Apple and
    Intel accelerators were loaded as CPU float32.
    """
    import torch

    if torch.cuda.is_available():
        bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        return "cuda", (torch.bfloat16 if bf16 else torch.float16)
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


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
    }
    if params_billion:
        out["selection"] = _generative.select_backend(hw, params_billion).to_dict()
    return out
