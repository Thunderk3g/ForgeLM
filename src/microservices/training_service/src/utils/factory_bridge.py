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


def causal_lm_load_plan(gpu_opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """How the training service should load a base causal LM on this machine.

    Both training paths used to load every model with a 4-bit bitsandbytes
    config and ``device_map="auto"``, whatever the hardware detector said. On
    NVIDIA that is what the service was tuned for, and it is kept: the plan
    says ``accelerator == "cuda"`` and the caller runs its existing code
    unchanged. Everywhere else the plan replaces it:

    * AMD (ROCm): 16-bit weights by default. 4-bit or 8-bit bitsandbytes
      loading only when ``gpu_opts`` (from ``AdaptiveTrainingConfig``) asks for
      it, which it does only when the model does not fit or the operator has
      overridden it after checking that 4-bit works on their card.
    * Intel (XPU) and Apple (MPS): no bitsandbytes; weights in the device's
      dtype, then moved to the device (``device_map="auto"`` does not place
      models on these devices).
    * CPU: float32, no bitsandbytes.

    Returns ``accelerator`` ("cuda", "rocm", "xpu", "mps", "cpu"), ``kwargs``
    for ``from_pretrained`` (besides the path and ``trust_remote_code``),
    ``move_to`` (device to call ``.to()`` with, or None), ``kbit`` (whether to
    call ``prepare_model_for_kbit_training``), ``training_precision`` ("bf16",
    "fp16" or None: the value for the service's ``model._bnb_precision``; None
    leaves the trainer in full precision, which is the safe choice on MPS and
    CPU) and ``reason``.
    """
    import torch

    gpu_opts = gpu_opts or {}
    device, dtype = torch_device_and_dtype()
    hip = getattr(getattr(torch, "version", None), "hip", None)
    dtype_label = {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(dtype)

    if device == "cuda" and not hip:
        return {"accelerator": "cuda", "kwargs": {}, "move_to": None, "kbit": True,
                "training_precision": dtype_label,
                "reason": "NVIDIA: the service's existing 4-bit loading, unchanged"}

    if device == "cuda":  # AMD ROCm reports through torch.cuda
        kwargs: Dict[str, Any] = {"torch_dtype": dtype, "device_map": "auto"}
        kbit = False
        if gpu_opts.get("load_in_4bit") or gpu_opts.get("load_in_8bit"):
            from transformers import BitsAndBytesConfig

            if gpu_opts.get("load_in_4bit"):
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True)
                reason = "AMD ROCm: 4-bit bitsandbytes, as the hardware configuration asked"
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                reason = "AMD ROCm: 8-bit bitsandbytes, as the hardware configuration asked"
            kbit = True
        else:
            reason = f"AMD ROCm: LoRA on {dtype_label or 'full-precision'} weights, no bitsandbytes"
        return {"accelerator": "rocm", "kwargs": kwargs, "move_to": None, "kbit": kbit,
                "training_precision": dtype_label, "reason": reason}

    if device in ("xpu", "mps"):
        return {"accelerator": device, "kwargs": {"torch_dtype": dtype}, "move_to": device, "kbit": False,
                "training_precision": "bf16" if device == "xpu" and dtype is torch.bfloat16 else None,
                "reason": f"{'Intel XPU' if device == 'xpu' else 'Apple MPS'}: "
                          f"{str(dtype).replace('torch.', '')} weights on the device, no bitsandbytes"}

    return {"accelerator": "cpu", "kwargs": {"torch_dtype": dtype}, "move_to": None, "kbit": False,
            "training_precision": None, "reason": "CPU: float32 weights, no bitsandbytes"}


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
