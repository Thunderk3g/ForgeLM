"""A measured dry run behind the memory estimate.

``forgelm.generative.estimate_memory_gb`` is a heuristic: its own docstring
says to confirm it with a few real steps before committing a long run. This
module is that confirmation.

    measure_training_memory   run a few real optimisation steps on synthetic
                              token ids; return peak memory and seconds per step
    fits                      does a measured figure fit in what is available
    compare_with_estimate     estimate, measurement and verdict side by side

How memory is read depends on the device, because each runtime exposes
something different:

    CUDA / ROCm   torch.cuda.max_memory_allocated: the allocator's own peak
    Intel XPU     torch.xpu.max_memory_allocated where the build has it
    Apple MPS     torch.mps.driver_allocated_memory, sampled around each step
                  (Metal keeps no peak counter)
    CPU           growth of the process's resident memory (psutil if it is
                  installed, otherwise the ``resource`` high-water mark)

Only the CPU path has been run. The CUDA, ROCm, XPU and MPS readers are
exercised in the tests against a stand-in for torch, not on real devices.

Nothing heavy is imported until a function is called, like the rest of
``forgelm``.
"""

from __future__ import annotations

import gc
import inspect
import math
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

GIB = 1024**3

from .generative import MPS_MEMORY_FRACTION  # one budget for unified memory, shared with select_backend


# --------------------------------------------------------------------------
# Small pure helpers
# --------------------------------------------------------------------------

def dtype_name_for(accelerator: str, bf16_supported: bool = False) -> str:
    """Name of the torch dtype a causal LM is loaded with on an accelerator.

    Used only when this module loads a model by id. The training service's
    ``factory_bridge.dtype_name_for`` states the same rule for models the
    service loads; the service must work without forgelm, so it cannot import
    this one. tests/factory/test_training_hardware.py checks the two agree.
    """
    if accelerator in ("cuda", "rocm"):  # ROCm goes through torch.cuda
        return "bfloat16" if bf16_supported else "float16"
    if accelerator == "xpu":
        return "bfloat16"
    if accelerator == "mps":
        return "float16"
    return "float32"


def fits(measured_gb: float, available_gb: float, headroom: float = 0.9) -> bool:
    """Whether ``measured_gb`` fits in ``available_gb`` leaving ``1 - headroom`` spare.

    The spare share is there because a dry run uses synthetic batches of one
    fixed length: real batches, evaluation and checkpoint saving add to it.

    Raises ``ValueError`` for figures that cannot be memory sizes, so that a
    failed measurement (NaN, a negative difference) is never read as "fits".
    """
    for name, value in (("measured_gb", measured_gb), ("available_gb", available_gb)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite number of at least 0, got {value!r}")
    if not (0 < headroom <= 1):
        raise ValueError(f"headroom must be above 0 and at most 1, got {headroom!r}")
    return measured_gb <= available_gb * headroom


def _best_device(torch) -> str:
    """The device a model loaded here goes on: CUDA API, then XPU, then MPS, then CPU.

    The same order as the training service's ``torch_device_and_dtype``.
    Each attribute is read defensively: older torch has no ``xpu`` or ``mps``.
    """
    try:
        if torch.cuda.is_available():
            return "cuda"
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
    return "cpu"


def _accelerator_of(torch, device_type: str) -> str:
    """"cuda", "rocm", "xpu", "mps" or "cpu" for a torch device type."""
    if device_type == "cuda":
        return "rocm" if getattr(getattr(torch, "version", None), "hip", None) else "cuda"
    return device_type if device_type in ("xpu", "mps") else "cpu"


def _vocab_size(model, tokenizer_or_vocab_size) -> int:
    """How many token ids the synthetic batch may draw from.

    Capped at the size of the model's input embedding: an id beyond it is an
    index error on CPU and a device-side assert on a GPU.
    """
    embedding_rows = None
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        try:
            embedding_rows = int(get_embeddings().weight.shape[0])
        except Exception:
            embedding_rows = None

    if tokenizer_or_vocab_size is None:
        size = embedding_rows or getattr(getattr(model, "config", None), "vocab_size", None)
        if not size:
            raise ValueError("cannot tell the vocabulary size from the model; pass a tokenizer or an integer")
    elif isinstance(tokenizer_or_vocab_size, bool):
        raise ValueError("tokenizer_or_vocab_size must be a tokenizer, an integer or None")
    elif isinstance(tokenizer_or_vocab_size, int):
        size = tokenizer_or_vocab_size
    else:
        # len(tokenizer) counts added tokens; vocab_size does not.
        try:
            size = len(tokenizer_or_vocab_size)
        except TypeError:
            size = getattr(tokenizer_or_vocab_size, "vocab_size", None)
        if not size:
            raise ValueError("tokenizer_or_vocab_size must be a tokenizer, an integer or None")
    size = int(size)
    if size < 2:
        raise ValueError(f"the vocabulary must have at least 2 tokens, got {size}")
    return min(size, embedding_rows) if embedding_rows else size


def _weight_bits(model) -> int:
    """Bits per weight of the base model: 4 or 8 for bitsandbytes layers, else the commonest dtype."""
    names = {type(m).__name__ for m in model.modules()}
    if "Linear4bit" in names:
        return 4
    if "Linear8bitLt" in names:
        return 8
    by_bits: Dict[int, int] = {}
    for p in model.parameters():
        bits = p.element_size() * 8
        by_bits[bits] = by_bits.get(bits, 0) + p.numel()
    return max(by_bits, key=by_bits.get) if by_bits else 32


def _is_out_of_memory(torch, exc: BaseException) -> bool:
    """Whether an exception is a device running out of memory.

    CUDA raises a dedicated error class on recent torch; MPS and older CUDA
    builds raise a RuntimeError whose message says so.
    """
    for owner in (torch, getattr(torch, "cuda", None)):
        oom_class = getattr(owner, "OutOfMemoryError", None)
        if isinstance(oom_class, type) and isinstance(exc, oom_class):
            return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# --------------------------------------------------------------------------
# Reading memory
# --------------------------------------------------------------------------

def _process_memory_readers() -> Tuple[Optional[Callable[[], int]], Optional[Callable[[], int]], str]:
    """(current RSS in bytes, lifetime peak RSS in bytes, name of the source).

    Either callable may be None: psutil is optional, and ``resource`` does not
    exist on Windows.
    """
    current = high_water = None
    try:
        import psutil

        process = psutil.Process()
        current = lambda: int(process.memory_info().rss)  # noqa: E731
    except Exception:
        pass
    try:
        import resource

        # ru_maxrss is in kilobytes on Linux and in bytes on macOS.
        scale = 1 if sys.platform == "darwin" else 1024
        high_water = lambda: int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale  # noqa: E731
    except Exception:
        pass
    if current is not None:
        return current, high_water, "process RSS growth (psutil)"
    if high_water is not None:
        return None, high_water, "process RSS growth (resource high-water mark)"
    return None, None, "unavailable"


class _MemoryMeter:
    """Baseline and peak memory for one kind of device.

    ``start()`` takes the baseline, ``sample()`` is called at the points in a
    step where memory is highest, ``stop()`` returns ``(baseline, peak)`` in
    bytes. A runtime with its own peak counter ignores the samples; one
    without is only as good as its samples, and says so in ``notes``.

    ``torch`` is passed in so the tests can drive every branch with a
    stand-in: none of the accelerator branches can run on this machine.
    """

    def __init__(self, torch, kind: str, indices: Sequence[int] = (0,)):
        self.torch = torch
        self.indices = list(indices) or [0]
        self.notes: List[str] = []
        self.baseline: Optional[int] = None
        self.peak: Optional[int] = None
        self.reserved_peak: Optional[int] = None
        self._read: Optional[Callable[[], int]] = None  # sampled sources
        self._high_water: Optional[Callable[[], int]] = None
        self._high_water_at_start: Optional[int] = None
        self._counter = None  # torch.cuda or torch.xpu when they keep a peak themselves

        if kind in ("cuda", "rocm"):
            self._counter = torch.cuda
            self.source = "torch.cuda.max_memory_allocated"
            return
        if kind == "xpu":
            xpu = getattr(torch, "xpu", None)
            if all(hasattr(xpu, n) for n in ("max_memory_allocated", "reset_peak_memory_stats", "memory_allocated")):
                self._counter = xpu
                self.source = "torch.xpu.max_memory_allocated"
                return
            if hasattr(xpu, "memory_allocated"):
                self._read = lambda: sum(int(xpu.memory_allocated(i)) for i in self.indices)
                self.source = "torch.xpu.memory_allocated (sampled)"
                self.notes.append("this PyTorch build has no XPU peak counter: memory was sampled around each "
                                  "step, so a short-lived peak inside a step can be missed")
                return
            self.notes.append("this PyTorch build reports no XPU memory: process memory was measured instead, "
                              "which is not what the device holds")
        elif kind == "mps":
            mps = getattr(torch, "mps", None)
            if hasattr(mps, "driver_allocated_memory"):
                self._read = lambda: int(mps.driver_allocated_memory())
                self.source = "torch.mps.driver_allocated_memory (sampled)"
                self.notes.append("Metal keeps no peak counter: driver-allocated memory was sampled around each "
                                  "step, and it includes memory the allocator holds for reuse")
                return
            self.notes.append("this PyTorch build reports no MPS memory: process memory was measured instead, "
                              "which on unified memory includes the model but is not the driver's own figure")

        # CPU, and the fallback for the two cases above.
        self._read, self._high_water, self.source = _process_memory_readers()
        if self._read is None and self._high_water is not None:
            self.notes.append("psutil is not installed: growth is read from the process high-water mark, which "
                              "does not move if the process used more memory earlier than these steps did")
        elif self._read is None:
            self.notes.append("neither psutil nor the resource module is available: memory was not measured")

    def start(self) -> None:
        if self._counter is not None:
            total = 0
            for i in self.indices:
                self._counter.synchronize(i)
                self._counter.reset_peak_memory_stats(i)
                total += int(self._counter.memory_allocated(i))
            self.baseline = self.peak = total
            return
        if self._high_water is not None:
            self._high_water_at_start = self._high_water()
        if self._read is not None:
            self.baseline = self.peak = self._read()
        elif self._high_water_at_start is not None:
            self.baseline = self.peak = self._high_water_at_start

    def sample(self) -> None:
        if self._read is not None and self.peak is not None:
            self.peak = max(self.peak, self._read())

    def stop(self) -> Tuple[Optional[int], Optional[int]]:
        if self._counter is not None:
            self.peak = 0
            for i in self.indices:
                self._counter.synchronize(i)
                self.peak += int(self._counter.max_memory_allocated(i))
            reserved = getattr(self._counter, "max_memory_reserved", None)
            if reserved is not None:
                self.reserved_peak = sum(int(reserved(i)) for i in self.indices)
            return self.baseline, self.peak
        self.sample()
        if self._high_water is not None and self._high_water_at_start is not None and self.peak is not None:
            now = self._high_water()
            # The lifetime peak moved during the steps, so the steps set it.
            # This catches a peak inside a step that fell between two samples.
            if now > self._high_water_at_start:
                self.peak = max(self.peak, now)
        return self.baseline, self.peak


# --------------------------------------------------------------------------
# The dry run
# --------------------------------------------------------------------------

def _synchronize(torch, device_type: str) -> None:
    """Wait for queued device work, so that a step's time is the step's time."""
    try:
        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "mps":
            torch.mps.synchronize()
        elif device_type == "xpu":
            torch.xpu.synchronize()
    except Exception:
        pass  # a build without synchronize: timings are then approximate


def _empty_cache(torch, device_type: str) -> None:
    try:
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "mps":
            torch.mps.empty_cache()
        elif device_type == "xpu":
            torch.xpu.empty_cache()
    except Exception:
        pass


def _load_model(torch, model_id: str, device: Optional[str]):
    """Load a causal LM by Hugging Face id (or local path) for the measurement."""
    import transformers

    device = device or _best_device(torch)
    device_type = str(device).split(":")[0]
    bf16 = False
    if device_type == "cuda":
        try:
            bf16 = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16 = False
    dtype = getattr(torch, dtype_name_for(_accelerator_of(torch, device_type), bf16))
    # The argument was renamed from torch_dtype to dtype in transformers 4.56.
    try:
        version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
    except ValueError:
        version = (0, 0)
    dtype_argument = "dtype" if version >= (4, 56) else "torch_dtype"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **{dtype_argument: dtype})
    return model.to(device)


def _loss_of(torch, output, input_ids):
    """The training loss from whatever the model returned.

    Hugging Face models return an object with ``.loss`` when given labels. A
    plain module returns logits, from which the usual next-token loss is
    computed here.
    """
    loss = getattr(output, "loss", None)
    if loss is not None:
        return loss
    logits = getattr(output, "logits", None)
    if logits is None:
        logits = output[0] if isinstance(output, (tuple, list)) else output
    if logits.dim() == 0:
        return logits  # the module computed its own scalar loss
    shifted = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    return torch.nn.functional.cross_entropy(shifted.float(), input_ids[:, 1:].reshape(-1))


def measure_training_memory(model, tokenizer_or_vocab_size, batch_size: int, seq_len: int, steps: int = 3,
                            device: Optional[str] = None, lora: Optional[Any] = None, *, lr: float = 1e-4,
                            restore: bool = False, amp_dtype: Optional[str] = None, seed: int = 0) -> Dict[str, Any]:
    """Run a few real optimisation steps and report peak memory and seconds per step.

    ``model``
        A loaded model, or a Hugging Face id or local path (a string), which
        is loaded here and freed again afterwards. A loaded model is measured
        where it is unless ``device`` says otherwise, in which case it is
        moved there and stays there.
    ``tokenizer_or_vocab_size``
        A tokenizer, an integer, or ``None`` to use the size of the model's
        input embedding. Only the number of token ids is needed: the batch is
        synthetic, random ids of shape ``(batch_size, seq_len)`` used as their
        own labels.
    ``lora``
        Keyword arguments for ``peft.LoraConfig`` (or a ``peft`` config
        object). The adapter is attached for the measurement and removed
        again, which leaves the base model as it was. ``None`` trains whatever
        in the model has ``requires_grad`` set, so a model that already
        carries an adapter is measured as it will be trained.
    ``restore``
        The steps are real: AdamW changes the trainable weights. With
        ``restore=True`` they are copied to CPU memory first and put back
        afterwards. Off by default because the copy is as large as the
        trainable weights.
    ``amp_dtype``
        ``"bf16"`` or ``"fp16"`` to run the forward pass under autocast, the
        way a Trainer with ``bf16=True`` / ``fp16=True`` does. ``None`` runs
        the model in the precision it was loaded in.

    Gradient checkpointing is the caller's choice: enable it on the model
    before calling if the real run will use it.

    The result is a dictionary. The fields a caller usually wants:

        seconds_per_step   median wall time of a step (the first is often slower)
        peak_gb            the peak the device's runtime reported: memory
                           allocated on a GPU, growth of the process's resident
                           memory on a CPU
        total_gb           what training this model needs on that device, the
                           figure to pass to ``fits``. On a GPU it is
                           ``peak_gb`` (the weights live on the device and are
                           in the count). On a CPU it is the weights plus the
                           growth, because the weights were in memory before
                           the baseline was taken.
        completed          False when the device ran out of memory, in which
                           case ``out_of_memory`` is True and ``error`` says why

    An out-of-memory error on a GPU is reported, not raised: finding that out
    is what a dry run is for. On a CPU the operating system kills the process
    instead, and nothing can report that.
    """
    import torch

    for name, value, least in (("batch_size", batch_size, 1), ("seq_len", seq_len, 2), ("steps", steps, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < least:
            raise ValueError(f"{name} must be an integer of at least {least}, got {value!r}")
    if amp_dtype not in (None, "bf16", "fp16"):
        raise ValueError(f"amp_dtype must be None, 'bf16' or 'fp16', got {amp_dtype!r}")

    loaded_here = isinstance(model, str)
    model_id = model if loaded_here else None
    if loaded_here:
        model = _load_model(torch, model, device)
    else:
        first = next(model.parameters(), None)
        if first is None:
            raise ValueError("the model has no parameters")
        if device is not None and torch.device(device) != first.device:
            model = model.to(device)

    vocab = _vocab_size(model, tokenizer_or_vocab_size)
    config = getattr(model, "config", None)
    max_positions = getattr(config, "max_position_embeddings", None) or getattr(config, "n_positions", None)
    if isinstance(max_positions, int) and seq_len > max_positions:
        if loaded_here:
            del model
        raise ValueError(f"seq_len {seq_len} is longer than the model's {max_positions} positions")

    was_training = model.training
    train_model = model
    frozen_by_peft: list = []
    if lora is not None:
        import peft

        # peft freezes every base weight when it attaches an adapter and does
        # not thaw them when the adapter is removed, so remember which were trainable.
        frozen_by_peft = [p for p in model.parameters() if p.requires_grad]

        if isinstance(lora, peft.PeftConfig):
            lora_config = lora
        else:
            kwargs = {} if lora is True else dict(lora)
            kwargs.setdefault("task_type", "CAUSAL_LM")
            lora_config = peft.LoraConfig(**kwargs)
        train_model = peft.get_peft_model(model, lora_config)

    train_model.train()
    trainable = [p for p in train_model.parameters() if p.requires_grad]

    def clean_up(saved=None) -> None:
        """Leave the caller's model as it was found, as far as that was asked for."""
        nonlocal model, train_model
        for p in trainable:
            p.grad = None
        if saved is not None:
            with torch.no_grad():
                for p, original in zip(trainable, saved):
                    p.copy_(original.to(p.device))
        if lora is not None:
            # Removes the adapter layers without merging them: the base modules come back untouched.
            train_model.unload()
            for p in frozen_by_peft:
                p.requires_grad_(True)
        model.train(was_training)
        if loaded_here:
            train_model = model = None
        gc.collect()
        _empty_cache(torch, device_type)

    embeddings = getattr(train_model, "get_input_embeddings", None)
    try:
        input_device = embeddings().weight.device if callable(embeddings) else trainable[0].device
    except Exception:
        input_device = next(train_model.parameters()).device
    device_type = input_device.type
    if not trainable:
        clean_up()
        raise ValueError("nothing in the model is trainable: set requires_grad or pass lora=...")

    accelerator = _accelerator_of(torch, device_type)
    total_params = sum(p.numel() for p in train_model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    weight_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    weight_bits = _weight_bits(model)

    # A model spread over several GPUs (device_map="auto") is measured on all of them.
    indices = sorted({(p.device.index if p.device.index is not None else 0)
                      for p in train_model.parameters() if p.device.type == device_type}) or [0]

    def fingerprint() -> list:
        # A few thousand evenly spaced values from each trainable tensor,
        # copied to CPU: enough to tell that the optimiser moved the weights,
        # without a second copy of them on a device that may be nearly full.
        # copy=True matters: for float32 weights on a CPU the conversion would
        # otherwise return a view, which changes along with the weights.
        samples = []
        for p in trainable:
            flat = p.detach().reshape(-1)
            samples.append(flat[:: max(1, flat.numel() // 4096)][:4096].to("cpu", torch.float32, copy=True))
        return samples

    saved = None
    if restore and lora is None:
        saved = [p.detach().to("cpu", copy=True) for p in trainable]
    before = fingerprint()

    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab, (batch_size, seq_len), generator=generator).to(input_device)
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp_dtype)

    if device_type == "cpu":
        # Weights loaded from disk can be memory-mapped and not yet resident.
        # Reading them now keeps their arrival out of the growth measured below.
        with torch.no_grad():
            for p in model.parameters():
                p.sum()

    meter = _MemoryMeter(torch, accelerator, indices)
    step_seconds: List[float] = []
    losses: List[float] = []
    error: Optional[str] = None
    out_of_memory = False
    loss = output = None

    meter.start()
    try:
        for _ in range(steps):
            _synchronize(torch, device_type)
            started = time.perf_counter()
            if autocast_dtype is not None:
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    output = _forward(train_model, input_ids)
                    loss = _loss_of(torch, output, input_ids)
            else:
                output = _forward(train_model, input_ids)
                loss = _loss_of(torch, output, input_ids)
            meter.sample()  # every activation the backward pass needs is alive here
            loss.backward()
            meter.sample()
            optimizer.step()
            meter.sample()
            optimizer.zero_grad(set_to_none=True)
            _synchronize(torch, device_type)
            step_seconds.append(time.perf_counter() - started)
            losses.append(float(loss.detach()))
    except Exception as exc:
        if not _is_out_of_memory(torch, exc):
            del loss, output, optimizer
            clean_up(saved)
            raise
        out_of_memory = True
        error = f"{type(exc).__name__}: {exc}"
    baseline, peak = meter.stop()

    # Drop the graph and the gradients before anything else touches the device:
    # after an out-of-memory error there is no room left to work in.
    del loss, output, optimizer, input_ids
    for p in trainable:
        p.grad = None
    if out_of_memory:
        _empty_cache(torch, device_type)
    after = fingerprint()
    weights_changed = any(not torch.equal(a, b) for a, b in zip(before, after))
    clean_up(saved)

    def gib(value: Optional[int]) -> Optional[float]:
        return None if value is None else round(value / GIB, 4)

    increase = None if (baseline is None or peak is None) else max(0, peak - baseline)
    on_cpu_counter = meter.source.startswith("process RSS") or meter.source == "unavailable"
    if on_cpu_counter:
        # Process memory: only the growth belongs to the steps, and the
        # weights, already resident, are added back for the total.
        peak_bytes = increase
        total_bytes = None if increase is None else weight_bytes + increase
    else:
        peak_bytes = peak
        total_bytes = peak

    notes = list(meter.notes)
    if losses and not all(math.isfinite(x) for x in losses):
        notes.append("the loss was not finite in at least one step; the memory figures still hold, "
                     "but check the precision before training (16-bit weights without autocast overflow easily)")
    if steps < 2:
        notes.append("one step only: seconds_per_step includes one-off warm-up cost")

    ordered = sorted(step_seconds)
    median = None
    if ordered:
        middle = len(ordered) // 2
        median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2

    return {
        "completed": not out_of_memory,
        "out_of_memory": out_of_memory,
        "error": error,
        "model": model_id,
        "device": str(input_device),
        "device_indices": indices,
        "accelerator": accelerator,
        "measure": meter.source,
        "seconds_per_step": median,
        "step_seconds": [round(s, 6) for s in step_seconds],
        "peak_gb": gib(peak_bytes),
        "total_gb": gib(total_bytes),
        "baseline_gb": gib(baseline),
        "peak_increase_gb": gib(increase),
        "peak_reserved_gb": gib(meter.reserved_peak),
        "model_weights_gb": gib(weight_bytes),
        "model_resident": not loaded_here,
        "weight_bits": weight_bits,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "lora": lora is not None,
        "amp_dtype": amp_dtype,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "steps": len(step_seconds),
        "steps_requested": steps,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "weights_changed": weights_changed,
        "restored": saved is not None or lora is not None,
        "notes": notes,
    }


def _forward(model, input_ids):
    """Call the model the way a causal-LM trainer does, or as a plain module.

    A Hugging Face causal LM takes ``labels`` and returns the loss. A plain
    module that maps ids to logits does not, and is called with the ids alone.
    """
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        parameters = {}
    takes_labels = "labels" in parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    if takes_labels or not parameters:
        return model(input_ids=input_ids, labels=input_ids)
    return model(input_ids)


# --------------------------------------------------------------------------
# Estimate, measurement and verdict together
# --------------------------------------------------------------------------

def available_memory_gb(measurement: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """Memory the measured device has for training, and what that figure is.

    GPU: the device's total memory (what other processes hold is not
    subtracted). Apple MPS: 65% of system RAM. CPU: RAM available now,
    honouring a container limit, plus the model's weights when the model is
    still loaded, since they are already counted as used.
    """
    accelerator = measurement.get("accelerator", "cpu")
    indices = measurement.get("device_indices") or [0]
    if accelerator in ("cuda", "rocm", "xpu"):
        import torch

        backend = torch.cuda if accelerator != "xpu" else getattr(torch, "xpu", None)
        try:
            total = sum(backend.get_device_properties(i).total_memory for i in indices)
        except Exception:
            return None, "the device's total memory could not be read"
        return round(total / GIB, 2), "total memory of the device(s) the model is on"

    from .hardware import _memory, _read

    total_gb, available_gb, limited = _memory(_read)
    if accelerator == "mps":
        return round(total_gb * MPS_MEMORY_FRACTION, 2), f"{MPS_MEMORY_FRACTION:.0%} of unified memory"
    basis = "RAM available now" + (" (container limit applied)" if limited else "")
    if measurement.get("model_resident"):
        return round(available_gb + (measurement.get("model_weights_gb") or 0.0), 2), basis + " plus the loaded weights"
    return available_gb, basis


def compare_with_estimate(measurement: Dict[str, Any], available_gb: Optional[float] = None,
                          params_billion: Optional[float] = None, bits: Optional[int] = None,
                          headroom: float = 0.9, gradient_checkpointing: bool = True,
                          full_finetune: Optional[bool] = None) -> Dict[str, Any]:
    """Set a dry-run measurement beside ``forgelm.generative.estimate_memory_gb``.

    Everything the estimate needs defaults to what the measurement recorded:
    the model's parameter count, the bits per weight it was loaded with, the
    batch size and sequence length, and whether most of the model was
    trainable (a full fine-tune) or only an adapter. ``available_gb`` defaults
    to ``available_memory_gb(measurement)``.

    Returns the estimate, the measurement, the memory available, whether each
    fits, and a one-word ``verdict`` taken from the measurement, never from
    the estimate: ``"fits"``, ``"does_not_fit"`` or ``"unknown"`` (memory
    could not be measured, or what is available is not known).
    """
    from .generative import estimate_memory_gb

    if params_billion is None:
        params_billion = measurement["total_params"] / 1e9
    if bits is None:
        bits = measurement.get("weight_bits") or 16
    bits = bits if bits in (4, 8, 16, 32) else 32  # float64 weights: the estimate has no wider case
    if full_finetune is None:
        full_finetune = measurement["trainable_params"] > 0.5 * measurement["total_params"]
    estimate = estimate_memory_gb(params_billion, bits, measurement["batch_size"], measurement["seq_len"],
                                  gradient_checkpointing, full_finetune)

    basis = "given by the caller"
    if available_gb is None:
        available_gb, basis = available_memory_gb(measurement)

    measured = measurement.get("total_gb")
    # estimate_memory_gb counts parameters in billions (decimal GB); measurements and available
    # memory are in GiB. Convert so the comparison is like with like (the difference is about 7%).
    estimate_gib = estimate["total_gb"] * 1e9 / 1024**3
    estimate_fits = None if available_gb is None else fits(estimate_gib, available_gb, headroom)
    measured_fits: Optional[bool]
    if measurement.get("out_of_memory"):
        measured_fits = False
    elif measured is None or available_gb is None:
        measured_fits = None
    else:
        measured_fits = fits(measured, available_gb, headroom)
    verdict = {True: "fits", False: "does_not_fit", None: "unknown"}[measured_fits]

    if measurement.get("out_of_memory"):
        measured_text = "the dry run ran out of memory"
    elif measured is None:
        measured_text = "memory could not be measured"
    else:
        measured_text = f"measured {measured:.2f} GiB ({measurement.get('measure')})"
    available_text = "available memory unknown" if available_gb is None else \
        f"{available_gb:.2f} GiB available ({basis}), {headroom:.0%} of it usable"
    summary = (f"estimate {estimate_gib:.2f} GiB, {measured_text}, {available_text}: "
               f"{verdict.replace('_', ' ')}")

    return {
        "estimate": estimate,
        "measurement": measurement,
        "available_gb": available_gb,
        "available_basis": basis,
        "headroom": headroom,
        "estimate_fits": estimate_fits,
        "measured_fits": measured_fits,
        # Positive when the estimate was the more cautious of the two.
        "estimate_minus_measured_gb": None if measured is None else round(estimate_gib - measured, 2),
        "verdict": verdict,
        "summary": summary,
    }
