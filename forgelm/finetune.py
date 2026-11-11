"""LoRA fine-tuning of a small generative model on any PyTorch device.

The training service fine-tunes through Unsloth, which needs an NVIDIA GPU.
This module is the portable path: a plain PyTorch loop over Hugging Face
``peft`` adapters that runs unchanged on CUDA, ROCm, XPU, MPS and CPU, with no
bitsandbytes, xformers or flash-attention dependency to break elsewhere.

    FineTuneRecipe      what to train and how, as plain data
    encode_chat         one conversation -> token ids and loss labels
    finetune_lora       train an adapter, save it with its tokenizer and recipe
    merge_adapter       fold the adapter into full weights (needed for GGUF)
    load_model          a model and its tokenizer, ready to generate
    generate            greedy decoding of one conversation
    evaluate_exact      exact-match accuracy, keeping every raw output
    write_mlx_dataset, write_mlx_config, mlx_lora_command
                        the same recipe expressed for Apple's mlx-lm trainer

Nothing here imports torch at module level. :mod:`forgelm.gguf` takes the
merged model the rest of the way to a quantised GGUF file.

What has and has not been run: the PyTorch path is tested on CPU (float32).
The bfloat16 + autocast branch used for CUDA and XPU has only been exercised
on CPU, and the MLX functions only build files and an argument list; neither
has been run on the hardware it is meant for.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# The value PyTorch's cross-entropy skips. Every position that must not
# contribute to the loss (the prompt, the padding) carries this label.
IGNORE_INDEX = -100


# --------------------------------------------------------------------------
# Recipe
# --------------------------------------------------------------------------

@dataclass
class FineTuneRecipe:
    """Everything that decides the result of a fine-tune, and nothing else.

    ``method``, ``rank``, ``alpha``, ``dropout`` and ``target_modules`` are
    handed to :func:`forgelm.generative.lora_config`, so the adapter built here
    is the one the training service builds for the same request.

    ``warmup`` is a fraction of the optimizer steps when below 1 and a number
    of steps otherwise. ``epochs`` may be fractional.
    """

    base_model: str
    method: str = "lora"
    rank: int = 16
    alpha: Optional[int] = None  # None means 2 * rank, as in forgelm.generative
    dropout: float = 0.0
    target_modules: Optional[List[str]] = None
    learning_rate: float = 2e-4
    epochs: float = 3
    batch_size: int = 4
    grad_accumulation: int = 1
    max_seq_len: int = 512
    warmup: float = 0.1
    seed: int = 0
    train_on_responses_only: bool = True
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    def __post_init__(self):
        # Checked here rather than deep in the loop: a batch size of 0 would
        # otherwise surface as a division error after the model has loaded.
        if not self.base_model:
            raise ValueError("base_model must name a Hugging Face model id or a local directory")
        for name, minimum in (("rank", 1), ("batch_size", 1), ("grad_accumulation", 1), ("max_seq_len", 2)):
            if not (getattr(self, name) >= minimum):
                raise ValueError(f"{name} must be at least {minimum}, got {getattr(self, name)!r}")
        if not (self.epochs > 0):
            raise ValueError(f"epochs must be greater than 0, got {self.epochs!r}")
        if not (self.learning_rate > 0):
            raise ValueError(f"learning_rate must be greater than 0, got {self.learning_rate!r}")
        if not (self.warmup >= 0):
            raise ValueError(f"warmup must be 0 or more, got {self.warmup!r}")

    @property
    def effective_alpha(self) -> int:
        return self.alpha if self.alpha is not None else 2 * self.rank

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        return str(path)

    @classmethod
    def load(cls, path: str) -> "FineTuneRecipe":
        data = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------
# Chat records
# --------------------------------------------------------------------------

def check_record(record: dict, index: int = 0) -> List[dict]:
    """Return the messages of one training record, or say what is wrong with it."""
    messages = record.get("messages") if isinstance(record, dict) else None
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"record {index}: expected {{'messages': [...]}} with at least a user and an "
                         "assistant message")
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("role"), str) or not isinstance(m.get("content"), str):
            raise ValueError(f"record {index}: every message needs a string 'role' and a string 'content'")
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"record {index}: the last message must be the assistant's answer, "
                         f"got role {messages[-1]['role']!r}")
    return messages


def load_chat_records(path: str) -> List[dict]:
    """Read a JSONL file of chat records, one conversation per line."""
    records = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}, line {n}: not valid JSON ({exc.msg})") from None
        check_record(record, n)
        records.append(record)
    if not records:
        raise ValueError(f"{path} holds no records")
    return records


def _token_ids(tokenizer, text: str) -> List[int]:
    # The chat template has already written the special tokens it wants.
    # Letting the tokenizer add its own as well gives two BOS tokens on models
    # whose template starts with one, which trains on a prompt the model will
    # never see at inference.
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def encode_chat(tokenizer, messages: Sequence[dict], max_seq_len: int,
                train_on_responses_only: bool = True) -> Dict[str, object]:
    """Turn one conversation into ``input_ids`` and ``labels``.

    With ``train_on_responses_only`` the loss covers the final assistant turn
    only: its text and the end-of-turn tokens the template writes after it, so
    the model also learns to stop. The prompt length is measured by rendering
    the conversation without its last message and with
    ``add_generation_prompt=True``, which is exactly what the model is given
    at inference.

    Sequences longer than ``max_seq_len`` are cut on the right. That can cut
    into the answer, or remove it entirely; ``n_target`` says how many
    supervised tokens are left so the caller can drop an example with none.
    """
    full_text = tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=False)
    full_ids = _token_ids(tokenizer, full_text)
    n_prompt = 0
    if train_on_responses_only:
        prompt_text = tokenizer.apply_chat_template(list(messages[:-1]), tokenize=False, add_generation_prompt=True)
        if not full_text.startswith(prompt_text):
            # Without this the mask would cover the wrong tokens and nothing
            # would fail: the loss would simply be computed on prompt text.
            raise ValueError(
                "the chat template does not render the prompt as a prefix of the full conversation, so the "
                "answer cannot be located; set train_on_responses_only=False for this model")
        prompt_ids = _token_ids(tokenizer, prompt_text)
        # Count the tokens the two renderings share, rather than trusting
        # len(prompt_ids): a tokenizer may merge the last prompt character with
        # the first answer character. A token that straddles the boundary
        # carries answer text, so it is supervised.
        limit = min(len(prompt_ids), len(full_ids))
        while n_prompt < limit and prompt_ids[n_prompt] == full_ids[n_prompt]:
            n_prompt += 1
    input_ids = full_ids[:max_seq_len]
    labels = [IGNORE_INDEX if i < n_prompt else tok for i, tok in enumerate(input_ids)]
    return {"input_ids": input_ids, "labels": labels, "truncated": len(full_ids) > max_seq_len,
            "n_target": max(0, len(input_ids) - n_prompt)}


def collate(examples: Sequence[dict], pad_token_id: int) -> Dict[str, np.ndarray]:
    """Right-pad a batch to its longest member.

    Padding goes on the right whatever ``tokenizer.padding_side`` says: with a
    causal mask, tokens on the right cannot influence the real tokens to their
    left. Padded positions get attention 0 and label -100.
    """
    width = max(len(e["input_ids"]) for e in examples)
    input_ids = np.full((len(examples), width), pad_token_id, dtype=np.int64)
    labels = np.full((len(examples), width), IGNORE_INDEX, dtype=np.int64)
    attention = np.zeros((len(examples), width), dtype=np.int64)
    for row, e in enumerate(examples):
        n = len(e["input_ids"])
        input_ids[row, :n] = e["input_ids"]
        labels[row, :n] = e["labels"]
        attention[row, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def batch_schedule(n_examples: int, batch_size: int, epochs: float, seed: int) -> List[np.ndarray]:
    """Index batches for the whole run: a fresh shuffle each epoch, cut to ``epochs``."""
    rng = np.random.default_rng(seed)
    per_epoch = math.ceil(n_examples / batch_size)
    wanted = max(1, int(round(per_epoch * epochs)))
    batches: List[np.ndarray] = []
    while len(batches) < wanted:
        order = rng.permutation(n_examples)
        batches += [order[i:i + batch_size] for i in range(0, n_examples, batch_size)]
    return batches[:wanted]


def warmup_steps(warmup: float, total_steps: int) -> int:
    """``warmup`` below 1 is a fraction of the run; 1 or more is a step count."""
    steps = int(round(warmup * total_steps)) if warmup < 1 else int(warmup)
    # Leave at least one step after the warm-up, or the peak rate is never used.
    return max(0, min(steps, total_steps - 1))


def lr_multiplier(step: int, total_steps: int, warmup: int) -> float:
    """Linear warm-up to 1, then linear decay, for optimizer step ``step`` (0-based).

    The first step is already above 0 and the last is still above 0, so no
    step is spent on a learning rate of exactly zero.
    """
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup))


# --------------------------------------------------------------------------
# Device and precision
# --------------------------------------------------------------------------

def resolve_device(device: Optional[str] = None) -> str:
    """The torch device string: cuda (NVIDIA or AMD ROCm) > xpu > mps > cpu."""
    from .students import pick_device

    # forgelm.generative calls AMD's backend "rocm"; PyTorch calls it "cuda".
    return "cuda" if device == "rocm" else pick_device(device)


def training_dtype(device: str) -> str:
    """The weight precision to load the frozen base model in.

    float32 on CPU and MPS: half precision is slower than float32 on most
    CPUs, and float16 training on MPS is a known source of NaNs. bfloat16 on
    CUDA and XPU where the device supports it, which halves memory; float16 is
    never used because training in it needs a gradient scaler.
    """
    kind = device.split(":")[0]
    if kind == "cuda":
        import torch

        return "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
    if kind == "xpu":
        return "bfloat16"
    return "float32"


def _load_causal_lm(name: str, dtype: str):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    # transformers renamed the argument from torch_dtype to dtype in 4.56.
    version = tuple(int(p) for p in re.findall(r"\d+", transformers.__version__)[:2])
    key = "dtype" if version >= (4, 56) else "torch_dtype"
    return AutoModelForCausalLM.from_pretrained(name, **{key: getattr(torch, dtype)})


def load_model(name: str, device: Optional[str] = None, dtype: Optional[str] = None):
    """``(model, tokenizer)`` ready for :func:`generate`, on the best device present.

    For a full model directory (for example the output of
    :func:`merge_adapter`) or a Hub id. The precision defaults to what
    :func:`training_dtype` picks for the device, so a model is scored in the
    precision it was trained in.
    """
    from transformers import AutoTokenizer

    device = resolve_device(device)
    model = _load_causal_lm(name, dtype or training_dtype(device)).to(device)
    model.eval()
    return model, AutoTokenizer.from_pretrained(name)


def _pad_id(tokenizer) -> int:
    # Any id works as padding because padded positions are masked out of both
    # attention and loss; the end-of-sequence token is the conventional choice
    # for causal models that ship without a pad token.
    for candidate in (getattr(tokenizer, "pad_token_id", None), getattr(tokenizer, "eos_token_id", None)):
        if candidate is not None:
            return int(candidate)
    return 0


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def finetune_lora(recipe: FineTuneRecipe, train_records: Sequence[dict], out_dir: str,
                  eval_records: Optional[Sequence[dict]] = None, device: Optional[str] = None,
                  on_progress: Optional[Callable[[int, int, float], None]] = None,
                  dtype: Optional[str] = None) -> Dict[str, object]:
    """Train a PEFT adapter on chat records and save it to ``out_dir``.

    ``train_records`` and ``eval_records`` are conversations,
    ``{"messages": [{"role": ..., "content": ...}, ...]}``, each ending with
    the assistant answer to learn. ``on_progress(step, total_steps, loss)`` is
    called after every optimizer step. ``dtype`` overrides the precision
    :func:`training_dtype` picks for the device.

    Written to ``out_dir``: the adapter (``adapter_model.safetensors``,
    ``adapter_config.json``), the tokenizer, ``recipe.json`` and
    ``training_log.json``. The returned dict holds the loss of every optimizer
    step, the evaluation loss after each epoch when ``eval_records`` is given,
    the step count, wall-clock seconds and the device used.
    """
    import torch
    from peft import get_peft_model
    from transformers import AutoTokenizer

    from .generative import build_peft_config

    device = resolve_device(device)
    dtype = dtype or training_dtype(device)
    torch.manual_seed(recipe.seed)

    tokenizer = AutoTokenizer.from_pretrained(recipe.base_model)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"{recipe.base_model} has no chat template; fine-tune an instruct or chat model, "
                         "or set tokenizer.chat_template before training")
    pad_id = _pad_id(tokenizer)

    def encode_all(records, what):
        encoded, truncated, dropped = [], 0, 0
        for i, record in enumerate(records):
            e = encode_chat(tokenizer, check_record(record, i), recipe.max_seq_len, recipe.train_on_responses_only)
            truncated += bool(e["truncated"])
            if e["n_target"] == 0:
                # Nothing left to learn from: the prompt alone fills the
                # window. Keeping it would put a batch with no supervised
                # token, and so a NaN loss, into the run.
                dropped += 1
                continue
            encoded.append(e)
        if not encoded:
            raise ValueError(f"no {what} example has an answer inside max_seq_len={recipe.max_seq_len}; "
                             "raise max_seq_len or shorten the prompts")
        return encoded, truncated, dropped

    train, n_truncated, n_dropped = encode_all(train_records, "training")
    evaluation = encode_all(eval_records, "evaluation")[0] if eval_records else []

    model = _load_causal_lm(recipe.base_model, dtype)
    peft_config = build_peft_config(recipe.method, rank=recipe.rank, alpha=recipe.alpha, dropout=recipe.dropout,
                                    target_modules=recipe.target_modules)
    # peft keeps the adapter weights in float32 even on a half-precision base
    # (autocast_adapter_dtype), which is what makes bfloat16 training stable:
    # small AdamW updates would be rounded away in a 16-bit adapter.
    model = get_peft_model(model, peft_config)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())

    batches = batch_schedule(len(train), recipe.batch_size, recipe.epochs, recipe.seed)
    per_epoch = math.ceil(len(train) / recipe.batch_size)
    accum = recipe.grad_accumulation
    total_steps = math.ceil(len(batches) / accum)
    n_warmup = warmup_steps(recipe.warmup, total_steps)
    optimizer = torch.optim.AdamW(trainable, lr=recipe.learning_rate, weight_decay=recipe.weight_decay)

    # A float32 adapter on a bfloat16 base needs autocast to mix the two.
    mixed = dtype != "float32"

    def forward_loss(batch_examples):
        arrays = collate(batch_examples, pad_id)
        tensors = {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}
        with torch.autocast(device_type=device.split(":")[0], dtype=getattr(torch, dtype), enabled=mixed):
            return model(**tensors, use_cache=False).loss.float()

    def evaluation_loss() -> float:
        model.eval()
        total, weight = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(evaluation), recipe.batch_size):
                chunk = evaluation[i:i + recipe.batch_size]
                # Each batch loss is a mean over its supervised tokens, so it
                # is weighted by that count to get a per-token mean overall.
                n = sum(e["n_target"] for e in chunk)
                total += float(forward_loss(chunk)) * n
                weight += n
        model.train()
        return total / max(1, weight)

    loss_history: List[float] = []
    eval_history: List[Dict[str, float]] = []
    tokens_seen = 0
    started = time.time()
    model.train()
    optimizer.zero_grad()
    step, group_loss = 0, 0.0
    for b, indices in enumerate(batches):
        # The last group of a run can be shorter than ``accum``; divide by its
        # real size so its gradient is a mean like every other step's.
        group_size = min(accum, len(batches) - (b // accum) * accum)
        chunk = [train[i] for i in indices]
        loss = forward_loss(chunk)
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss became {float(loss)} at batch {b}; lower the learning rate"
                               + (" or train in float32" if mixed else ""))
        (loss / group_size).backward()
        group_loss += float(loss.detach()) / group_size
        tokens_seen += sum(len(e["input_ids"]) for e in chunk)
        if (b + 1) % accum == 0 or b + 1 == len(batches):
            if recipe.max_grad_norm and recipe.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, recipe.max_grad_norm)
            for group in optimizer.param_groups:
                group["lr"] = recipe.learning_rate * lr_multiplier(step, total_steps, n_warmup)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            loss_history.append(group_loss)
            group_loss = 0.0
            if on_progress:
                on_progress(step, total_steps, loss_history[-1])
        if evaluation and ((b + 1) % per_epoch == 0 or b + 1 == len(batches)):
            eval_history.append({"step": step, "epoch": round((b + 1) / per_epoch, 3),
                                 "loss": evaluation_loss()})
    seconds = time.time() - started

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    recipe.save(str(out / "recipe.json"))
    result = {
        "loss_history": loss_history, "eval_loss_history": eval_history, "steps": step,
        "seconds": seconds, "device": device, "dtype": dtype,
        "n_train": len(train), "n_eval": len(evaluation), "n_truncated": n_truncated,
        "n_dropped_no_answer": n_dropped, "tokens_seen": tokens_seen,
        "trainable_parameters": n_trainable, "total_parameters": n_total, "out_dir": str(out),
    }
    (out / "training_log.json").write_text(json.dumps(result, indent=2))
    return result


def merge_adapter(base_model: Optional[str], adapter_dir: str, out_dir: str, dtype: str = "float32") -> Dict[str, object]:
    """Fold a trained adapter into the base weights and save a standalone model.

    GGUF conversion reads ordinary model weights, not a base plus an adapter,
    so this is the step between :func:`finetune_lora` and
    :func:`forgelm.gguf.convert_to_gguf`. ``base_model`` may be ``None`` when
    ``adapter_dir`` was written by :func:`finetune_lora`, whose ``recipe.json``
    names it.

    The merge runs on the CPU in ``dtype``. float32 keeps the sum of base and
    adapter exact; for a model too large to hold in float32, pass "float16" or
    "bfloat16" and accept the rounding.
    """
    from peft import PeftModel
    from transformers import AutoTokenizer

    adapter = Path(adapter_dir)
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"{adapter} holds no adapter_config.json; pass the directory finetune_lora wrote")
    if base_model is None:
        if (adapter / "recipe.json").exists():
            base_model = FineTuneRecipe.load(str(adapter / "recipe.json")).base_model
        else:
            base_model = json.loads((adapter / "adapter_config.json").read_text()).get("base_model_name_or_path")
        if not base_model:
            raise ValueError(f"{adapter} does not record its base model; pass base_model")
    merged = PeftModel.from_pretrained(_load_causal_lm(base_model, dtype), str(adapter)).merge_and_unload()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    # The tokenizer saved beside the adapter is the one the adapter was trained
    # with, chat template included; fall back to the base model's otherwise.
    source = str(adapter) if (adapter / "tokenizer_config.json").exists() else base_model
    AutoTokenizer.from_pretrained(source).save_pretrained(str(out))
    weights = sorted(out.glob("*.safetensors"))
    return {"out_dir": str(out), "base_model": base_model, "dtype": dtype,
            "parameters": sum(p.numel() for p in merged.parameters()),
            "weights_bytes": sum(f.stat().st_size for f in weights)}


# --------------------------------------------------------------------------
# Generation and evaluation
# --------------------------------------------------------------------------

def generate(model, tokenizer, messages: Sequence[dict], max_new_tokens: int = 64, return_details: bool = False,
             **generate_kwargs):
    """Greedy-decode the assistant's reply to ``messages``.

    Greedy with no repetition penalty, whatever the model's own generation
    config says, so that two runs, or a PyTorch model and its GGUF export, can
    be compared output for output. Returns the reply text, or with
    ``return_details`` a dict that also holds token counts and seconds.
    """
    import torch

    prompt = tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([_token_ids(tokenizer, prompt)], device=model.device)
    # The sampling knobs are set to None so that values in the model's own
    # generation_config.json neither apply nor draw a warning under greedy.
    settings = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, repetition_penalty=1.0,
                    temperature=None, top_p=None, top_k=None, pad_token_id=_pad_id(tokenizer))
    settings.update(generate_kwargs)
    started = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **settings)
    seconds = time.time() - started
    new = out[0, ids.shape[1]:].tolist()
    # Count the tokens of the reply, not the end-of-turn token that stopped
    # it: llama.cpp leaves that token out of its completion count, and the two
    # throughput figures are compared side by side.
    stop = _stop_ids(model, tokenizer)
    n_reply = next((i for i, t in enumerate(new) if t in stop), len(new))
    text = tokenizer.decode(new[:n_reply], skip_special_tokens=True).strip()
    if not return_details:
        return text
    return {"text": text, "prompt_tokens": int(ids.shape[1]), "new_tokens": n_reply, "seconds": seconds}


def _stop_ids(model, tokenizer) -> set:
    """Token ids that end a reply: the generation config's EOS ids and the tokenizer's."""
    ids = set()
    config = getattr(model, "generation_config", None)
    for value in (getattr(config, "eos_token_id", None), getattr(tokenizer, "eos_token_id", None)):
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, (list, tuple)):
            ids.update(int(v) for v in value)
    return ids


def normalise_answer(text: str) -> str:
    """Case, surrounding whitespace, quotes and a final full stop do not count as a difference."""
    return re.sub(r"\s+", " ", text).strip().strip("\"'`").rstrip(".").strip().casefold()


def split_record(record: dict):
    """``(prompt messages, reference answer)`` of an evaluation record."""
    messages = check_record(record)
    return messages[:-1], messages[-1]["content"]


def evaluate_chat(chat: Callable[[List[dict], int], object], records: Sequence[dict],
                  normalise: Optional[Callable[[str], str]] = None, max_new_tokens: int = 64,
                  on_progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, object]:
    """Exact-match accuracy of any ``chat(messages, max_new_tokens)`` function.

    ``chat`` returns the reply text, or a dict with ``text`` and optionally
    ``prompt_tokens``, ``new_tokens`` and ``seconds`` (the shape both
    :func:`generate` and :meth:`forgelm.gguf.GGUFChat.chat_details` return).
    The same function scores a PyTorch model and a GGUF file, so their numbers
    are computed the same way.

    The raw outputs are returned beside the accuracy: an exact-match score
    alone cannot tell a wrong answer from a right answer with a stray word.
    """
    normalise = normalise or normalise_answer
    outputs, references, correct = [], [], []
    prompt_tokens = new_tokens = 0
    seconds = 0.0
    for i, record in enumerate(records):
        prompt, reference = split_record(record)
        started = time.time()
        reply = chat(prompt, max_new_tokens)
        elapsed = time.time() - started
        if isinstance(reply, dict):
            text = reply["text"]
            prompt_tokens += int(reply.get("prompt_tokens", 0))
            new_tokens += int(reply.get("new_tokens", 0))
            seconds += float(reply.get("seconds", elapsed))
        else:
            text = reply
            seconds += elapsed
        outputs.append(text)
        references.append(reference)
        correct.append(normalise(text) == normalise(reference))
        if on_progress:
            on_progress(i + 1, len(records))
    n = len(records)
    return {"accuracy": float(np.mean(correct)) if n else float("nan"), "n": n, "outputs": outputs,
            "references": references, "correct": correct, "prompt_tokens": prompt_tokens,
            "new_tokens": new_tokens, "seconds": seconds,
            "new_tokens_per_second": new_tokens / seconds if seconds > 0 and new_tokens else None,
            "seconds_per_example": seconds / n if n else None}


def evaluate_exact(model, tokenizer, records: Sequence[dict], normalise: Optional[Callable[[str], str]] = None,
                   max_new_tokens: int = 64, on_progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, object]:
    """Exact-match accuracy of a PyTorch model on chat records, with every raw output.

    Each record's final assistant message is the reference; the messages
    before it are the prompt. ``normalise`` is applied to both sides before
    comparing and defaults to :func:`normalise_answer`.
    """
    model.eval()
    return evaluate_chat(lambda messages, limit: generate(model, tokenizer, messages, limit, return_details=True),
                         records, normalise, max_new_tokens, on_progress)


# --------------------------------------------------------------------------
# Apple Silicon: the same recipe for mlx-lm
# --------------------------------------------------------------------------
#
# Flags and file formats follow the mlx-lm LoRA documentation and the argument
# parser it describes, read on 2026-11-11:
#   https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/LORA.md
#   https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/lora.py
#
# None of this can be run without Apple hardware. The tests check the files
# and the argument list against that documentation, not against mlx-lm itself.

MLX_CONFIG_NAME = "lora_config.yaml"
# mlx-lm trains LoRA, DoRA or the full weights. QLoRA is not a separate type
# there: it is what LoRA becomes when --model points at a quantised model.
_MLX_FINE_TUNE_TYPE = {"lora": "lora", "qlora": "lora", "dora": "dora"}


def write_mlx_dataset(records: Sequence[dict], data_dir: str, valid_records: Optional[Sequence[dict]] = None,
                      valid_fraction: float = 0.1, seed: int = 0) -> Dict[str, object]:
    """Write ``train.jsonl`` and ``valid.jsonl`` in mlx-lm's ``chat`` format.

    One JSON object per line with a ``messages`` list, which is also the
    format of the records themselves; keys mlx-lm does not read are left out.
    Without ``valid_records``, ``valid_fraction`` of the records (at least one,
    when there are two or more) is held out with a seeded shuffle. mlx-lm
    treats ``valid.jsonl`` as optional and reports validation loss when it is
    there.

    Not run against mlx-lm here: there is no Apple hardware to run it on.
    """
    rows = [{"messages": [{"role": m["role"], "content": m["content"]} for m in check_record(r, i)]}
            for i, r in enumerate(records)]
    if valid_records is not None:
        train = rows
        valid = [{"messages": [{"role": m["role"], "content": m["content"]} for m in check_record(r, i)]}
                 for i, r in enumerate(valid_records)]
    else:
        n_valid = min(len(rows) - 1, max(1, int(round(len(rows) * valid_fraction)))) if valid_fraction > 0 else 0
        order = np.random.default_rng(seed).permutation(len(rows))
        held_out = set(order[:max(0, n_valid)].tolist())
        train = [r for i, r in enumerate(rows) if i not in held_out]
        valid = [r for i, r in enumerate(rows) if i in held_out]
    if not train:
        raise ValueError("no training records to write")
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {"n_train": len(train), "n_valid": len(valid)}
    for name, part in (("train", train), ("valid", valid)):
        path = out / f"{name}.jsonl"
        if part:
            # json.dumps escapes newlines inside strings, which keeps every
            # example on one line as mlx-lm requires.
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
            written[name] = str(path)
        elif path.exists():
            path.unlink()  # a stale file from an earlier call would be read as this run's data
    return written


def mlx_iterations(recipe: FineTuneRecipe, n_train: int) -> int:
    """mlx-lm counts batches (``--iters``), not epochs."""
    return max(1, int(round(math.ceil(n_train / recipe.batch_size) * recipe.epochs)))


def _yaml_number(value: float) -> str:
    # Plain decimals only: standard YAML reads "2e-05" as a string, and
    # although mlx-lm patches its loader to accept it, a decimal is read
    # correctly by every loader.
    text = f"{value:.12f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def write_mlx_config(recipe: FineTuneRecipe, data_dir: str, n_train: int) -> str:
    """Write the YAML file that carries what mlx-lm takes only from a config.

    Rank, scale and dropout have no command-line flag in mlx-lm, and neither
    has the learning-rate schedule. mlx-lm's ``scale`` multiplies the adapter
    output directly, so it is ``alpha / rank`` in the terms used by ``peft``
    and by this recipe; passing alpha as the scale would multiply the update
    by the rank.
    """
    iters = mlx_iterations(recipe, n_train)
    # mlx-lm advances the schedule once per optimizer update, and with
    # gradient accumulation it updates every ``grad_accumulation`` batches.
    updates = max(1, iters // recipe.grad_accumulation)
    n_warmup = warmup_steps(recipe.warmup, updates)
    lines = [
        "# Written by forgelm.finetune.write_mlx_config. Options that mlx_lm.lora",
        "# accepts only from a config file; the rest is on the command line.",
        "lora_parameters:",
        f"  rank: {recipe.rank}",
        f"  scale: {_yaml_number(recipe.effective_alpha / recipe.rank)}",
        f"  dropout: {_yaml_number(recipe.dropout)}",
    ]
    if recipe.target_modules:
        bare = [m for m in recipe.target_modules if "." not in m]
        if bare:
            # mlx-lm matches the module's path inside a transformer block, so
            # a peft-style suffix such as "q_proj" would match nothing and
            # train no adapter at all.
            raise ValueError(
                f"mlx-lm selects layers by their path inside a block (for example 'self_attn.q_proj', "
                f"'mlp.gate_proj'), not by a suffix: {bare} cannot be translated. Give dotted paths, or leave "
                "target_modules unset to adapt every linear layer in each block")
        lines.append("  keys: [" + ", ".join(json.dumps(m) for m in recipe.target_modules) + "]")
    lines += ["lr_schedule:", "  name: linear_schedule"]
    if n_warmup:
        lines += [f"  warmup: {n_warmup}", "  warmup_init: 0.0"]
    lines += [f"  arguments: [{_yaml_number(recipe.learning_rate)}, 0.0, {max(1, updates - n_warmup)}]",
              "optimizer_config:", "  adamw:", f"    weight_decay: {_yaml_number(recipe.weight_decay)}"]
    path = Path(data_dir) / MLX_CONFIG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def mlx_lora_command(recipe: FineTuneRecipe, data_dir: str, adapter_dir: str, n_train: Optional[int] = None,
                     python: str = "python") -> List[str]:
    """The argument list that trains this recipe with Apple's mlx-lm.

    Returns ``[python, "-m", "mlx_lm", "lora", ...]`` for a Mac with
    ``pip install "mlx-lm[train]"``. It writes nothing. ``data_dir`` must hold
    the ``train.jsonl`` from :func:`write_mlx_dataset` and the
    ``lora_config.yaml`` from :func:`write_mlx_config`; the command passes that
    file with ``--config`` because it is the only way to set the rank.
    ``n_train`` is read from ``train.jsonl`` when not given, to turn epochs
    into ``--iters``.

    Differences from the PyTorch path that the flags cannot remove: mlx-lm has
    no gradient clipping, and ``--mask-prompt`` treats the final message as
    the completion, which matches ``train_on_responses_only``.

    This command has not been run: there is no Apple hardware here. The unit
    tests check the argument list against mlx-lm's documented flags only.
    """
    method = recipe.method.lower()
    if method not in _MLX_FINE_TUNE_TYPE:
        raise ValueError(f"mlx-lm cannot train {recipe.method!r}; it supports {sorted(_MLX_FINE_TUNE_TYPE)}")
    if n_train is None:
        train_file = Path(data_dir) / "train.jsonl"
        if not train_file.exists():
            raise FileNotFoundError(f"{train_file} not found: call write_mlx_dataset first, or pass n_train")
        n_train = sum(1 for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip())
    argv = [
        python, "-m", "mlx_lm", "lora",
        "--model", recipe.base_model,
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(adapter_dir),
        "--fine-tune-type", _MLX_FINE_TUNE_TYPE[method],
        # mlx-lm adapts only the last 16 blocks by default; peft adapts all.
        "--num-layers", "-1",
        "--optimizer", "adamw",
        "--batch-size", str(recipe.batch_size),
        "--iters", str(mlx_iterations(recipe, n_train)),
        "--learning-rate", repr(float(recipe.learning_rate)),
        "--grad-accumulation-steps", str(recipe.grad_accumulation),
        "--max-seq-length", str(recipe.max_seq_len),
        "--seed", str(recipe.seed),
        "--config", str(Path(data_dir) / MLX_CONFIG_NAME),
    ]
    if recipe.train_on_responses_only:
        argv.append("--mask-prompt")
    return argv


def prepare_mlx_run(recipe: FineTuneRecipe, records: Sequence[dict], data_dir: str, adapter_dir: str,
                    valid_records: Optional[Sequence[dict]] = None) -> List[str]:
    """Write the dataset and config for mlx-lm and return the command to run on the Mac."""
    written = write_mlx_dataset(records, data_dir, valid_records, seed=recipe.seed)
    write_mlx_config(recipe, data_dir, written["n_train"])
    return mlx_lora_command(recipe, data_dir, adapter_dir, n_train=written["n_train"])
