"""Teachers: sources of labels that are too slow or costly to serve directly.

A teacher turns unlabelled text into (soft) labels. It can be a local LLM, a
hosted LLM behind an OpenAI-compatible endpoint, a larger fine-tuned model, a
zero-shot model, or any Python callable. The factory only needs ``label``.

Soft labels matter: a teacher that reports "60% complaint, 40% query" passes
more information to the student than one that reports "complaint".
"""

from __future__ import annotations

import json
import os
import string
import time
import urllib.request
from typing import Callable, List, Optional, Sequence

import numpy as np


class Teacher:
    """Base class. ``label`` returns an ``(n, n_classes)`` probability matrix."""

    name = "teacher"

    def __init__(self, labels: Sequence[str]):
        self.labels = list(labels)
        self.calls = 0
        self.seconds = 0.0

    def _label(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def label(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        start = time.perf_counter()
        probs = np.asarray(self._label(texts), dtype=np.float64)
        self.seconds += time.perf_counter() - start
        self.calls += len(texts)
        if probs.shape != (len(texts), len(self.labels)):
            raise ValueError(f"teacher returned shape {probs.shape}, expected {(len(texts), len(self.labels))}")
        return probs

    def stats(self) -> dict:
        return {"teacher": self.name, "items_labelled": self.calls, "seconds": round(self.seconds, 2),
                "seconds_per_item": round(self.seconds / self.calls, 4) if self.calls else None}


class CallableTeacher(Teacher):
    """Wrap any function ``texts -> probabilities`` (or ``-> label indices``)."""

    def __init__(self, fn: Callable[[List[str]], Sequence], labels: Sequence[str], name: str = "callable"):
        super().__init__(labels)
        self.fn, self.name = fn, name

    def _label(self, texts):
        out = np.asarray(self.fn(texts))
        if out.ndim == 1:  # hard labels -> one-hot
            onehot = np.zeros((len(texts), len(self.labels)))
            onehot[np.arange(len(texts)), out.astype(int)] = 1.0
            return onehot
        return out


class StudentTeacher(Teacher):
    """Use an already-trained (usually larger) student as the teacher."""

    def __init__(self, student, name: Optional[str] = None):
        super().__init__(student.labels)
        self.student = student
        self.name = name or f"student:{student.kind}"

    def _label(self, texts):
        return self.student.predict_proba(texts)


class LocalLLMTeacher(Teacher):
    """A local causal LM used as a classifier by reading answer-letter logits.

    One forward pass per text, no generation: the prompt lists the options as
    A, B, C... and the probabilities are the softmax over the logits of those
    letters at the answer position. That gives soft labels for free and makes
    a 1-3B model usable as a teacher on a CPU.

    For label sets larger than ``max_options`` pass a ``shortlister`` (any
    object with ``predict_proba``, e.g. a zero-shot or few-shot student). The
    LLM then chooses among each text's top candidates only. This keeps prompts
    short and is how a small-context teacher handles dozens of classes.
    """

    def __init__(self, model: str, labels: Sequence[str], descriptions: Optional[Sequence[str]] = None,
                 instruction: str = "Classify the text into exactly one of the options.",
                 device: Optional[str] = None, batch_size: int = 8, max_text_tokens: int = 256,
                 max_options: int = 12, shortlister=None, dtype: Optional[str] = None):
        super().__init__(labels)
        self.name = f"llm:{model}"
        self.model_name, self.instruction = model, instruction
        self.descriptions = list(descriptions) if descriptions else [l.replace("_", " ") for l in self.labels]
        self.device, self.batch_size, self.max_text_tokens = device, batch_size, max_text_tokens
        self.max_options = min(max_options, 26)
        self.shortlister = shortlister
        self.dtype = dtype
        if len(self.labels) > self.max_options and shortlister is None:
            raise ValueError(
                f"{len(self.labels)} labels exceed max_options={self.max_options}; pass a shortlister "
                "(e.g. a ZeroShotStudent) so the LLM chooses among each text's top candidates")
        self._model = self._tok = self._letter_ids = None
        self.resolved_dtype = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .students import pick_device

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._tok.padding_side = "left"
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        device = pick_device(self.device)
        if self.dtype:
            dtype = getattr(torch, self.dtype)
        elif device == "cpu":
            # bf16 is only a win on CPUs with AMX-BF16 (measured 2.5x faster here, 95% label agreement
            # with fp32); elsewhere it is emulated and slower.
            from .hardware import parse_cpu_flags, _read

            dtype = torch.bfloat16 if "amx_bf16" in parse_cpu_flags(_read("/proc/cpuinfo")) else torch.float32
        else:
            dtype = torch.bfloat16
        self.resolved_dtype = str(dtype).replace("torch.", "")
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype).to(device).eval()
        ids = [self._tok.encode(ch, add_special_tokens=False) for ch in string.ascii_uppercase[: self.max_options]]
        if any(len(i) != 1 for i in ids) or len({i[0] for i in ids}) != len(ids):
            raise ValueError(f"tokenizer of {self.model_name} does not map option letters to distinct single tokens")
        self._letter_ids = [i[0] for i in ids]

    def _prompt(self, text: str, option_idx: Sequence[int]) -> str:
        tok = self._tok
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) > self.max_text_tokens:
            text = tok.decode(ids[: self.max_text_tokens]) + " ..."
        options = "\n".join(f"{string.ascii_uppercase[k]}. {self.descriptions[j]}" for k, j in enumerate(option_idx))
        user = f"{self.instruction}\n\nText: {text}\n\nOptions:\n{options}\n\nAnswer with the letter only."
        messages = [{"role": "user", "content": user}]
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return user + "\nAnswer:"

    def _label(self, texts):
        import torch

        self._load()
        n, c = len(texts), len(self.labels)
        if c <= self.max_options:
            candidates = [list(range(c))] * n
        else:
            short = np.asarray(self.shortlister.predict_proba(texts))
            candidates = [list(np.argsort(-row)[: self.max_options]) for row in short]
        out = np.zeros((n, c))
        device = next(self._model.parameters()).device
        # Longest prompts first keeps padding (and wasted compute) low per batch.
        prompts = [self._prompt(t, cand) for t, cand in zip(texts, candidates)]
        order = sorted(range(n), key=lambda i: -len(prompts[i]))
        with torch.inference_mode():
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                enc = self._tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
                logits = self._model(**enc).logits[:, -1, :].float()
                for row, i in enumerate(idx):
                    cand = candidates[i]
                    letter_logits = logits[row, self._letter_ids[: len(cand)]]
                    out[i, cand] = torch.softmax(letter_logits, -1).cpu().numpy()
        return out


class ChatCompletionsTeacher(Teacher):
    """A teacher behind an OpenAI-compatible ``/chat/completions`` endpoint.

    Works with local servers that speak this protocol (llama.cpp server,
    Ollama, vLLM, LM Studio) as well as hosted APIs, so a regulated deployment
    can keep labelling on-premises. Returns one-hot labels; answers that do not
    parse to a known label are returned as a uniform row so they can be
    filtered out by confidence.

    The API key is read from the environment variable named by ``api_key_env``
    and never stored.
    """

    def __init__(self, base_url: str, model: str, labels: Sequence[str],
                 descriptions: Optional[Sequence[str]] = None,
                 instruction: str = "Classify the text into exactly one of the options.",
                 api_key_env: str = "FORGELM_TEACHER_API_KEY", timeout: float = 120.0, max_tokens: int = 16):
        super().__init__(labels)
        self.name = f"chat:{model}"
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.instruction = model, instruction
        self.descriptions = list(descriptions) if descriptions else [l.replace("_", " ") for l in self.labels]
        self.api_key_env, self.timeout, self.max_tokens = api_key_env, timeout, max_tokens
        self.unparsed = 0

    def _ask(self, text: str) -> str:
        options = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(self.descriptions))
        prompt = (f"{self.instruction}\n\nText: {text}\n\nOptions:\n{options}\n\n"
                  "Reply with the option number only.")
        body = json.dumps({"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
        return payload["choices"][0]["message"]["content"] or ""

    def _parse(self, answer: str) -> Optional[int]:
        digits = "".join(ch if ch.isdigit() else " " for ch in answer).split()
        if digits and 1 <= int(digits[0]) <= len(self.labels):
            return int(digits[0]) - 1
        lowered = answer.strip().lower()
        for i, (label, desc) in enumerate(zip(self.labels, self.descriptions)):
            if lowered in (label.lower(), desc.lower()):
                return i
        return None

    def _label(self, texts):
        c = len(self.labels)
        out = np.full((len(texts), c), 1.0 / c)
        for i, text in enumerate(texts):
            choice = self._parse(self._ask(text))
            if choice is None:
                self.unparsed += 1
                continue
            out[i] = 0.0
            out[i, choice] = 1.0
        return out

    def stats(self):
        return dict(super().stats(), unparsed=self.unparsed)
