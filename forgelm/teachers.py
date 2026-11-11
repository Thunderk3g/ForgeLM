"""Teachers: sources of labels that are too slow or costly to serve directly.

A teacher turns unlabelled text into (soft) labels. It can be a local LLM, a
hosted LLM behind an OpenAI-compatible endpoint, a larger fine-tuned model, a
zero-shot model, or any Python callable. The factory only needs ``label``.

Soft labels matter: a teacher that reports "60% complaint, 40% query" passes
more information to the student than one that reports "complaint".
"""

from __future__ import annotations

import hashlib
import json
import os
import string
import time
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
    LLM then chooses among each text's top candidates only, presented in a
    shuffled order. This keeps prompts short and is how a small-context
    teacher handles dozens of classes.

    Few-shot prompting with retrieved demonstrations (off by default): pass
    the few labelled examples you have as ``demonstrations`` (``(text,
    label_index)`` pairs), ``k_demos`` > 0, and a ``demo_embedder`` (any
    object whose ``encode(texts)`` returns L2-normalised vectors, e.g.
    :class:`forgelm.students.Embedder`). Each query is then shown its
    ``k_demos`` most similar demonstrations by cosine similarity, each as its
    text followed by the description of its label. Descriptions, not letters,
    because the letters are reassigned for every query. They are ordered
    from least to most similar, so the closest example sits right before the
    question. A demonstration whose text is identical to the query is never
    shown for it, so labelling a set that contains the demonstrations does
    not hand the model its own answer. With a shortlister, the labels of the
    shown demonstrations are always among the options: an example the model
    is told is similar should not point to an answer it cannot give.
    """

    def __init__(self, model: str, labels: Sequence[str], descriptions: Optional[Sequence[str]] = None,
                 instruction: str = "Classify the text into exactly one of the options.",
                 device: Optional[str] = None, batch_size: int = 8, max_text_tokens: int = 256,
                 max_options: int = 12, shortlister=None, dtype: Optional[str] = None,
                 demonstrations: Optional[Sequence[Tuple[str, int]]] = None, k_demos: int = 0,
                 demo_embedder=None):
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
        self.demonstrations = [(str(t), int(y)) for t, y in (demonstrations or [])]
        self.k_demos = int(k_demos)
        self.demo_embedder = demo_embedder
        if self.k_demos < 0:
            raise ValueError("k_demos must be >= 0")
        if self.k_demos > 0:
            if not self.demonstrations:
                raise ValueError("k_demos > 0 needs demonstrations: (text, label_index) pairs")
            if demo_embedder is None:
                raise ValueError("k_demos > 0 needs a demo_embedder (e.g. forgelm.students.Embedder) to retrieve "
                                 "the nearest demonstrations")
            bad = sorted({y for _, y in self.demonstrations if not 0 <= y < len(self.labels)})
            if bad:
                raise ValueError(f"demonstration label indices out of range 0..{len(self.labels) - 1}: {bad}")
        self._demo_vectors: Optional[np.ndarray] = None
        # How many demonstrations each labelled item was shown (count -> items), for stats().
        self.demo_counts: Dict[int, int] = {}
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

    def _truncate(self, text: str) -> str:
        ids = self._tok.encode(text, add_special_tokens=False)
        if len(ids) > self.max_text_tokens:
            return self._tok.decode(ids[: self.max_text_tokens]) + " ..."
        return text

    def _prompt(self, text: str, option_idx: Sequence[int], demos: Sequence[int] = ()) -> str:
        """The prompt for one text. ``demos`` are indices into ``self.demonstrations``, in display order.

        Without demonstrations the prompt is exactly what this method produced
        before few-shot support existed, so zero-shot results stay comparable.
        """
        tok = self._tok
        text = self._truncate(text)
        options = "\n".join(f"{string.ascii_uppercase[k]}. {self.descriptions[j]}" for k, j in enumerate(option_idx))
        if demos:
            shown = "\n\n".join(f"Text: {self._truncate(self.demonstrations[d][0])}\n"
                                f"Correct answer: {self.descriptions[self.demonstrations[d][1]]}" for d in demos)
            user = (f"{self.instruction}\n\nLabelled examples:\n\n{shown}\n\nNow classify this text.\n\n"
                    f"Text: {text}\n\nOptions:\n{options}\n\nAnswer with the letter only.")
        else:
            user = f"{self.instruction}\n\nText: {text}\n\nOptions:\n{options}\n\nAnswer with the letter only."
        messages = [{"role": "user", "content": user}]
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return user + "\nAnswer:"

    def _select_demos(self, texts: Sequence[str]) -> List[List[int]]:
        """For each text, the indices of its ``k_demos`` nearest demonstrations, least similar first."""
        if self.k_demos <= 0:
            return [[] for _ in texts]
        if self._demo_vectors is None:  # embed the demonstrations once, on first use
            self._demo_vectors = np.asarray(self.demo_embedder.encode([t for t, _ in self.demonstrations]),
                                            dtype=np.float64)
        queries = np.asarray(self.demo_embedder.encode(list(texts)), dtype=np.float64)
        chosen_all = []
        for text, sims in zip(texts, queries @ self._demo_vectors.T):
            chosen = []
            for d in np.argsort(-sims, kind="stable"):  # stable: ties resolve the same way on every run
                if self.demonstrations[d][0] == text:
                    continue  # never show a text its own label
                chosen.append(int(d))
                if len(chosen) == self.k_demos:
                    break
            chosen_all.append(chosen[::-1])
        return chosen_all

    def _candidates(self, texts: Sequence[str], demos: Optional[Sequence[Sequence[int]]] = None) -> List[List[int]]:
        """The label indices offered to the LLM for each text, in the order they are lettered A, B, C..."""
        n, c = len(texts), len(self.labels)
        if c <= self.max_options:
            return [list(range(c)) for _ in range(n)]
        demos = demos if demos is not None else [[] for _ in range(n)]
        short = np.asarray(self.shortlister.predict_proba(texts))
        candidates = []
        for text, row, shown in zip(texts, short, demos):
            ranked = np.argsort(-row)
            top = ranked[: self.max_options]
            # The labels of the shown demonstrations (most similar first) must be options. Fill the rest
            # with the shortlister's best, then put the set back in shortlister order before shuffling, so
            # that when the demonstrations' labels were in its top choices anyway nothing changes.
            required: List[int] = []
            for d in reversed(list(shown)):
                lab = self.demonstrations[d][1]
                if lab not in required:
                    required.append(lab)
            required = required[: self.max_options]
            if any(lab not in top for lab in required):
                fill = [int(j) for j in ranked if int(j) not in required][: self.max_options - len(required)]
                keep = set(required) | set(fill)
                top = np.array([int(j) for j in ranked if int(j) in keep], dtype=ranked.dtype)
            # Shuffle (deterministically per text) so option A is not always the shortlister's
            # first choice; otherwise the LLM's preference for early options would be
            # indistinguishable from agreement with the shortlister.
            seed = int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "little")
            candidates.append([int(j) for j in np.random.default_rng(seed).permutation(top)])
        return candidates

    def _label(self, texts):
        import torch

        self._load()
        n, c = len(texts), len(self.labels)
        demos = self._select_demos(texts)
        candidates = self._candidates(texts, demos)
        for shown in demos:
            self.demo_counts[len(shown)] = self.demo_counts.get(len(shown), 0) + 1
        out = np.zeros((n, c))
        device = next(self._model.parameters()).device
        # Longest prompts first keeps padding (and wasted compute) low per batch.
        prompts = [self._prompt(t, cand, shown) for t, cand, shown in zip(texts, candidates, demos)]
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

    def stats(self):
        items = sum(self.demo_counts.values())
        shown = sum(k * v for k, v in self.demo_counts.items())
        return dict(super().stats(), k_demos=self.k_demos,
                    demonstrations_per_item=round(shown / items, 3) if items else None,
                    demonstrations_histogram={int(k): v for k, v in sorted(self.demo_counts.items())})


class ChatCompletionsTeacher(Teacher):
    """A teacher behind an OpenAI-compatible ``/chat/completions`` endpoint.

    Works with local servers that speak this protocol (llama.cpp server,
    Ollama, vLLM, LM Studio) as well as hosted APIs, so a regulated deployment
    can keep labelling on-premises. Returns one-hot labels; answers that do not
    parse to a known label are returned as a uniform row, which
    :func:`forgelm.distill.distill` drops before training and counts in its
    report.

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
