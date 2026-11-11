"""Student models: small text classifiers that need no LLM at inference.

Every student has the same four-method surface (``fit``, ``predict_proba``,
``save``, ``load``) so the rest of the factory (distillation, calibration,
benchmarking, the eval contract) does not care which one it is holding.

``fit`` accepts either integer labels or an ``(n, n_classes)`` matrix of soft
labels. Soft labels are how a teacher's uncertainty reaches the student.

The ladder, cheapest first:

    zero_shot    label descriptions only, no training data
    tfidf        word + character n-grams, logistic regression
    static       static token embeddings (Model2Vec), logistic regression
    encoder      frozen sentence encoder, logistic regression
    finetuned    encoder fine-tuned end to end

The first four keep a plain logistic-regression head that is stored as numpy
arrays, so serving needs numpy plus the embedder and nothing else.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .metrics import EPS


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _hard_and_weights(y, n_classes: int, min_weight: float = 0.02):
    """Turn hard or soft labels into (row_index, label, weight) triples.

    scikit-learn's LogisticRegression has no soft-label mode. Replicating each
    row once per class, weighted by the teacher's probability, optimises the
    soft-label cross-entropy. To keep that affordable with many classes, each
    row keeps only the classes with probability >= ``min_weight`` (always
    including its most likely class) and the kept weights are renormalised to
    sum to one, so every row counts once and none is dropped. This is exact
    when a teacher puts all its mass on a few options (an LLM choosing among a
    shortlist) and a top-k approximation for dense distributions.
    """
    y = np.asarray(y)
    if y.ndim == 1:
        return np.arange(len(y)), y.astype(np.int64), np.ones(len(y))
    if y.shape[1] != n_classes:
        raise ValueError(f"soft labels have {y.shape[1]} columns, expected {n_classes}")
    keep = y >= min_weight
    keep[np.arange(len(y)), y.argmax(1)] = True
    kept = np.where(keep, y, 0.0).astype(np.float64)
    kept /= kept.sum(1, keepdims=True)
    rows, cols = np.nonzero(keep)
    return rows, cols.astype(np.int64), kept[rows, cols]


class LinearHead:
    """Multinomial logistic regression, stored and served as plain numpy."""

    def __init__(self, C: float = 10.0, max_iter: int = 2000):
        self.C, self.max_iter = C, max_iter
        self.coef: Optional[np.ndarray] = None
        self.intercept: Optional[np.ndarray] = None

    def fit(self, X, y, n_classes: int) -> "LinearHead":
        from sklearn.linear_model import LogisticRegression

        rows, labels, weights = _hard_and_weights(y, n_classes)
        clf = LogisticRegression(C=self.C, max_iter=self.max_iter)
        clf.fit(X[rows], labels, sample_weight=weights)
        n_features = X.shape[1]
        coef = np.zeros((n_classes, n_features), dtype=np.float32)
        intercept = np.full(n_classes, -30.0, dtype=np.float32)  # unseen classes get ~0 probability
        if len(clf.classes_) == 2 and clf.coef_.shape[0] == 1:
            neg, pos = clf.classes_
            coef[pos], intercept[pos] = clf.coef_[0], clf.intercept_[0]
            coef[neg], intercept[neg] = 0.0, 0.0
        else:
            coef[clf.classes_] = clf.coef_
            intercept[clf.classes_] = clf.intercept_
        self.coef, self.intercept = coef, intercept
        return self

    def predict_proba(self, X) -> np.ndarray:
        z = X @ self.coef.T
        z = np.asarray(z) + self.intercept
        return _softmax(z.astype(np.float64))

    def state(self) -> Dict[str, np.ndarray]:
        return {"coef": self.coef, "intercept": self.intercept}

    @classmethod
    def from_state(cls, state) -> "LinearHead":
        head = cls()
        head.coef, head.intercept = state["coef"], state["intercept"]
        return head


class Student:
    """Common surface. Subclasses set ``kind`` and implement the four methods."""

    kind = "abstract"
    #: packages needed to *serve* this student (training may need more)
    serve_requires: List[str] = ["numpy"]

    def __init__(self):
        self.labels: List[str] = []
        self.temperature: float = 1.0

    @property
    def n_classes(self) -> int:
        return len(self.labels)

    def fit(self, texts: Sequence[str], y, labels: Sequence[str]) -> "Student":
        raise NotImplementedError

    def _raw_proba(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        probs = self._raw_proba(list(texts))
        if abs(self.temperature - 1.0) > 1e-9:
            from .metrics import apply_temperature

            probs = apply_temperature(probs, self.temperature)
        return probs

    def predict(self, texts: Sequence[str]) -> List[str]:
        return [self.labels[i] for i in self.predict_proba(texts).argmax(1)]

    def calibrate(self, texts: Sequence[str], y) -> float:
        """Fit the confidence temperature on held-out, human-labelled data."""
        from .metrics import fit_temperature

        self.temperature = 1.0
        self.temperature = fit_temperature(y, self._raw_proba(list(texts)))
        return self.temperature

    # persistence ---------------------------------------------------------
    def _meta(self) -> dict:
        return {"kind": self.kind, "labels": self.labels, "temperature": self.temperature}

    def save(self, directory: str) -> str:
        raise NotImplementedError

    @classmethod
    def load(cls, directory: str) -> "Student":
        raise NotImplementedError

    def size_bytes(self, directory: str) -> int:
        return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())


# --------------------------------------------------------------------------
# Rung 1: TF-IDF
# --------------------------------------------------------------------------

class TfidfStudent(Student):
    """Word and character n-gram TF-IDF with logistic regression.

    The baseline every other student has to beat. It trains in seconds on a
    CPU, is fully inspectable (each prediction decomposes into n-gram
    weights), and is often embarrassingly competitive once there are a few
    hundred labels per class.
    """

    kind = "tfidf"
    serve_requires = ["numpy", "scikit-learn"]

    def __init__(self, C: float = 20.0, max_features: int = 200_000):
        super().__init__()
        self.C, self.max_features = C, max_features
        self.vectorizer = None
        self.head: Optional[LinearHead] = None

    def fit(self, texts, y, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        self.labels = list(labels)
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1,
                                     max_features=self.max_features)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                                     min_df=1, max_features=self.max_features)),
        ])
        X = self.vectorizer.fit_transform(list(texts))
        self.head = LinearHead(C=self.C).fit(X, y, self.n_classes)
        return self

    def _raw_proba(self, texts):
        return self.head.predict_proba(self.vectorizer.transform(texts))

    def top_features(self, label: str, k: int = 15) -> List[tuple]:
        """The n-grams that most push a text toward ``label``."""
        names = self.vectorizer.get_feature_names_out()
        w = self.head.coef[self.labels.index(label)]
        return [(str(names[i]), float(w[i])) for i in np.argsort(-w)[:k]]

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        np.savez_compressed(os.path.join(directory, "head.npz"), **self.head.state())
        Path(directory, "student.json").write_text(json.dumps(self._meta(), indent=2))
        return directory

    @classmethod
    def load(cls, directory):
        meta = json.loads(Path(directory, "student.json").read_text())
        self = cls()
        self.labels, self.temperature = meta["labels"], meta["temperature"]
        # The vectorizer is a scikit-learn pickle: load only artefacts you produced.
        with open(os.path.join(directory, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)
        self.head = LinearHead.from_state(np.load(os.path.join(directory, "head.npz")))
        return self


# --------------------------------------------------------------------------
# Embedders shared by rungs 0, 2 and 3
# --------------------------------------------------------------------------

class Embedder:
    """Text -> L2-normalised vectors. ``backend`` selects the implementation."""

    def __init__(self, model: str, backend: str = "auto", device: Optional[str] = None,
                 batch_size: int = 64, max_seq_length: Optional[int] = None):
        self.model_name, self.batch_size, self.device = model, batch_size, device
        self.max_seq_length = max_seq_length
        if backend == "auto":
            lowered = model.lower()
            backend = "static" if ("potion" in lowered or "model2vec" in lowered or "m2v" in lowered) else "sentence"
        self.backend = backend
        self._impl = None

    def _load(self):
        if self._impl is not None:
            return self._impl
        if self.backend == "static":
            from model2vec import StaticModel

            self._impl = StaticModel.from_pretrained(self.model_name)
        elif self.backend == "sentence":
            from sentence_transformers import SentenceTransformer

            self._impl = SentenceTransformer(self.model_name, device=self.device)
            if self.max_seq_length:
                self._impl.max_seq_length = self.max_seq_length
        else:
            raise ValueError(f"unknown embedder backend {self.backend!r}")
        return self._impl

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        impl = self._load()
        texts = list(texts)
        if self.backend == "static":
            vecs = np.asarray(impl.encode(texts), dtype=np.float32)
        else:
            vecs = impl.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                               show_progress_bar=False).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, EPS)


class EmbeddingStudent(Student):
    """Frozen embeddings with a logistic-regression head.

    With a static embedder (Model2Vec ``potion-*``) the whole model is a
    lookup table and a matrix multiply: no torch at inference and tens of
    thousands of texts per second on one CPU core. With a sentence encoder it
    is slower and usually more accurate when labels are scarce.
    """

    kind = "embedding"

    def __init__(self, model: str = "minishlab/potion-base-32M", backend: str = "auto", C: float = 10.0,
                 device: Optional[str] = None, max_seq_length: Optional[int] = None):
        super().__init__()
        self.embedder = Embedder(model, backend, device, max_seq_length=max_seq_length)
        self.C = C
        self.head: Optional[LinearHead] = None

    @property
    def serve_requires(self):  # type: ignore[override]
        return ["numpy", "model2vec"] if self.embedder.backend == "static" else ["numpy", "sentence-transformers"]

    def fit(self, texts, y, labels, embeddings: Optional[np.ndarray] = None):
        self.labels = list(labels)
        X = embeddings if embeddings is not None else self.embedder.encode(texts)
        self.head = LinearHead(C=self.C).fit(X, y, self.n_classes)
        return self

    def proba_from_embeddings(self, X: np.ndarray) -> np.ndarray:
        return self.head.predict_proba(X)

    def _raw_proba(self, texts):
        return self.head.predict_proba(self.embedder.encode(texts))

    def _meta(self):
        return dict(super()._meta(), model=self.embedder.model_name, backend=self.embedder.backend, C=self.C,
                    max_seq_length=self.embedder.max_seq_length)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        np.savez_compressed(os.path.join(directory, "head.npz"), **self.head.state())
        Path(directory, "student.json").write_text(json.dumps(self._meta(), indent=2))
        return directory

    @classmethod
    def load(cls, directory):
        meta = json.loads(Path(directory, "student.json").read_text())
        self = cls(meta["model"], meta["backend"], meta.get("C", 10.0), max_seq_length=meta.get("max_seq_length"))
        self.labels, self.temperature = meta["labels"], meta["temperature"]
        self.head = LinearHead.from_state(np.load(os.path.join(directory, "head.npz")))
        return self


class ZeroShotStudent(Student):
    """No training data: classify by similarity to a description of each label.

    Useful as a first look and as a cheap teacher. Expect poor calibration and
    weak accuracy when there are many fine-grained classes; measure before
    trusting it.
    """

    kind = "zero_shot"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", backend: str = "auto", scale: float = 20.0,
                 device: Optional[str] = None):
        super().__init__()
        self.embedder = Embedder(model, backend, device)
        self.scale = scale
        self.label_vectors: Optional[np.ndarray] = None
        self.descriptions: List[str] = []

    @property
    def serve_requires(self):  # type: ignore[override]
        return ["numpy", "model2vec"] if self.embedder.backend == "static" else ["numpy", "sentence-transformers"]

    def fit(self, texts=None, y=None, labels: Sequence[str] = (), descriptions: Optional[Sequence[str]] = None):
        self.labels = list(labels)
        self.descriptions = list(descriptions) if descriptions else [l.replace("_", " ") for l in self.labels]
        self.label_vectors = self.embedder.encode(self.descriptions)
        return self

    def proba_from_embeddings(self, X: np.ndarray) -> np.ndarray:
        return _softmax((X @ self.label_vectors.T).astype(np.float64) * self.scale)

    def _raw_proba(self, texts):
        return self.proba_from_embeddings(self.embedder.encode(texts))

    def _meta(self):
        return dict(super()._meta(), model=self.embedder.model_name, backend=self.embedder.backend,
                    scale=self.scale, descriptions=self.descriptions)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        np.savez_compressed(os.path.join(directory, "labels.npz"), label_vectors=self.label_vectors)
        Path(directory, "student.json").write_text(json.dumps(self._meta(), indent=2))
        return directory

    @classmethod
    def load(cls, directory):
        meta = json.loads(Path(directory, "student.json").read_text())
        self = cls(meta["model"], meta["backend"], meta["scale"])
        self.labels, self.temperature, self.descriptions = meta["labels"], meta["temperature"], meta["descriptions"]
        self.label_vectors = np.load(os.path.join(directory, "labels.npz"))["label_vectors"]
        return self


# --------------------------------------------------------------------------
# Rung 4: fine-tuned encoder
# --------------------------------------------------------------------------

def pick_device(preferred: Optional[str] = None) -> str:
    """cuda (NVIDIA or AMD ROCm) > xpu (Intel) > mps (Apple) > cpu."""
    if preferred:
        return preferred
    import torch

    if torch.cuda.is_available():
        return "cuda"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FineTunedStudent(Student):
    """An encoder fine-tuned end to end for sequence classification.

    The most accurate rung once there are roughly 50+ labels per class, and
    the only one here that needs torch at inference (unless exported to ONNX
    with :mod:`forgelm.compress`). Soft labels train with KL divergence.

    The training loop is plain PyTorch on purpose: it runs unchanged on CUDA,
    ROCm, XPU, MPS and CPU, with no bitsandbytes / xformers / flash-attn
    dependency to break on non-NVIDIA hardware.
    """

    kind = "finetuned"
    serve_requires = ["numpy", "torch", "transformers"]

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", epochs: int = 5, lr: float = 5e-5,
                 batch_size: int = 32, max_length: int = 128, weight_decay: float = 0.01, warmup: float = 0.1,
                 device: Optional[str] = None, seed: int = 0,
                 on_progress: Optional[Callable[[int, int, float], None]] = None):
        super().__init__()
        self.model_name, self.epochs, self.lr = model, epochs, lr
        self.batch_size, self.max_length = batch_size, max_length
        self.weight_decay, self.warmup, self.device, self.seed = weight_decay, warmup, device, seed
        self.on_progress = on_progress
        self.model = None
        self.tokenizer = None
        self.history: List[float] = []

    def _load_base(self, n_labels: int):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=n_labels)

    def fit(self, texts, y, labels):
        import torch

        self.labels = list(labels)
        torch.manual_seed(self.seed)
        self._load_base(self.n_classes)
        device = pick_device(self.device)
        model = self.model.to(device)
        texts = list(texts)
        y_arr = np.asarray(y)
        soft = y_arr.ndim == 2
        target = torch.tensor(y_arr, dtype=torch.float32 if soft else torch.long)

        decay, no_decay = [], []
        for name, param in model.named_parameters():
            (no_decay if param.ndim < 2 or "norm" in name.lower() else decay).append(param)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": self.weight_decay},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=self.lr)
        steps_per_epoch = (len(texts) + self.batch_size - 1) // self.batch_size
        total = max(1, steps_per_epoch * self.epochs)
        warm = max(1, int(self.warmup * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))

        rng = np.random.default_rng(self.seed)
        step = 0
        model.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(texts))
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                enc = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt").to(device)
                logits = model(**enc).logits
                tgt = target[idx].to(device)
                if soft:
                    loss = -(tgt * torch.log_softmax(logits, -1)).sum(-1).mean()
                else:
                    loss = torch.nn.functional.cross_entropy(logits, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                self.history.append(float(loss.detach()))
                if self.on_progress:
                    self.on_progress(step, total, self.history[-1])
        model.eval()
        return self

    def _raw_proba(self, texts):
        import torch

        device = next(self.model.parameters()).device
        out = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(texts), 64):
                enc = self.tokenizer(texts[start:start + 64], padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt").to(device)
                out.append(torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def _meta(self):
        return dict(super()._meta(), model=self.model_name, max_length=self.max_length)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.model.config.id2label = dict(enumerate(self.labels))
        self.model.config.label2id = {l: i for i, l in enumerate(self.labels)}
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        Path(directory, "student.json").write_text(json.dumps(self._meta(), indent=2))
        return directory

    @classmethod
    def load(cls, directory, device: Optional[str] = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta = json.loads(Path(directory, "student.json").read_text())
        self = cls(meta["model"], max_length=meta["max_length"], device=device)
        self.labels, self.temperature = meta["labels"], meta["temperature"]
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory).to(pick_device(device)).eval()
        return self


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

STUDENT_KINDS = {
    "zero_shot": ZeroShotStudent,
    "tfidf": TfidfStudent,
    "embedding": EmbeddingStudent,
    "finetuned": FineTunedStudent,
}


def make_student(kind: str, **kwargs) -> Student:
    """Build a student by ladder name.

    ``static`` and ``encoder`` are the two flavours of ``embedding`` and pick a
    sensible default model each.
    """
    if kind == "static":
        kwargs.setdefault("model", "minishlab/potion-base-32M")
        return EmbeddingStudent(backend="static", **kwargs)
    if kind == "encoder":
        kwargs.setdefault("model", "BAAI/bge-small-en-v1.5")
        return EmbeddingStudent(backend="sentence", **kwargs)
    if kind not in STUDENT_KINDS:
        raise KeyError(f"unknown student kind {kind!r}; choose from {sorted(STUDENT_KINDS) + ['static', 'encoder']}")
    return STUDENT_KINDS[kind](**kwargs)


def load_student(directory: str) -> Student:
    meta = json.loads(Path(directory, "student.json").read_text())
    return STUDENT_KINDS[meta["kind"]].load(directory)
