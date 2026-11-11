"""Shared helpers for the reproducible experiments.

Run every experiment from the repository root, e.g.

    python experiments/e1_banking77_ladder.py

Datasets and models are fetched from the Hugging Face Hub on first use.
Result JSON goes to experiments/results/ (committed); large intermediates go
to experiments/artifacts/ (git-ignored).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "experiments" / "results"
ARTIFACTS = ROOT / "experiments" / "artifacts"
RESULTS.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def hub_file(repo: str, filename: str, repo_type: str = "dataset") -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type=repo_type)


def load_banking77():
    """Banking77 (PolyAI, CC BY 4.0): 10,003 train / 3,080 test, 77 intents."""
    train = pd.read_parquet(hub_file("PolyAI/banking77", "data/train-00000-of-00001.parquet"))
    test = pd.read_parquet(hub_file("PolyAI/banking77", "data/test-00000-of-00001.parquet"))
    readme = Path(hub_file("PolyAI/banking77", "README.md")).read_text()
    names = re.findall(r"'\d+':\s*(\S+)", readme)
    assert len(names) == 77, len(names)
    return train, test, names


def load_unfair_tos():
    """LexGLUE UNFAIR-ToS: terms-of-service sentences, 8 unfair-term types (multi-label).

    Returned as a binary task: does the sentence contain a potentially unfair term?
    """
    out = []
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(hub_file("coastalcph/lex_glue", f"unfair_tos/{split}-00000-of-00001.parquet"))
        df["label"] = df["labels"].apply(lambda x: int(len(x) > 0))
        out.append(df[["text", "label"]].reset_index(drop=True))
    return out


def stratified_sample(labels: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    idx = []
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        idx.extend(rng.choice(members, size=min(per_class, len(members)), replace=False))
    return np.array(sorted(idx))


# Display names of the E1 students. report.py joins E1's rows to the E1/E1c serving rows on these strings.
STUDENT_LABELS = {"tfidf": "tfidf + LR", "static": "static potion-32M + LR", "encoder": "frozen bge-small + LR",
                  "finetuned": "fine-tuned MiniLM-L6 (22M)", "modernbert": "fine-tuned ModernBERT-base (149M)"}


def banking77_anchor(labels: np.ndarray) -> np.ndarray:
    """The Banking77 calibration anchor: 5 training texts per class, seed 1234.

    E1 saves anchor probabilities for this draw, and E5 fits temperatures on them against these labels.
    """
    return stratified_sample(labels, 5, np.random.default_rng(1234))


def e2_draws(n_train: int, n_test: int, pool: int = 2000, anchor: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """E2's draws, seed 7: the unlabelled training pool first, then the test anchor, both sorted.

    E8 reuses E2's cached teacher labels, so it must get the same anchor items.
    """
    rng = np.random.default_rng(7)
    pool_idx = np.sort(rng.choice(n_train, size=pool, replace=False))
    anchor_idx = np.sort(rng.choice(n_test, size=anchor, replace=False))
    return pool_idx, anchor_idx


def environment() -> dict:
    """Hardware and package versions, stored with every result file."""
    from forgelm.hardware import probe

    hw = probe()
    keep = ("torch", "transformers", "sentence_transformers", "model2vec", "sklearn", "onnxruntime", "lightgbm",
            "interpret", "tabicl", "peft")
    return {"cpu": hw.cpu_model, "cores": hw.logical_cores, "ram_gb": hw.ram_total_gb,
            "device_class": hw.device_class, "cpu_flags": hw.cpu_flags, "python": hw.python,
            "packages": {k: v for k, v in hw.packages.items() if k in keep},
            "note": "shared 4-core cloud VM; latency figures are indicative, not a controlled benchmark"}


def write_result(name: str, payload: dict, smoke: bool = False, warning: str | None = None) -> Path:
    payload = dict(payload, environment=environment(), generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if warning:
        payload["warning"] = warning
    # Smoke output never goes to RESULTS: the evidence registry treats results/ as measured.
    path = (ARTIFACTS / f"{name}_smoke.json") if smoke else (RESULTS / f"{name}.json")
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {path}")
    return path


def rounded(d: dict, nd: int = 4) -> dict:
    return {k: (round(v, nd) if isinstance(v, float) else v) for k, v in d.items()}
