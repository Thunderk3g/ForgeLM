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


def write_result(name: str, payload: dict) -> Path:
    payload = dict(payload, environment=environment(), generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {path}")
    return path


def rounded(d: dict, nd: int = 4) -> dict:
    return {k: (round(v, nd) if isinstance(v, float) else v) for k, v in d.items()}
