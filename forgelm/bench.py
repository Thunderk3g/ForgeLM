"""Deployment cost measurement: latency, throughput, size, memory.

Numbers are only meaningful for the machine they were measured on, so every
result carries the hardware it came from (see :mod:`forgelm.provenance`).
"""

from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np


def directory_size_mb(path: str) -> float:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far (high-water mark)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def current_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return float("nan")


def benchmark(predict: Callable[[Sequence[str]], object], texts: Sequence[str], batch_size: int = 32,
              single_item_runs: int = 100, warmup: int = 3) -> Dict[str, float]:
    """Single-item latency (p50/p95) and batched throughput for ``predict``.

    Single-item latency is what an interactive caller sees; batched throughput
    is what an overnight scoring job gets.
    """
    texts = list(texts)
    for i in range(warmup):
        predict(texts[i:i + 1])
    lat = []
    for i in range(min(single_item_runs, len(texts))):
        start = time.perf_counter()
        predict(texts[i:i + 1])
        lat.append((time.perf_counter() - start) * 1000.0)
    start = time.perf_counter()
    n = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        predict(batch)
        n += len(batch)
    elapsed = time.perf_counter() - start
    return {
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "throughput_per_s": float(n / elapsed) if elapsed > 0 else float("inf"),
        "batch_size": batch_size,
        "n_items": n,
    }
