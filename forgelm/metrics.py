"""Evaluation: quality, calibration, abstention and cascades.

Accuracy alone is the wrong acceptance test for compliance work. A classifier
that is right 93% of the time but cannot say *which* 7% it is unsure about
forces a human to re-check everything. These metrics measure whether the
confidence is trustworthy and how much work can safely be automated.

Only numpy is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

EPS = 1e-12


def _as_probs(probs) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"expected (n_samples, n_classes) probabilities, got shape {p.shape}")
    return p


def classification_report(y_true: Sequence[int], probs, n_classes: Optional[int] = None) -> Dict[str, float]:
    """Accuracy, macro precision / recall / F1, and calibration in one dict."""
    p = _as_probs(probs)
    y = np.asarray(y_true, dtype=np.int64)
    k = n_classes or p.shape[1]
    pred = p.argmax(1)
    precisions, recalls, f1s = [], [], []
    for c in range(k):
        tp = float(np.sum((pred == c) & (y == c)))
        fp = float(np.sum((pred == c) & (y != c)))
        fn = float(np.sum((pred != c) & (y == c)))
        if tp + fn == 0:  # class absent from this evaluation set
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "ece": expected_calibration_error(y, p),
        "brier": brier_score(y, p),
        "nll": negative_log_likelihood(y, p),
    }


def binary_report(y_true: Sequence[int], prob_positive, threshold: float = 0.5) -> Dict[str, float]:
    """Precision / recall / F1 of the positive class plus ranking quality.

    For compliance screening the positive class is the violation, and recall
    is usually the number that matters.
    """
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(prob_positive, dtype=np.float64)
    pred = (s >= threshold).astype(int)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    two_col = np.stack([1 - s, s], axis=1)
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "threshold": float(threshold),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "accuracy": float(np.mean(pred == y)),
        "roc_auc": roc_auc(y, s),
        "ece": expected_calibration_error(y, two_col),
        "brier": float(np.mean((s - y) ** 2)),
    }


def roc_auc(y_true, scores) -> float:
    """Area under the ROC curve via the rank statistic (ties get mean rank)."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def expected_calibration_error(y_true, probs, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width bins.

    Read it as "on average, stated confidence is off by this much". It is
    biased upward on small samples, so compare models on the same test set.
    """
    p = _as_probs(probs)
    y = np.asarray(y_true, dtype=np.int64)
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(np.float64)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score(y_true, probs) -> float:
    p = _as_probs(probs)
    y = np.asarray(y_true, dtype=np.int64)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def negative_log_likelihood(y_true, probs) -> float:
    p = _as_probs(probs)
    y = np.asarray(y_true, dtype=np.int64)
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + EPS)))


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def apply_temperature(probs, temperature: float) -> np.ndarray:
    """Rescale confidence without changing the predicted class."""
    logp = np.log(_as_probs(probs) + EPS) / max(temperature, 1e-6)
    logp -= logp.max(1, keepdims=True)
    out = np.exp(logp)
    return out / out.sum(1, keepdims=True)


def fit_temperature(y_true, probs, lo: float = 0.05, hi: float = 20.0, iters: int = 60) -> float:
    """Temperature that minimises NLL on a held-out set (golden-section search).

    Fit this on real, human-labelled data that was not used for training.
    Fitting it on teacher-labelled or synthetic data calibrates the student to
    the teacher's mistakes.
    """
    p = _as_probs(probs)
    y = np.asarray(y_true, dtype=np.int64)
    phi = (np.sqrt(5) - 1) / 2
    a, b = np.log(lo), np.log(hi)

    def loss(log_t: float) -> float:
        return negative_log_likelihood(y, apply_temperature(p, float(np.exp(log_t))))

    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = loss(d)
    return float(np.exp((a + b) / 2))


# --------------------------------------------------------------------------
# Abstention and cascades
# --------------------------------------------------------------------------

@dataclass
class OperatingPoint:
    threshold: float
    coverage: float  # fraction answered by the small model
    selective_accuracy: float  # accuracy on the answered fraction
    target_met: bool

    def to_dict(self) -> dict:
        return {"threshold": self.threshold, "coverage": self.coverage,
                "selective_accuracy": self.selective_accuracy, "target_met": self.target_met}


def risk_coverage(y_true, probs, points: int = 21) -> list[dict]:
    """Accuracy on the most-confident fraction, for a sweep of coverages."""
    p = _as_probs(probs)
    y = np.asarray(y_true, dtype=np.int64)
    order = np.argsort(-p.max(1), kind="mergesort")
    correct = (p.argmax(1) == y)[order]
    cum = np.cumsum(correct) / np.arange(1, len(y) + 1)
    out = []
    for cov in np.linspace(0.05, 1.0, points):
        n = max(1, int(round(cov * len(y))))
        out.append({"coverage": float(n / len(y)), "accuracy": float(cum[n - 1])})
    return out


def threshold_for_accuracy(y_true, probs, target_accuracy: float) -> OperatingPoint:
    """Lowest confidence threshold whose answered set meets ``target_accuracy``.

    Choose it on a validation set, then report coverage on the test set with
    :func:`apply_threshold`; a threshold tuned and reported on the same data
    is optimistic.
    """
    p = _as_probs(probs)
    y = np.asarray(y_true, dtype=np.int64)
    conf = p.max(1)
    order = np.argsort(-conf, kind="mergesort")
    correct = (p.argmax(1) == y)[order]
    cum = np.cumsum(correct) / np.arange(1, len(y) + 1)
    ok = np.where(cum >= target_accuracy)[0]
    if len(ok) == 0:
        return OperatingPoint(1.0 + 1e-9, 0.0, float("nan"), False)
    n = int(ok[-1]) + 1
    return OperatingPoint(float(conf[order][n - 1]), n / len(y), float(cum[n - 1]), True)


def apply_threshold(y_true, probs, threshold: float) -> OperatingPoint:
    p = _as_probs(probs)
    y = np.asarray(y_true, dtype=np.int64)
    mask = p.max(1) >= threshold
    acc = float(np.mean(p.argmax(1)[mask] == y[mask])) if mask.any() else float("nan")
    return OperatingPoint(float(threshold), float(mask.mean()), acc, True)


def simulate_cascade(
    y_true,
    small_probs,
    big_pred: Sequence[int],
    threshold: float,
    small_cost: float = 1.0,
    big_cost: float = 100.0,
) -> Dict[str, float]:
    """Small model answers when confident; everything else goes to the big one.

    ``big_pred`` are the larger model's (or the human reviewers') decisions on
    the same items. Costs are relative units per item. The result says whether
    escalation helps at all: if the big model is not more accurate than the
    small one on the *escalated* items, the cascade only adds cost.
    """
    p = _as_probs(small_probs)
    y = np.asarray(y_true, dtype=np.int64)
    big = np.asarray(big_pred, dtype=np.int64)
    keep = p.max(1) >= threshold
    final = np.where(keep, p.argmax(1), big)
    esc = ~keep
    return {
        "threshold": float(threshold),
        "escalation_rate": float(esc.mean()),
        "cascade_accuracy": float(np.mean(final == y)),
        "small_only_accuracy": float(np.mean(p.argmax(1) == y)),
        "big_only_accuracy": float(np.mean(big == y)),
        "small_accuracy_on_escalated": float(np.mean(p.argmax(1)[esc] == y[esc])) if esc.any() else float("nan"),
        "big_accuracy_on_escalated": float(np.mean(big[esc] == y[esc])) if esc.any() else float("nan"),
        "relative_cost_vs_big_only": float((small_cost + esc.mean() * big_cost) / big_cost),
    }
