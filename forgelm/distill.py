"""Teacher -> student distillation with a human-labelled anchor.

The anchor is the one rule this module enforces. A student trained on teacher
labels and then scored on teacher labels tells you how well it copies the
teacher, not whether it is right. Every distillation run therefore takes a
small set of real, human-labelled examples that is never shown to the teacher
for labelling and never used for training. Both teacher and student are
scored on it, so the report can say three separate things:

    how good the teacher is,
    how much the student lost (or gained) relative to the teacher,
    how often the student simply repeats the teacher.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .metrics import classification_report


@dataclass
class DistillReport:
    n_unlabelled: int
    n_used_for_training: int
    soft_labels: bool
    min_confidence: float
    teacher_on_anchor: Dict[str, float]
    student_on_anchor: Dict[str, float]
    student_teacher_agreement: float
    teacher_stats: dict
    label_distribution: Dict[str, int]
    warnings: List[str] = field(default_factory=list)

    @property
    def student_minus_teacher_accuracy(self) -> float:
        return self.student_on_anchor["accuracy"] - self.teacher_on_anchor["accuracy"]

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["student_minus_teacher_accuracy"] = self.student_minus_teacher_accuracy
        return d


def _fingerprint(text: str) -> str:
    return hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()


def distill(
    teacher,
    student,
    unlabelled_texts: Sequence[str],
    anchor_texts: Sequence[str],
    anchor_y: Sequence[int],
    soft: bool = True,
    min_confidence: float = 0.0,
    teacher_probs: Optional[np.ndarray] = None,
    teacher_anchor_probs: Optional[np.ndarray] = None,
) -> DistillReport:
    """Label ``unlabelled_texts`` with the teacher, train the student, score both.

    ``min_confidence`` drops teacher labels below that confidence. Filtering
    removes noise but also removes the hard cases the student most needs, so
    treat it as a knob to measure, not a default to trust.

    ``teacher_probs`` / ``teacher_anchor_probs`` let a caller reuse cached
    teacher outputs (teacher calls are the expensive part).
    """
    unlabelled = list(unlabelled_texts)
    anchor_texts = list(anchor_texts)
    labels = list(teacher.labels)
    warnings: List[str] = []

    anchor_prints = {_fingerprint(t) for t in anchor_texts}
    keep = [i for i, t in enumerate(unlabelled) if _fingerprint(t) not in anchor_prints]
    if len(keep) < len(unlabelled):
        warnings.append(f"removed {len(unlabelled) - len(keep)} unlabelled texts that also appear in the anchor set")
    if teacher_probs is None:
        probs = teacher.label([unlabelled[i] for i in keep])
    else:
        probs = np.asarray(teacher_probs)[keep]
    texts = [unlabelled[i] for i in keep]

    if min_confidence > 0:
        confident = probs.max(1) >= min_confidence
        texts = [t for t, k in zip(texts, confident) if k]
        probs = probs[confident]
    if len(texts) == 0:
        raise ValueError("no teacher labels survived the confidence filter")

    hard = probs.argmax(1)
    counts = np.bincount(hard, minlength=len(labels))
    missing = [labels[i] for i in np.where(counts == 0)[0]]
    if missing:
        warnings.append(f"teacher never predicted {len(missing)} class(es): {missing[:5]}"
                        + (" ..." if len(missing) > 5 else ""))

    student.fit(texts, probs if soft else hard, labels)

    y = np.asarray(anchor_y)
    t_anchor = teacher.label(anchor_texts) if teacher_anchor_probs is None else np.asarray(teacher_anchor_probs)
    s_anchor = student.predict_proba(anchor_texts)
    return DistillReport(
        n_unlabelled=len(unlabelled),
        n_used_for_training=len(texts),
        soft_labels=soft,
        min_confidence=min_confidence,
        teacher_on_anchor=classification_report(y, t_anchor),
        student_on_anchor=classification_report(y, s_anchor),
        student_teacher_agreement=float(np.mean(s_anchor.argmax(1) == t_anchor.argmax(1))),
        teacher_stats=teacher.stats(),
        label_distribution={labels[i]: int(c) for i, c in enumerate(counts)},
        warnings=warnings,
    )


def uncertainty_order(probs: np.ndarray) -> np.ndarray:
    """Indices sorted most-uncertain first (smallest top-2 margin).

    Use a cheap provisional student to rank an unlabelled pool, then spend the
    teacher (or human) budget on the head of this ordering.
    """
    p = np.sort(np.asarray(probs), axis=1)
    margin = p[:, -1] - p[:, -2]
    return np.argsort(margin, kind="mergesort")


# --------------------------------------------------------------------------
# Synthetic data: prompt construction and quality checks
# --------------------------------------------------------------------------

def attribute_prompts(label: str, description: str, attributes: Dict[str, Sequence[str]], n: int,
                      seed: int = 0, domain: str = "") -> List[str]:
    """Attribute-conditioned generation prompts for one class.

    Asking a model for "an example of class X" repeatedly yields near-copies.
    Sampling a different combination of attributes (channel, tone, length,
    language, customer type ...) per request is the published remedy; it does
    not remove the gap to real data, so synthetic text should supplement real
    examples rather than replace them.
    """
    rng = np.random.default_rng(seed)
    keys = sorted(attributes)
    prompts = []
    for _ in range(n):
        picked = {k: attributes[k][int(rng.integers(len(attributes[k])))] for k in keys}
        traits = "; ".join(f"{k}: {v}" for k, v in picked.items())
        prompts.append(
            f"Write one realistic {domain + ' ' if domain else ''}text that belongs to the category "
            f"\"{label}\" ({description}). Constraints - {traits}. "
            "Return only the text, with no label, quotes or explanation.")
    return prompts


def diversity_report(texts: Sequence[str]) -> Dict[str, float]:
    """Cheap collapse detectors for a synthetic set.

    Low distinct-n or a high exact-duplicate rate means the generator is
    repeating itself and the student will overfit to its phrasing.
    """
    texts = [" ".join(t.lower().split()) for t in texts]
    if not texts:
        return {"n": 0}

    def distinct(n: int) -> float:
        grams, total = set(), 0
        for t in texts:
            toks = t.split()
            for i in range(len(toks) - n + 1):
                grams.add(tuple(toks[i:i + n]))
                total += 1
        return len(grams) / total if total else 0.0

    lengths = [len(t.split()) for t in texts]
    return {
        "n": len(texts),
        "exact_duplicate_rate": 1.0 - len(set(texts)) / len(texts),
        "distinct_1": distinct(1),
        "distinct_2": distinct(2),
        "mean_words": float(np.mean(lengths)),
        "std_words": float(np.std(lengths)),
    }
