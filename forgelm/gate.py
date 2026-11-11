"""The gate: is the costlier option really better on this test set?

The planner's first advice for a new classification task is "try something
cheap before something costly": prompt an LLM before fine-tuning one, and try
an LLM-free student before keeping an LLM in the serving path. That advice is
only worth something if it can be checked on the user's own labelled test
items, and if the check says when the test set cannot answer the question.

This module compares systems that were scored on the *same* test items:

* :func:`paired_bootstrap` gives an interval for the accuracy difference.
  Resampling is paired: each bootstrap replicate draws one set of test items
  and scores both systems on it, so the item-to-item difficulty they share
  cancels out. That makes the interval much tighter than comparing two
  independent accuracy intervals.
* :func:`mcnemar_exact` asks the same question with an exact test on the
  items where the two systems disagree about correctness.
* :func:`decide` ranks the options by cost and recommends the cheapest one
  that no costlier option beats by at least ``min_gain`` with an interval
  that excludes zero. When an interval is too wide to rule out a gain worth
  paying for, the report says the test set is too small to tell, instead of
  presenting a coin toss as a finding.

Only numpy and the standard library are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# Bootstrap replicates are drawn in chunks so that memory stays near this many
# indices whatever the test-set size (5,000 replicates of 10,000 items would
# otherwise be a 400 MB index array).
_CHUNK_INDICES = 2_000_000


def _as_labels(pred) -> np.ndarray:
    """Predicted labels from labels or from an ``(n, n_classes)`` probability matrix."""
    p = np.asarray(pred)
    if p.ndim == 2:
        return p.argmax(1)
    if p.ndim != 1:
        raise ValueError(f"expected labels (n,) or probabilities (n, n_classes), got shape {p.shape}")
    return p


def _correct(y_true, pred) -> np.ndarray:
    y = np.asarray(y_true)
    p = _as_labels(pred)
    if p.shape[0] != y.shape[0]:
        raise ValueError(f"{p.shape[0]} predictions for {y.shape[0]} test items")
    if y.shape[0] == 0:
        raise ValueError("empty test set")
    return p == y


def _bootstrap_means(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Means of ``values`` over ``n_boot`` resamples of the items, with replacement.

    The resamples depend only on ``(len(values), n_boot, seed)``, so two calls
    with the same seed on the same test set see the same resampled items.
    """
    n = len(values)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = max(1, _CHUNK_INDICES // n)
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        means[start:start + m] = values[rng.integers(0, n, size=(m, n))].mean(1)
    return means


def _interval(samples: np.ndarray, confidence: float):
    tail = 100.0 * (1.0 - confidence) / 2.0
    lo, hi = np.percentile(samples, [tail, 100.0 - tail])
    return float(lo), float(hi)


def paired_bootstrap(y_true, pred_a, pred_b, n_boot: int = 5000, seed: int = 0,
                     confidence: float = 0.95) -> dict:
    """Accuracy of A and B, the difference B - A, and a percentile interval for it.

    ``pred_a`` and ``pred_b`` are predicted labels or probability matrices for
    the same test items in the same order. Identical systems give a difference
    of exactly 0 and an interval of [0, 0].
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    ca, cb = _correct(y_true, pred_a), _correct(y_true, pred_b)
    diff = cb.astype(np.float64) - ca.astype(np.float64)
    lo, hi = _interval(_bootstrap_means(diff, n_boot, seed), confidence)
    return {"n": int(len(diff)), "accuracy_a": float(ca.mean()), "accuracy_b": float(cb.mean()),
            "difference": float(diff.mean()), "ci_low": lo, "ci_high": hi,
            "confidence": confidence, "n_boot": int(n_boot), "seed": int(seed)}


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact two-sided McNemar test on the items where A and B differ in correctness.

    Under "A and B are equally accurate" each discordant item is a fair coin
    flip, so the count won by B is Binomial(discordant, 1/2). The p-value is
    twice the smaller tail, capped at 1. Computed with exact integers
    (``math.comb``), so it stays exact for any test-set size.
    """
    ca, cb = _correct(y_true, pred_a), _correct(y_true, pred_b)
    a_only = int(np.sum(ca & ~cb))
    b_only = int(np.sum(cb & ~ca))
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1))
        p = min(1.0, 2 * tail / 2 ** n)  # int / int: Python rounds the exact ratio once
    return {"a_only_correct": a_only, "b_only_correct": b_only, "discordant": n, "p_value": float(p)}


@dataclass
class Option:
    """One way of solving the task, with what it costs to run.

    Give ``predictions`` (labels or probabilities on the shared test items)
    whenever possible; an option with only an ``accuracy`` is listed but
    cannot be compared, because a difference between two accuracies measured
    without pairing has no usable interval here. When both are given the
    predictions are used.

    ``labels_needed`` (human labels the option needs) only breaks ties
    between options that cost the same to run.
    """

    name: str
    predictions: Optional[Sequence] = None
    accuracy: Optional[float] = None
    needs_llm_at_inference: bool = False
    seconds_per_item: float = 0.0
    labels_needed: int = 0
    notes: str = ""

    def cost_key(self):
        # An LLM in the serving path is the dominant cost (hardware, latency, and the operational risk
        # of running a generative model in production), so it outranks any measured speed difference.
        return (bool(self.needs_llm_at_inference), float(self.seconds_per_item), int(self.labels_needed), self.name)


# Verdicts on "costlier minus cheaper" accuracy, with the interval [lo, hi].
# The verdict reads the bootstrap interval alone; whether McNemar agrees is kept separately ("confirmed").
BETTER = "better"                      # gain >= min_gain and lo > 0: worth paying for, if McNemar agrees
SLIGHTLY_BETTER = "slightly_better"    # lo > 0 but gain < min_gain: real but too small to pay for
NO_MEANINGFUL_GAIN = "no_meaningful_gain"  # hi < min_gain: a gain worth paying for is ruled out
WORSE = "worse"                        # hi < 0
INCONCLUSIVE = "inconclusive"          # interval spans 0 and reaches min_gain: the test set cannot tell


def _verdict(gain: float, lo: float, hi: float, min_gain: float) -> str:
    if lo > 0:
        return BETTER if gain >= min_gain else SLIGHTLY_BETTER
    if hi < 0:
        return WORSE
    if hi < min_gain:
        return NO_MEANINGFUL_GAIN
    return INCONCLUSIVE


@dataclass
class GateReport:
    """What :func:`decide` found. ``to_dict`` is JSON-ready; ``render`` is for people."""

    n_items: int
    min_gain: float
    confidence: float
    options: List[dict]
    comparisons: List[dict]
    recommended: Optional[str]
    reason: str
    too_small_to_tell: bool
    rejected: Dict[str, dict] = field(default_factory=dict)
    untested: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"n_items": self.n_items, "min_gain": self.min_gain, "confidence": self.confidence,
                "options": self.options, "comparisons": self.comparisons, "recommended": self.recommended,
                "reason": self.reason, "too_small_to_tell": self.too_small_to_tell,
                "rejected": self.rejected, "untested": self.untested}

    def render(self) -> str:
        pct = f"{self.confidence:.0%}"
        w = max([len("option")] + [len(o["name"]) for o in self.options])
        lines = [f"Gate on {self.n_items} test items (minimum gain worth paying for: {_pts(self.min_gain)})", "",
                 f"{'option':{w}} {'accuracy':>9} {pct + ' interval':>17} {'LLM':>4} {'s/item':>9} {'labels':>7}"]
        for o in self.options:
            acc = "-" if o["accuracy"] is None else f"{o['accuracy']:.3f}"
            ci = "-" if o["accuracy_ci"] is None else f"{o['accuracy_ci'][0]:.3f}-{o['accuracy_ci'][1]:.3f}"
            mark = "  <- recommended" if o["name"] == self.recommended else ""
            lines.append(f"{o['name']:{w}} {acc:>9} {ci:>17} {'yes' if o['needs_llm_at_inference'] else 'no':>4} "
                         f"{o['seconds_per_item']:>9.4f} {o['labels_needed']:>7}{mark}")
        if self.comparisons:
            lines += ["", f"Costlier minus cheaper, paired bootstrap {pct} interval and exact McNemar p:"]
            for c in self.comparisons:
                lines.append(f"  {c['costlier']} vs {c['cheaper']}: {_signed_pts(c['gain'])} "
                             f"[{_signed_pts(c['ci_low'])}, {_signed_pts(c['ci_high'])}], "
                             f"p={c['mcnemar_p']:.3g} -> {c['verdict']}"
                             + ("" if c["settled"] else " (not settled)"))
        lines += ["", self.reason]
        return "\n".join(lines)


def _pts(x: float) -> str:
    return f"{100 * x:.1f} points"


def _signed_pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def _describe(o: Option, acc: float) -> str:
    where = "needs an LLM at inference" if o.needs_llm_at_inference else "no LLM at inference"
    return f"'{o.name}' (accuracy {acc:.3f}, {where}, {o.seconds_per_item:.3g} s per item)"


def decide(options: Sequence[Option], y_true, min_gain: float = 0.01, alpha: float = 0.05,
           n_boot: int = 5000, seed: int = 0) -> GateReport:
    """Recommend the cheapest option that no costlier option clearly beats.

    Options are ranked by cost: no LLM at inference before LLM, then seconds
    per item, then labels needed. Walking from the cheapest, an option is
    rejected when some costlier option beats it by at least ``min_gain``
    (absolute accuracy, 0.01 = one point) with a ``1 - alpha`` paired
    bootstrap interval that excludes zero *and* an exact McNemar p-value of
    at most ``alpha``. The percentile bootstrap is optimistic on small
    samples, so a gain it alone finds is not enough to make the user pay for
    the costlier option. The first option not rejected is recommended. The
    costliest option can never be rejected, so with any comparable option
    there is always a recommendation.

    The report then compares every other option with the recommended one,
    and marks each comparison *settled* or not:

    * against a costlier option, settled when the interval rules out a gain
      worth paying for (its upper end is below ``min_gain``), or when the
      interval and McNemar agree that the costlier option is better;
    * against a cheaper option (set aside because something costlier beat
      it), settled when the interval and McNemar agree that the recommended
      option is better.

    ``too_small_to_tell`` is set when any comparison is not settled, or when
    the test set is so small that one item is worth more than ``min_gain``
    (it then cannot resolve the differences that matter). The recommendation
    stays with the cheaper side of an unsettled comparison, and the reason
    says so.
    """
    if not options:
        raise ValueError("decide needs at least one option")
    names = [o.name for o in options]
    if len(set(names)) != len(names):
        raise ValueError(f"option names must be unique, got {names}")
    if min_gain < 0:
        raise ValueError("min_gain must be >= 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    y = np.asarray(y_true)
    n = int(len(y))
    confidence = 1.0 - alpha
    ranked = sorted(options, key=Option.cost_key)
    testable = [o for o in ranked if o.predictions is not None]
    untested = [o.name for o in ranked if o.predictions is None]

    correct = {o.name: _correct(y, o.predictions) for o in testable}
    accuracy = {o.name: float(correct[o.name].mean()) if o.name in correct else
                (None if o.accuracy is None else float(o.accuracy)) for o in ranked}
    option_rows = []
    for rank, o in enumerate(ranked):
        ci = (_interval(_bootstrap_means(correct[o.name].astype(np.float64), n_boot, seed), confidence)
              if o.name in correct else None)
        option_rows.append({"name": o.name, "cost_rank": rank, "accuracy": accuracy[o.name], "accuracy_ci": ci,
                            "needs_llm_at_inference": bool(o.needs_llm_at_inference),
                            "seconds_per_item": float(o.seconds_per_item), "labels_needed": int(o.labels_needed),
                            "has_predictions": o.name in correct, "notes": o.notes})

    cache: Dict[tuple, dict] = {}

    def compare(cheaper: Option, costlier: Option) -> dict:
        key = (cheaper.name, costlier.name)
        if key not in cache:
            bs = paired_bootstrap(y, cheaper.predictions, costlier.predictions, n_boot, seed, confidence)
            mc = mcnemar_exact(y, cheaper.predictions, costlier.predictions)
            verdict = _verdict(bs["difference"], bs["ci_low"], bs["ci_high"], min_gain)
            # Both tests must agree before the costlier option counts as better: the percentile bootstrap
            # is optimistic on small samples, and the exact test guards against acting on that.
            confirmed = bs["ci_low"] > 0 and mc["p_value"] <= alpha
            # Seen from the cheaper option, the question is "is the costlier one worth paying for?". It is
            # answered when a worthwhile gain is ruled out or when a gain is confirmed. Comparisons in which
            # the costlier option is the recommended one are re-marked in decide(), where that is known.
            settled = bs["ci_high"] < min_gain or confirmed
            cache[key] = {"cheaper": cheaper.name, "costlier": costlier.name,
                          "accuracy_cheaper": bs["accuracy_a"], "accuracy_costlier": bs["accuracy_b"],
                          "gain": bs["difference"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
                          "mcnemar_p": mc["p_value"], "cheaper_only_correct": mc["a_only_correct"],
                          "costlier_only_correct": mc["b_only_correct"], "verdict": verdict,
                          "confirmed": confirmed, "settled": settled}
        return cache[key]

    if not testable:
        reason = ("No option has per-item predictions on the shared test set, so none can be compared with "
                  "another; give predictions, not just accuracies, to get a recommendation.")
        return GateReport(n, min_gain, confidence, option_rows, [], None, reason, True, {}, untested)

    recommended: Optional[Option] = None
    rejected: Dict[str, dict] = {}
    for i, cheap in enumerate(testable):
        # A cheaper option is set aside only on a gain worth paying for that both tests confirm. A gain
        # only the bootstrap sees leaves the comparison unsettled, and the cheaper option is kept.
        beaten = next((c for c in (compare(cheap, costly) for costly in testable[i + 1:])
                       if c["verdict"] == BETTER and c["confirmed"]), None)
        if beaten is None:
            recommended = cheap
            break
        rejected[cheap.name] = beaten
    assert recommended is not None  # the costliest testable option has nothing costlier to lose to

    comparisons = []
    for o in testable:
        if o is recommended:
            continue
        if Option.cost_key(o) < Option.cost_key(recommended):
            # Seen from the recommended (costlier) option, the question is "is it really better than this
            # cheaper one?", which only a confirmed gain answers. Copied so the cached entry, which may
            # also sit in ``rejected``, keeps its own reading.
            c = dict(compare(o, recommended))
            c["settled"] = c["confirmed"]
        else:
            c = compare(recommended, o)
        comparisons.append(c)
    coarse = min_gain > 0 and n * min_gain < 1
    too_small = coarse or any(not c["settled"] for c in comparisons)
    reason = _reason(recommended, accuracy, comparisons, rejected, untested, n, min_gain, alpha, coarse)
    return GateReport(n, min_gain, confidence, option_rows, comparisons, recommended.name, reason,
                      too_small, rejected, untested)


def _span(c: dict, pct: str) -> str:
    return (f"{_signed_pts(c['gain'])} points, {pct} interval {_signed_pts(c['ci_low'])} to "
            f"{_signed_pts(c['ci_high'])}, McNemar p={c['mcnemar_p']:.2g}")


def _reason(rec: Option, accuracy, comparisons, rejected, untested, n: int, min_gain: float, alpha: float,
            coarse: bool) -> str:
    pct = f"{1 - alpha:.0%}"
    parts = [f"Recommend {_describe(rec, accuracy[rec.name])}."]
    if rejected:
        beaten = "; ".join(f"'{c['costlier']}' beats '{name}' ({_span(c, pct)})" for name, c in rejected.items())
        parts.append("Cheaper options were beaten by enough to pay for, with the bootstrap interval and the "
                     f"exact McNemar test agreeing: {beaten}.")
    costlier = [c for c in comparisons if c["cheaper"] == rec.name]
    if not costlier:
        parts.append("It is the costliest option compared.")

    def settled_words(c: dict) -> str:
        # Only a gain both tests confirm is called "better"; an interval below zero that McNemar does not
        # confirm is still enough to rule out paying more, but is not called "worse".
        if c["verdict"] == SLIGHTLY_BETTER and c["confirmed"]:
            return "is better, but by less than the minimum gain worth paying for"
        if c["verdict"] == WORSE and c["mcnemar_p"] <= alpha:
            return "is worse"
        return "does not gain enough to be worth paying for"

    settled_up = [c for c in costlier if c["settled"]]
    if settled_up:
        parts.append("Of the costlier options, " + "; ".join(
            f"'{c['costlier']}' {settled_words(c)} ({_span(c, pct)})" for c in settled_up) + ".")

    unsettled = [c for c in comparisons if not c["settled"]]
    up = [c for c in unsettled if c["cheaper"] == rec.name]
    down = [c for c in unsettled if c["costlier"] == rec.name]
    # Two ways a comparison with a costlier option stays open: the interval spans both zero and the
    # minimum gain, or the interval excludes zero but the exact test does not confirm it.
    wide = [c for c in up if c["ci_low"] <= 0]
    disputed = [c for c in up if c["ci_low"] > 0]
    if wide:
        parts.append(f"The test set of {n} items is too small to tell whether paying more helps ("
                     + "; ".join(f"'{c['costlier']}': {_span(c, pct)}" for c in wide)
                     + "): the evidence neither shows a gain of " + _pts(min_gain)
                     + " or more nor rules it out, so the recommendation stays with the cheaper option until more "
                       "labelled test items show otherwise.")
    if disputed:
        parts.append(f"The test set of {n} items is too small to tell whether paying more helps ("
                     + "; ".join(f"'{c['costlier']}': {_span(c, pct)}" for c in disputed)
                     + f"): the bootstrap interval excludes zero but the exact McNemar test does not confirm the "
                       f"difference at p <= {alpha:.2g}; with so few items on which the two disagree the gain may be "
                       "chance, so the recommendation stays with the cheaper option until more labelled test items "
                       "show otherwise.")
    if down:
        parts.append(f"The test set of {n} items is also too small to tell whether the recommended option is "
                     "really better than " + "; ".join(f"'{c['cheaper']}' ({_span(c, pct)})" for c in down)
                     + "; that option was set aside only because a costlier one beat it.")
    if unsettled:
        widest = max(unsettled, key=lambda c: c["ci_high"] - c["ci_low"])
        half_width = (widest["ci_high"] - widest["ci_low"]) / 2.0
        target = max(abs(widest["gain"]), min_gain)
        # The interval narrows roughly as 1/sqrt(n): this is the test-set size at which its half-width
        # would match the difference in question. A rough planning figure, not a guarantee.
        needed = int(math.ceil(n * (half_width / target) ** 2)) if target > 0 else 0
        if needed > n:
            parts.append(f"Roughly {needed} test items would be needed to resolve a difference of this size.")
    if coarse:
        parts.append(f"With only {n} test items one item is worth {_pts(1.0 / n)}, more than the minimum gain of "
                     f"{_pts(min_gain)}, so this test set cannot resolve the differences that matter; treat the "
                     "recommendation as provisional (too small to tell).")
    if comparisons and not unsettled and not coarse:
        parts.append(f"The intervals on {n} test items are narrow enough to settle these comparisons.")
    if untested:
        parts.append("Listed but not compared, because only an accuracy was given and not per-item "
                     f"predictions: {', '.join(repr(u) for u in untested)}.")
    return " ".join(parts)
