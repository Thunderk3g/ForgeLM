"""E5 - abstention and cascades on Banking77.

Questions
  1. If the small model only answers when confident, how much of the traffic
     can it take at a stated accuracy target?
  2. Does escalating the rest to a larger model help? Tested with two "larger"
     models: the real local LLM teacher from E2, and a perfect oracle (the
     upper bound a human reviewer or ideal LLM could give).

Protocol
  * Students are the full-data models saved by E1. Temperature and the
    confidence threshold are both chosen on the 385-example calibration anchor
    and then applied, unchanged, to the test split.
  * The LLM comparison uses the 1,000 test items the E2 teacher labelled.
"""

from __future__ import annotations

import numpy as np

from common import ARTIFACTS, load_banking77, rounded, stratified_sample, write_result
from forgelm.metrics import (apply_temperature, apply_threshold, risk_coverage, simulate_cascade,
                             threshold_for_accuracy)

train, test, names = load_banking77()
test_y = test.label.values
anchor_idx = stratified_sample(train.label.values, 5, np.random.default_rng(1234))
anchor_y = train.label.values[anchor_idx]

llm = None
for path in ARTIFACTS.glob("e2_teacher_*.npz"):
    cached = np.load(path)
    llm = {"idx": cached["anchor_idx"], "pred": cached["anchor"].argmax(1), "name": path.stem}

out = []
labels_of = {"tfidf": "tfidf + LR", "static": "static potion-32M + LR", "encoder": "frozen bge-small + LR",
             "finetuned": "fine-tuned MiniLM-L6 (22M)", "modernbert": "fine-tuned ModernBERT-base (149M)"}
for key, label in labels_of.items():
    path = ARTIFACTS / f"e1_{key}_probs.npz"
    if not path.exists():
        continue
    data = np.load(path)
    t = float(data["temperature"])
    p_test, p_anchor = apply_temperature(data["test"], t), apply_temperature(data["anchor"], t)
    entry = {"student": label, "temperature": round(t, 3),
             "test_accuracy": round(float(np.mean(p_test.argmax(1) == test_y)), 4), "targets": [],
             "risk_coverage": [rounded(r) for r in risk_coverage(test_y, p_test, points=20)]}
    for target in (0.95, 0.97, 0.99):
        op = threshold_for_accuracy(anchor_y, p_anchor, target)
        on_test = apply_threshold(test_y, p_test, op.threshold)
        row = {"target_accuracy": target, "threshold": round(op.threshold, 4),
               "target_reachable_on_anchor": op.target_met,
               "anchor_coverage": round(op.coverage, 4), "test_coverage": round(on_test.coverage, 4),
               "test_selective_accuracy": round(on_test.selective_accuracy, 4)}
        row["oracle_cascade"] = rounded(simulate_cascade(test_y, p_test, test_y, op.threshold))
        if llm is not None:
            row["llm_cascade"] = rounded(simulate_cascade(test_y[llm["idx"]], p_test[llm["idx"]], llm["pred"],
                                                          op.threshold))
        entry["targets"].append(row)
    out.append(entry)
    print(label, [(r["target_accuracy"], r["test_coverage"], r["test_selective_accuracy"]) for r in entry["targets"]],
          flush=True)

write_result("e5_cascade", {
    "experiment": "E5 abstention and cascades (Banking77)",
    "threshold_chosen_on": "385-example calibration anchor (5 per class), applied unchanged to test",
    "llm_escalation_model": llm["name"] if llm else None,
    "caveat": "385 anchor examples give a noisy threshold; coverage at the 0.99 target is especially unstable",
    "students": out,
})
