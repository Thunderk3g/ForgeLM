"""E1c - serving cost of the full-data Banking77 students, measured alone.

E1 benchmarks its students as its last step, but other work may be using the
machine by then. This script reloads the students E1 (and E1b) saved and
measures single-text latency and batched throughput again, and should be run
with nothing else competing for the cores:

    python experiments/e1c_serving.py

It reads accuracy from the saved test probabilities, so it trains nothing.
"""

from __future__ import annotations

import numpy as np

from common import ARTIFACTS, load_banking77, rounded, write_result
from forgelm.bench import benchmark, directory_size_mb
from forgelm.students import load_student

train, test, names = load_banking77()
texts, y = test.text.tolist(), test.label.values
bench_texts = texts[:400]
labels_of = {"tfidf": "tfidf + LR", "static": "static potion-32M + LR", "encoder": "frozen bge-small + LR",
             "finetuned": "fine-tuned MiniLM-L6 (22M)", "modernbert": "fine-tuned ModernBERT-base (149M)"}
serving = []
for key, label in labels_of.items():
    directory = ARTIFACTS / f"e1_{key}"
    if not (directory / "student.json").exists():
        print(f"skip {label}: {directory} not found")
        continue
    student = load_student(str(directory))
    accuracy = float(np.mean(np.load(ARTIFACTS / f"e1_{key}_probs.npz")["test"].argmax(1) == y))
    result = benchmark(student.predict_proba, bench_texts, batch_size=32, single_item_runs=200)
    note = ("size is the classifier head only; the shared embedding model is separate"
            if key in ("static", "encoder") else "")
    serving.append({"student": label, "accuracy": round(accuracy, 4),
                    "artefact_mb": round(directory_size_mb(str(directory)), 2), "note": note, **rounded(result, 3)})
    print(serving[-1], flush=True)

write_result("e1c_serving", {"experiment": "E1c serving cost of the full-data Banking77 students, measured alone",
                             "serving": serving})
