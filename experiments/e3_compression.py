"""E3 - does int8 quantisation survive the parity gate?

Question: after exporting a fine-tuned classifier to ONNX and quantising it to
int8, is it still the same classifier, and how much faster and smaller is it?

Protocol
  * Inputs: the fine-tuned students saved by E1 (MiniLM-L6) and E1b
    (ModernBERT-base), both trained on full Banking77.
  * For each: PyTorch fp32 (reference), ONNX fp32, ONNX dynamic int8.
  * Gate: decision agreement with the PyTorch model >= 98% and accuracy drop
    <= 1 point on the 3,080-example test split.
"""

from __future__ import annotations

import os

import numpy as np

from common import ARTIFACTS, environment, load_banking77, rounded, write_result
from forgelm.bench import benchmark
from forgelm.compress import OnnxClassifier, export_and_gate
from forgelm.hardware import probe
from forgelm.metrics import classification_report
from forgelm.students import FineTunedStudent

train, test, names = load_banking77()
texts, y = test.text.tolist(), test.label.values
bench_texts = texts[:400]
hw = probe()
out = []
for key, label in (("finetuned", "MiniLM-L6 (22M)"), ("modernbert", "ModernBERT-base (149M)")):
    student_dir = ARTIFACTS / f"e1_{key}"
    if not (student_dir / "student.json").exists():
        print(f"skip {label}: {student_dir} not found (run E1 first)")
        continue
    student = FineTunedStudent.load(str(student_dir), device="cpu")
    student.temperature = 1.0
    ref = student.predict_proba(texts)
    entry = {"model": label, "pytorch_fp32": {"metrics": rounded(classification_report(y, ref)),
                                               "size_mb": round(os.path.getsize(student_dir / "model.safetensors") / 1e6, 2),
                                               **rounded(benchmark(student.predict_proba, bench_texts), 3)}}
    try:
        report = export_and_gate(str(student_dir), str(ARTIFACTS / f"e3_{key}_onnx"), texts, y, ref,
                                 has_vnni=hw.has_vnni)
    except Exception as exc:  # an export failure is itself a result worth recording
        entry["export_error"] = repr(exc)[:500]
        out.append(entry)
        print(label, "EXPORT FAILED", entry["export_error"], flush=True)
        continue
    for name in ("onnx_fp32", "onnx_int8"):
        clf = OnnxClassifier.from_student_dir(report[name]["path"], str(student_dir))
        clf.temperature = 1.0
        entry[name] = {"metrics": rounded(report[name]["metrics"]), "size_mb": round(report[name]["size_mb"], 2),
                       "parity": rounded(report[name]["parity"]), **rounded(benchmark(clf.predict_proba, bench_texts), 3)}
    entry["recommended"] = report["recommended"]
    out.append(entry)
    print(label, {n: (entry[n]["metrics"]["accuracy"], entry[n]["size_mb"], entry[n]["latency_ms_p50"],
                      entry[n].get("parity", {}).get("agreement")) for n in ("pytorch_fp32", "onnx_fp32", "onnx_int8")},
          entry["recommended"], flush=True)

write_result("e3_compression", {"experiment": "E3 ONNX export and int8 quantisation behind a parity gate",
                                "gate": {"min_agreement": 0.98, "max_accuracy_drop": 0.01},
                                "deployment_cpu_has_vnni": hw.has_vnni, "models": out})
