import json

import numpy as np
import pytest

from forgelm.compress import parity_gate
from forgelm.provenance import EvalContract, RunRecord, hash_directory, sha256_dataset


def _probs(pred, k=3):
    p = np.full((len(pred), k), 0.05)
    p[np.arange(len(pred)), pred] = 0.9
    return p


def test_parity_gate_passes_identical_models():
    y = np.array([0, 1, 2] * 40)
    ref = _probs(y)
    gate = parity_gate(ref, ref + 1e-4, y)
    assert gate.passed and gate.agreement == 1.0 and gate.accuracy_drop == 0.0


def test_parity_gate_fails_on_drift_and_says_why():
    y = np.array([0, 1, 2] * 40)
    ref = _probs(y)
    broken_pred = y.copy()
    broken_pred[:30] = (broken_pred[:30] + 1) % 3  # 25% of decisions changed, all for the worse
    gate = parity_gate(ref, _probs(broken_pred), y)
    assert not gate.passed
    assert gate.agreement == pytest.approx(0.75) and gate.accuracy_drop == pytest.approx(0.25)
    assert len(gate.reasons) == 2


def test_parity_gate_without_labels_checks_agreement_only():
    ref = _probs(np.array([0, 1] * 50), 2)
    cand = _probs(np.array([0, 1] * 49 + [1, 0]), 2)
    gate = parity_gate(ref, cand, min_agreement=0.99)
    assert not gate.passed and gate.reference_accuracy is None
    with pytest.raises(ValueError):
        parity_gate(ref, cand[:10])


def test_dataset_hash_is_sensitive_to_content_labels_and_order():
    base = sha256_dataset(["a", "b"], [0, 1])
    assert base == sha256_dataset(["a", "b"], [0, 1])
    assert base != sha256_dataset(["a", "b"], [1, 1])
    assert base != sha256_dataset(["b", "a"], [1, 0])
    assert sha256_dataset(["ab", "c"]) != sha256_dataset(["a", "bc"])


def test_eval_contract_checks_thresholds_and_data(tmp_path):
    texts, labels = ["x", "y", "z"], [0, 1, 0]
    contract = EvalContract.create("demo", texts, labels, {"accuracy": 0.9, "ece": 0.05})
    assert contract.check({"accuracy": 0.95, "ece": 0.02}, texts, labels)["passed"]
    bad = contract.check({"accuracy": 0.85, "ece": 0.08})
    assert not bad["passed"] and len(bad["failures"]) == 2
    assert not contract.check({"accuracy": 0.95})["passed"]  # ece missing
    swapped = contract.check({"accuracy": 0.95, "ece": 0.02}, texts, [1, 1, 0])
    assert "dataset hash" in swapped["failures"][0]
    path = contract.save(str(tmp_path / "c.json"))
    assert EvalContract.load(path).dataset_sha256 == contract.dataset_sha256


def test_eval_contract_fails_metrics_that_are_not_finite_numbers():
    # NaN is neither below a minimum nor above a limit, so a plain comparison
    # would pass it. roc_auc is NaN whenever the hold-out has only one class.
    contract = EvalContract("demo", "0" * 64, 3, {"roc_auc": 0.8, "ece": 0.05})
    for bad in (float("nan"), np.float64("nan"), float("inf"), float("-inf"), None, "0.9"):
        for metrics in ({"roc_auc": bad, "ece": 0.01}, {"roc_auc": 0.9, "ece": bad}):
            result = contract.check(metrics)
            assert not result["passed"], metrics
            assert len(result["failures"]) == 1 and "not a finite number" in result["failures"][0]
    assert "roc_auc is nan" in contract.check({"roc_auc": float("nan"), "ece": 0.01})["failures"][0]
    # numpy scalars and plain ints are still ordinary numbers
    assert contract.check({"roc_auc": np.float32(0.9), "ece": 0})["passed"]


def test_run_record_writes_card_and_bom(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"weights")
    record = RunRecord(
        name="complaint-router", task="text classification", recipe={"student": "static"},
        data={"train": {"n": 100, "sha256": "ab" * 32}}, metrics={"accuracy": 0.93, "n": 50},
        hardware={"device_class": "cpu", "cpu_model": "x", "ram_total_gb": 16},
        base_models=[{"name": "minishlab/potion-base-32M", "revision": "1e5a03f", "licence": "MIT"}],
        artefacts=hash_directory(str(tmp_path)), contract={"contract": "demo", "passed": False,
                                                           "failures": ["accuracy 0.93 below minimum 0.95"]})
    paths = record.write_all(str(tmp_path))
    assert len(paths) == 3
    card = (tmp_path / "MODEL_CARD.md").read_text()
    assert "minishlab/potion-base-32M" in card and "FAILED" in card and "accuracy: 0.9300" in card
    bom = json.loads((tmp_path / "ml-bom.cdx.json").read_text())
    assert bom["bomFormat"] == "CycloneDX" and bom["specVersion"] == "1.6"
    kinds = [c["type"] for c in bom["components"]]
    assert kinds == ["machine-learning-model", "machine-learning-model"]
    assert bom["dependencies"][0]["dependsOn"] == ["base:minishlab/potion-base-32M"]
    saved = json.loads((tmp_path / "run_record.json").read_text())
    assert saved["artefacts"]["model.bin"] == hash_directory(str(tmp_path))["model.bin"]
