import numpy as np
import pytest

from forgelm import metrics as M


def _confident(y, n_classes, p=0.9):
    probs = np.full((len(y), n_classes), (1 - p) / (n_classes - 1))
    probs[np.arange(len(y)), y] = p
    return probs


def test_report_perfect_predictions():
    y = np.array([0, 1, 2, 1, 0])
    rep = M.classification_report(y, _confident(y, 3, 0.99))
    assert rep["accuracy"] == 1.0 and rep["macro_f1"] == 1.0
    assert rep["ece"] == pytest.approx(0.01, abs=1e-6)


def test_macro_f1_ignores_absent_classes():
    y = np.array([0, 0, 1, 1])
    probs = _confident(np.array([0, 0, 1, 0]), 3)
    rep = M.classification_report(y, probs)
    assert rep["accuracy"] == 0.75
    assert 0 < rep["macro_f1"] < 1


def test_ece_detects_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    pred = np.where(rng.random(4000) < 0.7, y, 1 - y)  # 70% accurate
    over = _confident(pred, 2, 0.99)
    honest = _confident(pred, 2, 0.70)
    assert M.expected_calibration_error(y, over) > 0.25
    assert M.expected_calibration_error(y, honest) < 0.03


def test_temperature_scaling_recovers_known_temperature():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(5000, 4)) * 2.0
    true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(4, p=p) for p in true])
    sharpened = M.apply_temperature(true, 0.4)  # overconfident version
    t = M.fit_temperature(y, sharpened)
    assert t == pytest.approx(2.5, rel=0.15)
    assert M.expected_calibration_error(y, M.apply_temperature(sharpened, t)) < \
        M.expected_calibration_error(y, sharpened)
    assert (M.apply_temperature(sharpened, t).argmax(1) == sharpened.argmax(1)).all()


def test_roc_auc_matches_known_values():
    assert M.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert M.roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert np.isnan(M.roc_auc([1, 1], [0.2, 0.3]))


def test_binary_report_recall():
    rep = M.binary_report([1, 1, 1, 0, 0, 0], [0.9, 0.8, 0.2, 0.1, 0.6, 0.3], threshold=0.5)
    assert rep["recall"] == pytest.approx(2 / 3) and rep["precision"] == pytest.approx(2 / 3)


def test_threshold_for_accuracy_and_coverage():
    y = np.array([0] * 100)
    conf = np.linspace(0.5, 1.0, 100)
    probs = np.stack([conf, 1 - conf], 1)
    probs[:20] = probs[:20][:, ::-1]  # the 20 least confident are wrong
    op = M.threshold_for_accuracy(y, probs, 0.99)
    assert op.target_met and op.coverage == pytest.approx(0.8, abs=0.02)
    applied = M.apply_threshold(y, probs, op.threshold)
    assert applied.selective_accuracy >= 0.99
    never = M.threshold_for_accuracy(np.array([1] * 10), np.tile([0.9, 0.1], (10, 1)), 0.5)
    assert not never.target_met and never.coverage == 0.0


def test_cascade_reports_whether_escalation_helps():
    y = np.array([0, 0, 1, 1])
    small = np.array([[0.95, 0.05], [0.55, 0.45], [0.45, 0.55], [0.6, 0.4]])  # last one wrong, unsure
    helpful = M.simulate_cascade(y, small, big_pred=y, threshold=0.9)
    assert helpful["cascade_accuracy"] == 1.0 and helpful["escalation_rate"] == 0.75
    assert helpful["big_accuracy_on_escalated"] > helpful["small_accuracy_on_escalated"]
    useless = M.simulate_cascade(y, small, big_pred=1 - y, threshold=0.9)
    assert useless["cascade_accuracy"] < useless["small_only_accuracy"]
    assert 0 < helpful["relative_cost_vs_big_only"] < 1


def test_risk_coverage_is_monotone_for_ranked_errors():
    y = np.zeros(50, dtype=int)
    conf = np.linspace(0.51, 0.99, 50)
    probs = np.stack([conf, 1 - conf], 1)
    probs[:10] = probs[:10][:, ::-1]
    curve = M.risk_coverage(y, probs, points=5)
    assert curve[0]["accuracy"] >= curve[-1]["accuracy"]
    assert curve[-1]["coverage"] == 1.0
