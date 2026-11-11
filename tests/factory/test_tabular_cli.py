import json

import numpy as np
import pandas as pd
import pytest

from forgelm.cli import main
from forgelm.tabular import FOUNDATION_MODEL_LICENCES, recommend, run_ladder


@pytest.fixture
def risk_table():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "income": rng.normal(50, 15, n), "utilisation": rng.uniform(0, 1, n),
        "late_payments": rng.poisson(1.0, n).astype(float),
        "segment": rng.choice(["salaried", "self_employed", "retired"], n),
    })
    df.loc[rng.choice(n, 30, replace=False), "income"] = np.nan
    logit = -2 + 3 * df["utilisation"] + 0.8 * df["late_payments"] - 0.02 * df["income"].fillna(50) \
        + (df["segment"] == "self_employed") * 0.7
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df, y.values


def test_ladder_runs_baselines_on_mixed_columns_with_missing_values(risk_table):
    df, y = risk_table
    results = run_ladder(df.iloc[:450], y[:450], df.iloc[450:], y[450:], models=["logreg", "hist_gbdt"])
    assert [r.model for r in results] == ["logreg", "hist_gbdt"]
    for r in results:
        assert r.metrics["roc_auc"] > 0.7 and 0 <= r.metrics["brier"] <= 0.25
        assert r.to_dict()["n_train"] == 450


def test_ladder_skips_or_raises_for_missing_packages(risk_table, monkeypatch):
    import importlib.util

    df, y = risk_table
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "tabicl" else real(name, *a))
    assert run_ladder(df[:100], y[:100], df[100:150], y[100:150], models=["tabicl"]) == []
    with pytest.raises(ImportError):
        run_ladder(df[:100], y[:100], df[100:150], y[100:150], models=["tabicl"], skip_missing=False)


def _ladder_metrics(df, y, cut=450, models=("logreg", "hist_gbdt")):
    return [r.metrics for r in run_ladder(df.iloc[:cut], y[:cut], df.iloc[cut:], y[cut:], models=list(models))]


def test_ladder_metrics_do_not_depend_on_how_binary_labels_are_encoded(risk_table):
    df, y = risk_table
    reference = _ladder_metrics(df, y)
    encodings = {
        "{1, 2}": y + 1,
        "strings": np.where(y == 1, "yes", "no"),
        "object strings": np.where(y == 1, "yes", "no").astype(object),
        "pandas series": pd.Series(np.where(y == 1, "b", "a")),
        "floats": y.astype(float) * 10,
    }
    for name, labels in encodings.items():
        got = _ladder_metrics(df, labels)
        for ref, m in zip(reference, got):
            assert m == pytest.approx(ref), name
            assert "roc_auc" in m  # still scored as a binary task


def test_ladder_metrics_do_not_depend_on_how_multiclass_labels_are_encoded(risk_table):
    df, _ = risk_table
    y = df["segment"].map({"retired": 0, "salaried": 1, "self_employed": 2}).values
    features = df.drop(columns=["segment"]).assign(
        hint=y + np.random.default_rng(1).normal(0, 0.7, len(y)))
    reference = _ladder_metrics(features, y)
    assert reference[0]["accuracy"] > 0.6 and "roc_auc" not in reference[0]
    for labels in (y * 10 + 5, df["segment"].values, np.array(["a", "b", "c"])[y]):
        for ref, m in zip(reference, _ladder_metrics(features, labels)):
            assert m == pytest.approx(ref)


def test_ladder_scores_correctly_when_the_integer_labels_have_a_gap(risk_table):
    # Labels {0, 2}: the model has two probability columns, and column 1 is
    # class 2. Reading it as "class 1" would score every positive as wrong.
    df, y = risk_table
    assert _ladder_metrics(df, y * 2, models=["logreg"])[0] == pytest.approx(
        _ladder_metrics(df, y, models=["logreg"])[0])


def test_ladder_refuses_a_test_split_with_a_class_the_training_split_lacks(risk_table):
    df, y = risk_table
    y_test = y[450:].copy()
    y_test[:3] = 2
    with pytest.raises(ValueError, match=r"\[2\].*absent from the training split"):
        run_ladder(df.iloc[:450], y[:450], df.iloc[450:], y_test, models=["logreg"])
    # the same with string labels, and the error names the class
    names = np.array(["no", "yes", "unknown"])
    with pytest.raises(ValueError, match="unknown"):
        run_ladder(df.iloc[:450], names[y[:450]], df.iloc[450:], names[y_test], models=["logreg"])
    # a class that is only in the training split is fine: it is just never the right answer
    y3 = y.copy()
    y3[:5] = 2
    assert len(run_ladder(df.iloc[:450], y3[:450], df.iloc[450:], y[450:], models=["logreg"])) == 1
    # one class in the training split is not a classification task
    with pytest.raises(ValueError, match="at least two"):
        run_ladder(df.iloc[:450], np.zeros(450, dtype=int), df.iloc[450:], np.zeros(150, dtype=int),
                   models=["logreg"])


def test_recommend_rules():
    def models(**kw):
        return [r["model"] for r in recommend(**kw)]

    default = models(n_rows=2000, n_features=20)
    assert default[:2] == ["logreg", "lightgbm"] and "tabicl" in default and "tabpfn-v2" in default
    assert "tabpfn-2.5+" not in default
    assert "tabpfn-v2" not in models(n_rows=2000, n_features=20, n_classes=25)
    assert "tabicl" not in models(n_rows=50_000, n_features=20, gpu=False)
    assert "tabicl" in models(n_rows=50_000, n_features=20, gpu=True)
    assert "tabicl" not in models(n_rows=2000, n_features=20, temporal_or_grouped=True)
    assert models(n_rows=2000, n_features=20, needs_inherent_explainability=True)[0] == "ebm"
    assert any("under 300 rows" in m for m in models(n_rows=120, n_features=8))
    assert FOUNDATION_MODEL_LICENCES["tabpfn-2.5+"]["commercial"] is False
    assert FOUNDATION_MODEL_LICENCES["tabicl"]["commercial"] is True


def test_cli_doctor_and_plan(capsys):
    assert main(["doctor", "--no-torch", "--json"]) == 0
    assert "device_class" in json.loads(capsys.readouterr().out)
    assert main(["plan", "--classes", "12", "--labels-per-class", "100", "--hardware", "apple-16gb",
                 "--deploy", "cpu-server", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hardware_class"] == "apple" and out["deploy_class"] == "cpu"
    assert main(["plan", "--task", "generation", "--model-size-b", "8", "--hardware", "amd-16gb"]) == 0
    assert "rocm" in capsys.readouterr().out


def test_cli_train_predict_round_trip(toy_text, tmp_path, capsys):
    texts, y, labels = toy_text
    csv = tmp_path / "data.csv"
    pd.DataFrame({"text": texts, "label": [labels[i] for i in y]}).to_csv(csv, index=False)
    out = tmp_path / "model"
    assert main(["train", "--data", str(csv), "--student", "tfidf", "--out", str(out), "--holdout", "0.4"]) == 0
    for name in ("student.json", "eval_contract.json", "run_record.json", "MODEL_CARD.md", "ml-bom.cdx.json"):
        assert (out / name).is_file(), name
    capsys.readouterr()
    assert main(["predict", "--model", str(out), "my card was blocked", "nominee on my policy"]) == 0
    lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert [l["label"] for l in lines] == ["card", "policy"]
    assert main(["predict", "--model", str(out), "--threshold", "1.1", "card"]) == 0
    assert json.loads(capsys.readouterr().out)["label"] == "ABSTAIN"


def test_cli_train_fails_the_contract_when_accuracy_is_too_low(toy_text, tmp_path):
    texts, y, labels = toy_text
    csv = tmp_path / "data.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame({"text": texts, "label": [labels[i] for i in rng.permutation(y)]}).to_csv(csv, index=False)
    code = main(["train", "--data", str(csv), "--student", "tfidf", "--out", str(tmp_path / "m"),
                 "--holdout", "0.4", "--min-accuracy", "0.99"])
    assert code == 1
    assert json.loads((tmp_path / "m" / "run_record.json").read_text())["contract"]["passed"] is False


def test_cli_tabular(risk_table, tmp_path, capsys):
    df, y = risk_table
    df = df.assign(default=y)
    csv = tmp_path / "risk.csv"
    df.to_csv(csv, index=False)
    assert main(["tabular", "--data", str(csv), "--target", "default", "--models", "logreg", "hist_gbdt"]) == 0
    out = capsys.readouterr().out
    assert "logreg" in out and "hist_gbdt" in out and "roc_auc" in out


def test_cli_tabular_split_is_stratified_so_a_rare_class_stays_in_training(risk_table, tmp_path, monkeypatch):
    import forgelm.tabular

    df, y = risk_table
    target = np.where(y == 1, "default", "repaid").astype(object)
    target[:2] = "fraud"  # two rows: a plain shuffle puts both in the test part about 1 time in 16
    target[2] = "dispute"  # a single row must go to training
    csv = tmp_path / "risk.csv"
    df.assign(outcome=target).to_csv(csv, index=False)
    splits = []

    def record_split(train_df, y_train, test_df, y_test, models, seed=0):
        assert len(train_df) == len(y_train) and len(test_df) == len(y_test)
        splits.append((np.asarray(y_train), np.asarray(y_test), train_df, test_df))
        return []

    monkeypatch.setattr(forgelm.tabular, "run_ladder", record_split)
    for seed in range(20):
        assert main(["tabular", "--data", str(csv), "--target", "outcome", "--models", "logreg",
                     "--seed", str(seed)]) == 0
    counts = pd.Series(target).value_counts()
    for y_train, y_test, train_df, test_df in splits:
        assert set(y_train) == set(target)  # every class has training rows, whatever the seed
        assert not set(y_test) - set(y_train)
        assert len(y_train) + len(y_test) == len(target)
        assert (y_train == "fraud").sum() == 1 and (y_test == "fraud").sum() == 1
        assert (y_test == "dispute").sum() == 0
        for cls in ("default", "repaid"):  # each class is cut at the same fraction
            assert (y_test == cls).sum() == pytest.approx(0.25 * counts[cls], abs=1)
        # rows and labels still belong together, and no row is in both parts
        both = pd.concat([train_df, test_df])
        assert not both.duplicated().any()
    assert not np.array_equal(splits[0][0], splits[1][0])  # the seed still changes the split
    assert len(set(splits[0][0][:40])) > 1  # training rows are not grouped by class


def test_cli_tabular_trains_on_a_rare_class_end_to_end(risk_table, tmp_path, capsys):
    df, y = risk_table
    target = np.where(y == 1, "default", "repaid").astype(object)
    target[:2] = "fraud"
    csv = tmp_path / "risk.csv"
    df.assign(outcome=target).to_csv(csv, index=False)
    assert main(["tabular", "--data", str(csv), "--target", "outcome", "--models", "logreg"]) == 0
    captured = capsys.readouterr()
    assert "logreg" in captured.out and "accuracy" in captured.out and "stratified" in captured.out


def test_cli_tabular_reports_an_unusable_split_without_a_traceback(risk_table, tmp_path, capsys):
    df, _ = risk_table
    csv = tmp_path / "one_class.csv"
    df.assign(outcome="repaid").to_csv(csv, index=False)
    assert main(["tabular", "--data", str(csv), "--target", "outcome", "--models", "logreg"]) == 2
    captured = capsys.readouterr()
    assert "repaid" in captured.err and "at least two" in captured.err
    assert "Traceback" not in captured.err and captured.out == ""
