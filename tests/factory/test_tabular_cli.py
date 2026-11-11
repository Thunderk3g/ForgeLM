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
