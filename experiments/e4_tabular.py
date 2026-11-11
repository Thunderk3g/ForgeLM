"""E4 - tabular: classical baselines against an in-context foundation model.

Question: on BFSI-style risk tables, at what training-set size does each model
family win, how well calibrated is it, and what does it cost at inference?

Datasets (binary targets)
  * credit default: scikit-learn/credit-card-clients (UCI Taiwan credit, 30k rows, 23 numeric features)
  * census income: scikit-learn/adult-census-income (32.5k rows, numeric + categorical)

Protocol
  * A fixed random 5,000-row test set per dataset (not stratified).
  * Training sets of 100, 500, 2,000 and 10,000 rows, 3 seeds each.
  * Default hyper-parameters throughout (no search), so this compares
    out-of-the-box behaviour. A tuned GBDT would score somewhat higher.
  * TabICL runs on CPU; it is skipped at 10,000 rows if --tabicl-max is lower.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import hub_file, write_result
from forgelm.tabular import run_ladder

parser = argparse.ArgumentParser()
parser.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 2000, 10000])
parser.add_argument("--seeds", type=int, default=3)
parser.add_argument("--tabicl-max", type=int, default=10000)
args = parser.parse_args()


def load(name):
    if name == "credit_default":
        df = pd.read_csv(hub_file("scikit-learn/credit-card-clients", "UCI_Credit_Card.csv"))
        y = df["default.payment.next.month"].values
        X = df.drop(columns=["ID", "default.payment.next.month"])
        for col in ("SEX", "EDUCATION", "MARRIAGE"):
            X[col] = X[col].astype(str)
        return X, y
    df = pd.read_csv(hub_file("scikit-learn/adult-census-income", "adult.csv"))
    y = (df["income"].str.strip() == ">50K").astype(int).values
    return df.drop(columns=["income"]), y


rows = []
for dataset in ("credit_default", "census_income"):
    X, y = load(dataset)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    test_idx, rest = perm[:5000], perm[5000:]
    X_test, y_test = X.iloc[test_idx].reset_index(drop=True), y[test_idx]
    for n in args.sizes:
        for seed in range(args.seeds):
            pick = np.random.default_rng(100 + seed).choice(rest, size=n, replace=False)
            models = ["logreg", "hist_gbdt", "lightgbm", "ebm"] + (["tabicl"] if n <= args.tabicl_max else [])
            for r in run_ladder(X.iloc[pick].reset_index(drop=True), y[pick], X_test, y_test, models, seed=seed):
                rows.append({"dataset": dataset, "seed": seed, **r.to_dict()})
                print({k: rows[-1][k] for k in ("dataset", "model", "n_train", "seed", "roc_auc", "brier", "ece",
                                                "fit_seconds", "predict_seconds")}, flush=True)

frame = pd.DataFrame(rows)
summary = (frame.groupby(["dataset", "n_train", "model"])
           .agg(roc_auc=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"), brier=("brier", "mean"),
                ece=("ece", "mean"), fit_seconds=("fit_seconds", "mean"), predict_seconds=("predict_seconds", "mean"))
           .round(4).reset_index().to_dict(orient="records"))
write_result("e4_tabular", {
    "experiment": "E4 tabular ladder: logistic regression, GBDT, EBM, TabICL",
    "datasets": {"credit_default": "scikit-learn/credit-card-clients (UCI default of credit card clients)",
                 "census_income": "scikit-learn/adult-census-income (UCI Adult)"},
    "test_rows": 5000, "seeds": args.seeds, "hyperparameters": "library defaults, no search",
    "tabicl": {"checkpoint": "jingang/TabICL-clf (default of the tabicl package)", "licence": "BSD-3-Clause",
               "device": "cpu"},
    "summary": summary, "rows": rows,
})
