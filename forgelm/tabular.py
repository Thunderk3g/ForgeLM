"""Tabular models: classical baselines and in-context foundation models.

One fit/predict surface over logistic regression, gradient boosting, an
explainable boosting machine and a tabular foundation model, so they can be
compared on the same split with the same metrics.

Two things are built in rather than left to the caller:

* a licence registry. Several of the strongest tabular foundation models ship
  weights that may not be used commercially. The planner consults
  ``FOUNDATION_MODEL_LICENCES`` before recommending one.
* the baselines always run. A foundation model that has not been compared with
  a tuned GBDT and a linear model on *your* split is not evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .metrics import binary_report, classification_report

# Weights licence as read from each project's repository in November 2026.
# "commercial" False means the published weights may not be used in production
# or for commercial evaluation without a separate licence. Re-check before use.
FOUNDATION_MODEL_LICENCES: Dict[str, dict] = {
    "tabicl": {"commercial": True, "licence": "BSD-3-Clause", "max_rows": 100_000, "max_features": 100,
               "note": "benchmarked from 300 rows up; untested below that"},
    "tabpfn-v2": {"commercial": True, "licence": "Prior Labs License (Apache-2.0 + attribution)",
                  "max_rows": 10_000, "max_features": 500, "max_classes": 10},
    "tabpfn-2.5+": {"commercial": False, "licence": "non-commercial", "max_rows": 100_000, "max_features": 2_000,
                    "note": "covers 2.5, 2.6, 3 and 3.5; the default checkpoint of the tabpfn package"},
    "tabdpt": {"commercial": True, "licence": "Apache-2.0", "max_rows": 100_000, "max_classes": 160},
    "mitra": {"commercial": True, "licence": "Apache-2.0", "max_rows": 10_000, "max_features": 500,
              "max_classes": 10},
    "tabfm": {"commercial": False, "licence": "non-commercial"},
    "limix-2": {"commercial": False, "licence": "non-commercial"},
    "sap-rpt-1-oss": {"commercial": False, "licence": "research only"},
}


@dataclass
class TabularResult:
    model: str
    metrics: Dict[str, float]
    fit_seconds: float
    predict_seconds: float
    n_train: int
    note: str = ""

    def to_dict(self) -> dict:
        return {"model": self.model, "n_train": self.n_train, "fit_seconds": round(self.fit_seconds, 3),
                "predict_seconds": round(self.predict_seconds, 3), "note": self.note, **self.metrics}


def _split_columns(df):
    cat = [c for c in df.columns if str(df[c].dtype) in ("object", "category", "bool", "str", "string")]
    num = [c for c in df.columns if c not in cat]
    return num, cat


def make_model(name: str, df, seed: int = 0):
    """Build an unfitted scikit-learn-compatible classifier for ``df``'s schema."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    num, cat = _split_columns(df)
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), cat),
        ])
        return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=1.0))])

    ordinal = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), cat),
    ])
    if name == "hist_gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        cat_idx = list(range(len(num), len(num) + len(cat)))
        ordinal_nan = ColumnTransformer([
            ("num", "passthrough", num),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                   encoded_missing_value=np.nan, max_categories=250), cat),
        ])
        clf = HistGradientBoostingClassifier(categorical_features=cat_idx or None, early_stopping=True,
                                             random_state=seed, max_iter=500, learning_rate=0.06)
        return Pipeline([("pre", ordinal_nan), ("clf", clf)])
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, min_child_samples=10,
                             random_state=seed, verbose=-1)
        return Pipeline([("pre", ordinal), ("clf", clf)])
    if name == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier

        return Pipeline([("pre", ordinal), ("clf", ExplainableBoostingClassifier(random_state=seed, interactions=5))])
    if name == "tabicl":
        from tabicl import TabICLClassifier

        return Pipeline([("pre", ordinal), ("clf", TabICLClassifier(random_state=seed))])
    raise KeyError(f"unknown tabular model {name!r}")


AVAILABLE = ("logreg", "hist_gbdt", "lightgbm", "ebm", "tabicl")
_PACKAGE = {"logreg": "sklearn", "hist_gbdt": "sklearn", "lightgbm": "lightgbm", "ebm": "interpret",
            "tabicl": "tabicl"}


class SplitError(ValueError):
    """The train/test split cannot be scored; the caller has to re-split."""


def _encode_labels(y_train, y_test):
    """Recode labels to 0..k-1 by the sorted classes of the training split.

    Every model orders its ``predict_proba`` columns by the sorted classes it
    was fitted on. Fitting on 0..k-1 makes column ``i`` the probability of
    class ``i`` by construction, whatever the caller's encoding was ({1, 2},
    strings, or integers with a class missing from the training rows).
    """
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    classes = np.unique(y_train)  # sorted
    if len(classes) < 2:
        raise SplitError(f"the training split holds {len(classes)} class(es) {classes.tolist()}; "
                         "a classifier needs at least two")
    unseen = np.setdiff1d(np.unique(y_test), classes)
    if len(unseen):
        # A model has no probability column for a class it never saw, so every
        # such row would be scored as wrong against some other class's column.
        raise SplitError(f"the test split contains class(es) {unseen.tolist()} that are absent from the "
                         "training split; re-split so every class has training rows (stratify by class)")
    return np.searchsorted(classes, y_train), np.searchsorted(classes, y_test), classes


def run_ladder(train_df, y_train, test_df, y_test, models: Sequence[str] = AVAILABLE, seed: int = 0,
               skip_missing: bool = True) -> List[TabularResult]:
    """Fit each model on the same split and score it with the same metrics.

    Labels may use any encoding. With two classes the positive class is the
    one that sorts last (1 of {0, 1}, 2 of {1, 2}, "yes" of {"no", "yes"}).
    Raises :class:`SplitError` (a ``ValueError``) if the test split contains a
    class the training split does not.
    """
    import importlib.util

    results: List[TabularResult] = []
    y_train, y_test, classes = _encode_labels(y_train, y_test)
    binary = len(classes) == 2
    for name in models:
        module = {"sklearn": "sklearn", "lightgbm": "lightgbm", "interpret": "interpret", "tabicl": "tabicl"}[
            _PACKAGE[name]]
        if importlib.util.find_spec(module) is None:
            if skip_missing:
                continue
            raise ImportError(f"{name} needs the {module} package")
        model = make_model(name, train_df, seed)
        start = time.perf_counter()
        model.fit(train_df, y_train)
        fit_s = time.perf_counter() - start
        start = time.perf_counter()
        probs = np.asarray(model.predict_proba(test_df), dtype=np.float64)
        pred_s = time.perf_counter() - start
        if binary:
            metrics = binary_report(y_test, probs[:, 1])
        else:
            metrics = classification_report(y_test, probs)
        results.append(TabularResult(name, metrics, fit_s, pred_s, len(y_train)))
    return results


def recommend(n_rows: int, n_features: int, n_classes: int = 2, has_text_columns: bool = False,
              gpu: bool = False, commercial_use: bool = True, needs_inherent_explainability: bool = False,
              temporal_or_grouped: bool = False) -> List[dict]:
    """Ordered list of tabular model families to try, with the reason for each.

    Encodes the decision rule from docs/RESEARCH.md section 3. It decides
    what is worth *benchmarking*; it does not replace the benchmark.
    """
    out: List[dict] = []

    def add(model, role, why):
        out.append({"model": model, "role": role, "why": why})

    if needs_inherent_explainability:
        add("ebm", "champion", "additive and exactly explainable; required when each decision needs stated reasons")
        add("logreg", "baseline", "the reference a regulator already understands")
        add("lightgbm", "challenger", "accuracy ceiling to quantify what explainability costs")
    else:
        add("logreg", "baseline", "always fit; cheap, calibrated, exposes leakage when it scores too well")
        add("lightgbm", "baseline", "tuned GBDT is the bar every newer model must clear"
            + ("; untuned, it was badly calibrated with this little data (ECE 0.08-0.18 in E4), "
               "so tune it or calibrate it before reading its probabilities" if n_rows <= 2_000 else ""))
        add("ebm", "candidate",
            "at or within noise of the best model from 500 to 10,000 rows in E4 (at 100 rows it trailed by "
            "up to 0.04 AUC), well calibrated there, instant to score, and exactly explainable: often the "
            "model to ship")

    fm_ok = not temporal_or_grouped and n_rows <= 100_000 and n_features <= 500
    if temporal_or_grouped:
        add("(foundation models deprioritised)", "note",
            "published benchmarks show in-context models losing to GBDTs under temporal or grouped splits")
    if fm_ok:
        if gpu or n_rows <= 5_000:
            role = "challenger" if needs_inherent_explainability else "candidate"
            add("tabicl", role,
                "BSD-3 weights; in-context learner, no hyper-parameter search; best or tied from 100 to 2,000 "
                "rows in E4; it led an EBM by up to 0.033 AUC at 100 rows and by 0.01 or less from 500 rows up"
                + ("" if gpu else ", and scoring is slow on CPU (24-51 s per 5,000 rows in E4, against "
                                  "well under a second for the others)"))
            lim = FOUNDATION_MODEL_LICENCES["tabpfn-v2"]
            if n_rows <= lim["max_rows"] and n_features <= lim["max_features"] and n_classes <= lim["max_classes"]:
                add("tabpfn-v2", role, "permissive v2 weights only; pin the checkpoint, newer defaults are "
                                       "non-commercial")
        else:
            add("(tabicl needs a GPU at this size)", "note", "in-context inference cost grows with the training set")
    if n_rows < 300:
        add("(under 300 rows)", "note",
            "no model family is well evidenced here; use repeated cross-validation and recalibrate")
    if has_text_columns:
        add("text embeddings + lightgbm", "baseline", "embed free-text columns, append as features")
    if not commercial_use:
        add("tabpfn-2.5+", "candidate", "non-commercial weights; research or evaluation under licence only")
    else:
        noncommercial = [k for k, v in FOUNDATION_MODEL_LICENCES.items() if not v["commercial"]]
        add(f"(excluded: {', '.join(noncommercial)})", "note",
            "published weights are non-commercial; an enterprise licence is needed for production")
    return out
