"""Command line entry point: ``python -m forgelm`` or ``forgelm``.

    forgelm doctor                      what hardware and backends are here
    forgelm plan ...                    what to build for a task, and why
    forgelm train ...                   train an LLM-free text classifier from a CSV
    forgelm predict ...                 classify text with a trained student
    forgelm tabular ...                 benchmark tabular models on a CSV
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .hardware import REFERENCE_PROFILES, probe, reference_profile


def _hardware(name: Optional[str]):
    return reference_profile(name) if name else probe()


def cmd_doctor(args) -> int:
    hw = probe(use_torch=not args.no_torch)
    if args.json:
        print(json.dumps(hw.to_dict(), indent=2))
        return 0
    print(hw.summary())
    print()
    print("Installed backends:")
    groups = {
        "LLM fine-tuning": ["torch", "transformers", "peft", "trl", "bitsandbytes", "unsloth", "mlx_lm"],
        "LLM-free classifiers": ["sklearn", "model2vec", "sentence_transformers", "setfit"],
        "Tabular": ["lightgbm", "xgboost", "catboost", "interpret", "tabicl", "tabpfn"],
        "Serving / export": ["onnx", "onnxruntime", "onnxruntime_gpu", "openvino", "llama_cpp", "vllm"],
    }
    for title, names in groups.items():
        have = [f"{n} {hw.packages[n]}" for n in names if n in hw.packages]
        miss = [n for n in names if n not in hw.packages]
        print(f"  {title:21}: {', '.join(have) or '-'}")
        if miss:
            print(f"  {'':21}  not installed: {', '.join(miss)}")
    from .generative import capacity

    cap = capacity(hw)
    print()
    print(f"Generative fine-tuning : backend {cap['backend']} on {cap['device']}; up to about "
          f"{cap['max_params_billion']}B parameters ({cap['basis']}; {cap['usable_memory_gb']:.1f} GiB usable)")
    if cap["install_hint"]:
        print(f"                         to enable: pip install {' '.join(cap['install_hint'])}")
    print("Text classification    : every LLM-free student trains and serves on this machine "
          "(see `forgelm plan`)")
    return 0


def cmd_plan(args) -> int:
    from .planner import Constraints, TaskSpec, plan

    task = TaskSpec(kind=args.task, n_classes=args.classes, labelled_per_class=args.labels_per_class,
                    n_unlabelled=args.unlabelled, avg_tokens=args.avg_tokens, multilingual=args.multilingual,
                    needs_world_knowledge=args.needs_world_knowledge, n_rows=args.rows, n_features=args.features,
                    has_text_columns=args.text_columns, temporal_or_grouped=args.temporal,
                    model_params_billion=args.model_size_b, knowledge_changes_often=args.knowledge_changes)
    constraints = Constraints(llm_at_inference=args.llm_at_inference, teacher_available=args.teacher,
                              commercial_use=not args.non_commercial,
                              needs_inherent_explainability=args.explainable,
                              data_must_stay_on_prem=args.on_prem, max_latency_ms=args.max_latency_ms,
                              deploy=reference_profile(args.deploy) if args.deploy else None)
    result = plan(task, _hardware(args.hardware), constraints)
    print(json.dumps(result.to_dict(), indent=2, default=str) if args.json else result.render())
    return 0


def cmd_train(args) -> int:
    import numpy as np
    import pandas as pd

    from .metrics import classification_report
    from .provenance import EvalContract, RunRecord, hash_directory, sha256_dataset
    from .students import make_student

    df = pd.read_csv(args.data)
    for col in (args.text_col, args.label_col):
        if col not in df.columns:
            print(f"column {col!r} not in {list(df.columns)}", file=sys.stderr)
            return 2
    df = df.dropna(subset=[args.text_col, args.label_col])
    labels = sorted(df[args.label_col].astype(str).unique())
    index = {l: i for i, l in enumerate(labels)}
    y = df[args.label_col].astype(str).map(index).values
    texts = df[args.text_col].astype(str).tolist()

    rng = np.random.default_rng(args.seed)
    holdout: List[int] = []
    for c in range(len(labels)):
        members = rng.permutation(np.where(y == c)[0])
        holdout.extend(members[: max(1, int(round(len(members) * args.holdout)))])
    holdout_set = set(holdout)
    train_idx = [i for i in range(len(texts)) if i not in holdout_set]
    hold_idx = sorted(holdout_set)
    # Half of the hold-out calibrates confidence, half is the untouched test.
    cal_idx, test_idx = hold_idx[::2], hold_idx[1::2]
    if not test_idx or not train_idx:
        print("not enough data to hold out a test set; need at least 4 examples per class", file=sys.stderr)
        return 2

    kind = args.student
    if kind == "auto":
        per_class = len(train_idx) / len(labels)
        kind = "encoder" if per_class < 64 else "static"
        print(f"auto: {per_class:.0f} training labels per class -> '{kind}' student "
              "(run `forgelm plan` for the full ladder)")
    student = make_student(kind, **({"model": args.model} if args.model else {}))
    tr_texts = [texts[i] for i in train_idx]
    student.fit(tr_texts, y[train_idx], labels)
    student.calibrate([texts[i] for i in cal_idx], y[cal_idx])
    te_texts = [texts[i] for i in test_idx]
    metrics = classification_report(y[test_idx], student.predict_proba(te_texts))
    student.save(args.out)

    contract = EvalContract.create(f"{Path(args.out).name}-holdout", te_texts, y[test_idx].tolist(),
                                   {"accuracy": args.min_accuracy} if args.min_accuracy else {})
    contract.save(str(Path(args.out) / "eval_contract.json"))
    record = RunRecord(
        name=Path(args.out).name, task="text classification",
        recipe={"student": kind, "model": args.model, "seed": args.seed, "holdout": args.holdout,
                "temperature": student.temperature},
        data={"train": {"n": len(train_idx), "sha256": sha256_dataset(tr_texts, y[train_idx].tolist())},
              "calibration": {"n": len(cal_idx)},
              "test": {"n": len(test_idx), "sha256": contract.dataset_sha256}},
        metrics=metrics, hardware=probe(use_torch=False).to_dict(), artefacts=hash_directory(args.out),
        contract=contract.check(metrics),
        notes=[f"labels: {labels}"])
    record.write_all(args.out)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}, indent=2))
    print(f"saved to {args.out} (student, eval contract, run record, model card, ML-BOM)")
    return 0 if record.contract["passed"] else 1


def cmd_predict(args) -> int:
    from .students import load_student

    student = load_student(args.model)
    texts = args.text or [line.rstrip("\n") for line in sys.stdin if line.strip()]
    probs = student.predict_proba(texts)
    for text, row in zip(texts, probs):
        top = row.argmax()
        verdict = student.labels[top] if row[top] >= args.threshold else "ABSTAIN"
        print(json.dumps({"text": text, "label": verdict, "confidence": round(float(row[top]), 4)}))
    return 0


def cmd_tabular(args) -> int:
    import numpy as np
    import pandas as pd

    from .tabular import AVAILABLE, SplitError, run_ladder

    df = pd.read_csv(args.data)
    y_raw = df[args.target]
    classes = sorted(y_raw.astype(str).unique())
    y = y_raw.astype(str).map({c: i for i, c in enumerate(classes)}).values
    X = df.drop(columns=[args.target])
    # Split each class separately. One shuffle of all rows can put every row
    # of a rare class in the test part, and a model cannot be scored on a
    # class it was never trained on. Each class keeps at least one training
    # row; a class with a single row is therefore trained on but not tested.
    rng = np.random.default_rng(args.seed)
    tr_parts, te_parts = [], []
    for c in range(len(classes)):
        rows = rng.permutation(np.flatnonzero(y == c))
        cut = max(1, int(len(rows) * (1 - args.test_fraction)))
        tr_parts.append(rows[:cut])
        te_parts.append(rows[cut:])
    # Shuffle again so the rows are not grouped by class.
    tr, te = rng.permutation(np.concatenate(tr_parts)), rng.permutation(np.concatenate(te_parts))
    models = args.models or list(AVAILABLE)
    # Hand over the class names, not their positions, so that an error from
    # run_ladder names the values that are in the CSV. It encodes them in the
    # same sorted order as ``classes``.
    names = np.asarray(classes)
    try:
        results = run_ladder(X.iloc[tr].reset_index(drop=True), names[y[tr]],
                             X.iloc[te].reset_index(drop=True), names[y[te]], models, seed=args.seed)
    except SplitError as exc:
        print(f"cannot benchmark {args.data}: {exc}", file=sys.stderr)
        return 2
    key = "roc_auc" if len(classes) == 2 else "accuracy"
    print(f"{'model':12} {key:>9} {'brier':>8} {'ece':>8} {'fit s':>8} {'predict s':>10}")
    for r in sorted(results, key=lambda r: -r.metrics[key]):
        m = r.metrics
        print(f"{r.model:12} {m[key]:9.4f} {m['brier']:8.4f} {m['ece']:8.4f} {r.fit_seconds:8.2f} "
              f"{r.predict_seconds:10.2f}")
    print("random split, stratified by class; if rows are ordered in time or grouped by customer, split that way "
          "instead")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forgelm", description="ForgeLM model intelligence factory")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="report hardware and installed backends")
    d.add_argument("--json", action="store_true")
    d.add_argument("--no-torch", action="store_true", help="do not import torch (faster)")
    d.set_defaults(func=cmd_doctor)

    pl = sub.add_parser("plan", help="recommend what to build for a task")
    pl.add_argument("--task", default="text_classification",
                    choices=["text_classification", "tabular_classification", "generation", "document_qa"])
    pl.add_argument("--classes", type=int, default=2)
    pl.add_argument("--labels-per-class", type=float, default=0)
    pl.add_argument("--unlabelled", type=int, default=0)
    pl.add_argument("--avg-tokens", type=int, default=32)
    pl.add_argument("--multilingual", action="store_true")
    pl.add_argument("--needs-world-knowledge", action="store_true")
    pl.add_argument("--rows", type=int, default=0)
    pl.add_argument("--features", type=int, default=0)
    pl.add_argument("--text-columns", action="store_true")
    pl.add_argument("--temporal", action="store_true", help="data is ordered in time or grouped")
    pl.add_argument("--model-size-b", type=float, default=3.0, help="generative model size, billions of parameters")
    pl.add_argument("--knowledge-changes", action="store_true")
    pl.add_argument("--llm-at-inference", action="store_true", help="the deployed system may call an LLM")
    pl.add_argument("--teacher", action="store_true", help="an LLM is available at build time")
    pl.add_argument("--non-commercial", action="store_true")
    pl.add_argument("--explainable", action="store_true", help="each decision needs inherent explanations")
    pl.add_argument("--on-prem", action="store_true", help="data may not leave the premises")
    pl.add_argument("--max-latency-ms", type=float)
    pl.add_argument("--hardware", choices=REFERENCE_PROFILES, help="plan for a reference machine instead of this one")
    pl.add_argument("--deploy", choices=REFERENCE_PROFILES, help="deployment machine, if different")
    pl.add_argument("--json", action="store_true")
    pl.set_defaults(func=cmd_plan)

    t = sub.add_parser("train", help="train an LLM-free text classifier from a CSV")
    t.add_argument("--data", required=True)
    t.add_argument("--text-col", default="text")
    t.add_argument("--label-col", default="label")
    t.add_argument("--student", default="auto",
                   choices=["auto", "tfidf", "static", "encoder", "finetuned"])
    t.add_argument("--model", help="override the embedding / encoder model")
    t.add_argument("--out", required=True)
    t.add_argument("--holdout", type=float, default=0.2, help="fraction held out (half calibration, half test)")
    t.add_argument("--min-accuracy", type=float, help="fail (exit 1) if held-out accuracy is below this")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="classify text with a trained student")
    pr.add_argument("--model", required=True)
    pr.add_argument("--threshold", type=float, default=0.0, help="abstain below this confidence")
    pr.add_argument("text", nargs="*", help="texts to classify (default: one per line on stdin)")
    pr.set_defaults(func=cmd_predict)

    tb = sub.add_parser("tabular", help="benchmark tabular models on a CSV")
    tb.add_argument("--data", required=True)
    tb.add_argument("--target", required=True)
    tb.add_argument("--models", nargs="*")
    tb.add_argument("--test-fraction", type=float, default=0.25)
    tb.add_argument("--seed", type=int, default=0)
    tb.set_defaults(func=cmd_tabular)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
