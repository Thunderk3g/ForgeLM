"""ForgeLM Factory Service.

The HTTP face of the ``forgelm`` core: hardware probing, planning, LLM-free
classifier training and serving, tabular advice, generative-backend advice and
BFSI call screening.

Unlike the training service this one needs no GPU and no ML base image. Every
student it trains runs on a CPU, so it can be deployed on the same on-premises
application servers that will serve the models.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from forgelm import __version__
from forgelm.generative import PEFT_METHODS, capacity, lora_config, select_backend
from forgelm.hardware import REFERENCE_PROFILES, probe, reference_profile
from forgelm.packs import bfsi
from forgelm.planner import TASK_KINDS, Constraints, TaskSpec, plan
from forgelm.provenance import EvalContract, RunRecord, hash_directory, sha256_dataset

MODELS_DIR = Path(os.environ.get("FACTORY_MODELS_DIR", "/app/models/factory"))
MAX_TRAIN_ROWS = int(os.environ.get("FACTORY_MAX_TRAIN_ROWS", "50000"))
PREFIX = "/api/v1/factory"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

app = FastAPI(title="ForgeLM Factory Service", version=__version__)
_loaded: Dict[str, object] = {}
_lock = threading.Lock()


def _model_dir(name: str) -> Path:
    # Names become directory names, so they are restricted to a safe alphabet.
    if not _NAME.match(name):
        raise HTTPException(422, "name must be 1-64 characters: letters, digits, '-' or '_'")
    return MODELS_DIR / name


def _hardware(reference: Optional[str]):
    if reference is None:
        return probe(use_torch=False)
    if reference not in REFERENCE_PROFILES:
        raise HTTPException(422, f"unknown reference hardware; choose from {list(REFERENCE_PROFILES)}")
    return reference_profile(reference)


# ---- schemas -------------------------------------------------------------

class PlanRequest(BaseModel):
    task: str = "text_classification"
    n_classes: int = 2
    labelled_per_class: float = 0
    n_unlabelled: int = 0
    avg_tokens: int = 32
    multilingual: bool = False
    needs_world_knowledge: bool = False
    n_rows: int = 0
    n_features: int = 0
    has_text_columns: bool = False
    temporal_or_grouped: bool = False
    model_params_billion: float = 3.0
    knowledge_changes_often: bool = False
    llm_at_inference: bool = False
    teacher_available: bool = False
    commercial_use: bool = True
    needs_inherent_explainability: bool = False
    data_must_stay_on_prem: bool = False
    max_latency_ms: Optional[float] = None
    hardware: Optional[str] = Field(None, description="reference profile to plan for instead of this machine")
    deploy: Optional[str] = Field(None, description="reference profile of the deployment machine")


class TrainRequest(BaseModel):
    name: str
    texts: List[str]
    labels: List[str]
    student: str = Field("static", description="tfidf | static | encoder | finetuned")
    model: Optional[str] = None
    holdout: float = Field(0.2, gt=0.0, lt=0.9)
    min_accuracy: Optional[float] = None
    seed: int = 0


class PredictRequest(BaseModel):
    texts: List[str]
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="abstain below this confidence")


class ReviewCallRequest(BaseModel):
    utterances: List[str]
    classifier: Optional[str] = Field(None, description="name of a trained classifier using the BFSI taxonomy")
    flag_threshold: float = 0.6
    unsure_below: float = 0.5


class GenerativeAdviceRequest(BaseModel):
    params_billion: float = 3.0
    batch_size: int = 2
    seq_len: int = 1024
    peft_method: str = "lora"
    rank: int = 16
    hardware: Optional[str] = None


# ---- endpoints -----------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "service": "llmforge-factory", "version": __version__}


@app.get(f"{PREFIX}/hardware")
def hardware(reference: Optional[str] = None):
    hw = _hardware(reference)
    return {"profile": hw.to_dict(), "generative_capacity": capacity(hw),
            "reference_profiles": list(REFERENCE_PROFILES)}


@app.post(f"{PREFIX}/plan")
def make_plan(req: PlanRequest):
    if req.task not in TASK_KINDS:
        raise HTTPException(422, f"task must be one of {list(TASK_KINDS)}")
    task = TaskSpec(kind=req.task, n_classes=req.n_classes, labelled_per_class=req.labelled_per_class,
                    n_unlabelled=req.n_unlabelled, avg_tokens=req.avg_tokens, multilingual=req.multilingual,
                    needs_world_knowledge=req.needs_world_knowledge, n_rows=req.n_rows, n_features=req.n_features,
                    has_text_columns=req.has_text_columns, temporal_or_grouped=req.temporal_or_grouped,
                    model_params_billion=req.model_params_billion,
                    knowledge_changes_often=req.knowledge_changes_often)
    constraints = Constraints(llm_at_inference=req.llm_at_inference, teacher_available=req.teacher_available,
                              commercial_use=req.commercial_use,
                              needs_inherent_explainability=req.needs_inherent_explainability,
                              data_must_stay_on_prem=req.data_must_stay_on_prem, max_latency_ms=req.max_latency_ms,
                              deploy=_hardware(req.deploy) if req.deploy else None)
    result = plan(task, _hardware(req.hardware), constraints)
    return {"plan": json.loads(json.dumps(result.to_dict(), default=str)), "text": result.render()}


@app.get(f"{PREFIX}/usecases")
def usecases():
    return {"use_cases": bfsi.USE_CASES, "sales_call_taxonomy": bfsi.TAXONOMY,
            "disclaimer": "illustrative defaults; not a statement of regulatory requirements"}


@app.post(f"{PREFIX}/generative/advice")
def generative_advice(req: GenerativeAdviceRequest):
    if req.peft_method.lower() not in PEFT_METHODS:
        raise HTTPException(422, f"peft_method must be one of {sorted(PEFT_METHODS)}")
    hw = _hardware(req.hardware)
    name, cfg, notes = lora_config(req.peft_method, rank=req.rank)
    return {"backend": select_backend(hw, req.params_billion, req.batch_size, req.seq_len).to_dict(),
            "peft": {"config_class": name, "kwargs": cfg, "notes": notes}}


@app.post(f"{PREFIX}/classifiers")
def train_classifier(req: TrainRequest):
    """Train an LLM-free classifier synchronously.

    The linear-head students finish in seconds to a couple of minutes on a
    CPU. ``finetuned`` can take much longer; for large jobs use the CLI or a
    worker rather than holding an HTTP request open.
    """
    import numpy as np

    from forgelm.metrics import classification_report
    from forgelm.students import make_student

    out = _model_dir(req.name)
    if len(req.texts) != len(req.labels):
        raise HTTPException(422, "texts and labels must have the same length")
    if len(req.texts) > MAX_TRAIN_ROWS:
        raise HTTPException(413, f"at most {MAX_TRAIN_ROWS} rows per request")
    if req.student not in ("tfidf", "static", "encoder", "finetuned"):
        raise HTTPException(422, "student must be one of tfidf, static, encoder, finetuned")
    labels = sorted(set(req.labels))
    if len(labels) < 2:
        raise HTTPException(422, "need at least two distinct labels")
    index = {l: i for i, l in enumerate(labels)}
    y = np.array([index[l] for l in req.labels])
    rng = np.random.default_rng(req.seed)
    hold: List[int] = []
    for c in range(len(labels)):
        members = rng.permutation(np.where(y == c)[0])
        if len(members) < 4:
            raise HTTPException(422, f"label {labels[c]!r} has {len(members)} examples; at least 4 are needed "
                                     "to hold out calibration and test data")
        hold.extend(members[: max(2, int(round(len(members) * req.holdout)))])
    hold = sorted(hold)
    hold_set = set(hold)
    train_idx = [i for i in range(len(y)) if i not in hold_set]
    cal_idx, test_idx = hold[::2], hold[1::2]

    student = make_student(req.student, **({"model": req.model} if req.model else {}))
    tr_texts = [req.texts[i] for i in train_idx]
    te_texts = [req.texts[i] for i in test_idx]
    student.fit(tr_texts, y[train_idx], labels)
    student.calibrate([req.texts[i] for i in cal_idx], y[cal_idx])
    metrics = classification_report(y[test_idx], student.predict_proba(te_texts))
    student.save(str(out))

    contract = EvalContract.create(f"{req.name}-holdout", te_texts, y[test_idx].tolist(),
                                   {"accuracy": req.min_accuracy} if req.min_accuracy else {})
    contract.save(str(out / "eval_contract.json"))
    verdict = contract.check(metrics)
    RunRecord(name=req.name, task="text classification",
              recipe={"student": req.student, "model": req.model, "seed": req.seed, "holdout": req.holdout,
                      "temperature": student.temperature},
              data={"train": {"n": len(train_idx), "sha256": sha256_dataset(tr_texts, y[train_idx].tolist())},
                    "calibration": {"n": len(cal_idx)},
                    "test": {"n": len(test_idx), "sha256": contract.dataset_sha256}},
              metrics=metrics, hardware=probe(use_torch=False).to_dict(), artefacts=hash_directory(str(out)),
              contract=verdict).write_all(str(out))
    with _lock:
        _loaded[req.name] = student
    return {"name": req.name, "labels": labels, "n_train": len(train_idx), "n_calibration": len(cal_idx),
            "n_test": len(test_idx), "metrics": metrics, "temperature": student.temperature,
            "contract": verdict, "serve_requires": list(student.serve_requires)}


@app.get(f"{PREFIX}/classifiers")
def list_classifiers():
    out = []
    if MODELS_DIR.is_dir():
        for d in sorted(MODELS_DIR.iterdir()):
            record = d / "run_record.json"
            if record.is_file():
                data = json.loads(record.read_text())
                out.append({"name": d.name, "created_utc": data.get("created_utc"),
                            "student": data.get("recipe", {}).get("student"),
                            "accuracy": data.get("metrics", {}).get("accuracy"),
                            "contract_passed": (data.get("contract") or {}).get("passed")})
    return {"classifiers": out}


def _get_student(name: str):
    from forgelm.students import load_student

    directory = _model_dir(name)
    with _lock:
        if name not in _loaded:
            if not (directory / "student.json").is_file():
                raise HTTPException(404, f"no classifier named {name!r}")
            _loaded[name] = load_student(str(directory))
        return _loaded[name]


@app.get(f"{PREFIX}/classifiers/{{name}}")
def classifier_card(name: str):
    directory = _model_dir(name)
    if not (directory / "run_record.json").is_file():
        raise HTTPException(404, f"no classifier named {name!r}")
    return {"run_record": json.loads((directory / "run_record.json").read_text()),
            "model_card": (directory / "MODEL_CARD.md").read_text()}


@app.post(f"{PREFIX}/classifiers/{{name}}/predict")
def predict(name: str, req: PredictRequest):
    student = _get_student(name)
    if not req.texts:
        return {"predictions": []}
    probs = student.predict_proba(req.texts)
    out = []
    for row in probs:
        top = int(row.argmax())
        conf = float(row[top])
        out.append({"label": student.labels[top] if conf >= req.threshold else None,
                    "abstained": conf < req.threshold, "confidence": round(conf, 4)})
    return {"predictions": out}


@app.post(f"{PREFIX}/review-call")
def review_call(req: ReviewCallRequest):
    classifier = _get_student(req.classifier) if req.classifier else None
    review = bfsi.review_call(req.utterances, classifier, req.flag_threshold, req.unsure_below)
    return dict(review.to_dict(), disclaimer="flags are prompts for human review, not findings of a violation")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("SERVICE_HOST", "0.0.0.0"), port=int(os.environ.get("SERVICE_PORT", "8006")))
