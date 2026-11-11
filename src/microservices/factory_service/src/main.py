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
import math
import os
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from forgelm import __version__
from forgelm.generative import PEFT_METHODS, capacity, lora_config, select_backend
from forgelm.hardware import REFERENCE_PROFILES, probe, reference_profile
from forgelm.packs import bfsi
from forgelm.planner import TASK_KINDS, Constraints, TaskSpec, plan
from forgelm.provenance import EvalContract, RunRecord, hash_directory, sha256_dataset

MODELS_DIR = Path(os.environ.get("FACTORY_MODELS_DIR", "/app/models/factory"))
MAX_TRAIN_ROWS = int(os.environ.get("FACTORY_MAX_TRAIN_ROWS", "50000"))
# Serving limits: a prediction request is answered while the caller waits, so
# both the number of texts and the length of each one are bounded.
MAX_PREDICT_ROWS = int(os.environ.get("FACTORY_MAX_PREDICT_ROWS", "10000"))
MAX_TEXT_CHARS = int(os.environ.get("FACTORY_MAX_TEXT_CHARS", "20000"))
# The "model" field of a training request is handed to the student, which
# downloads a Hugging Face repo or opens a local path of that name. Only the
# repo ids listed here are accepted, so a request cannot make the service fetch
# arbitrary code-bearing repos or read arbitrary directories. The default is the
# set of models the ladder already uses.
_DEFAULT_ALLOWED_MODELS = ("minishlab/potion-base-32M,minishlab/potion-base-8M,BAAI/bge-small-en-v1.5,"
                           "sentence-transformers/all-MiniLM-L6-v2,intfloat/multilingual-e5-small,"
                           "answerdotai/ModernBERT-base")
ALLOWED_MODELS = tuple(m.strip() for m in
                       (os.environ.get("FACTORY_ALLOWED_MODELS") or _DEFAULT_ALLOWED_MODELS).split(",") if m.strip())
PREFIX = "/api/v1/factory"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")

app = FastAPI(title="ForgeLM Factory Service", version=__version__)
_loaded: Dict[str, object] = {}
_lock = threading.Lock()
# One lock per classifier name, created on first use under ``_lock``. Two
# requests that train the same name run one after the other; different names
# still train in parallel.
_train_locks: Dict[str, threading.Lock] = {}


def _json_safe(value):
    """Replace NaN and infinities, which a JSON response cannot carry, by their names."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def _validation_error(request, exc):
    # Python's json parser accepts NaN and Infinity in a request body. The
    # stock handler echoes the rejected value back, cannot encode it, and turns
    # what should be a 422 into a 500. Same response shape, with those values
    # written as text.
    return JSONResponse(status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))})


def _model_dir(name: str) -> Path:
    # Names become directory names, so they are restricted to a safe alphabet.
    # fullmatch, not match with "$": "$" also matches just before a trailing
    # newline, which would let "abc\n" through as a directory name.
    if not _NAME.fullmatch(name):
        raise HTTPException(422, "name must be 1-64 characters: letters, digits, '-' or '_'")
    return MODELS_DIR / name


def _n_holdout(n: int, holdout: float) -> int:
    """How many of a class's ``n`` examples go to calibration and test."""
    return max(2, int(round(n * holdout)))


def _check_batch(texts: List[str], what: str) -> None:
    if len(texts) > MAX_PREDICT_ROWS:
        raise HTTPException(413, f"at most {MAX_PREDICT_ROWS} {what} per request")
    if any(len(t) > MAX_TEXT_CHARS for t in texts):
        raise HTTPException(413, f"each of the {what} may be at most {MAX_TEXT_CHARS} characters long")


def _hardware(reference: Optional[str]):
    if reference is None:
        return probe(use_torch=False)
    if reference not in REFERENCE_PROFILES:
        raise HTTPException(422, f"unknown reference hardware; choose from {list(REFERENCE_PROFILES)}")
    return reference_profile(reference)


# ---- schemas -------------------------------------------------------------

class PlanRequest(BaseModel):
    task: str = "text_classification"
    # The bounds below keep nonsense (zero or negative sizes, infinities) out of
    # the planner's arithmetic, where it would surface as an HTTP 500.
    n_classes: int = Field(2, ge=1)
    labelled_per_class: float = Field(0, ge=0, allow_inf_nan=False)
    n_unlabelled: int = Field(0, ge=0)
    avg_tokens: int = Field(32, ge=1)
    multilingual: bool = False
    needs_world_knowledge: bool = False
    n_rows: int = Field(0, ge=0)
    n_features: int = Field(0, ge=0)
    has_text_columns: bool = False
    temporal_or_grouped: bool = False
    model_params_billion: float = Field(3.0, gt=0, le=2000)
    knowledge_changes_often: bool = False
    llm_at_inference: bool = False
    teacher_available: bool = False
    commercial_use: bool = True
    needs_inherent_explainability: bool = False
    data_must_stay_on_prem: bool = False
    max_latency_ms: Optional[float] = Field(None, gt=0, allow_inf_nan=False)
    hardware: Optional[str] = Field(None, description="reference profile to plan for instead of this machine")
    deploy: Optional[str] = Field(None, description="reference profile of the deployment machine")


class TrainRequest(BaseModel):
    name: str
    texts: List[str]
    labels: List[str]
    student: str = Field("static", description="tfidf | static | encoder | finetuned")
    model: Optional[str] = Field(None, description="one of the repo ids in FACTORY_ALLOWED_MODELS")
    holdout: float = Field(0.2, gt=0.0, lt=0.9)
    min_accuracy: Optional[float] = None
    seed: int = Field(0, ge=0)  # numpy refuses a negative seed


class PredictRequest(BaseModel):
    texts: List[str]
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="abstain below this confidence")


class ReviewCallRequest(BaseModel):
    utterances: List[str]
    classifier: Optional[str] = Field(None, description="name of a trained classifier using the BFSI taxonomy")
    flag_threshold: float = Field(0.6, ge=0.0, le=1.0)
    unsure_below: float = Field(0.5, ge=0.0, le=1.0)


class GenerativeAdviceRequest(BaseModel):
    params_billion: float = Field(3.0, gt=0, le=2000)
    batch_size: int = Field(2, ge=1, le=4096)
    seq_len: int = Field(1024, ge=1, le=1_000_000)
    peft_method: str = "lora"
    rank: int = Field(16, ge=1, le=1024)
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
    if req.model is not None:
        if req.model not in ALLOWED_MODELS:
            raise HTTPException(422, "model is not in the allow-list set by the FACTORY_ALLOWED_MODELS "
                                     f"environment variable; choose from {list(ALLOWED_MODELS)}")
        if req.student == "tfidf":
            raise HTTPException(422, "the tfidf student has no base model; leave 'model' out")
    labels = sorted(set(req.labels))
    if len(labels) < 2:
        raise HTTPException(422, "need at least two distinct labels")
    index = {l: i for i, l in enumerate(labels)}
    y = np.array([index[l] for l in req.labels])
    rng = np.random.default_rng(req.seed)
    # Each class gives up max(2, round(n * holdout)) examples for calibration
    # and test and must keep at least one for training. With a hold-out share
    # near 0.9 a small class can lose every example (4 examples at 0.89 hold
    # out all 4), so say up front how many examples this share needs.
    needed = next(n for n in range(4, 100) if n - _n_holdout(n, req.holdout) >= 1)
    hold: List[int] = []
    for c in range(len(labels)):
        members = rng.permutation(np.where(y == c)[0])
        if len(members) < needed:
            raise HTTPException(422, f"label {labels[c]!r} has {len(members)} examples; at least {needed} per "
                                     f"class are needed with holdout={req.holdout} to hold out calibration and "
                                     "test data and still leave one to train on")
        hold.extend(members[: _n_holdout(len(members), req.holdout)])
    hold = sorted(hold)
    hold_set = set(hold)
    train_idx = [i for i in range(len(y)) if i not in hold_set]
    cal_idx, test_idx = hold[::2], hold[1::2]
    if not (train_idx and cal_idx and test_idx):
        raise HTTPException(422, f"the split left no training, calibration or test examples; at least {needed} "
                                 f"examples per class are needed with holdout={req.holdout}")

    with _lock:
        name_lock = _train_locks.setdefault(req.name, threading.Lock())
    # This endpoint is a plain ``def``, so FastAPI runs it in a thread pool and
    # two requests for the same name can be here at once. They take turns.
    with name_lock:
        # Everything is written to a hidden sibling directory and moved into
        # place only when it is complete. A failed or interrupted run therefore
        # leaves the previous model untouched, and a retrain with a different
        # student kind cannot leave the old kind's files next to the new ones
        # (hash_directory would otherwise list them in the run record).
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex[:12]
        tmp = MODELS_DIR / f".{req.name}.tmp-{token}"
        old = MODELS_DIR / f".{req.name}.old-{token}"
        try:
            student = make_student(req.student, **({"model": req.model} if req.model else {}))
            tr_texts = [req.texts[i] for i in train_idx]
            te_texts = [req.texts[i] for i in test_idx]
            student.fit(tr_texts, y[train_idx], labels)
            student.calibrate([req.texts[i] for i in cal_idx], y[cal_idx])
            metrics = classification_report(y[test_idx], student.predict_proba(te_texts))
            student.save(str(tmp))

            contract = EvalContract.create(f"{req.name}-holdout", te_texts, y[test_idx].tolist(),
                                           {"accuracy": req.min_accuracy} if req.min_accuracy else {})
            contract.save(str(tmp / "eval_contract.json"))
            verdict = contract.check(metrics)
            RunRecord(name=req.name, task="text classification",
                      recipe={"student": req.student, "model": req.model, "seed": req.seed, "holdout": req.holdout,
                              "temperature": student.temperature},
                      data={"train": {"n": len(train_idx), "sha256": sha256_dataset(tr_texts, y[train_idx].tolist())},
                            "calibration": {"n": len(cal_idx)},
                            "test": {"n": len(test_idx), "sha256": contract.dataset_sha256}},
                      metrics=metrics, hardware=probe(use_torch=False).to_dict(),
                      artefacts=hash_directory(str(tmp)), contract=verdict).write_all(str(tmp))

            # The swap and the cache update happen under the module lock, which
            # the read endpoints also take, so a reader sees either the old
            # model or the new one and never a missing or half-moved directory.
            # The old directory is moved aside first (a rename, so it can be put
            # back if the second rename fails) and deleted afterwards.
            with _lock:
                had_old = out.exists()
                if had_old:
                    os.replace(out, old)
                try:
                    os.replace(tmp, out)
                except BaseException:
                    if had_old:
                        os.replace(old, out)
                    raise
                _loaded[req.name] = student
            shutil.rmtree(old, ignore_errors=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)  # only still there if something failed
    return {"name": req.name, "labels": labels, "n_train": len(train_idx), "n_calibration": len(cal_idx),
            "n_test": len(test_idx), "metrics": metrics, "temperature": student.temperature,
            "contract": verdict, "serve_requires": list(student.serve_requires)}


@app.get(f"{PREFIX}/classifiers")
def list_classifiers():
    out = []
    if MODELS_DIR.is_dir():
        for d in sorted(MODELS_DIR.iterdir()):
            # Hidden directories are trainings in progress (or their leftovers),
            # not finished classifiers.
            if d.name.startswith("."):
                continue
            with _lock:  # not while a retrain is swapping this directory
                record = d / "run_record.json"
                if not record.is_file():
                    continue
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
    with _lock:  # so the record and the card come from the same training run
        if not (directory / "run_record.json").is_file():
            raise HTTPException(404, f"no classifier named {name!r}")
        return {"run_record": json.loads((directory / "run_record.json").read_text()),
                "model_card": (directory / "MODEL_CARD.md").read_text()}


@app.post(f"{PREFIX}/classifiers/{{name}}/predict")
def predict(name: str, req: PredictRequest):
    _check_batch(req.texts, "texts")
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
    _check_batch(req.utterances, "utterances")
    classifier = _get_student(req.classifier) if req.classifier else None
    review = bfsi.review_call(req.utterances, classifier, req.flag_threshold, req.unsure_below)
    return dict(review.to_dict(), disclaimer="flags are prompts for human review, not findings of a violation")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("SERVICE_HOST", "0.0.0.0"), port=int(os.environ.get("SERVICE_PORT", "8006")))
