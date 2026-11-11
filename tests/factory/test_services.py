"""The factory microservice and the training-service bridge."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("FACTORY_MODELS_DIR", str(tmp_path / "models"))
    module = _load("factory_main", ROOT / "src/microservices/factory_service/src/main.py")
    return TestClient(module.app)


def test_health_hardware_and_usecases(client):
    assert client.get("/health").json()["service"] == "llmforge-factory"
    body = client.get("/api/v1/factory/hardware").json()
    assert body["profile"]["device_class"] and "max_params_billion" in body["generative_capacity"]
    ref = client.get("/api/v1/factory/hardware", params={"reference": "apple-16gb"}).json()
    assert ref["profile"]["device_class"] == "apple" and ref["generative_capacity"]["backend"] == "mlx_lm"
    assert client.get("/api/v1/factory/hardware", params={"reference": "nope"}).status_code == 422
    assert "sales_call_compliance" in client.get("/api/v1/factory/usecases").json()["use_cases"]


def test_plan_endpoint(client):
    r = client.post("/api/v1/factory/plan", json={"n_classes": 77, "labelled_per_class": 8, "hardware": "cpu-laptop"})
    assert r.status_code == 200
    body = r.json()
    assert body["plan"]["approach"] == "compact classifier" and "Candidates" in body["text"]
    assert client.post("/api/v1/factory/plan", json={"task": "poetry"}).status_code == 422
    gen = client.post("/api/v1/factory/plan", json={"task": "generation", "model_params_billion": 8,
                                                    "hardware": "amd-16gb", "deploy": "cpu-server"}).json()
    assert gen["plan"]["hardware_class"] == "amd" and gen["plan"]["deploy_class"] == "cpu"


def test_generative_advice(client):
    r = client.post("/api/v1/factory/generative/advice",
                    json={"params_billion": 8, "peft_method": "dora", "rank": 32, "hardware": "nvidia-24gb"})
    body = r.json()
    assert body["backend"]["backend"] == "unsloth" and body["peft"]["kwargs"]["use_dora"] is True
    assert client.post("/api/v1/factory/generative/advice", json={"peft_method": "vera"}).status_code == 422


def test_train_predict_list_and_card(client, toy_text):
    texts, y, labels = toy_text
    payload = {"name": "router-v1", "texts": texts, "labels": [labels[i] for i in y], "student": "tfidf",
               "holdout": 0.4}
    r = client.post("/api/v1/factory/classifiers", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["labels"] == labels and body["n_train"] + body["n_calibration"] + body["n_test"] == len(texts)
    assert body["contract"]["passed"] is True

    pred = client.post("/api/v1/factory/classifiers/router-v1/predict",
                       json={"texts": ["my card got blocked", "loan emi deducted twice"]}).json()["predictions"]
    assert [p["label"] for p in pred] == ["card", "loan"]
    abstain = client.post("/api/v1/factory/classifiers/router-v1/predict",
                          json={"texts": ["hello"], "threshold": 1.0}).json()["predictions"][0]
    assert abstain["abstained"] is True and abstain["label"] is None

    listing = client.get("/api/v1/factory/classifiers").json()["classifiers"]
    assert listing[0]["name"] == "router-v1" and listing[0]["student"] == "tfidf"
    card = client.get("/api/v1/factory/classifiers/router-v1").json()
    assert "router-v1" in card["model_card"] and card["run_record"]["data"]["train"]["n"] == body["n_train"]


def test_training_input_validation(client, toy_text):
    texts, y, labels = toy_text
    named = [labels[i] for i in y]
    post = lambda **kw: client.post("/api/v1/factory/classifiers",
                                    json={"name": "m", "texts": texts, "labels": named, "student": "tfidf", **kw})
    assert post(name="../../etc/passwd").status_code == 422  # names become directory names
    assert post(labels=named[:-1]).status_code == 422
    assert post(student="gpt").status_code == 422
    assert post(texts=texts[:4], labels=["a", "a", "a", "a"]).status_code == 422
    assert post(texts=texts[:5], labels=["a", "a", "a", "a", "b"]).status_code == 422  # too few per class
    assert client.post("/api/v1/factory/classifiers/missing/predict", json={"texts": ["x"]}).status_code == 404
    assert client.get("/api/v1/factory/classifiers/..%2Fsecret").status_code in (404, 422)


def test_review_call_endpoint(client):
    r = client.post("/api/v1/factory/review-call",
                    json={"utterances": ["This is just like a fixed deposit with the bank.",
                                         "You have a free look period."]})
    body = r.json()
    assert body["flags"][0]["category"] == "product_misrepresentation" and body["needs_review"] == [0]
    assert "free_look_period" not in body["missing_disclosures"] and "disclaimer" in body


# ---- training service bridge ----------------------------------------------

@pytest.fixture
def bridge():
    return _load("factory_bridge_under_test",
                 ROOT / "src/microservices/training_service/src/utils/factory_bridge.py")


def test_bridge_honours_the_request_instead_of_a_fixed_lora(bridge):
    pytest.importorskip("peft")
    req = SimpleNamespace(peft_method="dora", peft_rank=8, peft_alpha=24, peft_dropout=0.05,
                          target_modules=["q_proj", "v_proj"])
    cfg = bridge.resolve_peft_config(req)
    assert cfg.r == 8 and cfg.lora_alpha == 24 and cfg.use_dora is True
    assert set(cfg.target_modules) == {"q_proj", "v_proj"}
    assert bridge.resolve_peft_config(SimpleNamespace(peft_method="pissa")).init_lora_weights.startswith("pissa")
    assert type(bridge.resolve_peft_config(SimpleNamespace(peft_method="oft"))).__name__ == "OFTConfig"
    with pytest.raises(ValueError, match="Unsupported peft_method"):
        bridge.resolve_peft_config(SimpleNamespace(peft_method="made_up"))


def test_bridge_falls_back_to_the_legacy_config_without_forgelm(bridge, monkeypatch):
    pytest.importorskip("peft")
    monkeypatch.setattr(bridge, "FORGELM_AVAILABLE", False)
    cfg = bridge.resolve_peft_config(SimpleNamespace(peft_method="dora", peft_rank=64))
    assert cfg.r == 16 and cfg.lora_alpha == 32  # previous hard-coded behaviour
    assert bridge.backend_advice() == {"available": False, "reason": "forgelm package not installed in this image"}


def test_bridge_backend_advice_and_device(bridge):
    advice = bridge.backend_advice(3.0)
    assert advice["available"] and advice["device_class"] and advice["selection"]["backend"]
    assert {"lora", "dora", "pissa"} <= set(advice["peft_methods"])
    pytest.importorskip("torch")
    device, _dtype = bridge.torch_device_and_dtype()
    assert device in ("cuda", "xpu", "mps", "cpu")


# ---- training service: start_fine_tuning must not block --------------------

def _training_method_source(name):
    """One method's source, exec'd in isolation (the module itself needs the GPU stack to import)."""
    import ast
    import textwrap

    path = ROOT / "src/microservices/training_service/src/services/training_service.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainingService")
    defs = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    assert len(defs) == 1, f"{name} must be defined exactly once"
    assert not defs[0].decorator_list, f"{name} must not carry a decorator"
    return textwrap.dedent(ast.get_source_segment(path.read_text(), defs[0]))


def _training_namespace(*methods):
    """The module-level names the methods use, plus the methods themselves."""
    import os
    import queue
    import uuid

    namespace = {"os": os, "uuid": uuid, "queue": queue, "Path": Path, "FineTuneRequest": object,
                 "Dict": dict, "Any": object, "GPULockPriority": SimpleNamespace(HIGH=1),
                 "task_log_queues": {}, "task_completion_flags": {}, "queued_task_ids": set(),
                 "UNSLOTH_AVAILABLE": False, "TRANSFORMERS_AVAILABLE": True}
    for name in methods:
        exec(_training_method_source(name), namespace)
    return namespace


def test_start_fine_tuning_returns_before_training_finishes():
    import asyncio
    import threading
    import time

    namespace = _training_namespace("start_fine_tuning")
    queues, flags, queued = (namespace[k] for k in ("task_log_queues", "task_completion_flags", "queued_task_ids"))
    release, started, finished, status_files = threading.Event(), threading.Event(), [], []

    class Service:
        start_fine_tuning = namespace["start_fine_tuning"]
        logger = SimpleNamespace(error=lambda *a, **k: None)

        def _initialize_status_file(self, task_id, request):
            status_files.append((task_id, request))

        def fine_tune(self, request, task_id):
            started.set()
            release.wait(5)
            finished.append(task_id)

    async def scenario():
        service, request = Service(), object()
        t0 = time.perf_counter()
        task_id = await service.start_fine_tuning(request)
        returned_after = time.perf_counter() - t0
        # Visible the moment the id comes back, before the job has had any chance to run.
        assert not started.is_set()
        assert status_files == [(task_id, request)]
        assert queues[task_id].empty() and flags[task_id] is False and task_id in queued
        await asyncio.sleep(0.05)  # the event loop stays responsive while the job runs
        assert started.is_set() and not finished

        # A second job waits behind the first on the single worker thread. It must be
        # visible while it waits, not only once the first job has finished.
        second = await service.start_fine_tuning(request)
        await asyncio.sleep(0.05)
        assert finished == [] and second != task_id
        assert status_files[-1] == (second, request)
        assert second in queues and flags[second] is False and second in queued

        release.set()
        await asyncio.gather(*service._background_jobs)
        return task_id, second, returned_after

    task_id, second, returned_after = asyncio.run(scenario())
    assert returned_after < 1.0 and finished == [task_id, second]


def test_start_fine_tuning_still_schedules_the_job_when_the_status_file_cannot_be_written():
    import asyncio

    namespace = _training_namespace("start_fine_tuning")
    ran, errors = [], []

    class Service:
        start_fine_tuning = namespace["start_fine_tuning"]
        logger = SimpleNamespace(error=lambda message, *a, **k: errors.append(message))

        def _initialize_status_file(self, task_id, request):
            raise OSError("disk full")

        def fine_tune(self, request, task_id):
            ran.append(task_id)

    async def scenario():
        service = Service()
        task_id = await service.start_fine_tuning(object())
        await asyncio.gather(*service._background_jobs)
        return task_id

    task_id = asyncio.run(scenario())
    assert ran == [task_id] and task_id in namespace["task_log_queues"]
    assert len(errors) == 1 and "disk full" in errors[0]


def test_fine_tune_keeps_what_submission_registered_and_skips_a_job_cancelled_while_queued(tmp_path, monkeypatch):
    import queue

    monkeypatch.chdir(tmp_path)  # status files are looked up under ./logs
    namespace = _training_namespace("fine_tune", "cancel_training")
    queues, flags, queued = (namespace[k] for k in ("task_log_queues", "task_completion_flags", "queued_task_ids"))
    trained, status_files, cancelled = [], [], []

    class Service:
        fine_tune = namespace["fine_tune"]
        cancel_training = namespace["cancel_training"]
        logger = SimpleNamespace(error=lambda *a, **k: None, info=lambda *a, **k: None)
        active_trainings = {}

        def _initialize_status_file(self, task_id, request):
            status_files.append(task_id)

        def _update_status_file_error(self, task_id, message):
            cancelled.append((task_id, message))

        def _cleanup_training_resources(self, task_id):
            pass

        def _fine_tune_with_transformers(self, request, task_id, log_queue):
            trained.append((task_id, log_queue))
            return {"status": "completed", "task_id": task_id}

    def submit(task_id):
        """What start_fine_tuning leaves behind for a job that is waiting its turn."""
        queued.add(task_id)
        queues[task_id] = queue.Queue()
        flags[task_id] = False
        (tmp_path / "logs").mkdir(exist_ok=True)
        (tmp_path / "logs" / f"{task_id}_status.json").write_text("{}")
        return queues[task_id]

    service = Service()

    # Submitted, then run: the same queue is handed to the trainer and the status file is not rewritten.
    registered = submit("a")
    assert service.fine_tune(object(), "a")["status"] == "completed"
    assert trained == [("a", registered)] and queues["a"] is registered
    assert status_files == [] and "a" not in queued

    # Submitted, cancelled while waiting, then reached by the worker: it must not train.
    submit("b")
    assert service.cancel_training("b") is True
    assert flags["b"] is True and cancelled == [("b", "Training cancelled by user")]
    assert service.fine_tune(object(), "b")["status"] == "cancelled"
    assert [t for t, _ in trained] == ["a"]
    assert service.cancel_training("b") is False and service.cancel_training("unknown") is False

    # Called directly, without a submission: fine_tune still sets everything up itself.
    assert service.fine_tune(object(), "c")["status"] == "completed"
    assert status_files == ["c"] and flags["c"] is False and trained[-1] == ("c", queues["c"])
