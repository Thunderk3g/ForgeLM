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

def _start_fine_tuning_source():
    """The method's source, exec'd in isolation (the module itself needs the GPU stack to import)."""
    import ast
    import textwrap

    path = ROOT / "src/microservices/training_service/src/services/training_service.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainingService")
    defs = [n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "start_fine_tuning"]
    assert len(defs) == 1, "start_fine_tuning must be defined exactly once"
    return textwrap.dedent(ast.get_source_segment(path.read_text(), defs[0]))


def test_start_fine_tuning_returns_before_training_finishes():
    import asyncio
    import os
    import threading
    import time
    import uuid

    namespace = {"os": os, "uuid": uuid, "FineTuneRequest": object, "GPULockPriority": SimpleNamespace(HIGH=1)}
    exec(_start_fine_tuning_source(), namespace)
    release, started, finished = threading.Event(), threading.Event(), []

    class Service:
        start_fine_tuning = namespace["start_fine_tuning"]
        logger = SimpleNamespace(error=lambda *a, **k: None)

        def fine_tune(self, request, task_id):
            started.set()
            release.wait(5)
            finished.append(task_id)

    async def scenario():
        service = Service()
        t0 = time.perf_counter()
        task_id = await service.start_fine_tuning(object())
        returned_after = time.perf_counter() - t0
        await asyncio.sleep(0.05)  # the event loop stays responsive while the job runs
        assert started.is_set() and not finished
        release.set()
        await asyncio.gather(*service._background_jobs)
        return task_id, returned_after

    task_id, returned_after = asyncio.run(scenario())
    assert returned_after < 1.0 and finished == [task_id]
