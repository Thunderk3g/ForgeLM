"""Input validation, concurrency and file-layout guarantees of the factory microservice.

Everything here trains the ``tfidf`` student only, so no model is downloaded.
"""

import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
API = "/api/v1/factory"
# What RunRecord.write_all adds after the artefacts have been hashed.
RECORD_FILES = {"run_record.json", "MODEL_CARD.md", "ml-bom.cdx.json"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_service():
    return _load("factory_main_hardening", ROOT / "src/microservices/factory_service/src/main.py")


@pytest.fixture
def service(tmp_path, monkeypatch):
    """The service module, freshly imported with an empty models directory."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    monkeypatch.setenv("FACTORY_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("FACTORY_ALLOWED_MODELS", raising=False)
    monkeypatch.delenv("FACTORY_MAX_PREDICT_ROWS", raising=False)
    monkeypatch.delenv("FACTORY_MAX_TEXT_CHARS", raising=False)
    return _load_service()


@pytest.fixture
def client(service):
    from fastapi.testclient import TestClient

    return TestClient(service.app)


@pytest.fixture
def payload(toy_text):
    """Build a training request body; keyword arguments override fields."""
    texts, y, labels = toy_text
    base = {"name": "m", "texts": texts, "labels": [labels[i] for i in y], "student": "tfidf", "holdout": 0.4}
    return lambda **kw: {**base, **kw}


def _hidden(service):
    return [p.name for p in service.MODELS_DIR.iterdir() if p.name.startswith(".")]


# ---- (a) names ------------------------------------------------------------

def test_a_name_with_a_trailing_newline_is_rejected(service, client, payload):
    r = client.post(f"{API}/classifiers", json=payload(name="abc\n"))
    assert r.status_code == 422, r.text
    assert not service.MODELS_DIR.exists() or list(service.MODELS_DIR.iterdir()) == []
    assert client.get(f"{API}/classifiers/abc%0A").status_code == 422
    assert client.post(f"{API}/classifiers/abc%0A/predict", json={"texts": ["x"]}).status_code == 422
    for bad in ("", "-abc", "a" * 65, "a b", "a/b", ".hidden", "abc\r", "\nabc"):
        assert client.post(f"{API}/classifiers", json=payload(name=bad)).status_code == 422, repr(bad)
    for good in ("a", "A1_b-2", "a" * 64):
        assert service._model_dir(good) == service.MODELS_DIR / good


# ---- (b) one training at a time per name, and an all-or-nothing directory ---

def test_two_trainings_of_one_name_take_turns(service, payload, monkeypatch):
    import time

    import forgelm.students as students

    real, counter = students.make_student, threading.Lock()
    active, peak = [0], [0]

    def slow_make_student(kind, **kwargs):
        with counter:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.2)  # long enough for the other request to arrive if nothing stops it
        with counter:
            active[0] -= 1
        return real(kind, **kwargs)

    monkeypatch.setattr(students, "make_student", slow_make_student)
    start, results = threading.Barrier(2), []

    def train(seed):
        start.wait(5)
        results.append(service.train_classifier(service.TrainRequest(**payload(seed=seed))))

    threads = [threading.Thread(target=train, args=(seed,)) for seed in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(results) == 2 and peak[0] == 1

    # Whichever finished last, the directory is one complete model: the files on
    # disk are exactly the ones its run record lists, with the same hashes.
    directory = service.MODELS_DIR / "m"
    on_disk = {k: v for k, v in service.hash_directory(str(directory)).items() if k not in RECORD_FILES}
    assert on_disk == json.loads((directory / "run_record.json").read_text())["artefacts"]
    assert _hidden(service) == []


def test_different_names_train_at_the_same_time(service, payload, monkeypatch):
    import forgelm.students as students

    real, both_inside = students.make_student, threading.Barrier(2)

    def make_student(kind, **kwargs):
        both_inside.wait(10)  # only passes if both requests are training at once
        return real(kind, **kwargs)

    monkeypatch.setattr(students, "make_student", make_student)
    results, errors = [], []

    def train(name):
        try:
            results.append(service.train_classifier(service.TrainRequest(**payload(name=name)))["name"])
        except Exception as exc:  # a broken barrier lands here
            errors.append(exc)

    threads = [threading.Thread(target=train, args=(name,)) for name in ("one", "two")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == [] and sorted(results) == ["one", "two"]


def test_retraining_replaces_the_directory_and_the_served_model(service, client, payload, toy_text):
    texts, y, labels = toy_text
    assert client.post(f"{API}/classifiers", json=payload()).status_code == 200
    directory = service.MODELS_DIR / "m"
    # Stands in for a file only another student kind writes (a transformer's weights, say).
    (directory / "model.safetensors").write_text("left over from the previous kind")
    before = client.post(f"{API}/classifiers/m/predict", json={"texts": ["claim status for my policy"]}).json()
    assert before["predictions"][0]["label"] == "policy"

    # Retrain under the same name on two of the three classes.
    keep = [i for i in range(len(texts)) if labels[y[i]] != "policy"]
    r = client.post(f"{API}/classifiers", json=payload(texts=[texts[i] for i in keep],
                                                       labels=[labels[y[i]] for i in keep]))
    assert r.status_code == 200, r.text
    record = json.loads((directory / "run_record.json").read_text())
    assert not (directory / "model.safetensors").exists()
    assert "model.safetensors" not in record["artefacts"]
    assert {p.name for p in directory.iterdir()} == set(record["artefacts"]) | RECORD_FILES
    assert _hidden(service) == []
    # The cached model was replaced too: it no longer knows the dropped class.
    after = client.post(f"{API}/classifiers/m/predict", json={"texts": ["claim status for my policy"]}).json()
    assert after["predictions"][0]["label"] in ("card", "loan")
    assert service._loaded["m"].labels == ["card", "loan"]


def test_a_failed_retrain_leaves_the_previous_model_in_place(service, payload, monkeypatch):
    from fastapi.testclient import TestClient

    client = TestClient(service.app, raise_server_exceptions=False)
    assert client.post(f"{API}/classifiers", json=payload()).status_code == 200
    directory = service.MODELS_DIR / "m"
    files_before = service.hash_directory(str(directory))
    served_before = service._loaded["m"]

    def disk_full(self, directory):
        raise OSError("disk full")

    # The run record is the last thing written, so this fails as late as it can.
    monkeypatch.setattr(service.RunRecord, "write_all", disk_full)
    assert client.post(f"{API}/classifiers", json=payload(seed=7)).status_code == 500
    assert service.hash_directory(str(directory)) == files_before
    assert service._loaded["m"] is served_before
    assert _hidden(service) == []  # the temporary directory was cleaned up

    # A first training that fails leaves nothing behind and nothing to serve.
    assert client.post(f"{API}/classifiers", json=payload(name="fresh")).status_code == 500
    assert sorted(p.name for p in service.MODELS_DIR.iterdir()) == ["m"]
    assert "fresh" not in service._loaded
    assert client.post(f"{API}/classifiers/fresh/predict", json={"texts": ["x"]}).status_code == 404


def test_listing_ignores_temporary_directories(service, client, payload):
    assert client.post(f"{API}/classifiers", json=payload(name="real")).status_code == 200
    record = (service.MODELS_DIR / "real" / "run_record.json").read_text()
    for leftover in (".real.tmp-0123456789ab", ".real.old-0123456789ab"):
        (service.MODELS_DIR / leftover).mkdir()
        (service.MODELS_DIR / leftover / "run_record.json").write_text(record)
    assert [c["name"] for c in client.get(f"{API}/classifiers").json()["classifiers"]] == ["real"]


# ---- (c) numeric bounds -----------------------------------------------------

@pytest.mark.parametrize("path, body", [
    ("/generative/advice", {"params_billion": 0}),
    ("/generative/advice", {"params_billion": -3}),
    ("/generative/advice", {"params_billion": 2000.5}),
    ("/generative/advice", {"batch_size": 0}),
    ("/generative/advice", {"batch_size": 4097}),
    ("/generative/advice", {"seq_len": 0}),
    ("/generative/advice", {"seq_len": 1_000_001}),
    ("/generative/advice", {"rank": 0}),
    ("/generative/advice", {"rank": 1025}),
    ("/plan", {"model_params_billion": 0}),
    ("/plan", {"model_params_billion": -1}),
    ("/plan", {"model_params_billion": 2001}),
    ("/plan", {"n_classes": 0}),
    ("/plan", {"labelled_per_class": -1}),
    ("/plan", {"n_unlabelled": -1}),
    ("/plan", {"n_rows": -1}),
    ("/plan", {"n_features": -1}),
    ("/plan", {"avg_tokens": 0}),
    ("/plan", {"max_latency_ms": 0}),
    ("/plan", {"max_latency_ms": -5}),
    ("/review-call", {"utterances": ["hello"], "flag_threshold": 1.5}),
    ("/review-call", {"utterances": ["hello"], "flag_threshold": -0.1}),
    ("/review-call", {"utterances": ["hello"], "unsure_below": 1.01}),
    ("/review-call", {"utterances": ["hello"], "unsure_below": -1}),
])
def test_out_of_range_numbers_are_a_422_not_a_500(client, path, body):
    r = client.post(API + path, json=body)
    assert r.status_code == 422, r.text


def test_infinite_and_negative_values_python_json_lets_through_are_rejected(client, payload):
    # The standard json module writes and reads Infinity, so it can reach the service.
    post = lambda path, text: client.post(API + path, content=text, headers={"content-type": "application/json"})
    assert post("/plan", '{"labelled_per_class": Infinity}').status_code == 422
    assert post("/plan", '{"max_latency_ms": Infinity}').status_code == 422
    assert post("/plan", '{"model_params_billion": NaN}').status_code == 422
    assert post("/generative/advice", '{"params_billion": Infinity}').status_code == 422
    assert client.post(f"{API}/classifiers", json=payload(seed=-1)).status_code == 422  # numpy refuses it


def test_values_on_the_bounds_are_accepted(client):
    r = client.post(f"{API}/generative/advice", json={"params_billion": 2000, "batch_size": 4096, "seq_len": 1,
                                                      "rank": 1024, "hardware": "cpu-laptop"})
    assert r.status_code == 200, r.text
    r = client.post(f"{API}/generative/advice", json={"params_billion": 0.01, "batch_size": 1, "seq_len": 1_000_000,
                                                      "rank": 1, "hardware": "nvidia-24gb"})
    assert r.status_code == 200, r.text
    r = client.post(f"{API}/plan", json={"task": "generation", "model_params_billion": 2000, "max_latency_ms": 0.5,
                                         "hardware": "cpu-laptop"})
    assert r.status_code == 200, r.text
    r = client.post(f"{API}/plan", json={"n_classes": 1, "labelled_per_class": 0, "n_unlabelled": 0, "n_rows": 0,
                                         "n_features": 0, "avg_tokens": 1, "hardware": "cpu-laptop"})
    assert r.status_code == 200, r.text
    r = client.post(f"{API}/review-call", json={"utterances": ["hello"], "flag_threshold": 0, "unsure_below": 1})
    assert r.status_code == 200, r.text


# ---- (d) a hold-out share that leaves nothing to train on --------------------

def _per_class(toy_text, n):
    texts, y, labels = toy_text
    rows = [i for c in range(len(labels)) for i in [j for j in range(len(texts)) if y[j] == c][:n]]
    return {"texts": [texts[i] for i in rows], "labels": [labels[y[i]] for i in rows]}


def test_a_holdout_that_takes_every_example_is_a_422(service, client, payload, toy_text):
    # round(4 * 0.89) = 4: every class would lose all four of its examples.
    r = client.post(f"{API}/classifiers", json=payload(holdout=0.89, **_per_class(toy_text, 4)))
    assert r.status_code == 422, r.text
    assert "at least 5 per class" in r.json()["detail"] and "holdout=0.89" in r.json()["detail"]
    assert not service.MODELS_DIR.exists() or list(service.MODELS_DIR.iterdir()) == []

    # One short class is enough to refuse, and it is the one named.
    texts, y, labels = toy_text
    rows = [i for i in range(len(texts)) if labels[y[i]] != "loan"] + [i for i in range(len(texts))
                                                                       if labels[y[i]] == "loan"][:4]
    r = client.post(f"{API}/classifiers", json=payload(holdout=0.89, texts=[texts[i] for i in rows],
                                                       labels=[labels[y[i]] for i in rows]))
    assert r.status_code == 422 and "'loan' has 4 examples" in r.json()["detail"]

    # The smallest set the message asks for does train: one example per class is left.
    r = client.post(f"{API}/classifiers", json=payload(holdout=0.89, **_per_class(toy_text, 5)))
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["n_train"], body["n_calibration"], body["n_test"]) == (3, 6, 6)

    # At the default share four per class is still enough, as before.
    r = client.post(f"{API}/classifiers", json=payload(holdout=0.2, **_per_class(toy_text, 4)))
    assert r.status_code == 200 and r.json()["n_train"] == 6


def test_holdout_arithmetic_always_leaves_a_training_example(service):
    for holdout in (0.01, 0.2, 0.5, 0.75, 0.8, 0.85, 0.88, 0.89, 0.899, 0.8999999):
        needed = next(n for n in range(4, 100) if n - service._n_holdout(n, holdout) >= 1)
        for n in range(needed, 300):
            held = service._n_holdout(n, holdout)
            assert held >= 2 and n - held >= 1, (holdout, n)


# ---- (e) the model allow-list and the serving limits -------------------------

def test_model_outside_the_allow_list_is_rejected(service, client, payload, monkeypatch):
    import forgelm.students as students

    def must_not_be_called(kind, **kwargs):
        raise AssertionError(f"make_student was reached with {kwargs}")

    monkeypatch.setattr(students, "make_student", must_not_be_called)
    for model in ("/etc", "../../secrets", "someone/unknown-repo", "minishlab/potion-base-32M/..", ""):
        r = client.post(f"{API}/classifiers", json=payload(student="static", model=model))
        assert r.status_code == 422, (model, r.text)
        assert "FACTORY_ALLOWED_MODELS" in r.json()["detail"]
    # tfidf has no base model at all, so even a listed one is a request error.
    r = client.post(f"{API}/classifiers", json=payload(student="tfidf", model="BAAI/bge-small-en-v1.5"))
    assert r.status_code == 422 and "tfidf" in r.json()["detail"]
    assert set(service.ALLOWED_MODELS) == {
        "minishlab/potion-base-32M", "minishlab/potion-base-8M", "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2", "intfloat/multilingual-e5-small", "answerdotai/ModernBERT-base"}


def test_a_listed_model_reaches_the_student(service, client, payload, monkeypatch):
    import forgelm.students as students

    real, seen = students.make_student, []

    def make_student(kind, **kwargs):
        seen.append((kind, kwargs))
        return real("tfidf")  # no embedding model is loaded in tests

    monkeypatch.setattr(students, "make_student", make_student)
    r = client.post(f"{API}/classifiers", json=payload(student="static", model="minishlab/potion-base-8M"))
    assert r.status_code == 200, r.text
    assert client.post(f"{API}/classifiers", json=payload(student="static")).status_code == 200
    assert seen == [("static", {"model": "minishlab/potion-base-8M"}), ("static", {})]


def test_allow_list_and_limits_are_read_from_the_environment(service, payload, monkeypatch):
    from fastapi.testclient import TestClient

    assert service.MAX_PREDICT_ROWS == 10000
    monkeypatch.setenv("FACTORY_ALLOWED_MODELS", "acme/small, acme/large ,")
    monkeypatch.setenv("FACTORY_MAX_PREDICT_ROWS", "2")
    module = _load_service()
    assert module.ALLOWED_MODELS == ("acme/small", "acme/large") and module.MAX_PREDICT_ROWS == 2
    client = TestClient(module.app)
    r = client.post(f"{API}/classifiers", json=payload(student="static", model="minishlab/potion-base-32M"))
    assert r.status_code == 422 and "acme/small" in r.json()["detail"]
    assert client.post(f"{API}/review-call", json={"utterances": ["a", "b"]}).status_code == 200
    assert client.post(f"{API}/review-call", json={"utterances": ["a", "b", "c"]}).status_code == 413


def test_predict_and_review_call_cap_the_number_and_length_of_texts(service, client, payload, monkeypatch):
    assert client.post(f"{API}/classifiers", json=payload()).status_code == 200
    monkeypatch.setattr(service, "MAX_PREDICT_ROWS", 3)
    monkeypatch.setattr(service, "MAX_TEXT_CHARS", 40)
    ok, long_text = ["my card got blocked"] * 3, "x" * 41

    assert client.post(f"{API}/classifiers/m/predict", json={"texts": ok}).status_code == 200
    r = client.post(f"{API}/classifiers/m/predict", json={"texts": ok + ["one more"]})
    assert r.status_code == 413 and "at most 3" in r.json()["detail"]
    r = client.post(f"{API}/classifiers/m/predict", json={"texts": [long_text]})
    assert r.status_code == 413 and "40 characters" in r.json()["detail"]
    # The limit is checked before the classifier is looked up, so nothing is loaded for a refused request.
    assert client.post(f"{API}/classifiers/missing/predict", json={"texts": ok + ["one more"]}).status_code == 413

    assert client.post(f"{API}/review-call", json={"utterances": ok}).status_code == 200
    assert client.post(f"{API}/review-call", json={"utterances": ok + ["one more"]}).status_code == 413
    assert client.post(f"{API}/review-call", json={"utterances": [long_text]}).status_code == 413
    assert client.post(f"{API}/review-call", json={"utterances": ok + ["one more"],
                                                   "classifier": "m"}).status_code == 413


def test_an_empty_allow_list_variable_means_the_defaults(tmp_path, monkeypatch):
    """docker-compose passes FACTORY_ALLOWED_MODELS through as "" when it is unset."""
    import importlib.util
    import sys
    from pathlib import Path

    monkeypatch.setenv("FACTORY_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("FACTORY_ALLOWED_MODELS", "")
    path = Path(__file__).resolve().parents[2] / "src/microservices/factory_service/src/main.py"
    spec = importlib.util.spec_from_file_location("factory_main_empty_allow", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert "minishlab/potion-base-32M" in module.ALLOWED_MODELS
