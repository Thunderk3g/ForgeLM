import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from forgelm.distill import attribute_prompts, distill, diversity_report, uncertainty_order
from forgelm.students import TfidfStudent
from forgelm.teachers import CallableTeacher, ChatCompletionsTeacher, LocalLLMTeacher, StudentTeacher


def _keyword_teacher(labels, noise_idx=()):
    def fn(texts):
        out = []
        for i, t in enumerate(texts):
            guess = next((j for j, l in enumerate(labels) if l in t.lower()), 0)
            out.append((guess + 1) % len(labels) if i in noise_idx else guess)
        return np.array(out)
    return CallableTeacher(fn, labels, name="keyword-rules")


def test_distill_scores_teacher_and_student_on_the_anchor(toy_text):
    texts, y, labels = toy_text
    anchor_idx = list(range(0, 30, 5))
    pool = [t for i, t in enumerate(texts) if i not in anchor_idx]
    report = distill(_keyword_teacher(labels), TfidfStudent(), pool, [texts[i] for i in anchor_idx], y[anchor_idx],
                     soft=False)
    assert report.n_used_for_training == len(pool)
    assert report.teacher_on_anchor["accuracy"] == 1.0
    assert report.student_on_anchor["accuracy"] >= 0.8
    assert 0 <= report.student_teacher_agreement <= 1
    assert report.teacher_stats["items_labelled"] == len(pool) + len(anchor_idx)
    assert set(report.to_dict()) >= {"student_minus_teacher_accuracy", "label_distribution"}


def test_distill_removes_anchor_leakage(toy_text):
    texts, y, labels = toy_text
    anchor = texts[:3]
    report = distill(_keyword_teacher(labels), TfidfStudent(), texts, anchor, y[:3], soft=False)
    assert report.n_used_for_training == len(texts) - 3
    assert any("also appear in the anchor" in w for w in report.warnings)


def test_distill_warns_when_teacher_never_predicts_a_class(toy_text):
    texts, y, labels = toy_text
    lazy = CallableTeacher(lambda ts: np.array([0 if "card" in t else 1 for t in ts]), labels)
    report = distill(lazy, TfidfStudent(), texts[3:], texts[:3], y[:3], soft=False)
    assert any("never predicted" in w for w in report.warnings)


def test_confidence_filter_and_empty_result(toy_text):
    texts, y, labels = toy_text
    probs = np.full((len(texts), 3), 1 / 3)
    teacher = CallableTeacher(lambda ts: probs[: len(ts)], labels)
    with pytest.raises(ValueError):
        distill(teacher, TfidfStudent(), texts[3:], texts[:3], y[:3], min_confidence=0.9)


def test_student_as_teacher(toy_text):
    texts, y, labels = toy_text
    big = TfidfStudent().fit(texts, y, labels)
    probs = StudentTeacher(big).label(texts[:4])
    assert probs.shape == (4, 3) and np.allclose(probs.sum(1), 1)


def test_uncertainty_order_puts_ties_first():
    probs = np.array([[0.9, 0.1], [0.51, 0.49], [0.7, 0.3]])
    assert list(uncertainty_order(probs)) == [1, 2, 0]


def test_attribute_prompts_vary_and_diversity_report_flags_collapse():
    prompts = attribute_prompts("mis_selling", "agent promises guaranteed returns",
                                {"language": ["English", "Hinglish"], "tone": ["pushy", "polite", "rushed"]},
                                n=12, seed=1, domain="insurance sales call")
    assert len(set(prompts)) > 3 and all("mis_selling" in p for p in prompts)
    collapsed = diversity_report(["Guaranteed returns sir"] * 9 + ["this plan doubles your money"])
    varied = diversity_report([f"customer {i} asks about premium {i * 7} for plan {i % 3}" for i in range(10)])
    assert collapsed["exact_duplicate_rate"] == pytest.approx(0.8)
    assert varied["distinct_2"] > collapsed["distinct_2"]
    assert diversity_report([]) == {"n": 0}


def test_local_llm_teacher_requires_shortlister_for_many_labels():
    with pytest.raises(ValueError, match="shortlister"):
        LocalLLMTeacher("any/model", [f"label{i}" for i in range(40)])


class _Stub(BaseHTTPRequestHandler):
    answers = ["2", "The answer is 1.", "no idea"]
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.seen.append((self.headers.get("Authorization"), body))
        reply = _Stub.answers[(len(_Stub.seen) - 1) % len(_Stub.answers)]
        payload = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_chat_completions_teacher_against_a_local_endpoint(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("FORGELM_TEACHER_API_KEY", "test-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    try:
        teacher = ChatCompletionsTeacher(f"http://127.0.0.1:{server.server_port}/v1", "local-model",
                                         ["complaint", "query"])
        probs = teacher.label(["a", "b", "c"])
    finally:
        server.shutdown()
    assert probs[0].tolist() == [0.0, 1.0]  # "2"
    assert probs[1].tolist() == [1.0, 0.0]  # "The answer is 1."
    assert probs[2].tolist() == [0.5, 0.5]  # unparseable -> uniform, filtered by confidence later
    assert teacher.stats()["unparsed"] == 1
    auth, body = _Stub.seen[0]
    assert auth == "Bearer test-key" and body["model"] == "local-model" and body["temperature"] == 0
