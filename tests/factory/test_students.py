import numpy as np
import pytest

from forgelm.students import (EmbeddingStudent, LinearHead, TfidfStudent, ZeroShotStudent, load_student,
                              make_student)


def test_tfidf_student_learns_and_round_trips(toy_text, tmp_path):
    texts, y, labels = toy_text
    student = TfidfStudent().fit(texts, y, labels)
    assert (student.predict_proba(texts).argmax(1) == y).mean() == 1.0
    assert student.predict(["block my card please"]) == ["card"]
    assert any("card" in gram for gram, _ in student.top_features("card", 10))
    student.temperature = 1.7
    student.save(str(tmp_path / "m"))
    loaded = load_student(str(tmp_path / "m"))
    assert loaded.kind == "tfidf" and loaded.temperature == 1.7
    np.testing.assert_allclose(loaded.predict_proba(texts), student.predict_proba(texts), atol=1e-6)


def test_soft_labels_shift_the_decision(toy_text):
    texts, y, labels = toy_text
    soft = np.zeros((len(y), 3))
    soft[np.arange(len(y)), y] = 1.0
    hard_student = TfidfStudent().fit(texts, y, labels)
    np.testing.assert_array_equal(TfidfStudent().fit(texts, soft, labels).predict_proba(texts).argmax(1),
                                  hard_student.predict_proba(texts).argmax(1))
    # A teacher unsure between class 0 and 1 for the class-0 texts must lower confidence in class 0.
    unsure = soft.copy()
    unsure[y == 0] = [0.55, 0.45, 0.0]
    p_unsure = TfidfStudent().fit(texts, unsure, labels).predict_proba(texts)[y == 0, 0].mean()
    assert p_unsure < hard_student.predict_proba(texts)[y == 0, 0].mean()
    with pytest.raises(ValueError):
        TfidfStudent().fit(texts, np.ones((len(y), 5)) / 5, labels)


def test_linear_head_binary_and_unseen_class():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 1, (40, 3)), rng.normal(2, 1, (40, 3))]).astype(np.float32)
    y = np.array([0] * 40 + [2] * 40)  # class 1 never appears
    head = LinearHead().fit(X, y, n_classes=3)
    probs = head.predict_proba(X)
    assert probs.shape == (80, 3)
    assert probs[:, 1].max() < 1e-6
    assert (probs.argmax(1) == y).mean() > 0.95
    assert np.allclose(probs.sum(1), 1.0)


def test_embedding_student_with_injected_embedder(toy_text, fake_embedder, tmp_path):
    texts, y, labels = toy_text
    student = EmbeddingStudent("fake", backend="static")
    student.embedder = fake_embedder
    student.fit(texts, y, labels)
    assert (student.predict_proba(texts).argmax(1) == y).mean() > 0.9
    t = student.calibrate(texts, y)
    assert t > 0
    student.save(str(tmp_path / "e"))
    loaded = EmbeddingStudent.load(str(tmp_path / "e"))
    loaded.embedder = fake_embedder
    np.testing.assert_allclose(loaded.predict_proba(texts), student.predict_proba(texts), atol=1e-6)
    assert student.serve_requires == ["numpy", "model2vec"]


def test_zero_shot_uses_label_descriptions(fake_embedder):
    student = ZeroShotStudent("fake", backend="static")
    student.embedder = fake_embedder
    student.fit(labels=["card", "loan", "policy"],
                descriptions=["debit card pin", "loan emi interest", "policy premium nominee claim"])
    assert student.predict(["what is my emi this month", "nominee change"]) == ["loan", "policy"]


def test_make_student_names():
    assert make_student("tfidf").kind == "tfidf"
    assert make_student("static").embedder.backend == "static"
    assert make_student("encoder").embedder.backend == "sentence"
    with pytest.raises(KeyError):
        make_student("gpt")


@pytest.mark.slow
def test_static_student_end_to_end_with_real_model(toy_text):
    pytest.importorskip("model2vec")
    texts, y, labels = toy_text
    student = make_student("static", model="minishlab/potion-base-8M").fit(texts, y, labels)
    assert (student.predict_proba(texts).argmax(1) == y).mean() > 0.9
