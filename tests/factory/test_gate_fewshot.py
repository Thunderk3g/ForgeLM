"""The gate (paired bootstrap, exact McNemar, decide) and few-shot prompting in LocalLLMTeacher.

No model is loaded: the teacher gets a small fake tokenizer (and, for the one
end-to-end test, a fake causal LM built from torch tensors).
"""

import math
import string
import subprocess
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from forgelm.gate import Option, decide, mcnemar_exact, paired_bootstrap
from forgelm.teachers import LocalLLMTeacher

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Paired bootstrap and McNemar
# --------------------------------------------------------------------------

def _system(y, accuracy, seed, n_classes=5):
    """Predictions that are right on a random ``accuracy`` share of items."""
    rng = np.random.default_rng(seed)
    pred = y.copy()
    wrong = rng.random(len(y)) >= accuracy
    pred[wrong] = (y[wrong] + 1) % n_classes
    return pred


def test_identical_systems_differ_by_exactly_zero():
    y = np.random.default_rng(0).integers(0, 4, 300)
    pred = _system(y, 0.7, 1, 4)
    bs = paired_bootstrap(y, pred, pred, n_boot=500)
    assert bs["difference"] == 0.0 and bs["ci_low"] == 0.0 and bs["ci_high"] == 0.0
    assert bs["accuracy_a"] == bs["accuracy_b"] == pytest.approx(np.mean(pred == y))
    mc = mcnemar_exact(y, pred, pred)
    assert mc == {"a_only_correct": 0, "b_only_correct": 0, "discordant": 0, "p_value": 1.0}


def test_mcnemar_matches_hand_computed_binomial_tails():
    # 6 discordant items, A wins 1 and B wins 5: p = 2 * (C(6,0) + C(6,1)) / 2^6 = 14 / 64.
    y = np.zeros(10, dtype=int)
    a = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0])  # correct on items 0, 6..9
    b = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # correct on items 1..9
    mc = mcnemar_exact(y, a, b)
    assert (mc["a_only_correct"], mc["b_only_correct"], mc["discordant"]) == (1, 5, 6)
    assert mc["p_value"] == pytest.approx(14 / 64)
    # B wins all 10 discordant items: p = 2 / 2^10.
    y = np.zeros(10, dtype=int)
    assert mcnemar_exact(y, np.ones(10, dtype=int), y)["p_value"] == pytest.approx(2 / 1024)
    # Balanced discordance: the doubled tail exceeds 1 and is capped.
    assert mcnemar_exact([0, 0], [0, 1], [1, 0])["p_value"] == 1.0
    # Exact integer arithmetic: no overflow and no premature underflow on a large test set.
    big = mcnemar_exact(np.zeros(1000, int), np.ones(1000, int), np.zeros(1000, int))
    assert big["p_value"] == pytest.approx(2.0 ** -999, rel=1e-12)


def test_clearly_better_system_has_an_interval_above_zero():
    y = np.random.default_rng(0).integers(0, 5, 200)
    a = _system(y, 0.5, 1)
    b = y.copy()
    b[:20] = (y[:20] + 1) % 5  # 90% correct
    bs = paired_bootstrap(y, a, b, n_boot=2000, seed=3)
    assert bs["difference"] == pytest.approx(np.mean(b == y) - np.mean(a == y))
    assert bs["ci_low"] > 0.2 and bs["ci_low"] < bs["difference"] < bs["ci_high"]
    assert mcnemar_exact(y, a, b)["p_value"] < 1e-10
    # Reversing the roles flips the sign and mirrors the interval (same resampled items).
    rev = paired_bootstrap(y, b, a, n_boot=2000, seed=3)
    assert rev["difference"] == pytest.approx(-bs["difference"])
    assert rev["ci_low"] == pytest.approx(-bs["ci_high"]) and rev["ci_high"] == pytest.approx(-bs["ci_low"])


def test_probability_matrices_are_read_as_their_argmax():
    y = np.array([0, 1, 2, 1])
    probs = np.eye(3)[[0, 1, 1, 1]] * 0.8 + 0.2 / 3
    assert paired_bootstrap(y, probs, [0, 1, 1, 1], n_boot=100)["difference"] == 0.0
    with pytest.raises(ValueError, match="predictions"):
        paired_bootstrap(y, [0, 1], [0, 1, 1, 1])


# --------------------------------------------------------------------------
# decide()
# --------------------------------------------------------------------------

def test_decide_keeps_the_cheap_option_when_the_costly_one_is_not_clearly_better():
    y = np.random.default_rng(0).integers(0, 5, 2000)
    cheap = _system(y, 0.80, 1)
    llm = cheap.copy()
    fixed, broken = np.flatnonzero(cheap != y)[:6], np.flatnonzero(cheap == y)[:3]
    llm[fixed] = y[fixed]
    llm[broken] = (y[broken] + 1) % 5  # 6 items gained, 3 lost: +0.15 points
    report = decide([Option("llm few-shot", llm, needs_llm_at_inference=True, seconds_per_item=1.0),
                     Option("student", cheap, seconds_per_item=0.001, labels_needed=616)], y)
    assert report.recommended == "student"
    assert [o["name"] for o in report.options] == ["student", "llm few-shot"]  # ranked by cost
    (cmp,) = report.comparisons
    assert cmp["cheaper"] == "student" and cmp["costlier"] == "llm few-shot"
    assert cmp["gain"] == pytest.approx(0.0015)
    assert cmp["ci_low"] < 0 < cmp["ci_high"] < report.min_gain
    assert cmp["verdict"] == "no_meaningful_gain" and cmp["settled"]
    assert not report.too_small_to_tell
    assert "Recommend 'student'" in report.reason and "student" in report.render()


def test_decide_pays_for_the_costly_option_when_it_clearly_wins():
    y = np.random.default_rng(0).integers(0, 5, 1000)
    report = decide([Option("zero-shot", _system(y, 0.55, 1)),
                     Option("student", _system(y, 0.70, 2), seconds_per_item=0.002, labels_needed=616),
                     Option("llm", _system(y, 0.90, 3), needs_llm_at_inference=True, seconds_per_item=0.5)], y)
    assert report.recommended == "llm"
    assert set(report.rejected) == {"zero-shot", "student"}
    assert all(c["verdict"] == "better" and c["settled"] for c in report.rejected.values())
    assert not report.too_small_to_tell
    assert "beats 'student'" in report.reason


def test_an_llm_ranks_as_costlier_than_a_slower_llm_free_option():
    y = np.arange(200) % 4
    same = _system(y, 0.8, 1, 4)
    report = decide([Option("fast llm", same, needs_llm_at_inference=True, seconds_per_item=0.01),
                     Option("slow student", same, seconds_per_item=0.5)], y)
    assert report.recommended == "slow student"


def test_decide_says_too_small_to_tell_on_a_tiny_test_set():
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    cheap = np.array([0, 1, 2, 0, 1, 0, 0, 2, 2, 1])  # 7 / 10
    llm = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 1])  # 8 / 10
    report = decide([Option("cheap", cheap), Option("llm", llm, needs_llm_at_inference=True)], y)
    assert report.recommended == "cheap"
    assert report.too_small_to_tell
    assert "too small to tell" in report.reason
    lo, hi = report.options[0]["accuracy_ci"]
    assert hi - lo > 0.3  # the per-option interval is visibly wide too


def test_decide_says_too_small_to_tell_when_the_interval_spans_a_worthwhile_gain():
    # 400 items is enough to resolve one point in principle, but a +2 point gain here has an interval
    # running from below zero to well above the minimum gain: neither shown nor ruled out.
    y = np.random.default_rng(0).integers(0, 5, 400)
    cheap = _system(y, 0.70, 1)
    llm = _system(y, 0.72, 2)
    report = decide([Option("cheap", cheap), Option("llm", llm, needs_llm_at_inference=True)], y)
    (cmp,) = report.comparisons
    assert cmp["verdict"] == "inconclusive" and not cmp["settled"]
    assert cmp["ci_low"] < 0 < report.min_gain < cmp["ci_high"]
    assert report.recommended == "cheap" and report.too_small_to_tell
    assert "too small to tell whether paying more helps" in report.reason
    assert "test items would be needed" in report.reason


def test_accuracy_only_options_are_listed_but_not_compared():
    y = np.arange(100) % 2
    report = decide([Option("reported elsewhere", accuracy=0.99),
                     Option("measured", y.copy(), seconds_per_item=0.1)], y)
    assert report.recommended == "measured"
    assert report.untested == ["reported elsewhere"]
    assert "not compared" in report.reason
    nothing = decide([Option("a", accuracy=0.9)], y)
    assert nothing.recommended is None and nothing.too_small_to_tell
    with pytest.raises(ValueError, match="unique"):
        decide([Option("a", y), Option("a", y)], y)
    assert set(report.to_dict()) >= {"options", "comparisons", "recommended", "reason", "too_small_to_tell"}


# --------------------------------------------------------------------------
# Few-shot prompting in LocalLLMTeacher, without loading a model
# --------------------------------------------------------------------------

class FakeTokenizer:
    """Whitespace tokenizer: one id per distinct word."""

    chat_template = None

    def __init__(self):
        self.vocab, self.words = {}, []

    def encode(self, text, add_special_tokens=False):
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.words)
                self.words.append(w)
            ids.append(self.vocab[w])
        return ids

    def decode(self, ids):
        return " ".join(self.words[i] for i in ids)


class FakeChatTokenizer(FakeTokenizer):
    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages) + "<|assistant|>"


def _prompt_at_8585879(self, text, option_idx):
    """LocalLLMTeacher._prompt exactly as it was at commit 8585879, before few-shot support."""
    tok = self._tok
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > self.max_text_tokens:
        text = tok.decode(ids[: self.max_text_tokens]) + " ..."
    options = "\n".join(f"{string.ascii_uppercase[k]}. {self.descriptions[j]}" for k, j in enumerate(option_idx))
    user = f"{self.instruction}\n\nText: {text}\n\nOptions:\n{options}\n\nAnswer with the letter only."
    messages = [{"role": "user", "content": user}]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user + "\nAnswer:"


def _candidates_at_8585879(self, texts):
    """The candidate computation that was inline in LocalLLMTeacher._label at commit 8585879."""
    import hashlib

    n, c = len(texts), len(self.labels)
    if c <= self.max_options:
        return [list(range(c))] * n
    short = np.asarray(self.shortlister.predict_proba(texts))
    candidates = []
    for text, row in zip(texts, short):
        top = np.argsort(-row)[: self.max_options]
        seed = int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "little")
        candidates.append([int(j) for j in np.random.default_rng(seed).permutation(top)])
    return candidates


TEXTS = ["my card was declined", "one two three four five six seven eight nine ten eleven", "loan emi", ""]


@pytest.mark.parametrize("tok_cls", [FakeTokenizer, FakeChatTokenizer])
def test_prompt_without_demonstrations_is_unchanged_from_8585879(tok_cls, fake_embedder):
    labels = ["card_blocked", "loan_query", "policy_status"]
    plain = LocalLLMTeacher("fake/model", labels, max_text_tokens=6)
    # The same teacher with few-shot switched on must still build the old prompt when given no demonstrations.
    fewshot = LocalLLMTeacher("fake/model", labels, max_text_tokens=6, demonstrations=[("card", 0)], k_demos=1,
                              demo_embedder=fake_embedder)
    for teacher in (plain, fewshot):
        teacher._tok = tok_cls()
        for text in TEXTS:
            for options in ([0, 1, 2], [2, 0], [1]):
                assert teacher._prompt(text, options) == _prompt_at_8585879(teacher, text, options)
                assert teacher._prompt(text, options, []) == _prompt_at_8585879(teacher, text, options)


def _teacher_from_8585879():
    """LocalLLMTeacher as committed at 8585879, loaded from git; skips where git history is unavailable."""
    try:
        source = subprocess.run(["git", "show", "8585879:forgelm/teachers.py"], cwd=ROOT, capture_output=True,
                                text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git history with commit 8585879 is not available")
    module = types.ModuleType("teachers_at_8585879")
    exec(compile(source, "teachers_at_8585879.py", "exec"), module.__dict__)  # stdlib and numpy imports only
    return module.LocalLLMTeacher


class FixedShortlister:
    """Scores labels with a fixed ranking per text (label 0 best, then 1, ...) unless overridden."""

    def __init__(self, n_labels, per_text=None):
        self.n, self.per_text = n_labels, per_text or {}

    def predict_proba(self, texts):
        rows = []
        for t in texts:
            order = self.per_text.get(t, list(range(self.n)))
            scores = np.zeros(self.n)
            scores[order] = np.linspace(1.0, 0.1, self.n)
            rows.append(scores / scores.sum())
        return np.array(rows)


def test_zero_shot_prompt_and_candidates_match_the_committed_class_from_git():
    old_cls = _teacher_from_8585879()
    labels = [f"intent_{i}" for i in range(30)]
    rng = np.random.default_rng(0)
    short = FixedShortlister(30, {t: list(rng.permutation(30)) for t in TEXTS})
    old = old_cls("fake/model", labels, max_options=10, shortlister=short, max_text_tokens=6)
    new = LocalLLMTeacher("fake/model", labels, max_options=10, shortlister=short, max_text_tokens=6)
    for tok_cls in (FakeTokenizer, FakeChatTokenizer):
        old._tok, new._tok = tok_cls(), tok_cls()
        candidates = new._candidates(TEXTS)
        assert candidates == _candidates_at_8585879(old, TEXTS)
        for text, cand in zip(TEXTS, candidates):
            assert new._prompt(text, cand) == old_cls._prompt(old, text, cand)


def test_candidates_without_demonstrations_match_the_old_inline_code():
    labels = [f"intent_{i}" for i in range(40)]
    texts = [f"text number {i}" for i in range(25)]
    rng = np.random.default_rng(1)
    short = FixedShortlister(40, {t: list(rng.permutation(40)) for t in texts})
    teacher = LocalLLMTeacher("fake/model", labels, max_options=10, shortlister=short)
    assert teacher._candidates(texts) == _candidates_at_8585879(teacher, texts)
    assert teacher._candidates(texts, [[] for _ in texts]) == _candidates_at_8585879(teacher, texts)
    small = LocalLLMTeacher("fake/model", labels[:5])
    assert small._candidates(texts[:3]) == [[0, 1, 2, 3, 4]] * 3


DEMOS = [("card card pin", 0),   # same direction as the query below: the nearest
         ("card blocked", 1),    # one "card": second nearest
         ("pin reset", 2),       # one "pin": third
         ("loan emi", 3),        # unrelated
         ("card pin card", 4)]   # identical to the query: never shown for it
QUERY = "card pin card"


def _fewshot_teacher(fake_embedder, k=2, **kwargs):
    labels = ["card_declined", "card_blocked", "pin_reset", "loan_emi", "card_pin"]
    teacher = LocalLLMTeacher("fake/model", labels, demonstrations=DEMOS, k_demos=k, demo_embedder=fake_embedder,
                              **kwargs)
    teacher._tok = FakeTokenizer()
    return teacher


def test_fewshot_prompt_shows_the_nearest_demonstrations_closest_last(fake_embedder):
    teacher = _fewshot_teacher(fake_embedder, k=3)
    (shown,) = teacher._select_demos([QUERY])
    assert shown == [2, 1, 0]  # least to most similar; the identical text (index 4) is skipped
    prompt = teacher._prompt(QUERY, [0, 1, 2, 3, 4], shown)
    positions = [prompt.index(f"Text: {DEMOS[d][0]}\n") for d in shown]
    assert positions == sorted(positions)
    query_at = prompt.index(f"Now classify this text.\n\nText: {QUERY}\n")
    assert positions[-1] < query_at
    # Each demonstration is followed by its label's description, not a letter.
    assert "Text: card card pin\nCorrect answer: card declined" in prompt
    assert "Text: pin reset\nCorrect answer: pin reset" in prompt
    assert "loan emi" not in prompt.split("Options:")[0]
    assert prompt.endswith("Answer with the letter only.\nAnswer:")
    # Demonstrations are embedded once, on first use.
    calls = []
    teacher.demo_embedder = SimpleNamespace(encode=lambda texts: calls.append(list(texts)) or
                                            fake_embedder.encode(texts))
    teacher._select_demos([QUERY, "loan"])
    assert calls == [[QUERY, "loan"]]


def test_fewshot_candidates_include_the_demonstrations_labels(fake_embedder):
    labels = [f"intent_{i}" for i in range(30)]
    demos = [("card card pin", 20), ("card blocked", 25), ("loan emi", 3)]
    teacher = LocalLLMTeacher("fake/model", labels, max_options=4, shortlister=FixedShortlister(30),
                              demonstrations=demos, k_demos=2, demo_embedder=fake_embedder)
    teacher._tok = FakeTokenizer()
    shown = teacher._select_demos([QUERY])
    assert shown == [[1, 0]]
    (cand,) = teacher._candidates([QUERY], shown)
    assert len(cand) == 4 and len(set(cand)) == 4
    assert set(cand) == {20, 25, 0, 1}  # both demonstration labels, then the shortlister's best two
    assert teacher._candidates([QUERY], shown) == [cand]  # the shuffle is deterministic per text
    # When the demonstrations' labels are already shortlisted, the candidates are exactly the zero-shot ones.
    inside = LocalLLMTeacher("fake/model", labels, max_options=4, shortlister=FixedShortlister(30),
                             demonstrations=[("card card pin", 2), ("card blocked", 1)], k_demos=2,
                             demo_embedder=fake_embedder)
    assert inside._candidates([QUERY], inside._select_demos([QUERY])) == _candidates_at_8585879(inside, [QUERY])
    # More distinct demonstration labels than options: capped, the most similar demonstrations' labels kept.
    crowded = LocalLLMTeacher("fake/model", labels, max_options=2, shortlister=FixedShortlister(30),
                              demonstrations=demos, k_demos=3, demo_embedder=fake_embedder)
    (c3,) = crowded._candidates([QUERY], crowded._select_demos([QUERY]))
    assert set(c3) == {20, 25}


def test_fewshot_arguments_are_validated(fake_embedder):
    labels = ["a", "b"]
    with pytest.raises(ValueError, match="demonstrations"):
        LocalLLMTeacher("fake/model", labels, k_demos=2, demo_embedder=fake_embedder)
    with pytest.raises(ValueError, match="demo_embedder"):
        LocalLLMTeacher("fake/model", labels, demonstrations=[("x", 0)], k_demos=1)
    with pytest.raises(ValueError, match="out of range"):
        LocalLLMTeacher("fake/model", labels, demonstrations=[("x", 2)], k_demos=1, demo_embedder=fake_embedder)
    with pytest.raises(ValueError, match="k_demos"):
        LocalLLMTeacher("fake/model", labels, k_demos=-1)
    # Demonstrations without k_demos are simply unused: the default stays zero-shot.
    off = LocalLLMTeacher("fake/model", labels, demonstrations=[("x", 0)])
    assert off._select_demos(["y", "z"]) == [[], []]


def test_label_records_demonstrations_per_item_in_stats(fake_embedder):
    torch = pytest.importorskip("torch")

    class Batch(dict):
        def to(self, device):
            return self

    class Tok(FakeTokenizer):
        def __call__(self, prompts, return_tensors=None, padding=None):
            return Batch(input_ids=torch.zeros((len(prompts), 1), dtype=torch.long))

    class LM:
        """Always prefers option B."""

        def parameters(self):
            yield torch.zeros(1)

        def __call__(self, input_ids):
            logits = torch.zeros((input_ids.shape[0], 1, 32))
            logits[:, -1, 1] = 4.0
            return SimpleNamespace(logits=logits)

    teacher = _fewshot_teacher(fake_embedder, k=2)
    teacher._tok, teacher._model, teacher._letter_ids = Tok(), LM(), list(range(5))
    probs = teacher.label([QUERY, "loan emi please", "pin"])
    assert probs.shape == (3, 5)
    assert np.allclose(probs.sum(1), 1.0) and (probs.argmax(1) == 1).all()
    stats = teacher.stats()
    assert stats["k_demos"] == 2 and stats["demonstrations_per_item"] == 2.0
    assert stats["demonstrations_histogram"] == {2: 3}

    zero = _fewshot_teacher(fake_embedder, k=0)
    zero._tok, zero._model, zero._letter_ids = Tok(), LM(), list(range(5))
    zero.label(["pin"])
    assert zero.stats()["demonstrations_per_item"] == 0.0 and zero.stats()["k_demos"] == 0
    assert math.isclose(zero.stats()["items_labelled"], 1)
