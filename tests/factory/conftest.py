"""Fixtures for the forgelm factory tests.

These tests need no GPU, no network and no model downloads: they run in a few
seconds on any machine with numpy, pandas and scikit-learn.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TOPICS = {
    "card": ["my card was declined at the shop", "the card got blocked yesterday", "card stopped working abroad",
             "new card has not arrived yet", "please activate my debit card", "lost my credit card today",
             "card payment failed twice", "replace my damaged card", "card pin is not accepted",
             "contactless card not working"],
    "loan": ["what is the interest rate on a home loan", "i want to prepay my personal loan",
             "loan emi was deducted twice", "how do i apply for a car loan", "loan statement for tax filing",
             "reduce the tenure of my loan", "loan sanction letter is pending", "top up on existing loan",
             "foreclose my loan account", "loan eligibility for salaried person"],
    "policy": ["when does my insurance policy mature", "change the nominee on my policy",
               "premium due date for term policy", "surrender value of my policy", "policy document not received",
               "revive a lapsed policy", "add a rider to my policy", "free look cancellation of policy",
               "update address on policy", "claim status for my policy"],
}


@pytest.fixture
def toy_text():
    """30 short texts in 3 well-separated classes: (texts, y, labels)."""
    labels = sorted(TOPICS)
    texts, y = [], []
    for i, label in enumerate(labels):
        texts += TOPICS[label]
        y += [i] * len(TOPICS[label])
    return texts, np.array(y), labels


class FakeEmbedder:
    """Deterministic bag-of-keywords vectors; stands in for a real encoder."""

    model_name, backend, max_seq_length = "fake", "static", None
    vocab = ["card", "loan", "policy", "pin", "emi", "premium", "nominee", "interest", "debit", "claim"]

    def encode(self, texts):
        out = np.array([[t.lower().count(w) for w in self.vocab] for t in texts], dtype=np.float32) + 1e-3
        return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
