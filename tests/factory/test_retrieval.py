"""forgelm.retrieval: chunking, BM25, dense and hybrid search, metrics, citations.

No downloads: dense search is driven by the keyword-count FakeEmbedder from
conftest.py, and every metric value is checked against a hand computation.
"""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from forgelm.retrieval import (BM25Index, DenseIndex, FusedHit, HybridIndex, Passage, chunk_document, cite,
                               evaluate_rankings, evaluate_retrieval, extract_quotes, mrr_at_k, ndcg_at_k,
                               recall_at_k, tokenize, verify_quotes)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def topic_passages(toy_text):
    """The 30 conftest texts as passages with ids like "loan-2" (topic, position within topic)."""
    texts, y, labels = toy_text
    return [Passage(f"{labels[c]}-{int(np.sum(y[:i] == c))}", t, title=labels[c])
            for i, (t, c) in enumerate(zip(texts, y))]


def ids(hits):
    return [h[0].id for h in hits]


# --------------------------------------------------------------------------- import weight

def test_module_imports_with_numpy_only():
    code = ("import sys, forgelm.retrieval; "
            "heavy = [m for m in ('scipy', 'sklearn', 'torch', 'transformers', 'sentence_transformers', "
            "'model2vec', 'pandas') if m in sys.modules]; print(heavy); assert not heavy")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


# --------------------------------------------------------------------------- chunking

DOC = ("Article 1.  Scope\n\nThis regulation applies to every   credit institution\n"
       "authorised in the Union.\tArticle 2. Definitions: 'consumer' means a natural person acting "
       "for purposes outside a trade.  ")


@pytest.mark.parametrize("max_words,overlap", [(5, 2), (4, 0), (7, 6), (1000, 10)])
def test_chunk_offsets_reproduce_the_original_text(max_words, overlap):
    parts = chunk_document(DOC, max_words=max_words, overlap_words=overlap, doc_id="reg", title="Reg")
    assert parts, "a non-empty document must give passages"
    for n, p in enumerate(parts):
        assert DOC[p.start:p.end] == p.text  # every passage is an exact slice
        assert p.id == f"reg#{n}" and p.doc_id == "reg" and p.title == "Reg"
        assert len(p.text.split()) <= max_words
    # Stitch the document back from the passages alone (plus the whitespace
    # between them when they do not overlap).
    rebuilt = parts[0].text
    for prev, p in zip(parts, parts[1:]):
        if overlap:
            assert p.start < prev.end
            rebuilt += p.text[prev.end - p.start:]
        else:
            gap = DOC[prev.end:p.start]
            assert gap.strip() == "" and gap
            rebuilt += gap + p.text
    assert rebuilt == DOC.strip()
    assert parts[0].start == DOC.index("Article") and parts[-1].end == len(DOC.rstrip())


def test_chunk_overlap_and_edge_cases():
    words = " ".join(f"w{i}" for i in range(10))
    parts = chunk_document(words, max_words=4, overlap_words=1)
    assert [p.text for p in parts] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    # The last window stops at the last word: no tail passage contained in the previous one.
    assert [p.text for p in chunk_document(words, max_words=6, overlap_words=2)] == \
        ["w0 w1 w2 w3 w4 w5", "w4 w5 w6 w7 w8 w9"]
    assert chunk_document("   \n ") == []
    with pytest.raises(ValueError):
        chunk_document(words, max_words=3, overlap_words=3)
    with pytest.raises(ValueError):
        chunk_document(words, max_words=0, overlap_words=0)


# --------------------------------------------------------------------------- BM25

def test_tokenizer_is_lowercase_unicode_words():
    assert tokenize("Section 4(b): Kündigung, KYC-check_2 ") == ["section", "4", "b", "kündigung", "kyc", "check_2"]
    assert tokenize("") == [] and tokenize(" ,.; ") == []


def test_bm25_ranks_the_obvious_match_first(topic_passages):
    index = BM25Index().build(topic_passages)
    hits = index.search("change the nominee on my insurance policy", k=3)
    assert hits[0][0].text == "change the nominee on my policy"
    assert all(h[1] > 0 for h in hits) and [h[1] for h in hits] == sorted((h[1] for h in hits), reverse=True)
    assert ids(index.search("contactless debit card", k=1)) == ["card-9"]


def test_bm25_score_matches_the_formula():
    docs = [Passage("a", "loan loan rate"), Passage("b", "loan card"), Passage("c", "card card card pin")]
    k1, b = 1.2, 0.75
    index = BM25Index(k1, b).build(docs)
    n, lengths = 3, {"a": 3, "b": 2, "c": 4}
    avg = sum(lengths.values()) / 3
    idf_loan = math.log(1 + (n - 2 + 0.5) / (2 + 0.5))
    idf_rate = math.log(1 + (n - 1 + 0.5) / (1 + 0.5))

    def w(tf, dl, idf):
        return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg))

    expected_a = w(2, 3, idf_loan) + w(1, 3, idf_rate)
    expected_b = w(1, 2, idf_loan)
    scores = index.scores("loan rate")
    assert scores[0] == pytest.approx(expected_a, rel=1e-6)
    assert scores[1] == pytest.approx(expected_b, rel=1e-6)
    assert scores[2] == 0
    # A repeated query token counts once per occurrence.
    assert index.scores("rate rate")[0] == pytest.approx(2 * w(1, 3, idf_rate), rel=1e-6)
    assert ids(index.search("loan rate")) == ["a", "b"]  # "c" shares no term and is not returned


def test_bm25_empty_query_unseen_terms_and_duplicates():
    docs = [Passage("p1", "early repayment charge on a fixed rate mortgage"),
            Passage("p2", "card replacement fee"),
            Passage("p3", "early repayment charge on a fixed rate mortgage"),  # same text, different id
            Passage("p4", "")]
    index = BM25Index().build(docs)
    assert index.search("") == [] and index.search("   ?! ") == []
    assert index.search("cryptocurrency staking") == []
    hits = index.search("early repayment charge cryptocurrency", k=5)
    assert ids(hits) == ["p1", "p3"] and hits[0][1] == hits[1][1]
    assert ids(index.search("repayment", k=1)) == ["p1"]  # ties go to the earlier passage
    assert index.nbytes() > 0
    with pytest.raises(ValueError, match="duplicate passage id"):
        BM25Index().build([Passage("x", "a"), Passage("x", "b")])
    with pytest.raises(RuntimeError):
        BM25Index().search("a")


def test_bm25_large_k_and_empty_index():
    index = BM25Index().build([Passage("only", "one passage")])
    assert ids(index.search("passage", k=50)) == ["only"]
    assert BM25Index().build([]).search("anything") == []


# --------------------------------------------------------------------------- dense

class RecordingEmbedder:
    """Wraps the fake embedder and remembers what text it was asked to encode."""

    def __init__(self, inner):
        self.inner, self.model_name, self.seen = inner, "recording", []

    def encode(self, texts):
        self.seen.extend(texts)
        return self.inner.encode(texts)


def test_dense_index_with_fake_embedder(fake_embedder, topic_passages):
    index = DenseIndex(fake_embedder).build(topic_passages)
    hits = index.search("what is my loan emi", k=5)
    assert all(pid.startswith("loan-") for pid in ids(hits))
    assert hits[0][0].id == "loan-2"  # the only passage with both "loan" and "emi"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0, atol=1e-5)
    assert index.search("   ") == []
    assert index.nbytes() == index.vectors.nbytes
    assert index.name == "dense:fake"


def test_dense_index_prefixes_precomputed_vectors_and_batching(fake_embedder, topic_passages):
    rec = RecordingEmbedder(fake_embedder)
    index = DenseIndex(rec, query_prefix="query: ", passage_prefix="passage: ").build(topic_passages)
    assert rec.seen[0] == "passage: " + topic_passages[0].text
    rec.seen.clear()
    queries = ["premium due", "debit card pin", "car loan interest"]
    expected = index.search_batch(queries, k=4)
    assert rec.seen == ["query: " + q for q in queries]

    # Unnormalised precomputed vectors give the same ranking as encoding.
    raw = 3.0 * fake_embedder.encode([p.text for p in topic_passages])
    pre = DenseIndex(fake_embedder, max_scores_per_batch=1).build(topic_passages, vectors=raw)
    got = pre.search_batch(queries, k=4)  # one query per batch
    assert [ids(h) for h in got] == [ids(h) for h in expected]
    qv = fake_embedder.encode(["premium due"])[0] * 7
    assert ids(pre.search("", k=4, query_vector=qv)) == ids(expected[0])
    with pytest.raises(ValueError):
        DenseIndex(fake_embedder).build(topic_passages, vectors=raw[:3])


# --------------------------------------------------------------------------- hybrid

class FixedIndex:
    def __init__(self, name, order):
        self.name, self.order = name, order

    def search(self, query, k=10):
        return [(Passage(pid, pid), 1.0 / (r + 1)) for r, pid in enumerate(self.order[:k])]


def test_reciprocal_rank_fusion_order_and_scores():
    hybrid = HybridIndex([FixedIndex("lex", ["a", "b", "c"]), FixedIndex("dense", ["c", "a", "d"])])
    hits = hybrid.search("q", k=10)
    assert ids(hits) == ["a", "c", "b", "d"]
    assert isinstance(hits[0], FusedHit)
    assert hits[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert hits[1].score == pytest.approx(1 / 63 + 1 / 61)
    assert hits[2].score == pytest.approx(1 / 62) and hits[3].score == pytest.approx(1 / 63)
    assert hits[0].ranks == {"lex": 1, "dense": 2}
    assert hits[2].ranks == {"lex": 2, "dense": None}
    assert ids(hybrid.search("q", k=2)) == ["a", "c"]


def test_fusion_ties_names_and_constant():
    # x and y tie on fused score and on best rank (1 each); y is read first.
    hybrid = HybridIndex({"one": FixedIndex("same", ["y", "x"]), "two": FixedIndex("same", ["x", "y"])})
    assert ids(hybrid.search("q")) == ["y", "x"]
    # With rrf_k=0 all three score 1.0 (b: 1/1, c: 1/1, a: 1/2 + 1/2); a never
    # reached rank 1 so it comes last, and b beats c by being read first.
    hybrid = HybridIndex([FixedIndex("l", ["b", "a"]), FixedIndex("d", ["c", "a"])], rrf_k=0)
    hits = hybrid.search("q")
    assert ids(hits) == ["b", "c", "a"] and [h.score for h in hits] == [1.0, 1.0, 1.0]
    hybrid = HybridIndex([FixedIndex("same", ["x"]), FixedIndex("same", ["y"])])
    assert list(hybrid.components) == ["same", "same#2"]
    assert ids(hybrid.search("q")) == ["x", "y"]
    # Component depth limits what each component contributes.
    deep = HybridIndex([FixedIndex("a", ["p", "q", "r"]), FixedIndex("b", ["r"])], depth=1)
    assert ids(deep.search("q", k=1)) == ["p"]
    with pytest.raises(ValueError):
        HybridIndex([])


def test_hybrid_over_real_indexes(fake_embedder, topic_passages):
    bm25 = BM25Index().build(topic_passages)
    dense = DenseIndex(fake_embedder).build(topic_passages)
    hybrid = HybridIndex([bm25, dense])
    hits = hybrid.search("lapsed policy revival", k=3)
    assert hits[0].passage.id == "policy-5"
    assert set(hits[0].ranks) == {"bm25", "dense:fake"}
    assert hybrid.nbytes() == bm25.nbytes() + dense.nbytes()


# --------------------------------------------------------------------------- metrics

# Ranking a b c d e; relevant: b (1), d (2) and x (1), which was never retrieved.
RANKED = ["a", "b", "c", "d", "e"]
REL = {"b": 1, "d": 2, "x": 1, "e": 0}
L2, L3, L5 = math.log2(2), math.log2(3), math.log2(5)
IDCG_5 = 2 / L2 + 1 / L3 + 1 / math.log2(4)  # ideal order d, b, x


def test_recall_mrr_ndcg_hand_computed():
    assert recall_at_k(RANKED, REL, 1) == 0
    assert recall_at_k(RANKED, REL, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, REL, 5) == pytest.approx(2 / 3)  # binary: d's grade 2 counts as one
    assert mrr_at_k(RANKED, REL, 1) == 0
    assert mrr_at_k(RANKED, REL, 5) == pytest.approx(1 / 2)
    assert ndcg_at_k(RANKED, REL, 1) == 0
    assert ndcg_at_k(RANKED, REL, 2) == pytest.approx((1 / L3) / (2 / L2 + 1 / L3))
    assert ndcg_at_k(RANKED, REL, 5) == pytest.approx((1 / L3 + 2 / L5) / IDCG_5)
    assert ndcg_at_k(RANKED, REL, 5) == pytest.approx(0.476626, abs=1e-6)  # 1.4922829 / 3.1309298
    # Perfect graded order scores 1; swapping the grades scores less.
    assert ndcg_at_k(["d", "b", "x"], REL, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "d", "x"], REL, 3) == pytest.approx((1 + 2 / L3 + 1 / 2) / (2 + 1 / L3 + 1 / 2))
    # A passage listed twice is credited once, and the repeat still uses its slot.
    assert recall_at_k(["b", "b", "d"], REL, 2) == pytest.approx(1 / 3)
    assert ndcg_at_k(["b", "b", "d"], REL, 3) == pytest.approx((1 + 2 / 2) / (2 + 1 / L3 + 1 / 2))
    # No relevant passage: undefined.
    assert math.isnan(recall_at_k(RANKED, {"a": 0}, 5)) and math.isnan(ndcg_at_k(RANKED, {}, 5))


def test_evaluate_rankings_averages_and_skips_queries_without_relevant():
    rankings = {"q1": RANKED, "q2": ["r", "s"], "q3": ["a"], "q4": ["z"]}
    qrels = {"q1": REL, "q2": {"s": 1}, "q3": {"a": 0}}  # q3 judged but nothing relevant; q4 not judged
    m = evaluate_rankings(rankings, qrels, ks=(5, 1))
    assert m["n_queries"] == 2 and m["n_queries_without_relevant"] == 2
    assert m["recall@5"] == pytest.approx((2 / 3 + 1) / 2)
    assert m["mrr@5"] == pytest.approx((1 / 2 + 1 / 2) / 2)
    assert m["mrr@1"] == 0
    assert m["ndcg@5"] == pytest.approx(((1 / L3 + 2 / L5) / IDCG_5 + 1 / L3) / 2)
    empty = evaluate_rankings({"q": ["a"]}, {}, ks=(1,))
    assert empty["n_queries"] == 0 and math.isnan(empty["recall@1"])


def test_evaluate_retrieval_end_to_end(topic_passages):
    index = BM25Index().build(topic_passages)
    queries = {"q-emi": "loan emi deducted", "q-nominee": "nominee change", "q-none": "weather today"}
    qrels = {"q-emi": {"loan-2": 1}, "q-nominee": {"policy-1": 2}}
    m = evaluate_retrieval(index, queries, qrels, ks=(1, 10))
    assert m["recall@1"] == 1 and m["ndcg@10"] == 1 and m["mrr@10"] == 1
    assert m["n_queries"] == 2 and m["n_queries_without_relevant"] == 1


# --------------------------------------------------------------------------- citations

SOURCE = Passage("reg#3", "A creditor shall provide the consumer with the annual percentage rate\n"
                          "before the agreement is concluded.", doc_id="reg", title="Credit rules", start=410, end=513)
OTHER = Passage("faq#1", "Fees are listed in the tariff of charges.")


def test_cite_formats_numbered_sources():
    text = cite([(SOURCE, 0.9), OTHER])
    assert text.startswith("[1] id=reg#3 | title: Credit rules | document: reg | chars 410-513\nA creditor")
    assert "\n\n[2] id=faq#1\nFees are listed" in text
    assert cite([SOURCE], max_chars=10).endswith("A creditor [...]")


def test_verify_quotes_catches_an_invented_quote_and_accepts_a_real_one():
    answer = ('The rule says “provide the consumer with the annual percentage rate before the agreement” [1]. '
              'It also says "the creditor may charge any fee it chooses" [1], and fees are in the "tariff" [2].')
    assert extract_quotes(answer) == ["provide the consumer with the annual percentage rate before the agreement",
                                      "the creditor may charge any fee it chooses"]
    # The real quote spans a line break in the source; whitespace differences are allowed.
    assert verify_quotes(answer, [(SOURCE, 1.0), OTHER]) == ["the creditor may charge any fee it chooses"]
    # A quote is only supported by the passages actually given.
    assert len(verify_quotes(answer, [OTHER])) == 2
    # Changed case or wording is not verbatim; short quotes are not checked.
    assert verify_quotes('"A Creditor shall provide the consumer"', [SOURCE]) == ["A Creditor shall provide the consumer"]
    assert verify_quotes('"A creditor shall provide the consumer"', [SOURCE]) == []
    assert verify_quotes('The "APR" applies.', [OTHER]) == []
    assert verify_quotes("No quotations here.", []) == []
