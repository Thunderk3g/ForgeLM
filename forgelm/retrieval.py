"""Passage retrieval with quoted sources: lexical, dense, hybrid, and how to measure them.

The planner recommends "retrieval with quoted sources" for policy and
regulation question answering: knowledge that changes belongs in documents
the model reads at answer time, not in its weights, and an answer that quotes
its source can be checked. This module is the small, dependable core of that
recommendation:

* ``chunk_document`` splits long documents into overlapping passages and keeps
  each passage's character offsets, so an answer can cite the exact span.
* ``BM25Index`` is Okapi BM25 over a sparse term-passage matrix. It needs no
  model, indexes tens of thousands of passages in seconds and is a strong
  baseline on jargon-heavy text (clause numbers, product names, tickers).
* ``DenseIndex`` wraps ``forgelm.students.Embedder`` (a Model2Vec static model
  or a sentence encoder) and searches by cosine similarity with numpy.
* ``HybridIndex`` fuses any number of indexes by reciprocal rank fusion.
* ``evaluate_retrieval`` / ``evaluate_rankings`` compute recall@k, MRR@k and
  nDCG@k against relevance judgements. Retrieval should be measured on its own,
  before answer quality: an answer cannot cite a passage that was not found.
* ``cite`` formats retrieved passages as numbered sources for a prompt, and
  ``verify_quotes`` checks that every quotation in an answer occurs verbatim in
  those sources: the cheap guardrail behind "answer with verbatim citations".

Every index has the same surface: ``build(passages)``, ``search(query, k)`` and
``search_batch(queries, k)``. A search returns a list of tuples whose first two
items are ``(passage, score)``, best first. Exact brute-force search is used
throughout; up to a few hundred thousand passages it is fast enough on a CPU
and, unlike an approximate index, it cannot silently miss a passage.

Only numpy is needed at import. scipy is imported by ``BM25Index.build``; the
embedding backends are imported by ``Embedder`` when it first encodes.
"""

from __future__ import annotations

import math
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from .metrics import EPS

# ---------------------------------------------------------------------------
# Passages and chunking


@dataclass
class Passage:
    """One retrievable unit of text.

    ``doc_id``, ``start`` and ``end`` locate the passage in its source
    document: ``document_text[start:end] == text`` when it came from
    ``chunk_document``. They are what makes a citation checkable by a person.
    """

    id: str
    text: str
    doc_id: Optional[str] = None
    title: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


_WORD = re.compile(r"\S+")


def chunk_document(text: str, max_words: int = 200, overlap_words: int = 40, doc_id: str = "doc",
                   title: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> List[Passage]:
    """Split ``text`` into passages of at most ``max_words`` whitespace-separated words.

    Consecutive passages share ``overlap_words`` words, so a sentence that
    straddles a boundary is whole in at least one passage when it is shorter
    than the overlap. Each passage is an exact slice of the original text
    (``text[p.start:p.end] == p.text``): the original whitespace and line
    breaks are kept, which is what lets ``verify_quotes`` and a reader find a
    quoted span in the source document.

    Passage ids are ``"{doc_id}#{n}"`` with n counted from 0. A document with
    no words yields no passages.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be in [0, max_words)")
    spans = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    if not spans:
        return []
    stride = max_words - overlap_words
    out: List[Passage] = []
    first = 0
    while True:
        last = min(first + max_words, len(spans)) - 1
        start, end = spans[first][0], spans[last][1]
        out.append(Passage(id=f"{doc_id}#{len(out)}", text=text[start:end], doc_id=doc_id, title=title,
                           start=start, end=end, metadata=dict(metadata or {})))
        # Stop once a window reaches the last word; another step would only
        # produce a passage wholly contained in this one.
        if last == len(spans) - 1:
            return out
        first += stride


def _check_unique_ids(passages: Sequence[Passage]) -> None:
    # Evaluation and fusion identify passages by id; two passages sharing an
    # id would be counted, or fused, as one.
    seen = set()
    for p in passages:
        if p.id in seen:
            raise ValueError(f"duplicate passage id {p.id!r}")
        seen.add(p.id)


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` highest scores, best first; ties go to the lower index.

    A deterministic tie order matters: duplicated passages score identically,
    and an evaluation should not change from run to run because a partial sort
    picked a different one of them.
    """
    n = len(scores)
    k = min(k, n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k == n:
        cand = np.arange(n)
    else:
        kth = np.partition(scores, n - k)[n - k]
        above = np.flatnonzero(scores > kth)
        ties = np.flatnonzero(scores == kth)[: k - len(above)]
        cand = np.concatenate([above, ties])
    return cand[np.lexsort((cand, -scores[cand]))]


# ---------------------------------------------------------------------------
# Lexical retrieval

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    """Lowercase, then take maximal runs of Unicode word characters.

    Word characters are letters, digits and the underscore in any script, so
    "Section 4(b)" gives ["section", "4", "b"] and "Kündigung" stays one token.
    There is no stemming and no stop-word list: both are language-specific,
    and BM25's IDF already gives very common words little weight.
    """
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Okapi BM25 over a sparse passage-term weight matrix.

    For a query q and passage d::

        score(q, d) = sum over query tokens t of
                      idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * len(d) / avg_len))

        idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    This IDF is the Lucene / Elasticsearch variant: unlike Robertson's
    original it never goes negative, so a term that appears in most passages
    adds a little instead of penalising the passages that contain it. A token
    repeated in the query counts once per occurrence. ``k1`` (term-frequency
    saturation) and ``b`` (length normalisation) default to the Lucene values
    1.2 and 0.75.

    The whole BM25 weight of every (passage, term) pair is computed once at
    build time and stored in a compressed sparse column matrix, so a query is
    a slice of the columns of its terms and one sparse-dense product.
    Passages that share no term with the query are not returned: their score
    is zero and carries no ranking information.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 needs k1 >= 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b
        self.passages: List[Passage] = []
        self.vocab: Dict[str, int] = {}
        self.idf = np.empty(0)
        self._weights = None  # scipy.sparse.csc_matrix, passages x terms

    def __len__(self) -> int:
        return len(self.passages)

    def build(self, passages: Sequence[Passage]) -> "BM25Index":
        from scipy import sparse

        passages = list(passages)
        _check_unique_ids(passages)
        vocab: Dict[str, int] = {}
        term_ids: List[int] = []
        lengths = np.zeros(len(passages), dtype=np.float64)
        for i, p in enumerate(passages):
            toks = tokenize(p.text)
            lengths[i] = len(toks)
            term_ids.extend(vocab.setdefault(t, len(vocab)) for t in toks)
        n_docs, n_terms = len(passages), len(vocab)
        rows = np.repeat(np.arange(n_docs), lengths.astype(np.int64))
        # coo -> csr adds up the repeated (passage, term) entries: term frequency.
        tf = sparse.coo_matrix((np.ones(len(term_ids), dtype=np.float64), (rows, np.asarray(term_ids, np.int64))),
                               shape=(n_docs, n_terms)).tocsr()
        tf.sum_duplicates()
        df = np.bincount(tf.indices, minlength=n_terms).astype(np.float64)
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        avg_len = lengths.mean() if n_docs and lengths.mean() > 0 else 1.0
        norm = self.k1 * (1.0 - self.b + self.b * lengths / avg_len)
        doc_of_entry = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        tfv = tf.data
        weights = self.idf[tf.indices] * tfv * (self.k1 + 1.0) / (tfv + norm[doc_of_entry])
        self._weights = sparse.csr_matrix((weights.astype(np.float32), tf.indices, tf.indptr),
                                          shape=(n_docs, n_terms)).tocsc()
        self.passages, self.vocab = passages, vocab
        return self

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every passage for ``query`` (zeros when no query term is known)."""
        if self._weights is None:
            raise RuntimeError("call build() first")
        counts: Dict[int, int] = {}
        for t in tokenize(query):
            j = self.vocab.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        if not counts:
            return np.zeros(len(self.passages), dtype=np.float32)
        cols = np.fromiter(counts.keys(), dtype=np.int64)
        q = np.fromiter(counts.values(), dtype=np.float32)
        return np.asarray(self._weights[:, cols] @ q).ravel()

    def search(self, query: str, k: int = 10) -> List[Tuple[Passage, float]]:
        s = self.scores(query)
        return [(self.passages[i], float(s[i])) for i in _top_k(s, k) if s[i] > 0]

    def search_batch(self, queries: Sequence[str], k: int = 10) -> List[List[Tuple[Passage, float]]]:
        return [self.search(q, k) for q in queries]

    def nbytes(self) -> int:
        """Approximate memory of the index structures, excluding the passage texts.

        Counts the sparse matrix arrays, the IDF vector and the vocabulary
        dictionary (keys, values and table). Passage texts are left out because
        every index over the same passages holds the same texts.
        """
        if self._weights is None:
            return 0
        w = self._weights
        vocab = sys.getsizeof(self.vocab) + sum(sys.getsizeof(t) + sys.getsizeof(i) for t, i in self.vocab.items())
        return int(w.data.nbytes + w.indices.nbytes + w.indptr.nbytes + self.idf.nbytes + vocab)


# ---------------------------------------------------------------------------
# Dense retrieval


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


class DenseIndex:
    """Cosine-similarity search over embedding vectors.

    ``embedder`` is a ``forgelm.students.Embedder`` (or anything with an
    ``encode(texts) -> array`` method), or a model name from which one is
    built: Model2Vec ``potion-*`` models load as static embedders, anything
    else as a sentence-transformers encoder.

    Prefixes: several retrieval models are trained with a fixed string in
    front of the query, the passage or both, and lose accuracy without it.
    Pass it here and it is added for you:

    * BAAI/bge-*-en-v1.5: ``query_prefix="Represent this sentence for
      searching relevant passages: "``, no passage prefix.
    * intfloat/e5-*: ``query_prefix="query: "``, ``passage_prefix="passage: "``.
    * Model2Vec static models and all-MiniLM-style encoders: no prefix.

    Check the model card of any other model. Vectors are stored L2-normalised
    in float32, so the dot product is the cosine similarity; precomputed
    vectors (for example from a cache) can be passed to ``build`` and are
    normalised the same way. Search is exact: the query batch is multiplied by
    the whole passage matrix, in query batches sized so the score matrix stays
    under about 64 MB.
    """

    def __init__(self, embedder: Union[str, Any], query_prefix: str = "", passage_prefix: str = "",
                 name: Optional[str] = None, max_scores_per_batch: int = 1 << 24):
        if isinstance(embedder, str):
            from .students import Embedder

            embedder = Embedder(embedder)
        self.embedder = embedder
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.name = name or f"dense:{getattr(embedder, 'model_name', type(embedder).__name__)}"
        self.max_scores_per_batch = max_scores_per_batch
        self.passages: List[Passage] = []
        self.vectors: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.passages)

    def encode_passages(self, passages: Sequence[Passage]) -> np.ndarray:
        return _normalise(self.embedder.encode([self.passage_prefix + p.text for p in passages]))

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return _normalise(self.embedder.encode([self.query_prefix + q for q in queries]))

    def build(self, passages: Sequence[Passage], vectors: Optional[np.ndarray] = None) -> "DenseIndex":
        passages = list(passages)
        _check_unique_ids(passages)
        if vectors is None:
            vectors = self.encode_passages(passages) if passages else np.zeros((0, 0), np.float32)
        elif len(vectors) != len(passages):
            raise ValueError(f"{len(vectors)} vectors for {len(passages)} passages")
        self.passages, self.vectors = passages, _normalise(vectors)
        return self

    def search(self, query: str, k: int = 10) -> List[Tuple[Passage, float]]:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: Sequence[str], k: int = 10) -> List[List[Tuple[Passage, float]]]:
        """Top ``k`` passages for each query.

        A blank query returns no passages (an encoder would still produce a
        vector for it, but the ranking would mean nothing).
        """
        if self.vectors is None:
            raise RuntimeError("call build() first")
        out: List[List[Tuple[Passage, float]]] = [[] for _ in queries]
        todo = [i for i, q in enumerate(queries) if q.strip()]
        Q = self.encode_queries([queries[i] for i in todo]) if todo else None
        if not todo or not len(self.passages):
            return out
        step = max(1, self.max_scores_per_batch // len(self.passages))
        for lo in range(0, len(todo), step):
            S = Q[lo:lo + step] @ self.vectors.T
            for row, i in zip(S, todo[lo:lo + step]):
                out[i] = [(self.passages[j], float(row[j])) for j in _top_k(row, k)]
        return out

    def nbytes(self) -> int:
        """Memory of the stored vectors. The embedding model itself is not counted."""
        return 0 if self.vectors is None else int(self.vectors.nbytes)


# ---------------------------------------------------------------------------
# Hybrid retrieval


class FusedHit(NamedTuple):
    """A fused result: ``ranks`` maps each component name to the passage's
    1-based rank in that component's list, or None when it was not in it."""

    passage: Passage
    score: float
    ranks: Dict[str, Optional[int]]


class HybridIndex:
    """Reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009).

    Each component returns its top ``depth`` passages; a passage's fused
    score is ``sum over components of 1 / (rrf_k + rank)`` with ranks counted
    from 1, and a component that did not return the passage adds nothing.
    Only ranks are used, so a lexical score and a cosine similarity, which
    live on unrelated scales, need no calibration against each other. The
    constant 60 is the paper's; larger values flatten the gap between the
    first few ranks.

    ``indexes`` is a sequence (component names come from each index's
    ``name`` attribute) or a mapping of name to index. Components are matched
    by passage id, so they should index the same passages. Ties in fused
    score go to the passage with the better best rank, then to the one seen
    first when the components' lists are read in the order given.
    """

    def __init__(self, indexes: Union[Sequence[Any], Mapping[str, Any]], rrf_k: float = 60.0, depth: int = 100,
                 name: str = "hybrid"):
        items = list(indexes.items()) if isinstance(indexes, Mapping) else \
            [(getattr(ix, "name", f"index{i}"), ix) for i, ix in enumerate(indexes)]
        if not items:
            raise ValueError("HybridIndex needs at least one component index")
        names: List[str] = []
        for n, _ in items:
            base, j = n, 2
            while n in names:  # two components of the same kind still need distinct names
                n, j = f"{base}#{j}", j + 1
            names.append(n)
        self.components = dict(zip(names, (ix for _, ix in items)))
        self.rrf_k, self.depth, self.name = rrf_k, depth, name

    def build(self, passages: Sequence[Passage]) -> "HybridIndex":
        """Build every component on ``passages``. Components that need
        precomputed vectors should be built beforehand instead."""
        passages = list(passages)
        for ix in self.components.values():
            ix.build(passages)
        return self

    def search(self, query: str, k: int = 10) -> List[FusedHit]:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: Sequence[str], k: int = 10) -> List[List[FusedHit]]:
        depth = max(self.depth, k)
        per_component = {n: ix.search_batch(list(queries), depth) for n, ix in self.components.items()}
        out = []
        for qi in range(len(queries)):
            fused: Dict[str, List[Any]] = {}  # id -> [score, best rank, first seen, passage, ranks]
            for n, hits in per_component.items():
                for rank, hit in enumerate(hits[qi], start=1):
                    p = hit[0]
                    entry = fused.get(p.id)
                    if entry is None:
                        entry = fused[p.id] = [0.0, rank, len(fused), p, {m: None for m in self.components}]
                    entry[0] += 1.0 / (self.rrf_k + rank)
                    entry[1] = min(entry[1], rank)
                    entry[4][n] = rank
            ranked = sorted(fused.values(), key=lambda e: (-e[0], e[1], e[2]))[:k]
            out.append([FusedHit(e[3], e[0], e[4]) for e in ranked])
        return out

    def nbytes(self) -> int:
        return sum(int(getattr(ix, "nbytes", lambda: 0)()) for ix in self.components.values())


# ---------------------------------------------------------------------------
# Evaluation


Qrels = Mapping[str, Mapping[str, float]]


def recall_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    """Fraction of the relevant passages (relevance > 0) found in the top ``k``.

    Binary: a passage judged 2 counts the same as one judged 1. NaN when the
    query has no relevant passage, because the fraction is undefined.
    """
    relevant = {p for p, r in relevance.items() if r > 0}
    if not relevant:
        return float("nan")
    return len(relevant.intersection(pid for _, pid in _first_hits(ranked_ids, k))) / len(relevant)


def mrr_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    """1 / rank of the first relevant passage within the top ``k``; 0 if none is there."""
    for rank, pid in _first_hits(ranked_ids, k):
        if relevance.get(pid, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    """Normalised discounted cumulative gain at ``k`` with graded, linear gains.

    DCG@k = sum over ranks i = 1..k of rel_i / log2(i + 1), where rel_i is the
    judged relevance of the passage at rank i (0 if unjudged; negative
    judgements count as 0). The ideal DCG sorts all judged relevances from the
    qrels in decreasing order and applies the same formula, so a relevant
    passage the system never returned still lowers the score. Linear gain is
    what trec_eval and BEIR use; some papers use 2^rel - 1, which differs only
    when grades go above 1. NaN when the query has no relevant passage.
    """
    gains = sorted((r for r in relevance.values() if r > 0), reverse=True)[:k]
    if not gains:
        return float("nan")
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    dcg = sum(max(relevance.get(pid, 0), 0) / math.log2(rank + 1) for rank, pid in _first_hits(ranked_ids, k))
    return dcg / idcg


def _first_hits(ids: Iterable[str], k: int) -> List[Tuple[int, str]]:
    """(1-based rank, id) for the top ``k`` positions, skipping repeats.

    A ranking that lists a passage twice gets credit for it once, at its first
    position; the repeat still uses up its slot, so it cannot pull a later
    passage into the top k.
    """
    seen, out = set(), []
    for rank, pid in enumerate(ids, start=1):
        if rank > k:
            break
        if pid not in seen:
            seen.add(pid)
            out.append((rank, pid))
    return out


def evaluate_rankings(rankings: Mapping[str, Sequence[str]], qrels: Qrels, ks: Sequence[int] = (1, 5, 10)
                      ) -> Dict[str, Any]:
    """Average recall@k, MRR@k and nDCG@k over the ranked queries.

    ``rankings`` maps query id to passage ids, best first; ``qrels`` maps
    query id to {passage id: relevance}. A query with no relevant passage in
    the qrels is left out of every average, as trec_eval and BEIR do: none of
    the three metrics is defined for it. How many were left out is reported,
    so a run that silently lost its judgements is visible. Ids are compared
    as given: convert them to one type (str) first.
    """
    ks = sorted({int(k) for k in ks})
    per_metric: Dict[str, List[float]] = {f"{m}@{k}": [] for k in ks for m in ("recall", "mrr", "ndcg")}
    skipped = 0
    for qid, ranked in rankings.items():
        rel = qrels.get(qid, {})
        if not any(r > 0 for r in rel.values()):
            skipped += 1
            continue
        for k in ks:
            per_metric[f"recall@{k}"].append(recall_at_k(ranked, rel, k))
            per_metric[f"mrr@{k}"].append(mrr_at_k(ranked, rel, k))
            per_metric[f"ndcg@{k}"].append(ndcg_at_k(ranked, rel, k))
    n = len(rankings) - skipped
    out: Dict[str, Any] = {m: (float(np.mean(v)) if v else float("nan")) for m, v in per_metric.items()}
    out.update(n_queries=n, n_queries_without_relevant=skipped)
    return out


def evaluate_retrieval(index, queries: Mapping[str, str], qrels: Qrels, ks: Sequence[int] = (1, 5, 10)
                       ) -> Dict[str, Any]:
    """Search every query in ``queries`` ({query id: text}) and score the rankings.

    See ``evaluate_rankings`` for the metric definitions and which queries
    count. The index must provide ``search_batch``.
    """
    qids = list(queries)
    results = index.search_batch([queries[q] for q in qids], max(ks))
    return evaluate_rankings({q: [hit[0].id for hit in hits] for q, hits in zip(qids, results)}, qrels, ks)


# ---------------------------------------------------------------------------
# Citations

CITATION_INSTRUCTIONS = (
    "Answer only from the numbered sources below. Support every claim with a quotation copied exactly "
    "from a source, in double quotes, followed by its number, for example: \"exact words\" [2]. "
    "If the sources do not answer the question, say so."
)


def _as_passage(item) -> Passage:
    return item if isinstance(item, Passage) else item[0]


def cite(passages: Sequence[Union[Passage, Tuple[Passage, Any]]], max_chars: Optional[int] = None) -> str:
    """Format passages (or search hits) as numbered sources for a prompt.

    Each source starts with a header line ``[n] id=... | title: ... |
    document: ... | chars start-end`` (fields that are unknown are left out)
    followed by the passage text. The id and offsets let a reviewer go from an
    answer's [n] to the exact span of the source document. ``max_chars``
    truncates long passages and marks the cut with " [...]"; a quote taken
    from beyond the cut cannot have come from the prompt.
    """
    blocks = []
    for n, item in enumerate(passages, start=1):
        p = _as_passage(item)
        header = [f"[{n}] id={p.id}"]
        if p.title:
            header.append(f"title: {p.title}")
        if p.doc_id:
            header.append(f"document: {p.doc_id}")
        if p.start is not None and p.end is not None:
            header.append(f"chars {p.start}-{p.end}")
        body = p.text if max_chars is None or len(p.text) <= max_chars else p.text[:max_chars] + " [...]"
        blocks.append(" | ".join(header) + "\n" + body)
    return "\n\n".join(blocks)


# Straight or typographic double quotes, in any pairing, not spanning another
# quote mark. Single quotes are not used: they cannot be told apart from
# apostrophes ("the bank's 'fair value'").
_QUOTE = re.compile(r"[\"“”„]([^\"“”„]+)[\"“”„]")
_SPACE = re.compile(r"\s+")


def _canon(text: str) -> str:
    # NFC so that "e" + combining accent equals the precomposed letter, and
    # whitespace runs collapsed so that a quote re-flowed onto one line still
    # matches a passage that had a line break in it. Case and punctuation are
    # kept: a quote that changes either is not verbatim.
    return _SPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def extract_quotes(answer_text: str, min_chars: int = 20) -> List[str]:
    """Spans between double quotation marks with at least ``min_chars``
    characters after trimming. Shorter quotes are mostly terms ("APR"), not
    claims, and match by chance."""
    out = []
    for m in _QUOTE.finditer(answer_text):
        q = m.group(1).strip()
        if len(q) >= min_chars:
            out.append(q)
    return out


def verify_quotes(answer_text: str, passages: Sequence[Union[Passage, Tuple[Passage, Any]]],
                  min_chars: int = 20) -> List[str]:
    """Quoted spans in ``answer_text`` that occur in none of ``passages``.

    An empty list means every quotation of at least ``min_chars`` characters
    was found verbatim (up to whitespace and Unicode normalisation) in one of
    the given passages. It does not prove the answer is right, only that its
    quotations were not invented; check that the cited [n] is the passage that
    contains the quote if that matters. A quote that elides words with "..."
    is reported as unsupported: the check is deliberately strict.
    """
    sources = [_canon(_as_passage(item).text) for item in passages]
    return [q for q in extract_quotes(answer_text, min_chars) if not any(_canon(q) in s for s in sources)]
