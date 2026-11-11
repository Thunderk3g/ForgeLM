"""E9 - retrieval for financial question answering: lexical, static, encoder and hybrid.

The planner recommends retrieval with quoted sources for policy and regulation
question answering. That advice is only as good as the retriever: an answer
cannot quote a passage that was never found. The question here is

    for financial questions, how do lexical (BM25), static-embedding,
    encoder-embedding and hybrid retrieval compare in recall and ranking
    quality, and what does each cost on a CPU?

Data
  BEIR FiQA-2018 (Hugging Face Hub: "BeIR/fiqa" for corpus and queries,
  "BeIR/fiqa-qrels" for relevance judgements; licence cc-by-sa-4.0 per the
  dataset card). Questions and answers come from finance Q&A forums; the 648
  test queries have 1,706 passages judged relevant (all graded 1).

Protocol
  * Corpus: every passage judged relevant to a test query, plus distractors
    drawn uniformly at random (seed 0) from the rest of the 57,638-passage
    corpus, up to 15,000 passages. The passages are shuffled with the same
    seed, so no index can win a tie by position. Three in four distractors of
    the full corpus are left out, so this corpus is EASIER than the benchmark
    and the scores are NOT comparable with published FiQA numbers. They are
    comparable with each other: every system sees the same passages.
  * Passage text is title + text, as BEIR does (FiQA titles are empty).
  * Systems, all exact (brute-force) search:
      - BM25, k1 = 1.2, b = 0.75 (Lucene defaults), lowercase Unicode-word
        tokenizer, no stemming or stop words (forgelm.retrieval.BM25Index);
      - dense, minishlab/potion-base-32M, a Model2Vec static model (no prefix);
      - dense, BAAI/bge-small-en-v1.5, max 512 tokens, with the query
        instruction the model card documents for short-query-to-passage
        retrieval: "Represent this sentence for searching relevant passages: "
        (passages get no prefix);
      - hybrid: reciprocal rank fusion (constant 60) of BM25 and bge-small,
        each contributing its top 100.
  * Metrics over the test queries (forgelm.retrieval.evaluate_rankings):
    recall@10 (fraction of a query's relevant passages in the top 10),
    nDCG@10 (linear gain, ideal DCG from the qrels), MRR@10, and recall@100.
    A 95% bootstrap interval (2,000 resamples of queries) is given for each
    system's nDCG@10 and for the paired differences that matter.
  * Cost: index build seconds (encoding all passages, after the model is
    loaded; model load is reported separately), mean and 95th-percentile
    milliseconds per query searched one at a time (query encoding included),
    and index size in MB (vectors or sparse matrix plus vocabulary, not the
    passage texts, which every system shares). The embedding model's own
    weight memory is reported next to it.

  --smoke runs the same pipeline on 30 queries and about 300 passages and
  writes experiments/artifacts/e9_retrieval_smoke.json. It proves the script
  runs; its numbers measure nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from common import ARTIFACTS, environment, hub_file, rounded, write_result
from forgelm.retrieval import BM25Index, DenseIndex, HybridIndex, Passage, evaluate_rankings, ndcg_at_k
from forgelm.students import Embedder

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="30 queries, about 300 passages")
parser.add_argument("--max-passages", type=int, default=15000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--bge-max-seq-length", type=int, default=512)
parser.add_argument("--no-cache", action="store_true", help="re-encode passages even if a cached encoding exists")
args = parser.parse_args()
if args.smoke:
    args.max_passages = 300
N_SMOKE_QUERIES = 30
KS = (1, 5, 10, 100)
SEARCH_K = 100


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


# --------------------------------------------------------------------------- data

def load_fiqa():
    corpus = pd.read_parquet(hub_file("BeIR/fiqa", "corpus/corpus-00000-of-00001.parquet"))
    queries = pd.read_parquet(hub_file("BeIR/fiqa", "queries/queries-00000-of-00001.parquet"))
    qrels = pd.read_csv(hub_file("BeIR/fiqa-qrels", "test.tsv"), sep="\t", dtype=str)
    qrels["score"] = qrels["score"].astype(int)
    return corpus, queries, qrels


corpus, queries_df, qrels_df = load_fiqa()
rng = np.random.default_rng(args.seed)

qrels: dict = {}
for qid, pid, score in qrels_df[["query-id", "corpus-id", "score"]].itertuples(index=False):
    qrels.setdefault(qid, {})[pid] = score
test_qids = sorted(qrels, key=int)
if args.smoke:
    test_qids = sorted(rng.choice(test_qids, size=N_SMOKE_QUERIES, replace=False).tolist(), key=int)
qrels = {q: qrels[q] for q in test_qids}
query_text = dict(zip(queries_df["_id"].astype(str), queries_df["text"].astype(str)))
missing_q = [q for q in test_qids if q not in query_text]
assert not missing_q, f"test queries without text: {missing_q[:5]}"

corpus["_id"] = corpus["_id"].astype(str)
by_id = corpus.set_index("_id")
relevant = sorted({p for rel in qrels.values() for p, r in rel.items() if r > 0})
missing_p = [p for p in relevant if p not in by_id.index]
assert not missing_p, f"judged passages missing from the corpus: {missing_p[:5]}"
pool = np.setdiff1d(corpus["_id"].to_numpy(dtype=str), np.asarray(relevant, dtype=str))
n_distractors = max(0, min(args.max_passages - len(relevant), len(pool)))
distractors = rng.choice(pool, size=n_distractors, replace=False).tolist()
chosen = relevant + distractors
rng.shuffle(chosen)  # no system can win a tie by passage position


def passage_text(pid):
    row = by_id.loc[pid]
    title = row["title"] if isinstance(row["title"], str) else ""
    text = row["text"] if isinstance(row["text"], str) else ""
    return (title + "\n" + text).strip()


passages = [Passage(pid, passage_text(pid)) for pid in chosen]
log(f"{len(test_qids)} queries, {len(passages)} passages "
    f"({len(relevant)} judged relevant, {n_distractors} distractors)")
words = np.array([len(p.text.split()) for p in passages])


# --------------------------------------------------------------------------- systems

def model_megabytes(embedder):
    """Weight memory of the embedding model (what it costs to keep loaded)."""
    impl = embedder._load()
    if embedder.backend == "static":
        return round(float(np.asarray(impl.embedding).nbytes) / 1e6, 1)
    return round(sum(p.numel() * p.element_size() for p in impl.parameters()) / 1e6, 1)


def torch_threads():
    try:
        import torch

        return torch.get_num_threads()
    except ImportError:
        return None


def build_dense(name, model, backend, query_prefix="", max_seq_length=None):
    embedder = Embedder(model, backend, max_seq_length=max_seq_length)
    start = time.perf_counter()
    embedder.encode(["warm-up"])  # loads the model; reported separately from the build
    load_seconds = time.perf_counter() - start
    index = DenseIndex(embedder, query_prefix=query_prefix, name=name)
    cache = None if (args.smoke or args.no_cache) else \
        ARTIFACTS / f"e9_vectors_{model.replace('/', '_')}_{len(passages)}_{args.seed}_{max_seq_length}.npz"
    if cache is not None and cache.exists():
        c = np.load(cache)
        assert c["ids"].tolist() == [p.id for p in passages], "cached vectors are for different passages"
        index.build(passages, vectors=c["vectors"])
        seconds, cached = float(c["seconds"]), True
    else:
        start = time.perf_counter()
        index.build(passages)
        seconds, cached = time.perf_counter() - start, False
        if cache is not None:
            np.savez(cache, vectors=index.vectors, ids=np.array([p.id for p in passages]), seconds=seconds)
    log(f"{name}: built in {seconds:.1f}s{' (from cache)' if cached else ''}, model load {load_seconds:.1f}s")
    return index, {"build_seconds": seconds, "build_seconds_from_cache": cached,
                   "model_load_seconds": round(load_seconds, 2), "model": model, "backend": backend,
                   "query_prefix": query_prefix or None, "max_seq_length": max_seq_length, "model_weights_mb": model_megabytes(embedder),
                   "dimensions": int(index.vectors.shape[1])}


def run_queries(index):
    """Search each query on its own, as a user would, timing every call."""
    rankings, ms = {}, []
    for q in test_qids:
        start = time.perf_counter()
        hits = index.search(query_text[q], k=SEARCH_K)
        ms.append((time.perf_counter() - start) * 1000)
        rankings[q] = [h[0].id for h in hits]
    return rankings, np.array(ms)


def bootstrap_ci(values, n=2000, seed=0):
    values = np.asarray(values)
    idx = np.random.default_rng(seed).integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


systems = {}
start = time.perf_counter()
bm25 = BM25Index(k1=1.2, b=0.75).build(passages)
bm25_build = time.perf_counter() - start
log(f"bm25: built in {bm25_build:.2f}s, vocabulary {len(bm25.vocab)}")
systems["BM25"] = (bm25, {"build_seconds": bm25_build, "k1": 1.2, "b": 0.75, "vocabulary": len(bm25.vocab),
                          "model_weights_mb": 0.0})

potion, info = build_dense("dense:potion-base-32M", "minishlab/potion-base-32M", "static")
systems["dense, potion-base-32M (static)"] = (potion, info)

bge, info = build_dense("dense:bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", "sentence",
                        query_prefix=BGE_QUERY_INSTRUCTION, max_seq_length=args.bge_max_seq_length)
systems["dense, bge-small-en-v1.5 (encoder)"] = (bge, info)

hybrid = HybridIndex([bm25, bge], rrf_k=60, depth=100)
systems["hybrid, BM25 + bge-small (RRF 60)"] = (hybrid, {
    "build_seconds": bm25_build + info["build_seconds"], "rrf_k": 60, "depth_per_component": 100,
    "model_weights_mb": info["model_weights_mb"], "note": "build = BM25 build + bge-small encoding"})

rows, per_query_ndcg = [], {}
for name, (index, info) in systems.items():
    rankings, ms = run_queries(index)
    m = evaluate_rankings(rankings, qrels, KS)
    per_query_ndcg[name] = [ndcg_at_k(rankings[q], qrels[q], 10) for q in test_qids]
    row = {"system": name, "recall@10": m["recall@10"], "ndcg@10": m["ndcg@10"], "mrr@10": m["mrr@10"],
           "recall@100": m["recall@100"], "ndcg@10_95ci": bootstrap_ci(per_query_ndcg[name]),
           "index_build_seconds": round(info["build_seconds"], 2),
           "query_ms_mean": round(float(ms.mean()), 2), "query_ms_p95": round(float(np.percentile(ms, 95)), 2),
           "index_size_mb": round(index.nbytes() / 1e6, 2), "n_queries": m["n_queries"],
           "all_metrics": rounded({k: v for k, v in m.items() if "@" in k}),
           "settings": {k: v for k, v in info.items() if k != "build_seconds"}}
    rows.append(rounded(row))
    log(name, {k: row[k] for k in ("recall@10", "ndcg@10", "mrr@10", "index_build_seconds", "query_ms_mean",
                                    "index_size_mb")})

names = list(systems)
paired = {}
for a, b in ((names[3], names[2]), (names[3], names[0]), (names[2], names[0]), (names[1], names[0])):
    diff = np.array(per_query_ndcg[a]) - np.array(per_query_ndcg[b])
    paired[f"{a} minus {b}"] = {"ndcg@10_difference": round(float(diff.mean()), 4), "95ci": bootstrap_ci(diff)}

result = {
    "experiment": "E9 retrieval for financial question answering on BEIR FiQA-2018 (test queries)",
    "question": "how do lexical, static-embedding, encoder-embedding and hybrid retrieval compare, "
                "and what do they cost on a CPU?",
    "dataset": {
        "name": "BeIR/fiqa (corpus, queries) + BeIR/fiqa-qrels (test.tsv)",
        "licence": "cc-by-sa-4.0 per the BeIR/fiqa dataset card",
        "full_corpus_passages": len(corpus), "passages_used": len(passages),
        "judged_relevant_passages": len(relevant), "random_distractors": n_distractors, "seed": args.seed,
        "queries": len(test_qids), "passage_words_mean": round(float(words.mean()), 1),
        "passage_words_p95": int(np.percentile(words, 95)),
        "comparability": f"SUBSAMPLED CORPUS: all {len(relevant)} judged-relevant passages plus {n_distractors} "
                         f"random distractors out of {len(corpus)}. With most distractors removed this is "
                         "easier than the full benchmark; scores are NOT comparable with published FiQA "
                         "results, only with each other.",
    },
    "rows": rows,
    "paired_ndcg@10_differences": paired,
    "settings": {"tokenizer": "lowercase, maximal runs of Unicode word characters (\\w+), no stemming or stop words",
                 "bge_query_instruction": BGE_QUERY_INSTRUCTION, "search_k": SEARCH_K,
                 "timing": "one query at a time, query encoding included, after the model is loaded",
                 "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "torch_threads": torch_threads()},
}

if args.smoke:
    path = ARTIFACTS / "e9_retrieval_smoke.json"
    payload = dict(result, environment=environment(), generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   warning=f"SMOKE RUN: {len(test_qids)} queries, {len(passages)} passages. Proves the pipeline "
                           "runs; measures nothing.")
    path.write_text(json.dumps(payload, indent=2, default=float))
    log("wrote", path)
else:
    write_result("e9_retrieval", result)
