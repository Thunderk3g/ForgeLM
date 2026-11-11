"""E6 - compliance screening: flagging potentially unfair contract terms.

A public stand-in for guideline-adherence checks: each sentence of a terms-of-
service document is either ordinary or contains a potentially unfair term
(LexGLUE UNFAIR-ToS, about 11% positive). The costly error is a missed
violation, so the question is not accuracy but:

    to catch 90% of the unfair terms, how much of the text must a person read?

Protocol
  * Human-supervised ladder trained on the 5,532-sentence training split.
  * The decision threshold is chosen on the validation split to reach 90%
    recall there, then applied unchanged to the 1,607-sentence test split.
  * A local 1.5B LLM is also scored zero-shot on the test split, and as a
    teacher for students trained on 1,000 sentences with no human labels.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import ARTIFACTS, load_unfair_tos, write_result
from forgelm.metrics import binary_report
from forgelm.students import EmbeddingStudent, FineTunedStudent, TfidfStudent
from forgelm.teachers import LocalLLMTeacher

parser = argparse.ArgumentParser()
parser.add_argument("--teacher-pool", type=int, default=1000)
parser.add_argument("--teacher-val", type=int, default=500, help="validation sentences the teacher scores")
parser.add_argument("--skip-teacher", action="store_true")
args = parser.parse_args()

train, val, test = load_unfair_tos()
labels = ["ordinary", "potentially_unfair"]
tr_x, tr_y = train.text.tolist(), train.label.values
va_x, va_y = val.text.tolist(), val.label.values
te_x, te_y = test.text.tolist(), test.label.values
TARGET_RECALL = 0.90


def threshold_for_recall(y, score, recall):
    """Highest threshold whose recall on (y, score) is at least ``recall``."""
    pos = np.sort(score[y == 1])
    k = int(np.floor((1 - recall) * len(pos)))
    return float(pos[k])


def screen(name, score_val, score_test, extra=None, val_y=None):
    thr = threshold_for_recall(va_y if val_y is None else val_y, score_val, TARGET_RECALL)
    at_thr = binary_report(te_y, score_test, thr)
    default = binary_report(te_y, score_test, 0.5)
    row = {"model": name, "roc_auc": round(at_thr["roc_auc"], 4),
           "f1_at_0.5": round(default["f1"], 4), "precision_at_0.5": round(default["precision"], 4),
           "recall_at_0.5": round(default["recall"], 4),
           "screen_threshold": round(thr, 4), "screen_recall": round(at_thr["recall"], 4),
           "screen_precision": round(at_thr["precision"], 4),
           "fraction_flagged_for_review": round(float(np.mean(score_test >= thr)), 4),
           "ece": round(default["ece"], 4)}
    row.update(extra or {})
    print(row, flush=True)
    return row


rows = []
start = time.perf_counter()
tf = TfidfStudent().fit(tr_x, tr_y, labels)
rows.append(screen("tfidf + LR", tf.predict_proba(va_x)[:, 1], tf.predict_proba(te_x)[:, 1],
                   {"train_seconds": round(time.perf_counter() - start, 1)}))
top = [g for g, _ in tf.top_features("potentially_unfair", 12)]

embedded = {}
for name, model, backend in (("static potion-32M + LR", "minishlab/potion-base-32M", "static"),
                             ("frozen bge-small + LR", "BAAI/bge-small-en-v1.5", "sentence")):
    start = time.perf_counter()
    st = EmbeddingStudent(model, backend, max_seq_length=256 if backend == "sentence" else None)
    emb = {"train": st.embedder.encode(tr_x), "val": st.embedder.encode(va_x), "test": st.embedder.encode(te_x)}
    st.fit(None, tr_y, labels, embeddings=emb["train"])
    embedded[name] = (st, emb)
    rows.append(screen(name, st.proba_from_embeddings(emb["val"])[:, 1], st.proba_from_embeddings(emb["test"])[:, 1],
                       {"train_seconds": round(time.perf_counter() - start, 1)}))

start = time.perf_counter()
ft = FineTunedStudent("sentence-transformers/all-MiniLM-L6-v2", epochs=3, lr=1e-4, batch_size=32,
                      max_length=128).fit(tr_x, tr_y, labels)
rows.append(screen("fine-tuned MiniLM-L6 (22M)", ft.predict_proba(va_x)[:, 1], ft.predict_proba(te_x)[:, 1],
                   {"train_seconds": round(time.perf_counter() - start, 1)}))

teacher_block = None
if not args.skip_teacher:
    descriptions = [
        "an ordinary clause with no unfair term",
        "a potentially unfair term: for example the provider may change or end the service or the contract at "
        "will, removes content at its discretion, limits its own liability, forces arbitration or a foreign "
        "court or law, or treats mere use of the service as consent",
    ]
    teacher = LocalLLMTeacher("Qwen/Qwen2.5-1.5B-Instruct", labels, descriptions,
                              instruction="You review consumer terms of service. Decide whether the clause below "
                                          "contains a term that is potentially unfair to the consumer.",
                              batch_size=8, max_text_tokens=200)
    cache = ARTIFACTS / f"e6_teacher_{args.teacher_pool}_{args.teacher_val}.npz"
    pool_idx = np.sort(np.random.default_rng(3).choice(len(tr_x), size=args.teacher_pool, replace=False))
    pool_x = [tr_x[i] for i in pool_idx]
    va_sub = np.sort(np.random.default_rng(4).choice(len(va_x), size=min(args.teacher_val, len(va_x)), replace=False))
    va_sub_x, va_sub_y = [va_x[i] for i in va_sub], va_y[va_sub]
    if cache.exists():
        c = np.load(cache)
        t_pool, t_val, t_test, seconds = c["pool"], c["val"], c["test"], float(c["seconds"])
    else:
        start = time.perf_counter()
        t_pool, t_val, t_test = teacher.label(pool_x), teacher.label(va_sub_x), teacher.label(te_x)
        seconds = time.perf_counter() - start
        np.savez_compressed(cache, pool=t_pool, val=t_val, test=t_test, seconds=seconds)
    n_items = len(pool_x) + len(va_sub_x) + len(te_x)
    rows.append(screen("zero-shot local LLM (Qwen2.5-1.5B), no training", t_val[:, 1], t_test[:, 1],
                       {"seconds_per_item": round(seconds / n_items, 3),
                        "note": f"threshold from {len(va_sub_x)} validation sentences"}, val_y=va_sub_y))
    distilled = []
    for name, (st_ref, emb) in embedded.items():
        st = EmbeddingStudent(st_ref.embedder.model_name, st_ref.embedder.backend)
        st.fit(None, t_pool, labels, embeddings=emb["train"][pool_idx])
        distilled.append(screen(f"{name}, trained on LLM soft labels only ({args.teacher_pool} sentences)",
                                st.proba_from_embeddings(emb["val"])[:, 1],
                                st.proba_from_embeddings(emb["test"])[:, 1]))
        ref = EmbeddingStudent(st_ref.embedder.model_name, st_ref.embedder.backend)
        ref.fit(None, tr_y[pool_idx], labels, embeddings=emb["train"][pool_idx])
        distilled.append(screen(f"{name}, same {args.teacher_pool} sentences with human labels",
                                ref.proba_from_embeddings(emb["val"])[:, 1],
                                ref.proba_from_embeddings(emb["test"])[:, 1]))
    teacher_block = {"model": "Qwen/Qwen2.5-1.5B-Instruct", "dtype": teacher.resolved_dtype or "cached run",
                     "seconds_per_item": round(seconds / n_items, 3),
                     "teacher_positive_rate_on_pool": round(float(np.mean(t_pool.argmax(1))), 4),
                     "true_positive_rate_on_pool": round(float(np.mean(tr_y[pool_idx])), 4),
                     "students": distilled}

write_result("e6_unfair_terms", {
    "experiment": "E6 compliance screening on LexGLUE UNFAIR-ToS (binary: potentially unfair term or not)",
    "dataset": {"name": "coastalcph/lex_glue, config unfair_tos", "train": len(tr_x), "validation": len(va_x),
                "test": len(te_x), "positive_rate_test": round(float(te_y.mean()), 4),
                "note": "licence not confirmed on the dataset card; benchmark use only"},
    "screening_rule": f"threshold set for {TARGET_RECALL:.0%} recall on validation, applied unchanged to test",
    "rows": rows, "tfidf_top_ngrams_for_unfair": top, "llm_teacher": teacher_block,
})
