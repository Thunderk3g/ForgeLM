"""E1 - the student ladder on Banking77.

Question: at 8, 32 and all labels per class, which LLM-free student is most
accurate, how well calibrated is it, and what does it cost to train and serve
on a CPU?

Protocol
  * 385 training examples (5 per class) are set aside as a calibration anchor
    and never trained on. Temperature is fitted on that anchor only.
  * Few-shot subsets are drawn from the remaining training data with 3 seeds.
  * All numbers are on the untouched 3,080-example test split.
  * No hyper-parameter search was run. The linear-head students use the
    defaults in forgelm/students.py. The fine-tuned encoder uses fixed,
    untuned settings chosen in advance: learning rate 1e-4, batch 32, 64
    tokens, and 30 / 12 / 5 epochs for 8 / 32 / all labels per class.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import ARTIFACTS, STUDENT_LABELS, banking77_anchor, load_banking77, rounded, stratified_sample, write_result
from forgelm.bench import benchmark, directory_size_mb
from forgelm.metrics import apply_temperature, classification_report, fit_temperature
from forgelm.students import Embedder, EmbeddingStudent, FineTunedStudent, TfidfStudent, ZeroShotStudent

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=3)
parser.add_argument("--skip-finetune", action="store_true")
parser.add_argument("--only-modernbert", action="store_true",
                    help="fine-tune ModernBERT-base on the full set only; writes e1b_modernbert_full.json")
args = parser.parse_args()

train, test, names = load_banking77()
y_all = train.label.values
anchor_idx = banking77_anchor(y_all)
pool_idx = np.setdiff1d(np.arange(len(train)), anchor_idx)
texts_all = train.text.tolist()
anchor_texts, anchor_y = [texts_all[i] for i in anchor_idx], y_all[anchor_idx]
test_texts, test_y = test.text.tolist(), test.label.values
bench_texts = test_texts[:400]

rows = []


def score(student_name, k, seed, probs_test, probs_anchor, train_s, extra=None):
    rep = classification_report(test_y, probs_test)
    t = fit_temperature(anchor_y, probs_anchor)
    rep_cal = classification_report(test_y, apply_temperature(probs_test, t))
    row = {"student": student_name, "labels_per_class": k, "seed": seed, "train_seconds": round(train_s, 2),
           "temperature": round(t, 3), "ece_calibrated": rep_cal["ece"], "nll_calibrated": rep_cal["nll"],
           **rep}
    row.update(extra or {})
    rows.append(row)
    print(rounded({kk: v for kk, v in row.items() if kk in
                   ("student", "labels_per_class", "seed", "accuracy", "macro_f1", "ece", "ece_calibrated",
                    "train_seconds")}), flush=True)
    return t


# ---- embeddings are computed once per embedder and reused across regimes ----
embedders = {"static": ("minishlab/potion-base-32M", "static"),
             "encoder": ("BAAI/bge-small-en-v1.5", "sentence")}
cache = {}
for key, (model, backend) in ({} if args.only_modernbert else embedders).items():
    start = time.perf_counter()
    emb = Embedder(model, backend)
    cache[key] = {"train": emb.encode(texts_all), "test": emb.encode(test_texts), "embedder": emb,
                  "embed_seconds_per_1k": 0.0}
    cache[key]["embed_seconds_per_1k"] = (time.perf_counter() - start) / (len(texts_all) + len(test_texts)) * 1000
    print(f"embedded with {model} in {time.perf_counter() - start:.1f}s", flush=True)

# ---- rung 0: zero-shot -------------------------------------------------------
if not args.only_modernbert:
    zs = ZeroShotStudent("BAAI/bge-small-en-v1.5")
    zs.embedder = cache["encoder"]["embedder"]
    zs.fit(labels=names)
    score("zero_shot (bge-small, label names)", 0, 0, zs.proba_from_embeddings(cache["encoder"]["test"]),
          zs.proba_from_embeddings(cache["encoder"]["train"][anchor_idx]), 0.0)

regimes = [] if args.only_modernbert else [(8, args.seeds), (32, args.seeds), ("full", 1)]
saved = {}
for k, n_seeds in regimes:
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        idx = pool_idx if k == "full" else pool_idx[stratified_sample(y_all[pool_idx], k, rng)]
        tr_texts, tr_y = [texts_all[i] for i in idx], y_all[idx]

        start = time.perf_counter()
        tf = TfidfStudent().fit(tr_texts, tr_y, names)
        tf_raw_test = tf.predict_proba(test_texts)
        tf.temperature = score("tfidf + LR", k, seed, tf_raw_test, tf.predict_proba(anchor_texts),
                               time.perf_counter() - start)

        for key, label in (("static", "static potion-32M + LR"), ("encoder", "frozen bge-small + LR")):
            start = time.perf_counter()
            st = EmbeddingStudent(*embedders[key])
            st.embedder = cache[key]["embedder"]
            st.fit(None, tr_y, names, embeddings=cache[key]["train"][idx])
            head_s = time.perf_counter() - start
            embed_s = cache[key]["embed_seconds_per_1k"] * len(idx) / 1000
            p_test = st.proba_from_embeddings(cache[key]["test"])
            st.temperature = score(label, k, seed, p_test, st.proba_from_embeddings(cache[key]["train"][anchor_idx]),
                                   head_s + embed_s)
            if k == "full":
                saved[key] = (st, p_test)

        if not args.skip_finetune:
            epochs = {8: 30, 32: 12, "full": 5}[k]
            start = time.perf_counter()
            ft = FineTunedStudent("sentence-transformers/all-MiniLM-L6-v2", epochs=epochs, lr=1e-4, batch_size=32,
                                  max_length=64, seed=seed).fit(tr_texts, tr_y, names)
            train_s = time.perf_counter() - start
            p_test = ft.predict_proba(test_texts)
            ft.temperature = score("fine-tuned MiniLM-L6 (22M)", k, seed, p_test, ft.predict_proba(anchor_texts),
                                   train_s, {"epochs": epochs})
            if k == "full":
                saved["finetuned"] = (ft, p_test)
        if k == "full":
            saved["tfidf"] = (tf, tf_raw_test)

if args.only_modernbert:
    idx = pool_idx
    start = time.perf_counter()
    mb = FineTunedStudent("answerdotai/ModernBERT-base", epochs=3, lr=5e-5, batch_size=32, max_length=64,
                          on_progress=lambda s, t, l: print(f"  modernbert step {s}/{t} loss {l:.3f}", flush=True)
                          if s % 50 == 0 else None).fit([texts_all[i] for i in idx], y_all[idx], names)
    train_s = time.perf_counter() - start
    p_test = mb.predict_proba(test_texts)
    mb.temperature = score("fine-tuned ModernBERT-base (149M)", "full", 0, p_test, mb.predict_proba(anchor_texts),
                           train_s, {"epochs": 3})
    saved["modernbert"] = (mb, p_test)

# ---- serving cost on the full-data students ---------------------------------
serving = []
for key, (student, p_test) in saved.items():
    out_dir = ARTIFACTS / f"e1_{key}"
    student.save(str(out_dir))
    np.savez_compressed(ARTIFACTS / f"e1_{key}_probs.npz", test=p_test, anchor=student._raw_proba(anchor_texts),
                        temperature=student.temperature)
    b = benchmark(student.predict_proba, bench_texts, batch_size=32, single_item_runs=100)
    size = directory_size_mb(str(out_dir))
    note = ""
    if key in ("static", "encoder"):
        note = "size is the classifier head only; the shared embedding model is separate"
    serving.append({"student": STUDENT_LABELS[key], "artefact_mb": round(size, 2), "note": note, **rounded(b, 3)})
    print(serving[-1], flush=True)

write_result("e1b_modernbert_full" if args.only_modernbert else "e1_banking77_ladder", {
    "experiment": "E1 student ladder on Banking77",
    "dataset": {"name": "PolyAI/banking77", "licence": "CC BY 4.0", "train_pool": int(len(pool_idx)),
                "calibration_anchor": int(len(anchor_idx)), "test": int(len(test_y)), "classes": 77},
    "embedding_models": {"static": "minishlab/potion-base-32M (MIT)", "encoder": "BAAI/bge-small-en-v1.5 (MIT)",
                         "finetuned": "sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0)"},
    "embed_seconds_per_1k_texts": {k: round(v["embed_seconds_per_1k"], 3) for k, v in cache.items()},
    "rows": rows, "serving": serving,
})
