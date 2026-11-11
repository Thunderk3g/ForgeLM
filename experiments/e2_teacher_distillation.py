"""E2 - distilling a local LLM teacher into LLM-free students (Banking77).

Question: with zero human training labels, how close does a student trained
only on an LLM's labels get to that LLM, and how does it compare with simply
labelling 8 examples per class by hand?

Protocol
  * Teacher: Qwen2.5-1.5B-Instruct on CPU, used through answer-letter logits
    (one forward pass per text, soft labels). 77 intents do not fit a short
    prompt, so a zero-shot embedding model shortlists 10 candidate intents per
    text and the LLM chooses among them.
  * The teacher labels an unlabelled pool drawn from the training split. The
    true labels of that pool are used only to report the teacher's accuracy.
  * Anchor: 1,000 test examples with human labels, never shown for labelling.
  * Students: static embeddings + LR and frozen encoder + LR, trained on soft
    or hard teacher labels, with and without a confidence filter.

Teacher outputs are cached in experiments/artifacts/ so re-runs are cheap.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import ARTIFACTS, e2_draws, load_banking77, rounded, stratified_sample, write_result
from forgelm.distill import distill
from forgelm.metrics import classification_report
from forgelm.students import Embedder, EmbeddingStudent, ZeroShotStudent
from forgelm.teachers import CallableTeacher, LocalLLMTeacher

parser = argparse.ArgumentParser()
parser.add_argument("--pool", type=int, default=2000)
parser.add_argument("--anchor", type=int, default=1000)
parser.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B-Instruct")
args = parser.parse_args()

train, test, names = load_banking77()
pool_idx, anchor_idx = e2_draws(len(train), len(test), args.pool, args.anchor)
pool_texts, pool_y = [train.text.iloc[i] for i in pool_idx], train.label.values[pool_idx]
anchor_texts, anchor_y = [test.text.iloc[i] for i in anchor_idx], test.label.values[anchor_idx]
descriptions = [n.replace("_", " ") for n in names]

shortlister = ZeroShotStudent("BAAI/bge-small-en-v1.5").fit(labels=names, descriptions=descriptions)
short_anchor = shortlister.predict_proba(anchor_texts)
top10 = np.argsort(-short_anchor, axis=1)[:, :10]
shortlist_recall = float(np.mean([anchor_y[i] in top10[i] for i in range(len(anchor_y))]))

cache_path = ARTIFACTS / f"e2_teacher_{args.teacher.split('/')[-1]}_{args.pool}_{args.anchor}.npz"
teacher = LocalLLMTeacher(args.teacher, names, descriptions,
                          instruction="A bank customer wrote the text below. Choose the intent that best "
                                      "describes what the customer wants.",
                          max_options=10, shortlister=shortlister, batch_size=8)
if cache_path.exists():
    cached = np.load(cache_path)
    pool_probs, anchor_probs = cached["pool"], cached["anchor"]
    teacher_seconds = float(cached["seconds"])
    print("loaded cached teacher labels", flush=True)
else:
    start = time.perf_counter()
    pool_probs = teacher.label(pool_texts)
    print(f"teacher labelled pool in {time.perf_counter() - start:.0f}s", flush=True)
    anchor_probs = teacher.label(anchor_texts)
    teacher_seconds = time.perf_counter() - start
    np.savez_compressed(cache_path, pool=pool_probs, anchor=anchor_probs, seconds=teacher_seconds,
                        anchor_idx=anchor_idx, pool_idx=pool_idx)

teacher_pool = classification_report(pool_y, pool_probs)
teacher_anchor = classification_report(anchor_y, anchor_probs)
print("teacher on anchor", rounded(teacher_anchor), flush=True)


def product(llm, prior):
    """Product of experts: the LLM's choice among the shortlist, weighted by the shortlister's own belief."""
    mixed = llm * prior
    return mixed / mixed.sum(1, keepdims=True)


short_pool = shortlister.predict_proba(pool_texts)
ensemble_pool, ensemble_anchor = product(pool_probs, short_pool), product(anchor_probs, short_anchor)
teachers = {
    "llm": (pool_probs, anchor_probs),
    "zero-shot embedder": (short_pool, short_anchor),
    "llm x embedder (product)": (ensemble_pool, ensemble_anchor),
}
teacher_table = {name: {"on_pool": rounded(classification_report(pool_y, tp)),
                        "on_anchor": rounded(classification_report(anchor_y, ta))}
                 for name, (tp, ta) in teachers.items()}
for name, row in teacher_table.items():
    print("teacher", name, row["on_anchor"]["accuracy"], flush=True)

rows = []
embedders = {"static potion-32M + LR": ("minishlab/potion-base-32M", "static"),
             "frozen bge-small + LR": ("BAAI/bge-small-en-v1.5", "sentence")}
for teacher_name, (t_pool, t_anchor) in teachers.items():
    replay = CallableTeacher(lambda texts: None, names, name=teacher_name)
    for label, (model, backend) in embedders.items():
        for soft, min_conf in ((True, 0.0), (False, 0.0), (True, 0.5)):
            student = EmbeddingStudent(model, backend)
            rep = distill(replay, student, pool_texts, anchor_texts, anchor_y, soft=soft, min_confidence=min_conf,
                          teacher_probs=t_pool, teacher_anchor_probs=t_anchor)
            rows.append({"teacher": teacher_name, "student": label, "labels": "soft" if soft else "hard",
                         "min_confidence": min_conf, "n_train": rep.n_used_for_training,
                         **rounded(rep.student_on_anchor),
                         "teacher_accuracy": round(rep.teacher_on_anchor["accuracy"], 4),
                         "agreement_with_teacher": round(rep.student_teacher_agreement, 4),
                         "student_minus_teacher_accuracy": round(rep.student_minus_teacher_accuracy, 4),
                         "warnings": rep.warnings})
            print({k: rows[-1][k] for k in ("teacher", "student", "labels", "min_confidence", "n_train", "accuracy",
                                            "student_minus_teacher_accuracy")}, flush=True)

# Reference points on the same anchor: the same pool with its human labels, and 8 human labels per class.
references = []
for label, (model, backend) in embedders.items():
    emb = Embedder(model, backend)
    pool_emb, anchor_emb = emb.encode(pool_texts), emb.encode(anchor_texts)
    s = EmbeddingStudent(model, backend)
    s.embedder = emb
    s.fit(None, pool_y, names, embeddings=pool_emb)
    references.append({"student": label, "trained_on": f"same {args.pool} texts, human labels",
                       **rounded(classification_report(anchor_y, s.proba_from_embeddings(anchor_emb)))})
    accs = []
    for seed in range(3):
        few = stratified_sample(train.label.values, 8, np.random.default_rng(seed))
        s8 = EmbeddingStudent(model, backend)
        s8.embedder = emb
        s8.fit(None, train.label.values[few], names, embeddings=emb.encode([train.text.iloc[i] for i in few]))
        accs.append(classification_report(anchor_y, s8.proba_from_embeddings(anchor_emb))["accuracy"])
    references.append({"student": label, "trained_on": "8 human labels per class (616 labels, mean of 3 seeds)",
                       "accuracy": round(float(np.mean(accs)), 4), "accuracy_std": round(float(np.std(accs)), 4)})
    print(references[-2], references[-1], flush=True)

n_items = args.pool + args.anchor
write_result("e2_teacher_distillation", {
    "experiment": "E2 local LLM teacher -> LLM-free student (Banking77, zero human training labels)",
    "teacher": {"model": args.teacher, "licence": "Apache-2.0", "method": "answer-letter logits, 10 shortlisted options",
                "seconds_total": round(teacher_seconds, 1), "seconds_per_item": round(teacher_seconds / n_items, 3),
                "dtype": teacher.resolved_dtype or "cached run",
                "on_pool": rounded(teacher_pool), "on_anchor": rounded(teacher_anchor)},
    "teachers_compared": teacher_table,
    "shortlister": {"model": "BAAI/bge-small-en-v1.5 zero-shot", "top10_recall_on_anchor": round(shortlist_recall, 4),
                    "top1_accuracy_on_anchor": round(float(np.mean(short_anchor.argmax(1) == anchor_y)), 4)},
    "pool_size": args.pool, "anchor_size": args.anchor,
    "students_on_teacher_labels": rows, "references": references,
})
