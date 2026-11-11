"""E8 - prompt or train? (Banking77, 8 labelled examples per class)

Question: with 8 labelled examples per class, is it better to prompt a local
LLM (zero-shot, or few-shot with retrieved demonstrations) or to train an
LLM-free student, and at what cost? The platform advises "try prompting
before training"; this measures both on the same test items and lets the
gate (forgelm.gate) say which one to ship, with its uncertainty.

Protocol
  * Labelled set: 8 examples per class from the training split (616 texts),
    stratified_sample with numpy default_rng(0): the first of the three
    8-per-class draws E2 uses as its reference.
  * Test set: the same 1,000 test texts E2 used as its anchor (E2 draws its
    pool, then its anchor, from default_rng(7)). Never used for anything but
    scoring.
  * Systems, all scored on those items:
      (a) zero-shot embedding similarity to the label names (bge-small);
      (b) similarity-weighted vote of the 5 nearest labelled texts (bge-small,
          no training; the same 5 texts system (f) shows the LLM);
      (c) static embeddings (potion-base-32M) + logistic regression on the 616;
      (d) frozen bge-small + logistic regression on the 616;
      (e) Qwen2.5-1.5B-Instruct zero-shot through answer-letter logits, 10
          options shortlisted by (a). E2's cached anchor labels are reused when
          present: same model, prompt, shortlist and items;
      (f) the same LLM shown 5 demonstrations retrieved from the 616 by
          bge-small cosine similarity (closest last); the demonstrations'
          labels always join the 10 options.
  * Cost: seconds per item on this CPU for the whole prediction path,
    embedding included; for the LLM including shortlisting and retrieval.
    (e) taken from E2's run when its cache is reused. Shared 4-core VM, so the
    figures are indicative.
  * ECE is of the raw probabilities. No temperature is fitted: under this
    label budget there is no held-out calibration set besides the 616.
  * The gate: forgelm.gate.decide, minimum gain 1 point, 95% paired bootstrap
    (5,000 resamples) and exact McNemar on the shared test items.
  * --smoke: 2 labelled examples per class and the first 20 test items; writes
    to experiments/artifacts/, never to experiments/results/.

LLM outputs are cached in experiments/artifacts/ so a re-run is cheap.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import ARTIFACTS, environment, load_banking77, rounded, stratified_sample, write_result
from forgelm.gate import Option, decide
from forgelm.metrics import classification_report
from forgelm.students import Embedder, EmbeddingStudent, ZeroShotStudent
from forgelm.teachers import LocalLLMTeacher

parser = argparse.ArgumentParser()
parser.add_argument("--per-class", type=int, default=8)
parser.add_argument("--seed", type=int, default=0, help="seed of the labelled-set draw")
parser.add_argument("--test", type=int, default=1000, help="test items (E2's anchor size)")
parser.add_argument("--demos", type=int, default=5, help="demonstrations shown to the few-shot LLM")
parser.add_argument("--options", type=int, default=10, help="options shown to the LLM")
parser.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--min-gain", type=float, default=0.01)
parser.add_argument("--smoke", action="store_true", help="2 labelled per class, first 20 test items")
parser.add_argument("--no-llm", action="store_true", help="skip LLM forward passes; only build and measure prompts")
args = parser.parse_args()
if args.smoke:
    args.per_class = 2
LLM_RAM_GB = 7.0  # a 1.5B model in bf16 plus activations and the embedders, with margin


def available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    return float("nan")


run_llm = not args.no_llm
if run_llm and args.smoke and available_gb() < LLM_RAM_GB:
    print(f"only {available_gb():.1f} GB available; smoke run skips the LLM forward passes", flush=True)
    run_llm = False

train, test, names = load_banking77()
descriptions = [n.replace("_", " ") for n in names]
n_classes = len(names)

# The E2 anchor: replay E2's draws exactly (pool first, then anchor) so the test items are the same.
rng = np.random.default_rng(7)
rng.choice(len(train), size=2000, replace=False)
e2_anchor_idx = np.sort(rng.choice(len(test), size=1000, replace=False))
test_idx = e2_anchor_idx[: (20 if args.smoke else args.test)]
test_texts, test_y = [test.text.iloc[i] for i in test_idx], test.label.values[test_idx]

lab_idx = stratified_sample(train.label.values, args.per_class, np.random.default_rng(args.seed))
lab_texts, lab_y = [train.text.iloc[i] for i in lab_idx], train.label.values[lab_idx]
print(f"{len(lab_idx)} labelled texts, {len(test_idx)} test texts", flush=True)

tag = f"{args.per_class}pc_seed{args.seed}_{len(test_idx)}"
probs, seconds, notes = {}, {}, {}


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def knn_proba(query, keys, key_y, k, scale=20.0):
    """Similarity-weighted vote of the k nearest labelled texts (cosine; vectors are L2-normalised).

    The weights are a softmax of similarity at the same scale ZeroShotStudent uses, so a much closer
    neighbour outvotes several distant ones and the output is a usable probability.
    """
    sims = query @ keys.T
    out = np.zeros((len(query), n_classes))
    for i, row in enumerate(sims):
        top = np.argsort(-row, kind="stable")[:k]
        w = np.exp(scale * (row[top] - row[top].max()))
        np.add.at(out[i], key_y[top], w)
    return out / out.sum(1, keepdims=True)


# --- LLM-free systems -------------------------------------------------------------------------------------
bge = Embedder("BAAI/bge-small-en-v1.5", "sentence")
lab_bge = bge.encode(lab_texts)  # also loads the model, so the timings below exclude loading
test_bge, enc_s = timed(lambda: bge.encode(test_texts))

zero_shot = ZeroShotStudent("BAAI/bge-small-en-v1.5")
zero_shot.embedder = bge
zero_shot.fit(labels=names, descriptions=descriptions)
probs["a"], head_s = timed(lambda: zero_shot.proba_from_embeddings(test_bge))
seconds["a"] = enc_s + head_s

probs["b"], head_s = timed(lambda: knn_proba(test_bge, lab_bge, lab_y, args.demos))
seconds["b"] = enc_s + head_s

static = Embedder("minishlab/potion-base-32M", "static")
lab_static = static.encode(lab_texts)
test_static, static_s = timed(lambda: static.encode(test_texts))
student_c = EmbeddingStudent("minishlab/potion-base-32M", "static")
student_c.embedder = static
(_, train_c) = timed(lambda: student_c.fit(None, lab_y, names, embeddings=lab_static))
probs["c"], head_s = timed(lambda: student_c.proba_from_embeddings(test_static))
seconds["c"] = static_s + head_s
notes["c"] = f"trained in {train_c:.1f}s"

student_d = EmbeddingStudent("BAAI/bge-small-en-v1.5", "sentence")
student_d.embedder = bge
(_, train_d) = timed(lambda: student_d.fit(None, lab_y, names, embeddings=lab_bge))
probs["d"], head_s = timed(lambda: student_d.proba_from_embeddings(test_bge))
seconds["d"] = enc_s + head_s
notes["d"] = f"trained in {train_d:.1f}s"

# --- LLM systems ------------------------------------------------------------------------------------------
instruction = ("A bank customer wrote the text below. Choose the intent that best describes what the customer "
               "wants.")  # E2's instruction, so (e) can reuse E2's labels
model_tag = args.llm.split("/")[-1]


def make_teacher(k_demos: int) -> LocalLLMTeacher:
    return LocalLLMTeacher(args.llm, names, descriptions, instruction=instruction, max_options=args.options,
                           shortlister=zero_shot, batch_size=args.batch_size,
                           demonstrations=list(zip(lab_texts, lab_y.tolist())), k_demos=k_demos,
                           demo_embedder=bge if k_demos else None)


zs_teacher, fs_teacher = make_teacher(0), make_teacher(args.demos)
llm_info = {"model": args.llm, "licence": "Apache-2.0", "options": args.options, "demos": args.demos,
            "ran_forward_passes": run_llm}

e2_cache = ARTIFACTS / f"e2_teacher_{model_tag}_2000_1000.npz"
zs_cache = ARTIFACTS / f"e8_llm_zeroshot_{model_tag}_{len(test_idx)}.npz"
fs_cache = ARTIFACTS / f"e8_llm_fewshot_{model_tag}_k{args.demos}_o{args.options}_{tag}.npz"
reuse_e2 = args.options == 10 and e2_cache.exists()
if run_llm and not (fs_cache.exists() and (reuse_e2 or zs_cache.exists())):
    # One-off setup kept out of the per-item timings: loading the weights, and embedding the 616
    # demonstrations (like training a student, done once). Both teachers share one loaded model.
    _, load_s = timed(fs_teacher._load)
    _, demo_s = timed(lambda: fs_teacher._select_demos(lab_texts[:1]))
    for attr in ("_model", "_tok", "_letter_ids", "resolved_dtype"):
        setattr(zs_teacher, attr, getattr(fs_teacher, attr))
    llm_info.update(load_seconds=round(load_s, 1), demo_embedding_seconds=round(demo_s, 2))

if reuse_e2:
    cached = np.load(e2_cache)
    if not np.array_equal(cached["anchor_idx"], e2_anchor_idx):
        raise SystemExit(f"{e2_cache} has different anchor indices from E2's draw; delete it or investigate")
    probs["e"] = cached["anchor"][: len(test_idx)]
    seconds["e"] = float(cached["seconds"]) / (len(cached["pool"]) + len(cached["anchor"]))
    notes["e"] = "E2's cached labels and timing (bf16, whole machine)"
    llm_info["zero_shot_source"] = str(e2_cache.name)
elif zs_cache.exists():
    cached = np.load(zs_cache)
    probs["e"], seconds["e"] = cached["probs"], float(cached["seconds_per_item"])
    notes["e"] = f"cached in {zs_cache.name}"
    llm_info["zero_shot_source"] = zs_cache.name
elif run_llm:
    probs["e"], total = timed(lambda: zs_teacher.label(test_texts))
    seconds["e"] = total / len(test_texts)
    np.savez_compressed(zs_cache, probs=probs["e"], seconds_per_item=seconds["e"], test_idx=test_idx)
    notes["e"] = f"computed here ({zs_teacher.resolved_dtype})"
    llm_info["zero_shot_source"] = zs_cache.name

if fs_cache.exists():
    cached = np.load(fs_cache)
    if not (np.array_equal(cached["test_idx"], test_idx) and np.array_equal(cached["lab_idx"], lab_idx)):
        raise SystemExit(f"{fs_cache} was computed on different items; delete it")
    probs["f"], seconds["f"] = cached["probs"], float(cached["seconds_per_item"])
    llm_info["few_shot_stats"] = json.loads(str(cached["stats"]))
    notes["f"] = f"cached in {fs_cache.name}"
elif run_llm:
    probs["f"], total = timed(lambda: fs_teacher.label(test_texts))
    seconds["f"] = total / len(test_texts)
    llm_info["few_shot_stats"] = fs_teacher.stats()
    np.savez_compressed(fs_cache, probs=probs["f"], seconds_per_item=seconds["f"], test_idx=test_idx,
                        lab_idx=lab_idx, stats=json.dumps(llm_info["few_shot_stats"]))
    notes["f"] = f"computed here ({fs_teacher.resolved_dtype})"
if fs_teacher.resolved_dtype:
    llm_info["dtype"] = fs_teacher.resolved_dtype


def prompt_lengths(sample):
    """Token counts of the zero-shot and few-shot prompts, to explain the cost difference.

    Needs only the tokenizer. When no forward pass ran here, the real tokenizer is loaded on its own (no
    model weights), which also exercises the prompt construction end to end.
    """
    if fs_teacher._tok is None:
        from transformers import AutoTokenizer

        fs_teacher._tok = AutoTokenizer.from_pretrained(args.llm)
    zs_teacher._tok = fs_teacher._tok
    demos = fs_teacher._select_demos(sample)
    out = {}
    for name, teacher, shown in (("zero_shot", zs_teacher, [[] for _ in sample]), ("few_shot", fs_teacher, demos)):
        cands = teacher._candidates(sample, shown)
        lengths = [len(teacher._tok.encode(teacher._prompt(t, c, d))) for t, c, d in zip(sample, cands, shown)]
        out[name] = {"mean_tokens": round(float(np.mean(lengths)), 1), "max_tokens": int(np.max(lengths))}
    fs_cands = fs_teacher._candidates(sample, demos)
    out["few_shot_options_mean"] = round(float(np.mean([len(c) for c in fs_cands])), 2)
    # How often retrieval changed the shortlist: a demonstration label the zero-shot shortlist lacked.
    zs_cands = zs_teacher._candidates(sample)
    out["few_shot_shortlist_changed_share"] = round(float(np.mean([set(a) != set(b) for a, b in
                                                                   zip(fs_cands, zs_cands)])), 3)
    out["sample_size"] = len(sample)
    return out, fs_teacher._prompt(sample[0], fs_cands[0], demos[0])


llm_info["prompts"], example_prompt = prompt_lengths(test_texts[:50])
print("prompt lengths", llm_info["prompts"], flush=True)
print("example few-shot prompt:\n" + example_prompt, flush=True)

# --- Scores and the gate ----------------------------------------------------------------------------------
systems = {
    "a": ("zero-shot embedding similarity (bge-small)", False, 0),
    "b": (f"nearest-neighbour vote, k={args.demos} (bge-small)", False, len(lab_idx)),
    "c": ("static potion-32M + LR", False, len(lab_idx)),
    "d": ("frozen bge-small + LR", False, len(lab_idx)),
    "e": (f"LLM zero-shot, {args.options} options", True, 0),
    "f": (f"LLM few-shot, {args.demos} retrieved demos", True, len(lab_idx)),
}
rows, options = [], []
for key, (label, needs_llm, n_labels) in systems.items():
    if key not in probs:
        print(f"({key}) {label}: skipped (no LLM run and no cache)", flush=True)
        continue
    rep = classification_report(test_y, probs[key], n_classes=n_classes)
    rows.append({"system": label, "key": key, "needs_llm_at_inference": needs_llm, "labels_needed": n_labels,
                 "seconds_per_item": round(seconds[key], 5), "notes": notes.get(key, ""), **rounded(rep)})
    options.append(Option(label, probs[key].argmax(1), needs_llm_at_inference=needs_llm,
                          seconds_per_item=seconds[key], labels_needed=n_labels, notes=notes.get(key, "")))
    print(rounded({k: rows[-1][k] for k in ("system", "accuracy", "macro_f1", "ece", "seconds_per_item")}),
          flush=True)

gate = decide(options, test_y, min_gain=args.min_gain)
print(gate.render(), flush=True)
np.savez_compressed(ARTIFACTS / f"e8_probs_{model_tag}_k{args.demos}_o{args.options}_{tag}.npz", test_idx=test_idx, lab_idx=lab_idx, test_y=test_y,
                    **{k: v for k, v in probs.items()})

result = {
    "experiment": "E8 prompt or train? (Banking77, few labels per class)",
    "question": "With 8 labelled examples per class, prompt a local LLM or train an LLM-free student, and at "
                "what cost?",
    "smoke": args.smoke,
    "labelled_set": {"per_class": args.per_class, "n": int(len(lab_idx)), "seed": args.seed,
                     "source": "Banking77 train split, stratified_sample"},
    "test_set": {"n": int(len(test_idx)), "source": "E2's anchor items from the Banking77 test split"},
    "llm": llm_info, "systems": rows, "gate": gate.to_dict(), "gate_text": gate.render(),
}
if args.smoke:
    path = ARTIFACTS / "e8_prompt_or_train_smoke.json"
    path.write_text(json.dumps(dict(result, environment=environment(),
                                    generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                               indent=2, default=float))
    print(f"wrote {path}")
else:
    write_result("e8_prompt_or_train", result)
