"""E7 - specialising a small instruct model on a CPU, and whether it survives GGUF.

Question
  Can a small instruct model be specialised with LoRA on a CPU, in a time
  someone would accept, and does the result survive the trip to the format it
  would be served in: merged weights, GGUF, and 4-bit quantisation?

Protocol
  * Base model HuggingFaceTB/SmolLM2-360M-Instruct (Apache-2.0). If it cannot
    be downloaded, Qwen/Qwen2.5-0.5B-Instruct (Apache-2.0); the result names
    the model that was used.
  * Task: Banking77 (CC BY 4.0) restricted to its 10 most frequent intents in
    the training split, posed as generation. The system prompt lists the 10
    intent names; the model must answer with one of them and nothing else.
    Scored by exact match after forgelm.finetune.normalise_answer (case,
    surrounding spaces and quotes, and a final full stop do not count). A
    lenient score (the reply names exactly one intent, and it is the right
    one) is stored beside it to show how much of a miss is format, not content.
  * Training: 40 examples per intent from the training split (400), LoRA rank
    16, alpha 32, on every attention and MLP projection, 3 epochs, learning
    rate 2e-4, batch 4 with 2 accumulation steps (8 per update), linear
    warm-up over 10% then linear decay, gradient clipping at 1, float32, loss
    on the answer only. Seed 0 for data and training.
  * Evaluation: 300 test-split texts from the same intents (30 per intent),
    never trained on. Greedy decoding, no repetition penalty, at most 24 new
    tokens (the longest intent name is 17 tokens).
  * Measured: exact-match accuracy of (1) the base model zero-shot, (2) the
    fine-tuned model in PyTorch (merged float32 weights; the adapter itself is
    checked against them on the first 50 texts), (3) the merged model as GGUF
    f16 and (4) as GGUF Q4_K_M, both run with llama-cpp-python; agreement of
    (3) and (4) with (2), with forgelm.gguf.output_parity; file sizes; speed
    on CPU for (2) to (4), both on the task itself and as decode tokens per
    second for up to 64 tokens of open-ended text from an empty cache;
    training time and the loss curve; the chat-template round trip on the
    exported file; forgelm.generative.estimate_memory_gb beside the measured
    peak resident memory.
  * The converter is llama.cpp's convert_hf_to_gguf.py at the tag pinned in
    forgelm.gguf (downloaded and checksummed), or $LLAMA_CPP_CONVERT. If the
    GGUF stack is missing or fails, steps (3) and (4) are recorded as not run,
    with the reason, and the PyTorch results are still written.

  --smoke runs the same pipeline on 2 intents, 8 training examples each, 20
  evaluation texts, 1 epoch and at most 12 new tokens, to prove the script
  works end to end. Its result goes to experiments/artifacts/, never to
  experiments/results/, so it cannot be mistaken for a measurement.

Merged weights and GGUF files (about 2.5 GB for the 360M model) are deleted
at the end unless --keep-weights is given; the adapter is kept.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from common import ARTIFACTS, environment, load_banking77, rounded, write_result
from forgelm import gguf
from forgelm.finetune import (FineTuneRecipe, evaluate_chat, evaluate_exact, finetune_lora, generate, load_model,
                              merge_adapter, normalise_answer)
from forgelm.generative import estimate_memory_gb

PRIMARY_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="2 intents, 16 training examples, 20 test texts, 1 epoch")
parser.add_argument("--base-model", default=PRIMARY_MODEL)
parser.add_argument("--keep-weights", action="store_true", help="keep the merged weights and the GGUF files")
args = parser.parse_args()

if args.smoke:
    N_INTENTS, PER_INTENT, N_EVAL, EPOCHS, MAX_NEW, N_MERGE_CHECK = 2, 8, 20, 1, 12, 5
else:
    N_INTENTS, PER_INTENT, N_EVAL, EPOCHS, MAX_NEW, N_MERGE_CHECK = 10, 40, 300, 3, 24, 50
SEED = 0
# Threads for llama.cpp. torch already follows OMP_NUM_THREADS; using the same
# number for both keeps the speed comparison between them fair.
THREADS = int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count() or 1)
WORK = ARTIFACTS / ("e7_smoke" if args.smoke else "e7")
WORK.mkdir(parents=True, exist_ok=True)
started_all = time.time()


def log(*parts):
    print(f"[{time.time() - started_all:7.1f}s]", *parts, flush=True)


def peak_rss_gb() -> float:
    # ru_maxrss is in KiB on Linux: the high-water mark of this process since
    # it started, so each reading covers every stage before it.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 3)


def size_mb(path) -> float:
    path = Path(path)
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return round(sum(p.stat().st_size for p in files) / 1e6, 1)


# ---------------------------------------------------------------- data
train_df, test_df, names = load_banking77()
counts = train_df.label.value_counts()
# Most frequent first; ties broken by label id so the choice is deterministic.
intents = sorted(counts.index, key=lambda c: (-counts[c], c))[:N_INTENTS]
intent_names = [names[c] for c in intents]
SYSTEM = ("You are a banking support classifier. Read the customer's message and reply with exactly one intent "
          "name from this list, copied exactly, and nothing else:\n" + "\n".join(intent_names))


def record(text: str, label: int) -> dict:
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text},
                         {"role": "assistant", "content": names[label]}]}


rng = np.random.default_rng(SEED)
train_rows, eval_rows = [], []
per_intent_eval = int(np.ceil(N_EVAL / N_INTENTS))
for c in intents:
    train_rows += rng.choice(np.flatnonzero(train_df.label.values == c), PER_INTENT, replace=False).tolist()
    eval_rows += rng.choice(np.flatnonzero(test_df.label.values == c), per_intent_eval, replace=False).tolist()
eval_rows = sorted(rng.permutation(eval_rows)[:N_EVAL].tolist())
train_records = [record(train_df.text.iloc[i], int(train_df.label.iloc[i])) for i in rng.permutation(train_rows)]
eval_records = [record(test_df.text.iloc[i], int(test_df.label.iloc[i])) for i in eval_rows]
references = [r["messages"][-1]["content"] for r in eval_records]
log(f"{len(intents)} intents, {len(train_records)} training and {len(eval_records)} evaluation conversations")


def lenient(outputs) -> float:
    """Share of replies that name exactly one intent, and the right one."""
    hits = 0
    for out, ref in zip(outputs, references):
        found = [n for n in intent_names if n.casefold() in normalise_answer(out)]
        # If one intent name were part of another, a reply naming the longer
        # one would also "contain" the shorter; count only the longest match.
        found = [n for n in found if not any(n != m and n in m for m in found)]
        hits += found == [ref]
    return hits / len(references)


def score_summary(scored: dict) -> dict:
    keep = ("accuracy", "n", "prompt_tokens", "new_tokens", "seconds", "new_tokens_per_second", "seconds_per_example")
    out = rounded({k: scored[k] for k in keep})
    out["lenient_accuracy"] = round(lenient(scored["outputs"]), 4)
    out["outputs"] = scored["outputs"]
    return out


def progress(what):
    def report(i, n):
        if i == n or i % 50 == 0:
            log(f"  {what}: {i}/{n}")
    return report


# Open-ended prompts for decode speed: answered at length by any of the
# models, unlike the task prompts, whose answers are a few tokens long.
SPEED_PROMPTS = [[{"role": "user", "content": q}] for q in (
    "Explain in a few sentences why people keep an emergency savings fund.",
    "Describe how a bank transfer works, step by step.",
    "Write a short note to a customer explaining a card payment fee.")]
SPEED_TOKENS = 16 if args.smoke else 64


def decode_speed(chat) -> dict:
    """New tokens per second over the speed prompts, each from an empty cache."""
    new = seconds = 0.0
    for messages in SPEED_PROMPTS:
        d = chat(messages, SPEED_TOKENS)
        new += d["new_tokens"]
        seconds += d["seconds"]
    return {"new_tokens": int(new), "seconds": round(seconds, 3),
            "new_tokens_per_second": round(new / seconds, 2) if seconds else None,
            "note": f"greedy, up to {SPEED_TOKENS} tokens per prompt, prompt processing included, no cache reuse"}


# ---------------------------------------------------------------- base model
def resolve_base(requested: str):
    from huggingface_hub import snapshot_download

    tried = []
    candidates = [requested] + ([FALLBACK_MODEL] if requested == PRIMARY_MODEL else [])
    for name in candidates:
        try:
            snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"])
            return name, tried
        except Exception as exc:  # any download failure sends us to the fallback
            tried.append({"model": name, "error": f"{type(exc).__name__}: {exc}"[:300]})
            log(f"could not download {name}: {exc}")
    raise SystemExit(f"no base model could be downloaded: {tried}")


base_model, download_failures = resolve_base(args.base_model)
log("base model", base_model)
result = {"experiment": "E7 LoRA fine-tuning on CPU and GGUF export (Banking77, 10 intents, generative)",
          "smoke": args.smoke, "base_model": base_model, "base_model_download_failures": download_failures,
          "intents": intent_names, "n_train": len(train_records), "n_eval": len(eval_records),
          "max_new_tokens": MAX_NEW, "threads": THREADS, "system_prompt": SYSTEM, "references": references,
          "scoring": "exact match after normalise_answer; lenient = reply names exactly one intent and it is right",
          "stages": {}, "peak_rss_gb": {}}

# (1) zero-shot
model, tokenizer = load_model(base_model, device="cpu")
n_params = sum(p.numel() for p in model.parameters())
result["parameters"] = n_params
log(f"(1) base zero-shot, {n_params / 1e6:.0f}M parameters")
zero_shot = evaluate_exact(model, tokenizer, eval_records, max_new_tokens=MAX_NEW, on_progress=progress("zero-shot"))
result["stages"]["1_base_zero_shot"] = score_summary(zero_shot)
log("    accuracy", zero_shot["accuracy"], "lenient", result["stages"]["1_base_zero_shot"]["lenient_accuracy"])
del model
gc.collect()
result["peak_rss_gb"]["after_zero_shot"] = peak_rss_gb()

# ---------------------------------------------------------------- fine-tune
# An effective batch of 8 as 4 x 2 accumulated: in float32 without gradient
# checkpointing the activations, not the weights, dominate memory (the smoke
# run peaked near 7 GB at 8 x ~100 tokens), and the full prompt is about twice
# as long.
recipe = FineTuneRecipe(base_model=base_model, method="lora", rank=16, alpha=32, dropout=0.0, learning_rate=2e-4,
                        epochs=EPOCHS, batch_size=4, grad_accumulation=2, max_seq_len=256, warmup=0.1, seed=SEED,
                        train_on_responses_only=True)
adapter_dir, merged_dir = WORK / "adapter", WORK / "merged"
log("(2) fine-tuning", recipe.to_dict())


def train_progress(step, total, loss):
    if step == 1 or step == total or step % 10 == 0:
        log(f"  step {step}/{total} loss {loss:.4f}")


trained = finetune_lora(recipe, train_records, str(adapter_dir), device="cpu", on_progress=train_progress)
gc.collect()
result["peak_rss_gb"]["after_training"] = peak_rss_gb()
losses = trained["loss_history"]
tenth = max(1, len(losses) // 10)
result["training"] = {
    "recipe": recipe.to_dict(), "device": trained["device"], "dtype": trained["dtype"], "steps": trained["steps"],
    "seconds": round(trained["seconds"], 1), "tokens_seen": trained["tokens_seen"],
    "tokens_per_second": round(trained["tokens_seen"] / trained["seconds"], 1),
    "trainable_parameters": trained["trainable_parameters"], "n_truncated": trained["n_truncated"],
    "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4), "loss_min": round(min(losses), 4),
    "loss_mean_first_10pct": round(float(np.mean(losses[:tenth])), 4),
    "loss_mean_last_10pct": round(float(np.mean(losses[-tenth:])), 4),
    "loss_history": [round(x, 4) for x in losses],
}
seq_len_seen = trained["tokens_seen"] / max(1, len(train_records) * EPOCHS)
result["memory"] = {
    # The training estimate for this size of model, float32 weights (what the
    # CPU path loads), no gradient checkpointing (the loop does not use it),
    # at the run's batch size and its mean sequence length.
    "estimate_memory_gb": estimate_memory_gb(n_params / 1e9, bits=32, batch_size=recipe.batch_size,
                                             seq_len=int(round(seq_len_seen)), gradient_checkpointing=False),
    "estimate_inputs": {"params_billion": round(n_params / 1e9, 3), "bits": 32, "batch_size": recipe.batch_size,
                        "seq_len": int(round(seq_len_seen)), "gradient_checkpointing": False},
    "measured_peak_rss_gb_through_training": result["peak_rss_gb"]["after_training"],
    "note": "peak RSS is the whole process high-water mark (Python, tokenizer, data, model, optimizer)",
}
log(f"    {trained['steps']} steps in {trained['seconds']:.0f}s, loss {losses[0]:.3f} -> {losses[-1]:.3f}")

# The adapter on top of the base model, before merging: the merged model
# must give the same answers, or the merge broke something.
from peft import PeftModel  # noqa: E402  (imported here: only this check needs it)

base, tokenizer = load_model(base_model, device="cpu")
adapted = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
# Banking77's test split is grouped by intent, so take evenly spaced records rather than the first N:
# the first 50 would cover only 2 of the 10 intents.
check_idx = sorted({int(i) for i in np.linspace(0, len(eval_records) - 1, N_MERGE_CHECK)})
check_records = [eval_records[i] for i in check_idx]
adapter_check = evaluate_exact(adapted, tokenizer, check_records, max_new_tokens=MAX_NEW)
del adapted, base
gc.collect()

merge_info = merge_adapter(base_model, str(adapter_dir), str(merged_dir))
result["peak_rss_gb"]["after_merge"] = peak_rss_gb()
model, tokenizer = load_model(str(merged_dir), device="cpu")
log("    evaluating the merged model in PyTorch")
tuned = evaluate_exact(model, tokenizer, eval_records, max_new_tokens=MAX_NEW, on_progress=progress("fine-tuned"))
result["stages"]["2_finetuned_pytorch"] = score_summary(tuned)
merge_parity = gguf.output_parity(adapter_check["outputs"], [tuned["outputs"][i] for i in check_idx])
result["merge_check"] = {"n": len(check_records), "agreement_adapter_vs_merged": round(merge_parity.agreement, 4),
                         "adapter_accuracy": round(adapter_check["accuracy"], 4)}
result["stages"]["2_finetuned_pytorch"]["decode_speed"] = decode_speed(
    lambda m, k: generate(model, tokenizer, m, k, return_details=True))
log("    accuracy", tuned["accuracy"], "merge agreement", merge_parity.agreement)
del model
gc.collect()

files = {"adapter": {"path": str(adapter_dir), "mb": size_mb(adapter_dir)},
         "merged_float32_safetensors": {"path": str(merged_dir), "mb": round(merge_info["weights_bytes"] / 1e6, 1)}}
result["files"] = files
result["gguf_versions"] = gguf.installed_versions()


# ---------------------------------------------------------------- GGUF
def not_run(stage: str, reason: str):
    result["stages"][stage] = {"status": "not run", "reason": reason}
    log(f"    {stage} not run: {reason}")


def run_gguf():
    script = os.environ.get(gguf.CONVERT_ENV) or gguf.fetch_convert_script(str(ARTIFACTS / "llama_cpp"))
    t = time.time()
    f16 = gguf.convert_to_gguf(str(merged_dir), str(WORK / "model.f16.gguf"), script, outtype="f16")
    files["gguf_f16"] = {"path": f16, "mb": size_mb(f16), "seconds_to_convert": round(time.time() - t, 1)}
    t = time.time()
    q4 = gguf.quantize_gguf(f16, str(WORK / "model.Q4_K_M.gguf"), "Q4_K_M", n_threads=THREADS)
    files["gguf_q4_k_m"] = {"path": q4, "mb": size_mb(q4), "seconds_to_quantise": round(time.time() - t, 1)}
    # "Q4_K_M" is a recipe, not a guarantee of 4.5 bits: k-quants work on rows
    # of 256 weights, and llama.cpp falls back to q5_0 / q8_0 for any tensor
    # whose rows are not a multiple of that. Bits per weight show what it did.
    hidden = json.loads((merged_dir / "config.json").read_text()).get("hidden_size")
    for key in ("gguf_f16", "gguf_q4_k_m"):
        files[key]["bits_per_weight"] = round(Path(files[key]["path"]).stat().st_size * 8 / n_params, 2)
    files["gguf_q4_k_m"]["hidden_size"] = hidden
    files["gguf_q4_k_m"]["hidden_size_multiple_of_256"] = bool(hidden and hidden % 256 == 0)
    # The converter's own peak is not recorded: RUSAGE_CHILDREN on Linux
    # counts the child's memory before exec, which is a copy of this
    # process's, so it would only repeat the number above.

    hf_tokenizer = tokenizer
    trips = {}
    for label, messages in (("generic_conversation", None), ("task_prompt", eval_records[0]["messages"][:-1])):
        trip = gguf.chat_template_round_trip(hf_tokenizer, q4, messages)
        trips[label] = {k: trip.get(k) for k in ("passed", "identical", "tokens_identical", "template_identical",
                                                 "read_with", "rendered_with", "reasons", "n_tokens")}
    result["chat_template_round_trip"] = trips
    log("    round trip", {k: v["passed"] for k, v in trips.items()})

    for stage, path in (("3_gguf_f16", f16), ("4_gguf_q4_k_m", q4)):
        chat = gguf.GGUFChat(path, n_threads=THREADS, n_ctx=1024)
        try:
            log(f"({stage[0]}) evaluating {Path(path).name}")
            scored = evaluate_chat(lambda m, k: chat.chat_details(m, k), eval_records, max_new_tokens=MAX_NEW,
                                   on_progress=progress(stage))
            summary = score_summary(scored)
            summary["note"] = "llama.cpp reuses the cached system-prompt prefix between texts, as it does in service"
            summary["decode_speed"] = decode_speed(lambda m, k: chat.chat_details(m, k, fresh=True))
            parity = gguf.output_parity(tuned["outputs"], scored["outputs"], labels=references)
            summary["parity_with_pytorch"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                              for k, v in parity.to_dict().items()}
            result["stages"][stage] = summary
            log("    accuracy", scored["accuracy"], "agreement with PyTorch", parity.agreement)
        finally:
            chat.close()
    s3, s4 = result["stages"]["3_gguf_f16"], result["stages"]["4_gguf_q4_k_m"]
    q_vs_f = gguf.output_parity(s3["outputs"], s4["outputs"], labels=references)
    result["q4_vs_f16_agreement"] = round(q_vs_f.agreement, 4)


try:
    run_gguf()
except Exception as exc:  # the PyTorch results above are still worth writing
    reason = f"{type(exc).__name__}: {exc}"
    traceback.print_exc()
    for stage in ("3_gguf_f16", "4_gguf_q4_k_m"):
        if stage not in result["stages"]:
            not_run(stage, reason[:2000])
result["peak_rss_gb"]["end"] = peak_rss_gb()
result["seconds_total"] = round(time.time() - started_all, 1)

if not args.keep_weights:
    # The adapter is kept (a few MB); the merged model and the GGUF files can
    # be rebuilt from it with `forgelm export-gguf --adapter`.
    shutil.rmtree(merged_dir, ignore_errors=True)
    for f in WORK.glob("*.gguf"):
        f.unlink()
    result["weights_deleted_after_run"] = True

headline = {k: v.get("accuracy", v.get("status")) for k, v in result["stages"].items()}
log("accuracy by stage", headline)
if args.smoke:
    path = ARTIFACTS / "e7_generative_cpu_smoke.json"
    payload = dict(result, environment=environment(), generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   warning="SMOKE RUN: 2 intents, tiny data, 1 epoch. Proves the pipeline runs; measures nothing.")
    path.write_text(json.dumps(payload, indent=2, default=float))
    log("wrote", path)
else:
    write_result("e7_generative_cpu", result)
