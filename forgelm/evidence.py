"""The evidence behind each planner rule.

Every recommendation the planner makes cites one or more of these entries, and
each entry states how strong it is:

    measured    reproduced in this repository (see experiments/results/)
    published   read in a primary source (paper, official docs, project repo)
    reported    seen only in a search summary, vendor post or secondary write-up
    assumption  a reasoned default with no direct evidence found

A rule that rests only on "reported" or "assumption" evidence is a hypothesis
to test on your own data, and the plan output says so.
"""

from __future__ import annotations

from typing import Dict

LEVELS = ("measured", "published", "reported", "assumption")

EVIDENCE: Dict[str, dict] = {
    # ---- measured in this repository -----------------------------------
    "e1-ladder": {
        "level": "measured",
        "claim": "PENDING",
        "source": "experiments/results/e1_banking77_ladder.json",
    },
    "e1-static": {
        "level": "measured",
        "claim": "PENDING",
        "source": "experiments/results/e1_banking77_ladder.json",
    },
    "e1-zeroshot": {
        "level": "measured",
        "claim": "PENDING",
        "source": "experiments/results/e1_banking77_ladder.json",
    },
    "e1-calibration": {
        "level": "measured",
        "claim": "PENDING",
        "source": "experiments/results/e1_banking77_ladder.json",
    },
    "e2-teacher": {
        "level": "measured",
        "claim": "PENDING",
        "source": "experiments/results/e2_teacher_distillation.json",
    },
    "e3-int8": {
        "level": "measured",
        "claim": "PENDING",
        "source": "experiments/results/e3_compression.json",
    },
    "e4-tabular": {
        "level": "measured",
        "claim": "PENDING",
        "source": "experiments/results/e4_tabular.json",
    },
    "e5-cascade": {
        "level": "measured",
        "claim": "PENDING",
        "source": "experiments/results/e5_cascade.json",
    },
    "e6-screening": {
        "level": "measured",
        "claim": "PENDING",
        "source": "experiments/results/e6_unfair_terms.json",
    },
    # ---- published -----------------------------------------------------
    "peft-benchmark": {
        "level": "published",
        "claim": "HF PEFT's own benchmark (Llama-3.2-3B, MetaMathQA->GSM8K, single runs): rsLoRA r64 51.3% vs "
                 "LoRA r64 47.4% at equal memory and time; DoRA 46.9% at +10% VRAM and +26% time; VeRA, IA3, "
                 "BOFT and LoRA-FA 7-13 points worse.",
        "source": "https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA/results",
    },
    "torchtune-dead": {
        "level": "published",
        "claim": "torchtune stopped active development in July 2025; HF AutoTrain is unmaintained.",
        "source": "https://github.com/meta-pytorch/torchtune/issues/2883",
    },
    "unsloth-hardware": {
        "level": "published",
        "claim": "Unsloth's beta releases add MLX, CPU, Intel XPU and a ROCm image; no independent benchmarks "
                 "of those paths were found.",
        "source": "https://github.com/unslothai/unsloth/releases",
    },
    "mlx-lm": {
        "level": "published",
        "claim": "mlx-lm fine-tunes with LoRA, DoRA and full training on Apple Silicon and uses QLoRA "
                 "automatically when the base model is quantised.",
        "source": "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md",
    },
    "gguf-quality": {
        "level": "published",
        "claim": "llama.cpp perplexity on Llama-3-8B: Q4_K_M +2.8%, Q5_K_M +0.9%, Q8_0 +0.04%, Q2_K +56% "
                 "relative to FP16.",
        "source": "https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md",
    },
    "ort-vnni": {
        "level": "published",
        "claim": "ONNX Runtime documents that u8s8 int8 can saturate on AVX2/AVX-512 CPUs without VNNI; "
                 "reduce_range is the fix.",
        "source": "https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/performance/model-optimizations/quantization.md",
    },
    "modernbert-export": {
        "level": "published",
        "claim": "ModernBERT ONNX export fails through its compiled / flash-attention paths; eager attention "
                 "with compilation disabled works.",
        "source": "https://github.com/huggingface/transformers/issues/35545",
    },
    "pruning-academic": {
        "level": "published",
        "claim": "Wanda / SparseGPT at 50% unstructured sparsity raise LLaMA-2-7B perplexity by about 25%, "
                 "far more than Q4_K_M quantisation, and need sparse kernels for any speedup.",
        "source": "https://github.com/locuslab/wanda",
    },
    "tabular-licences": {
        "level": "published",
        "claim": "TabPFN v2 weights permit commercial use; 2.5 and later, TabFM, LimiX-2 and sap-rpt-1-oss are "
                 "non-commercial. TabICL (BSD-3), TabDPT and Mitra (Apache-2.0) are permissive.",
        "source": "https://github.com/PriorLabs/TabPFN ; https://github.com/soda-inria/tabicl",
    },
    "ebm": {
        "level": "published",
        "claim": "Explainable Boosting Machines give exact additive explanations and are MIT licensed.",
        "source": "https://github.com/interpretml/interpret",
    },
    "setfit-many-classes": {
        "level": "reported",
        "claim": "SetFit becomes impractical to train above roughly 10 categories; FastFit targets many-class "
                 "few-shot.",
        "source": "https://arxiv.org/pdf/2502.11830 ; https://arxiv.org/abs/2404.12365",
    },
    # ---- reported (not read in full) ------------------------------------
    "lora-lr": {
        "level": "reported",
        "claim": "After per-method learning-rate tuning, LoRA variants land within 1-2% of each other.",
        "source": "https://arxiv.org/abs/2602.04998",
    },
    "amd-bnb": {
        "level": "reported",
        "claim": "bitsandbytes releases up to 0.49.2 have a 4-bit NaN bug on AMD GPUs.",
        "source": "https://unsloth.ai/docs/get-started/install/amd",
    },
    "encoders-beat-zero-shot-llm": {
        "level": "reported",
        "claim": "Fine-tuned small encoders beat zero-shot GPT-4-era LLMs on text classification (10-25 points "
                 "on Banking77; 88 vs ~70 micro-F1 on LEDGAR). No matched comparison against 2026 frontier "
                 "models was found.",
        "source": "https://arxiv.org/abs/2406.08660 ; https://arxiv.org/abs/2304.12202",
    },
    "llm-labeller": {
        "level": "reported",
        "claim": "Students trained on LLM labels land within about 0-4 F1 of the teacher across several "
                 "studies, with lower precision than students trained on human labels.",
        "source": "https://arxiv.org/pdf/2406.17633 ; https://arxiv.org/pdf/2411.19638",
    },
    "synthetic-gap": {
        "level": "reported",
        "claim": "Classifiers trained on purely synthetic text trail those trained on real text, more so on "
                 "subjective tasks; attribute-conditioned prompts narrow but do not close the gap.",
        "source": "https://arxiv.org/abs/2310.07849 ; https://arxiv.org/abs/2306.15895",
    },
    "cascade-negative": {
        "level": "reported",
        "claim": "Cascades report 7-17% deferral with accuracy gains in some studies, but one 2026 study found "
                 "compact LLM verifiers add nothing once an in-domain supervised encoder exists.",
        "source": "https://arxiv.org/abs/2604.13285 ; https://aclanthology.org/2026.emnlp-main.1049.pdf",
    },
    "tabular-shift": {
        "level": "reported",
        "claim": "Tabular foundation models lead on small-to-medium IID data but lose to GBDTs and tuned MLPs on "
                 "temporal, grouped, large and high-cardinality data; a GLM won 3 of 4 insurance-pricing "
                 "comparisons.",
        "source": "https://arxiv.org/abs/2606.30410 ; https://arxiv.org/abs/2605.22892",
    },
    "rag-over-finetune": {
        "level": "reported",
        "claim": "For injecting knowledge, retrieval consistently outperforms fine-tuning; in an insurance "
                 "regulation benchmark retrieval acted as a knowledge equaliser across 51 LLMs.",
        "source": "https://arxiv.org/abs/2312.05934 ; https://arxiv.org/abs/2603.07825",
    },
    "prompt-first": {
        "level": "reported",
        "claim": "Prompt optimisation (GEPA, MIPROv2) reaches useful gains in hundreds of evaluations; "
                 "alternating prompt and weight optimisation beats either alone.",
        "source": "https://arxiv.org/abs/2507.19457 ; https://arxiv.org/abs/2407.10930",
    },
    "compliance-judges": {
        "level": "reported",
        "claim": "On conversation-compliance benchmarks frontier LLM judges were modest and a fine-tuned compact "
                 "judge matched them; LLM QA verdicts flipped 5-13% under counterfactual edits; span-level "
                 "evidence is needed for audit.",
        "source": "https://arxiv.org/abs/2604.12312 ; https://arxiv.org/abs/2602.18448",
    },
    "india-data-residency": {
        "level": "reported",
        "claim": "IRDAI's 2025 information regulations require policy and claim records to be held in data "
                 "centres in India; RBI's FREE-AI report recommends an AI inventory, audit and explainability. "
                 "Not legal advice - confirm with counsel.",
        "source": "docs/RESEARCH.md section 5",
    },
    # ---- assumptions -----------------------------------------------------
    "vram-heuristic": {
        "level": "assumption",
        "claim": "The memory estimate is a heuristic fitted to published minimum-VRAM tables that disagree by "
                 "up to 2x; confirm with a short real run.",
        "source": "forgelm/generative.py",
    },
    "latency-rule": {
        "level": "assumption",
        "claim": "Latency guidance extrapolates from measurements on one 4-core x86 machine; benchmark on the "
                 "deployment hardware.",
        "source": "experiments/results/",
    },
}


def cite(key: str) -> dict:
    if key not in EVIDENCE:
        raise KeyError(f"unknown evidence key {key!r}")
    return dict(EVIDENCE[key], key=key)


def weakest_level(keys) -> str:
    """The weakest evidence level among ``keys`` (a chain is as strong as that)."""
    if not keys:
        return "assumption"
    return max((EVIDENCE[k]["level"] for k in keys), key=LEVELS.index)
