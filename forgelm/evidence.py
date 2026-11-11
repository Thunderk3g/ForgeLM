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
        "claim": "On Banking77 (77 intents, 3,080 test texts) a frozen 33M encoder with a logistic-regression head was the "
                 "most accurate LLM-free student at every data size tested: 85.8% with 8 labels per class, 90.9% with 32, "
                 "93.1% with all. TF-IDF trailed by 15 points at 8 labels but by only 2 with all labels (91.2%). A 22M "
                 "encoder fine-tuned end to end with fixed, untuned settings never led (78.3%, 88.5%, 90.5%). A "
                 "fine-tuned ModernBERT-base (149M) only matched the frozen encoder with all labels (93.1%), after "
                 "50 minutes of CPU training against 38 seconds, at about 3x its latency and with a 602 MB artefact.",
        "source": "experiments/results/e1_banking77_ladder.json ; experiments/results/e1b_modernbert_full.json",
    },
    "e1-static": {
        "level": "measured",
        "claim": "A static-embedding student (potion-base-32M with a logistic-regression head) reached 78.1% / 87.7% / "
                 "91.0% at 8 / 32 / all labels per class and served at 0.14-0.22 ms per text and 13,000-18,000 texts per "
                 "second on CPU (two runs on a shared VM) with no torch at inference, 2 to 8 points below the frozen "
                 "encoder at roughly 50-90 times its single-text speed.",
        "source": "experiments/results/e1_banking77_ladder.json ; experiments/results/e1c_serving.json",
    },
    "e1-zeroshot": {
        "level": "measured",
        "claim": "Zero-shot similarity to the label names (bge-small) reached 68.0% on Banking77 with no training data, "
                 "with confidence off by 0.30 (ECE) until calibrated: a first look and a shortlister, not a deployable "
                 "classifier. Eight labels per class with a frozen encoder beat it by 18 points.",
        "source": "experiments/results/e1_banking77_ladder.json",
    },
    "e1-calibration": {
        "level": "measured",
        "claim": "Before calibration every student's stated confidence was off by 0.06 to 0.45 (ECE), worst with few "
                 "labels. Temperature scaling on 385 held-out examples brought it to 0.011-0.044 for every student and "
                 "data size.",
        "source": "experiments/results/e1_banking77_ladder.json",
    },
    "e2-teacher": {
        "level": "measured",
        "claim": "On Banking77 with no human training labels, a local 1.5B LLM teacher scored 59.2% on a 1,000-text "
                 "human-labelled anchor, below the zero-shot embedding model that shortlisted its options (67.5%). "
                 "Students trained on 2,000 teacher-labelled texts landed within 4 points of their teacher (59-63% from "
                 "the LLM, 64-67% from the embedder), far below the same students trained on 8 human labels per class "
                 "(79.9% and 85.9%). A student is bounded by its teacher: measure the teacher on an anchor first.",
        "source": "experiments/results/e2_teacher_distillation.json",
    },
    "e3-int8": {
        "level": "measured",
        "claim": "Exporting fine-tuned Banking77 classifiers to ONNX fp32 kept every decision (100% agreement) and cut "
                 "single-text latency from 6.1 to 2.1 ms (MiniLM-L6) and 34 to 23 ms (ModernBERT-base) on CPU. Dynamic "
                 "int8 made MiniLM 4x smaller with 98.0% agreement and 0.9 points lost (just failing a 98% gate), but "
                 "broke ModernBERT: 87.1% agreement and 7.9 points lost. The parity gate caught both and recommended fp32.",
        "source": "experiments/results/e3_compression.json",
    },
    "e4-tabular": {
        "level": "measured",
        "claim": "On UCI credit-default and census-income data (5,000-row test sets, 3 draws per size, library "
                 "defaults), the in-context model TabICL was best or tied from 100 to 2,000 training rows but led an "
                 "Explainable Boosting Machine by at most 0.02 AUC, inside the seed spread from 500 rows up, and took "
                 "24-51 s to score 5,000 rows on CPU. The EBM was at or near the top at every size up to 10,000 rows "
                 "and well calibrated. LightGBM at defaults had ECE 0.11-0.18 with 2,000 rows or fewer.",
        "source": "experiments/results/e4_tabular.json",
    },
    "e5-cascade": {
        "level": "measured",
        "claim": "With calibrated full-data Banking77 students and a confidence threshold chosen on 385 held-out "
                 "examples, the frozen-encoder student answered 92.7% of test texts at 96.1% accuracy (target 97%) and "
                 "72.7% at 99.1% (target 99%); targets were met to within about 1.5 points, so a small anchor gives a "
                 "noisy threshold. Escalating the uncertain items to a 1.5B local LLM lowered accuracy in every "
                 "configuration, because that LLM was less accurate than the student even on the items the student "
                 "was unsure about.",
        "source": "experiments/results/e5_cascade.json",
    },
    "e6-screening": {
        "level": "measured",
        "claim": "Screening LexGLUE UNFAIR-ToS sentences for potentially unfair terms (10.7% positive), with the "
                 "threshold set for 90% recall on validation: TF-IDF and a fine-tuned 22M encoder each caught 90.7% "
                 "of unfair terms on test while sending 18-19% of sentences to review; frozen-embedding students sent "
                 "26-27%. A zero-shot 1.5B local LLM caught 83.1% while sending 42.7%, and students trained only on its "
                 "labels sent 56-60%. TF-IDF's top features ('we may', 'liable', 'terminate', 'courts') are directly "
                 "auditable.",
        "source": "experiments/results/e6_unfair_terms.json",
    },
    "e7-generative": {
        "level": "measured",
        "claim": "On a 4-core CPU, LoRA (rank 16, 3 epochs, 19 minutes) took SmolLM2-360M-Instruct from 1.3% to "
                 "89.0% exact match on a 10-intent Banking77 generation task. GGUF f16 export kept every answer "
                 "(100% agreement) at half the size and twice the speed; Q4_K_M lost 3.3 points (92.7% agreement) "
                 "at 271 MB and 0.25 s per text. The chat template survived export. The memory heuristic "
                 "estimated 2.69 GB against a measured 6.88 GB peak resident memory.",
        "source": "experiments/results/e7_generative_cpu.json",
    },
    "e8-prompt-or-train": {
        "level": "measured",
        "claim": "With 8 labels per class on Banking77 (1,000 test texts), a frozen bge-small encoder with a "
                 "logistic-regression head scored 87.5% at 0.003 s per text. A local 1.5B LLM scored 59.2% "
                 "zero-shot and 66.9% with 5 retrieved demonstrations, at 0.56-0.77 s per text: retrieved "
                 "demonstrations helped the LLM by 7.7 points but it stayed 20.6 points behind the trained small "
                 "model (paired bootstrap and exact McNemar agree, p < 1e-38).",
        "source": "experiments/results/e8_prompt_or_train.json",
    },
    "e9-retrieval": {
        "level": "measured",
        "claim": "On FiQA financial QA (648 test queries, 15,000-passage subsample, not comparable with published "
                 "scores), dense retrieval with bge-small reached nDCG@10 0.519 against 0.345 for BM25 and 0.269 "
                 "for static potion embeddings. Equal-weight reciprocal-rank fusion of BM25 and bge-small scored "
                 "0.459, significantly below dense alone (-0.060, 95% interval -0.080 to -0.041). BM25 answered "
                 "in 0.4 ms per query, bge-small in about 100 ms on CPU.",
        "source": "experiments/results/e9_retrieval.json",
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
                 "up to 2x, and in E7 it was 2.6x below the measured peak on a CPU run; confirm with a short real "
                 "run (forgelm.dryrun) before committing to a model size.",
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
