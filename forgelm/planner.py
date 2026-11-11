"""The planner: task + data + constraints + hardware -> an ordered plan.

This is the part of the platform that replaces "pick a base model and press
train". It answers, before any compute is spent:

    which *kind* of system fits the task (compact classifier, classical ML,
    retrieval, a fine-tuned LLM, or a cascade),
    which candidates to try, cheapest first,
    what each needs (packages, memory) and whether this machine can run it,
    how the result should be compiled and served on the deployment hardware,
    which guardrails are mandatory,
    and how strong the evidence behind each recommendation is.

It is rule-based on purpose. Every rule cites :mod:`forgelm.evidence`, so a
recommendation can be challenged, and replaced when a measurement disagrees.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from . import generative, tabular
from .evidence import EVIDENCE, weakest_level
from .hardware import HardwareProfile

TASK_KINDS = ("text_classification", "tabular_classification", "generation", "document_qa")


@dataclass
class TaskSpec:
    kind: str = "text_classification"
    # text classification
    n_classes: int = 2
    labelled_per_class: float = 0.0
    n_unlabelled: int = 0
    avg_tokens: int = 32
    multilingual: bool = False
    needs_world_knowledge: bool = False
    # tabular
    n_rows: int = 0
    n_features: int = 0
    has_text_columns: bool = False
    temporal_or_grouped: bool = False
    # generation
    model_params_billion: float = 3.0
    knowledge_changes_often: bool = False

    def __post_init__(self):
        if self.kind not in TASK_KINDS:
            raise ValueError(f"unknown task kind {self.kind!r}; choose from {TASK_KINDS}")


@dataclass
class Constraints:
    llm_at_inference: bool = False  # may the deployed system call an LLM?
    teacher_available: bool = False  # is an LLM available at build time?
    commercial_use: bool = True
    needs_inherent_explainability: bool = False
    data_must_stay_on_prem: bool = False
    max_latency_ms: Optional[float] = None  # per item, on the deployment hardware
    deploy: Optional[HardwareProfile] = None  # where it will run; defaults to the training machine


@dataclass
class Step:
    name: str
    action: str
    why: str
    evidence: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    feasible: bool = True
    detail: Dict[str, object] = field(default_factory=dict)

    @property
    def evidence_level(self) -> str:
        return weakest_level(self.evidence)


@dataclass
class Plan:
    task: TaskSpec
    hardware_class: str
    deploy_class: str
    approach: str
    approach_why: str
    steps: List[Step] = field(default_factory=list)
    deployment: List[str] = field(default_factory=list)
    guardrails: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    not_recommended: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for step, raw in zip(self.steps, d["steps"]):
            raw["evidence_level"] = step.evidence_level
            raw["evidence"] = [dict(EVIDENCE[k], key=k) for k in step.evidence]
        return d

    def render(self) -> str:
        out = [f"Task          : {self.task.kind}",
               f"Train on      : {self.hardware_class}",
               f"Deploy on     : {self.deploy_class}",
               f"Approach      : {self.approach}",
               f"                {self.approach_why}", "", "Candidates, cheapest first:"]
        for i, s in enumerate(self.steps, 1):
            flag = "" if s.feasible else "  [not runnable here]"
            out.append(f"  {i}. {s.name}{flag}")
            out.append(f"     do   : {s.action}")
            out.append(f"     why  : {s.why}")
            if s.missing:
                out.append(f"     need : pip install {' '.join(s.missing)}")
            if s.evidence:
                out.append(f"     basis: {s.evidence_level} ({', '.join(s.evidence)})")
        if self.deployment:
            out += ["", "Deployment:"] + [f"  - {d}" for d in self.deployment]
        if self.guardrails:
            out += ["", "Guardrails (not optional):"] + [f"  - {g}" for g in self.guardrails]
        if self.not_recommended:
            out += ["", "Not recommended:"] + [f"  - {n['what']}: {n['why']}" for n in self.not_recommended]
        if self.warnings:
            out += ["", "Warnings:"] + [f"  - {w}" for w in self.warnings]
        return "\n".join(out)


# --------------------------------------------------------------------------

_PIP = {"sklearn": "scikit-learn", "sentence_transformers": "sentence-transformers", "interpret": "interpret-core",
        "mlx_lm": "mlx-lm"}


def _step(hw: HardwareProfile, name: str, action: str, why: str, evidence: List[str],
          requires: Optional[List[str]] = None, feasible: bool = True, **detail) -> Step:
    requires = requires or []
    missing = [_PIP.get(p, p) for p in requires if not hw.has(p)]
    return Step(name, action, why, evidence, requires, missing, feasible, detail)


def _encoder_choice(task: TaskSpec, hw: HardwareProfile) -> Dict[str, str]:
    """Which encoder to fine-tune, given hardware, text length and language."""
    gpu = hw.device_class != "cpu" and hw.accelerator_memory_gb >= 6
    if task.multilingual:
        if gpu:
            return {"model": "jhu-clsp/mmBERT-base", "why": "multilingual, 8k context, MIT licence"}
        return {"model": "intfloat/multilingual-e5-small", "why": "118M multilingual encoder that trains on CPU"}
    if task.avg_tokens > 400:
        return {"model": "answerdotai/ModernBERT-base",
                "why": "8,192-token context; older encoders truncate at 512 tokens"}
    if gpu:
        return {"model": "microsoft/deberta-v3-base",
                "why": "most sample-efficient encoder in controlled comparisons; ModernBERT-base if speed matters"}
    return {"model": "sentence-transformers/all-MiniLM-L6-v2",
            "why": "22M parameters: fine-tunes on a CPU in minutes and serves in a few milliseconds"}


def _plan_text_classification(task: TaskSpec, c: Constraints, hw: HardwareProfile, deploy: HardwareProfile) -> Plan:
    k = task.labelled_per_class
    total = k * task.n_classes
    if task.needs_world_knowledge and c.llm_at_inference:
        approach = "hybrid cascade"
        why = ("the label depends on knowledge outside the text, so a compact classifier screens and an LLM "
               "(with retrieval) handles what it cannot")
    else:
        approach = "compact classifier"
        why = ("classification with stable labels is the case where small supervised models match or beat "
               "LLMs at a fraction of the cost, and need no LLM at inference")
    plan = Plan(task, hw.device_class, deploy.device_class, approach, why)
    S = plan.steps

    if k == 0:
        S.append(_step(hw, "zero-shot by label description",
                       "ZeroShotStudent: embed a one-line description of each label, classify by similarity",
                       "the only option with no labels; treat the output as a first look and as a shortlister",
                       ["e1-zeroshot"], ["sentence_transformers"], student="zero_shot"))
        S.append(_step(hw, "label about 8 examples per class by hand",
                       "EmbeddingStudent('BAAI/bge-small-en-v1.5', backend='sentence') on the hand-labelled set",
                       "the cheapest large gain measured here: 8 labels per class with a frozen encoder "
                       "reached 85.8% on a 77-class task, against 68.0% zero-shot and 59-67% for students "
                       "trained on 2,000 texts labelled by a small local LLM or a zero-shot model",
                       ["e1-ladder", "e2-teacher"], ["sentence_transformers", "sklearn"], student="encoder"))
        if c.teacher_available and task.n_unlabelled > 0:
            S.append(_step(hw, "teacher-label the unlabelled pool, then train a student",
                           "distill(LocalLLMTeacher or ChatCompletionsTeacher, EmbeddingStudent, pool, anchor)",
                           "turns an LLM's judgement into a model that needs no LLM at inference, but the "
                           "student is bounded by the teacher: score the teacher on a human-labelled anchor "
                           "first, and only distil if it beats what a few hand labels give",
                           ["e2-teacher", "llm-labeller"], ["sentence_transformers", "torch"],
                           student="embedding", teacher=True))
        elif c.teacher_available:
            S.append(_step(hw, "generate synthetic examples, then train a student",
                           "attribute_prompts(...) -> teacher generates -> diversity_report -> EmbeddingStudent",
                           "no unlabelled text exists; synthetic data trails real data, so collect even a small "
                           "real pool as soon as possible",
                           ["synthetic-gap"], ["sentence_transformers"], student="embedding", synthetic=True))
        plan.warnings.append("with zero human labels there is nothing to measure against: label at least "
                             "20-50 real examples per class as an anchor before deploying anything")
    else:
        S.append(_step(hw, "TF-IDF + logistic regression",
                       "TfidfStudent().fit(...)",
                       "the baseline everything else must beat; trains in seconds and every prediction "
                       "decomposes into n-gram weights. With enough labels it can be the model to ship: on "
                       "unfair-term screening it matched a fine-tuned encoder",
                       ["e1-ladder", "e6-screening"], ["sklearn"], student="tfidf"))
        S.append(_step(hw, "static embeddings + logistic regression",
                       "EmbeddingStudent('minishlab/potion-base-32M')",
                       "a lookup table and a matrix multiply: sub-millisecond on one CPU core, no torch at "
                       "inference",
                       ["e1-static"], ["model2vec", "sklearn"], student="static"))
        S.append(_step(hw, "frozen sentence encoder + logistic regression",
                       "EmbeddingStudent('BAAI/bge-small-en-v1.5', backend='sentence')"
                       if not task.multilingual else
                       "EmbeddingStudent('intfloat/multilingual-e5-small', backend='sentence')",
                       "strongest option when labels are scarce: the encoder is not updated, so a handful of "
                       "examples per class cannot overfit it",
                       ["e1-ladder"], ["sentence_transformers", "sklearn"], student="encoder"))
        if k < 16 and task.n_classes <= 10:
            S.append(_step(hw, "SetFit (contrastive few-shot)",
                           "setfit.SetFitModel + Trainer on the same few examples",
                           "contrastive fine-tuning squeezes more from very few labels when there are few "
                           "classes; pair count explodes with many classes",
                           ["setfit-many-classes"], ["setfit"], student="setfit"))
        enc = _encoder_choice(task, hw)
        if k >= 32 or total >= 2000:
            cpu_big = hw.device_class == "cpu" and "MiniLM" not in enc["model"] and "e5-small" not in enc["model"]
            S.append(_step(hw, f"fine-tune an encoder ({enc['model'].split('/')[-1]})",
                           f"FineTunedStudent('{enc['model']}')",
                           f"{enc['why']}. Whether it beats a frozen encoder depends on the task: on short "
                           "banking intents it did not (90.5% against 93.1% with all labels), on longer legal "
                           "sentences with 5,500 labels it did (ROC AUC 0.969 against 0.949). Run both and "
                           "compare on the same hold-out",
                           ["e1-ladder", "e6-screening", "encoders-beat-zero-shot-llm"], ["torch", "transformers"],
                           student="finetuned", model=enc["model"],
                           note="trains on CPU but slowly; budget an hour or more" if cpu_big else ""))
        else:
            plan.not_recommended.append({
                "what": "fine-tuning an encoder end to end",
                "why": f"about {k:g} labels per class is too few; frozen embeddings generalise better here. "
                       "Revisit at roughly 32+ per class"})
        if c.teacher_available and task.n_unlabelled >= 500:
            S.append(_step(hw, "enlarge the training set with teacher labels",
                           "distill(teacher, student, unlabelled_pool, anchor) with soft labels",
                           "worth it only if the teacher, scored on the anchor, beats the student trained on "
                           "the human labels alone; a 1.5B local LLM did not in the measurement here, so use the "
                           "strongest teacher available",
                           ["e2-teacher", "llm-labeller"], ["torch", "transformers"], teacher=True))

    # ---- deployment ------------------------------------------------------
    dcls = deploy.device_class
    if c.max_latency_ms is not None and c.max_latency_ms < 2 and dcls == "cpu":
        plan.deployment.append(f"latency budget {c.max_latency_ms} ms on CPU: only TF-IDF and static embeddings "
                               "meet it; a transformer encoder will not")
        plan.warnings.append("latency figures were measured on one 4-core x86 machine; re-measure on the target")
    if dcls == "cpu":
        plan.deployment.append("static / TF-IDF students: serve the numpy head directly (no torch, no ONNX)")
        plan.deployment.append("encoder students: export to ONNX fp32 and serve with ONNX Runtime (lossless and "
                               "1.4-3x faster per text than PyTorch in E3)")
        vnni = "has VNNI" if deploy.has_vnni else "has no VNNI: quantise with reduce_range"
        plan.deployment.append(f"int8: only after the parity gate passes; this CPU {vnni}. Dynamic int8 lost 7.9 "
                               "points on ModernBERT in E3, so never ship it unchecked")
    elif dcls == "apple":
        plan.deployment.append("encoder students: PyTorch MPS in fp32, or ONNX Runtime CPU (arm64) for portability")
    else:
        plan.deployment.append("encoder students: fp16 / bf16 on the GPU; ONNX Runtime CPU int8 as the fallback tier")
    if c.llm_at_inference:
        S.append(_step(hw, "escalate low-confidence items to a larger model",
                       "threshold_for_accuracy on validation -> simulate_cascade",
                       "keeps the LLM for the minority of items the small model is unsure about, and only "
                       "helps if the larger model is more accurate on exactly those items: in the measurement "
                       "here, escalating to a 1.5B local LLM lowered accuracy at every threshold",
                       ["e5-cascade", "cascade-negative"]))

    plan.guardrails += [
        "hold out a human-labelled anchor set that is never used for training or teacher labelling",
        "calibrate confidence (temperature scaling) on held-out real data before setting any threshold",
        "set an abstention threshold for a stated accuracy target; route the rest to review",
        "re-run the evaluation on the exported artefact (parity gate) before it replaces the original",
        "record the run: data hash, recipe, hardware, metrics, artefact hashes",
    ]
    if c.needs_inherent_explainability:
        plan.warnings.append("neural encoders are not inherently explainable: keep the TF-IDF model as the "
                             "explainable reference, or require span-level evidence from the reviewer tier")
    if c.data_must_stay_on_prem:
        plan.guardrails.append("teacher must run on-premises (LocalLLMTeacher, or ChatCompletionsTeacher pointed "
                               "at an in-house server)")
    if task.multilingual:
        plan.warnings.append("code-mixed and romanised text (for example Hinglish, ASR transcripts) degrades "
                             "multilingual encoders; evaluate on transcripts from your own channel")
    return plan


def _plan_tabular(task: TaskSpec, c: Constraints, hw: HardwareProfile, deploy: HardwareProfile) -> Plan:
    plan = Plan(task, hw.device_class, deploy.device_class, "classical ML",
                "structured data with labelled outcomes: gradient boosting and linear models remain the bar, "
                "with in-context foundation models as a challenger on small IID data")
    recs = tabular.recommend(task.n_rows, task.n_features, task.n_classes, task.has_text_columns,
                             gpu=hw.device_class != "cpu", commercial_use=c.commercial_use,
                             needs_inherent_explainability=c.needs_inherent_explainability,
                             temporal_or_grouped=task.temporal_or_grouped)
    pkg = {"logreg": "sklearn", "lightgbm": "lightgbm", "ebm": "interpret", "tabicl": "tabicl",
           "tabpfn-v2": "tabpfn", "tabpfn-2.5+": "tabpfn"}
    ev = {"tabicl": ["e4-tabular", "tabular-licences"], "tabpfn-v2": ["tabular-licences"],
          "ebm": ["ebm"], "lightgbm": ["e4-tabular", "tabular-shift"], "logreg": ["e4-tabular"]}
    for r in recs:
        if r["role"] == "note":
            (plan.not_recommended if r["model"].startswith("(excluded") else plan.warnings).append(
                {"what": r["model"].strip("()"), "why": r["why"]} if r["model"].startswith("(excluded")
                else f"{r['model'].strip('()')}: {r['why']}")
            continue
        plan.steps.append(_step(hw, f"{r['model']} ({r['role']})", f"tabular.run_ladder(models=['{r['model']}'])",
                                r["why"], ev.get(r["model"], []), [pkg[r["model"]]] if r["model"] in pkg else []))
    plan.deployment.append("GBDT / linear / EBM: serve the native library on CPU; no accelerator needed")
    plan.deployment.append("in-context models keep the training rows (or a cache of them) in the serving "
                           "process: customer data must be allowed there, and deleting a record means refitting")
    plan.guardrails += [
        "split the way the model will be used: by time or by group, not randomly, if the data has either",
        "report calibration (Brier, reliability) as well as AUC; risk scores are consumed as probabilities",
        "pin foundation-model checkpoints: package defaults change between releases, some to non-commercial",
        "record the run: data hash, recipe, hardware, metrics, artefact hashes",
    ]
    return plan


def _plan_generation(task: TaskSpec, c: Constraints, hw: HardwareProfile, deploy: HardwareProfile) -> Plan:
    plan = Plan(task, hw.device_class, deploy.device_class, "fine-tuned small LLM, after cheaper baselines",
                "weight updates are the last step, not the first: a prompt-optimisation baseline and, for "
                "knowledge, retrieval are cheaper and often enough")
    S = plan.steps
    S.append(_step(hw, "freeze an evaluation contract",
                   "EvalContract(dataset, metric, threshold)",
                   "every later stage (prompting, tuning, quantising) is accepted or rejected against the same "
                   "frozen test", []))
    S.append(_step(hw, "prompt-optimisation baseline",
                   "optimise the prompt of the untuned model (for example DSPy MIPROv2 or GEPA) on the contract",
                   "hundreds of evaluations instead of a training run; also the starting point if you do train",
                   ["prompt-first"]))
    if task.knowledge_changes_often:
        S.append(_step(hw, "retrieval over the source documents",
                       "RAG with quoted sources",
                       "for knowledge that changes, retrieval beats baking facts into weights and stays auditable",
                       ["rag-over-finetune"]))
    choice = generative.select_backend(hw, task.model_params_billion)
    name, cfg, notes = generative.lora_config("lora", rank=32)
    S.append(_step(hw, f"LoRA fine-tune a {task.model_params_billion:g}B model with {choice.backend}",
                   f"{choice.backend} on {choice.device}, {choice.load_bits}-bit weights, {choice.precision}; "
                   f"plain LoRA r={cfg['r']} (alpha {cfg['lora_alpha']}) on attention and MLP projections. Rank-stabilised scaling "
                   f"(rsLoRA) is an option to try, together with a learning-rate sweep, because it changes "
                   f"the adapter scale",
                   "; ".join(choice.reasons),
                   ["peft-benchmark", "lora-lr", "vram-heuristic"],
                   ["torch", "transformers", "peft", "trl"] if choice.backend != "mlx_lm" else ["mlx_lm"],
                   feasible=choice.feasible, memory=choice.memory, available_gb=choice.available_gb,
                   backend=choice.to_dict(), lora=cfg))
    plan.warnings += choice.warnings
    dchoice = generative.select_backend(deploy, task.model_params_billion) if deploy is not hw else choice
    plan.deployment.append(f"format: {dchoice.serve_format}")
    plan.deployment.append(f"runtime: {dchoice.serve_runtime}")
    plan.deployment.append("re-run the evaluation contract on the quantised artefact; measure against your own "
                           "16-bit model rather than quoting perplexity tables")
    plan.not_recommended += [
        {"what": "DoRA, PiSSA, OLoRA as defaults",
         "why": "no consistent gain over well-tuned LoRA; available as advanced options. Sweep the learning "
                "rate before changing the method"},
        {"what": "LoftQ", "why": "not offered: it needs the unquantised base model, which a fine-tune request "
                                 "does not carry"},
        {"what": "VeRA, IA3, BOFT, LoRA-FA", "why": "7-13 points worse than LoRA in the PEFT project benchmark"},
        {"what": "pruning (Wanda / SparseGPT) for deployment",
         "why": "costs far more quality than 4-bit quantisation and needs sparse kernels to be faster"},
        {"what": "torchtune, AutoTrain", "why": "no longer maintained"},
    ]
    plan.guardrails += [
        "check the chat template round-trips between training and the serving runtime",
        "pin the base-model revision: an adapter is only valid for the exact weights it was trained on",
        "record the run: data hash, recipe, hardware, metrics, artefact hashes",
    ]
    return plan


def _plan_document_qa(task: TaskSpec, c: Constraints, hw: HardwareProfile, deploy: HardwareProfile) -> Plan:
    plan = Plan(task, hw.device_class, deploy.device_class, "retrieval with a reasoning model",
                "interpreting policies and regulations is a knowledge task: the answer must be grounded in, and "
                "cite, the current text. Fine-tuning does not inject knowledge reliably")
    S = plan.steps
    S.append(_step(hw, "retrieval with quoted sources",
                   "hybrid lexical + dense retrieval, reranker, answer with verbatim citations",
                   "retrieval equalises knowledge across models; citations make every answer checkable",
                   ["rag-over-finetune"]))
    S.append(_step(hw, "compact classifiers around the retriever",
                   "FineTunedStudent / EmbeddingStudent for obligation-sentence detection, document routing, "
                   "query intent",
                   "the high-volume, stable sub-tasks do not need an LLM", ["e1-ladder", "compliance-judges"],
                   ["sentence_transformers", "sklearn"]))
    S.append(_step(hw, "human review for consequential answers",
                   "route answers used for customer communication or filings to a reviewer",
                   "published accuracy on regulatory QA is still imperfect", ["compliance-judges"]))
    plan.not_recommended.append({"what": "fine-tuning an LLM on the regulation text",
                                 "why": "retrieval outperforms fine-tuning for knowledge, and regulations change"})
    if c.data_must_stay_on_prem:
        plan.guardrails.append("serve the reasoning model on-premises (GGUF via llama.cpp, or vLLM on a GPU)")
    plan.guardrails += ["version the document corpus; log which passages each answer cited",
                        "evaluate retrieval (recall@k) separately from answer quality"]
    return plan


def plan(task: TaskSpec, hardware: Optional[HardwareProfile] = None,
         constraints: Optional[Constraints] = None) -> Plan:
    """Build a plan for ``task`` on ``hardware`` (probed if omitted)."""
    from .hardware import probe

    hw = hardware or probe()
    c = constraints or Constraints()
    deploy = c.deploy or hw
    builder = {"text_classification": _plan_text_classification, "tabular_classification": _plan_tabular,
               "generation": _plan_generation, "document_qa": _plan_document_qa}[task.kind]
    result = builder(task, c, hw, deploy)
    weak = [s.name for s in result.steps if s.evidence and s.evidence_level in ("reported", "assumption")]
    if weak:
        result.warnings.append("these steps rest on evidence that was not reproduced here or read in full; "
                               "treat them as hypotheses to test: " + "; ".join(weak))
    return result
