# Research findings and what the platform does about them

Compiled 11 November 2026 from six parallel research passes (PEFT and training
frameworks; LLM-free classifiers and distillation; tabular foundation models;
compression and runtimes; BFSI and compliance; customization platforms), then
checked against experiments run in this repository.

## How to read the evidence tags

| Tag | Meaning |
|---|---|
| **[measured]** | Reproduced here. Numbers are in `experiments/results/`, summarised in `experiments/RESULTS.md`. |
| **[read]** | A researcher fetched and read the primary source (project repository, official docs). |
| **[snippet]** | Seen only in a search-result summary. arxiv.org, huggingface.co, regulator and vendor sites were not reachable during the research passes, so **no paper or regulation was read in full**. Check before quoting. |
| **[recall]** | From model memory, unchecked. |

The same levels are machine-readable in `forgelm/evidence.py`, and every
planner recommendation cites them. Anything tagged [snippet] or [recall] is a
hypothesis to test, not a result.

---

## 1. PEFT methods and training frameworks

**Findings**

- The best primary evidence is the PEFT project's own benchmark (Llama-3.2-3B,
  MetaMathQA to GSM8K, one run per method, learning rates untuned) **[read]**:

  | Method | GSM8K accuracy | Peak VRAM | Train time |
  |---|---|---|---|
  | LoRA r32 | 47.8% | 22.3 GB | 942 s |
  | LoRA r64 | 47.4% | 22.6 GB | 967 s |
  | rsLoRA r64 | 51.3% | 22.6 GB | 966 s |
  | DoRA r32 | 46.9% | 24.5 GB | 1,191 s |
  | OFT r32 | 49.1% | 22.3 GB | 1,192 s |
  | VeRA | 35.4% | 21.6 GB | 1,011 s |
  | IA3 | 39.8% | 23.2 GB | 944 s |
  | BOFT | 34.3% | 24.7 GB | 6,926 s |
  | Full fine-tune | 47.2% | 38.9 GB | 2,147 s |

  Differences of two points or less are noise at one run each. rsLoRA is the
  only variant that gained at no memory or time cost, but it also takes a
  larger effective step at the same learning rate, and learning rates were not
  tuned. DoRA cost about 10% more VRAM and 26% more time for no gain. VeRA,
  IA3, BOFT and LoRA-FA lost 7 to 13 points.
- After per-method learning-rate tuning, LoRA variants reportedly land within
  1 to 2% of each other **[snippet]**. Under reinforcement-learning objectives
  DoRA reportedly leads and PiSSA collapses **[snippet]**.
- Framework status **[read]**: torchtune stopped development in July 2025.
  HF AutoTrain is unmaintained. Unsloth, Axolotl, TRL, MLX-LM and Ludwig are
  active. Unsloth's beta releases now claim MLX, CPU, Intel XPU and ROCm
  support; no independent benchmark of those paths was found. Unsloth's core is
  Apache-2.0 but its Studio UI is AGPL-3.0.
- Minimum VRAM for 4-bit QLoRA, two published tables that disagree by up to 2x:
  3B about 1.5 to 3.5 GB, 8B about 6 GB, 14B 8.5 to 12 GB, 32B 24 to 26 GB,
  70B 41 to 48 GB **[read / snippet]**. Real peaks at realistic batch sizes are
  much higher than these minimums.
- Small models with permissive licences: Qwen3 (0.6B to 8B), SmolLM2/3,
  Phi-4-mini; Gemma 3, Llama 3.2 and LFM2 carry restrictive terms **[recall /
  snippet]**. The evidence that fine-tuned small models match large ones is
  vendor-run and unreplicated **[snippet]**.

**What the platform does**

- `forgelm.generative.lora_config` maps method names onto HF `peft`. Plain LoRA
  is the default. rsLoRA is opt-in, not automatic: it changes the adapter scale
  from alpha/rank to alpha/sqrt(rank), which at rank 64 is an 8x larger
  effective update, so the benchmark gain above may be a learning-rate effect.
  Use it together with a learning-rate sweep. DoRA, PiSSA and OLoRA are
  advanced options; VeRA, IA3 and BOFT are not offered. LoftQ is not offered
  because it needs the unquantised base model, which the request path does not
  have.
- The training service's hand-written DoRA / OFT / PiSSA layers (never called)
  were replaced by that mapping. The Transformers path now honours the
  requested method, rank, alpha, dropout and target modules; before, it applied
  a fixed r=16 LoRA, and for `dora`, `pissa` or `oft` it applied no adapter at
  all, silently running a full fine-tune.
- `forgelm.generative.select_backend` chooses Unsloth, PEFT + TRL or MLX-LM
  from the probed hardware, not from NVIDIA VRAM tiers.
- The memory estimate is a labelled heuristic, tested against both published
  tables. It is not a substitute for a short real run.

## 2. Small classifiers that need no LLM at inference

**Findings**

- Vendor benchmark, 26 datasets, 1,000 training examples **[read]**: static
  embeddings + logistic regression 76.3 F1 at about 33,000 texts/s; SetFit 74.7
  at about 1,000/s; fully fine-tuned MiniLM 77.1 at 274/s; TF-IDF 72.9. Static
  models lose badly on some tasks (emotion 66 against 81).
- SetFit is maintained (v1.2.0) but is reported impractical above roughly 10
  classes; FastFit targets many-class few-shot **[read / snippet]**.
- ModernBERT (149M / 395M, 8,192 tokens, Apache-2.0). One controlled study
  found DeBERTa-v3 more sample-efficient, with ModernBERT faster and more
  learning-rate sensitive **[snippet]**.
- Zero-shot without an LLM: an independent benchmark puts embedding models at
  about 0.62 macro-F1 and NLI cross-encoders at about 0.59, below a small
  reranker; vendor numbers for GLiClass and GLiNER2 contradict each other
  **[snippet]**.
- LLM-as-labeller: students land within roughly 0 to 4 F1 of the teacher in
  several studies, with lower precision than students trained on human labels
  **[snippet]**.
- Purely synthetic training text trails real text, more so on subjective
  tasks; attribute-conditioned prompts narrow the gap **[snippet]**.
- Cascades report 7 to 17% deferral with accuracy gains in some studies; one
  2026 study found compact LLM verifiers add nothing once an in-domain
  supervised encoder exists **[snippet]**.
- ONNX int8 is about 3x faster than PyTorch on short texts **[read]**. On
  ModernBERT-large, one report measured only 32 to 67% decision agreement
  between int8 and fp32 **[read]**; another reported a small loss **[snippet]**.

**What this repository measured** (see `experiments/RESULTS.md`)

- E1: the student ladder on Banking77 at 8, 32 and all labels per class.
- E2: a local 1.5B LLM as teacher, with zero human training labels.
- E3: ONNX fp32 and int8 behind a parity gate, MiniLM and ModernBERT.
- E5: abstention thresholds and cascades.
- E6: screening for unfair contract terms at a fixed recall target.

**What the platform does**

- `forgelm.students` implements the ladder behind one interface, with soft
  labels throughout. The linear-head students are stored as numpy arrays, so
  serving needs no torch.
- `forgelm.distill` enforces a human-labelled anchor that the teacher never
  labels and the student never trains on.
- `forgelm.metrics` covers calibration, abstention and cascade simulation, and
  reports whether escalation actually helps.
- `forgelm.compress.export_and_gate` refuses an int8 export that drifts.

## 3. Tabular foundation models

**Findings**

- Licences are the first filter **[read]**. TabPFN v2 weights permit commercial
  use. TabPFN 2.5 and later (the default of the `tabpfn` package), TabFM,
  LimiX-2 and sap-rpt-1-oss are non-commercial. TabICL (BSD-3), TabDPT and
  Mitra (Apache-2.0) are permissive. TabPFN's distillation to an MLP or tree
  ensemble is an enterprise feature, not in the open package.
- On TabArena-Lite the best commercially usable AutoGluon preset scores 1745
  Elo against 1480 for a tuned, ensembled RealMLP; tuned LightGBM and CatBoost
  sit near 1420 to 1430 **[read / snippet]**. That table is published by the
  team that maintains the benchmark.
- Negative results **[snippet]**: foundation models lead only on small-to-medium
  IID data and lose to GBDTs under temporal or grouped splits, on large and
  high-cardinality data, and under distribution shift. In an insurance-pricing
  comparison a GLM won three of four tests.
- In-context models keep the training rows in the serving process **[read]**.
  Customer data must be allowed there, and deleting a record means refitting.
- Below 300 rows no family is well evidenced; TabICL is explicitly untested
  there **[read]**.

**What the platform does**

- `forgelm.tabular.FOUNDATION_MODEL_LICENCES` gates recommendations; the
  planner excludes non-commercial weights unless told the use is non-commercial.
- `forgelm.tabular.run_ladder` always fits logistic regression and a GBDT next
  to any foundation model, on the same split, with calibration metrics.
- When each decision needs stated reasons, an Explainable Boosting Machine is
  the champion and everything else is a challenger.
- E4 measured this ladder on credit-default and census-income data. TabICL was
  best or tied from 100 to 2,000 rows, but by at most 0.02 AUC over an EBM,
  within the seed spread from 500 rows up, and it took 24 to 51 s to score
  5,000 rows on a CPU against well under a second for the rest. The EBM was at
  or near the top at every size and well calibrated; LightGBM at its defaults
  was badly calibrated with 2,000 rows or fewer. The planner therefore lists
  the EBM as a primary candidate, not an afterthought, and warns about
  untuned LightGBM probabilities on small data. **[measured]**

## 4. Compression and inference runtimes

**Findings**

- GGUF quality on Llama-3-8B, perplexity relative to FP16 **[read]**: Q8_0
  +0.04%, Q5_K_M +0.9%, Q4_K_M +2.8%, Q3_K_M +10.5%, Q2_K +56%. Larger models
  tolerate more. Perplexity is not comparable across projects, and fine-tunes
  raise it: measure against your own 16-bit model.
- AutoAWQ is deprecated and AutoGPTQ unmaintained; GPTQModel and llm-compressor
  are the current paths. TGI is in maintenance mode. Intel's IPEX is retired in
  favour of PyTorch's native XPU backend. ONNX Runtime removed its ROCm
  provider; DirectML is in sustained engineering **[read]**.
- Pruning (Wanda, SparseGPT) at 50% sparsity raises perplexity about 25%, far
  more than 4-bit quantisation, and needs sparse kernels for any speedup
  **[read]**. It is not a deployment technique today.
- ONNX Runtime documents an accuracy trap: on AVX2 / AVX-512 CPUs without VNNI,
  int8 multiplies can saturate; `reduce_range` is the fix **[read]**.
- ModernBERT ONNX export fails through its compiled and flash-attention paths;
  eager attention works **[read]**.

**Hardware matrix**

| Target | Generative LLM | Encoder classifier |
|---|---|---|
| CPU only | train elsewhere; GGUF Q4_K_M on llama.cpp | ONNX Runtime fp32; int8 only after a parity check |
| NVIDIA | QLoRA with Unsloth or PEFT; vLLM or llama.cpp | fp16 on GPU |
| AMD Radeon | ROCm PyTorch, LoRA on 16-bit weights; GGUF on llama.cpp (HIP / Vulkan) | ROCm fp16, or the CPU path |
| Intel Arc | PyTorch XPU; OpenVINO int4 or GGUF | OpenVINO int8 |
| Apple Silicon | mlx-lm; MLX 4-bit or GGUF (Metal) | PyTorch MPS fp32, or ONNX Runtime arm64 |

4-bit loading on AMD, Intel and Apple is recent; bitsandbytes releases up to
0.49.2 are reported to produce NaNs on AMD **[snippet]**. The default therefore
avoids it there and warns when it cannot.

**What the platform does**

- `forgelm.hardware.probe` reports the vendor runtime and the CPU's VNNI / AMX
  support. `forgelm doctor` prints it.
- `forgelm.compress.quantize_dynamic_int8` takes the *deployment* CPU's VNNI
  flag and sets `reduce_range` when it is absent.
- Plans separate the training machine from the deployment machine.

## 5. BFSI and compliance

Not legal advice. Regulatory statements below came from search summaries and
secondary write-ups; have counsel confirm each against the regulation text.

**Regulatory context [snippet]**

- IRDAI: Protection of Policyholders' Interests Regulations 2024 and a master
  circular (September 2024), 30-day free-look. Unfair-business-practice
  grievances were 22% of life complaints in FY25. Information regulations of
  2025 require policy and claim records to be held in data centres in India.
  An AI working group was formed in June 2026; no draft was found.
- RBI: the FREE-AI report (August 2025) recommends a board-approved AI policy,
  an AI inventory, risk-based audit and explainability, and encourages
  indigenous small models. It is advisory until directions issue.
- DPDP rules were notified in November 2025; main obligations apply from May
  2027.
- EU AI Act: life and health insurance risk assessment and pricing are
  high-risk. US SR 11-7 was replaced by SR 26-2 in April 2026.
- Nothing found mandates on-premises or small models. Data-location rules,
  inventory, explainability and audit requirements make self-hostable models
  with loggable, span-level outputs the lower-friction route.

**Which system for which task**

| Use case | Recommended | Why |
|---|---|---|
| Sales-call compliance, mis-selling | rules, then compact classifier, then larger model on flagged spans, then a person | LLM judges alone were modest; a fine-tuned compact judge matched them; audit needs spans **[snippet]** |
| Policy and regulation Q&A | retrieval with a reasoning model and quoted sources | retrieval beats fine-tuning for knowledge **[snippet]** |
| Document classification | compact fine-tuned classifier | cheap, plentiful labels; 95%+ on public document benchmarks **[snippet]** |
| Risk categorisation, underwriting triage | classical ML (GLM, GBDT, EBM) | not beaten in published insurance tests **[snippet]**; E4 **[measured]** |
| Marketing-material checks | rules, retrieval, larger model, human sign-off | no benchmark exists; unproven |
| Complaint classification | compact classifier with abstention | E1, E5 **[measured]** |
| Regulatory change monitoring | obligation-sentence classifier, retrieval, larger model, a person | sentence detection is solved; end-to-end mapping is not **[snippet]** |

**Gaps in the evidence**

- No public dataset of Indian-language insurance sales calls or mis-selling.
- No head-to-head of multilingual encoders on Hinglish call transcripts, and no
  study of how ASR errors propagate into compliance classification.
- Most encoder-versus-LLM comparisons use GPT-4-era models.

**What the platform does**

- `forgelm.packs.bfsi` holds the use-case map, a sales-call taxonomy, rule
  screens that return the triggering span, and `review_call`, which routes
  flagged and uncertain utterances to a reviewer rather than deciding.
- The pack's checklist and seed utterances are illustrative and unvalidated on
  real calls. Your compliance function must set them.

## 6. Customization platforms

**Findings [read unless noted]**

- Platform mortality is high: AutoTrain unmaintained, NVIDIA's data-flywheel
  blueprint deprecated, Argilla and distilabel without their original authors,
  OpenAI self-serve fine-tuning winding down **[snippet]**, Lamini and
  Predibase absorbed by acquirers **[snippet]**.
- Open-source trainers (LLaMA-Factory, H2O LLM Studio, Transformer Lab) target
  generative models. No live open-source tool was found that distils an LLM
  teacher into an encoder classifier with calibration as a first-class flow.
- Prompt optimisation (GEPA, MIPROv2) is a cheaper first step than training,
  and alternating prompt and weight optimisation reportedly beats either alone
  **[snippet]**.
- Routing is not a free win: one 2026 benchmark found many routers fail to beat
  the best single model **[snippet]**.
- AI-BOM tooling assumes models live on a public hub. Model signing works
  without public infrastructure.
- Recurring user complaints: dependency conflicts, chat-template mismatch on
  export, GPU lock-in, and no evaluation harness **[snippet]**.

**What the platform does**

- Classifier students are first-class, not an afterthought to LLM tuning.
- `EvalContract` freezes the acceptance test before training; every later
  stage is checked against it.
- `RunRecord` writes a run record, a model card and a CycloneDX ML-BOM from the
  local run, with no hub or tracking server.
- The generation plan puts a prompt-optimisation baseline, and retrieval for
  changing knowledge, ahead of any weight update.
- The core has one hard dependency (numpy). Each backend is an optional extra,
  so a broken GPU stack cannot break planning, classifiers or tabular work.

---

## Not done, and why

| Item | Status |
|---|---|
| GPU training paths (Unsloth, QLoRA, MLX) | Not run. This environment has 4 CPU cores and no accelerator. The backend rules are tested against reference hardware profiles, not real devices. |
| TabPFN | Not run. v2 weights need a licence acceptance; later weights are non-commercial. |
| SetFit, GLiClass, NLI zero-shot | Not run. Listed in the plan with [snippet] evidence. |
| Hosted LLM teacher | Not run. `ChatCompletionsTeacher` is tested against a local stub endpoint only. |
| Prompt optimisation, retrieval | Recommended by the planner; not implemented here. Integrate DSPy / an existing RAG stack. |
| Real insurance call data | None available publicly. The BFSI pack is unvalidated. |
