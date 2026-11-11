# ForgeLM

Turn existing intelligence (a large model, a few labels, documents, a table of
outcomes) into a small, task-specific model that runs on the hardware you
have, and know before you start which kind of model the task needs.

ForgeLM began as an LLM fine-tuning platform: FastAPI microservices and a
React dashboard around Unsloth on NVIDIA GPUs. That platform is still here.
Next to it is `forgelm`, a Python package that plans the work, trains models
that need no LLM at inference, distils from LLM teachers, calibrates and
abstains, fine-tunes small LLMs on any PyTorch device, exports and checks the
result, and records what was done. Every recommendation it makes cites the
evidence behind it, most of it measured in this repository.

- [Quick start](#quick-start)
- [What the measurements say](#what-the-measurements-say)
- [Choosing an approach](#choosing-an-approach)
- [Using it](#using-it)
- [Hardware](#hardware)
- [The platform services](#the-platform-services)
- [Repository layout](#repository-layout)
- [Development](#development)
- [Status and limits](#status-and-limits)

## Quick start

Python 3.10 or later. Nothing here needs a GPU.

```bash
pip install -e ".[classify,tabular,train,generative,retrieval,api,dev]"
```

See what this machine can do:

```bash
forgelm doctor
```

Ask what to build. A plan lists candidates cheapest first, what each needs,
how to deploy the result, the guardrails that are not optional, and how strong
the evidence behind each step is:

```bash
forgelm plan --task text_classification --classes 77 --labels-per-class 8
```

Train a classifier from a CSV of `text` and `label` columns, then use it:

```bash
forgelm train --data complaints.csv --text-col text --label-col category --student encoder --out models/complaints
```

```bash
forgelm predict --model models/complaints --threshold 0.8 "my premium was debited twice"
```

`train` holds out data it never trains on, calibrates the model's confidence,
and writes the model with an evaluation contract, a run record, a model card
and a CycloneDX ML-BOM. `--threshold` makes the model answer "ABSTAIN" when it
is unsure.

## What the measurements say

Nine experiments on public data, all on a 4-core CPU. Full tables are in
[experiments/RESULTS.md](experiments/RESULTS.md).

| Question | Finding |
|---|---|
| Which small classifier? (E1, Banking77, 77 intents) | A frozen sentence encoder with a linear head was best at every data size: 85.8% with 8 labels per class, 93.1% with all. Fine-tuning ModernBERT for 50 minutes only matched it. |
| Prompt a local LLM, or train? (E8) | With 8 labels per class the trained small model scored 87.5% at 3 ms per text. A local 1.5B LLM scored 66.9% with retrieved examples, at 770 ms. |
| Use the LLM as a teacher? (E2) | Students trained on its labels were worse than students trained on 8 human labels per class. A student cannot beat its teacher. |
| Escalate uncertain cases to the LLM? (E5) | A confidence threshold hit its accuracy target to within about 1.5 points. Escalating to the 1.5B LLM never helped. |
| Screen contracts for unfair terms (E6) | To catch 90% of them, TF-IDF or a small fine-tuned model sends 18-19% of sentences for review; the zero-shot LLM sends 43%. |
| Tabular risk models (E4) | An in-context foundation model (TabICL) led by at most 0.033 AUC and was slow to score. The explainable EBM was first or second from 500 rows up. |
| Compress for CPU (E3) | ONNX fp32 is lossless and up to 3x faster. Dynamic int8 silently cost ModernBERT 7.9 points; the parity gate caught it. |
| Fine-tune an LLM on a CPU (E7) | A 360M model went from 1.3% to 89.0% in 19 minutes. GGUF f16 was lossless; Q4_K_M lost 3.3 points. |
| Retrieval for financial QA (E9) | A dense encoder beat BM25, and also beat an equal-weight hybrid of the two. |

The local LLM in these experiments is small (1.5B parameters). A larger
teacher may change the teacher and escalation results; measure it on your own
anchor set with `forgelm.distill` and `forgelm.gate` before relying on it.

## Choosing an approach

| You have | Start with | Then try |
|---|---|---|
| Text classes, no labels | zero-shot by label description, as a first look | label about 8 examples per class |
| Text classes, a few labels per class | frozen encoder + linear head (`--student encoder`) | nearest-neighbour vote; static embeddings if latency matters |
| Text classes, hundreds of labels per class | TF-IDF and the frozen encoder | a fine-tuned encoder; keep whichever wins on the hold-out |
| Text that needs a policy or regulation answer | dense retrieval with quoted, verified citations, and a reasoning model | BM25 as a model-free fallback |
| A table of outcomes (risk, lapse, fraud) | logistic regression, LightGBM, EBM | TabICL on small data, if its licence and scoring cost fit |
| A generation task | a frozen evaluation set and a prompt baseline | LoRA fine-tune of a small model, exported to GGUF |

`forgelm plan` makes this choice for your numbers and hardware, and
`forgelm.gate.decide` settles it on your own test set: it recommends the
cheapest option that no costlier option beats by a margin worth paying for,
with a bootstrap interval and an exact McNemar test.

## Using it

### Classifiers that need no LLM at inference

| Student | What it is | Needs at inference |
|---|---|---|
| `zero_shot` | similarity to a one-line description of each label | an embedding model |
| `tfidf` | word and character n-grams, logistic regression | scikit-learn |
| `static` | static token embeddings (Model2Vec), logistic regression | numpy, model2vec |
| `encoder` | frozen sentence encoder, logistic regression | sentence-transformers |
| `finetuned` | encoder fine-tuned end to end | torch, or ONNX Runtime after export |

```python
from forgelm.students import make_student

# y holds class indices into label_names; predict returns label names
student = make_student("encoder").fit(texts, y, label_names)
student.calibrate(holdout_texts, holdout_y)
student.predict(["my card was declined"])
```

`forgelm.compress.export_and_gate` exports a fine-tuned student to ONNX and
refuses an int8 version that changes its decisions.

### Distilling from an LLM teacher

```python
from forgelm.distill import distill
from forgelm.students import make_student
from forgelm.teachers import LocalLLMTeacher

teacher = LocalLLMTeacher("Qwen/Qwen2.5-1.5B-Instruct", label_names, descriptions)
report = distill(teacher, make_student("static"), unlabelled_texts, anchor_texts, anchor_labels)
print(report.teacher_on_anchor["accuracy"], report.student_on_anchor["accuracy"])
```

The anchor is a small human-labelled set the teacher never labels for
training; both teacher and student are scored on it. `ChatCompletionsTeacher`
uses any OpenAI-compatible endpoint, including an on-premises llama.cpp, vLLM
or Ollama server.

### Tabular models

```bash
forgelm tabular --data risk.csv --target default
```

Fits the baselines and the candidates on the same split and reports ROC AUC,
Brier score and calibration. The in-context models with non-commercial
weights (TabPFN 2.5 and later, among others) are never recommended for
commercial use.

### Fine-tuning a small LLM and exporting it

```bash
forgelm finetune --data train.jsonl --base-model HuggingFaceTB/SmolLM2-360M-Instruct --out runs/intent --merge
```

```bash
pip install -e ".[export]"
```

```bash
forgelm export-gguf --model runs/intent/merged --quant Q4_K_M --fetch-converter --out runs/intent/gguf
```

`train.jsonl` holds one `{"messages": [...]}` conversation per line. Training
runs on CUDA, ROCm, Intel XPU, Apple MPS or CPU without bitsandbytes. The
export checks that the chat template inside the GGUF file renders the same
prompt as the original tokenizer, the usual silent failure after export. On a
Mac, `--backend mlx` prepares an `mlx_lm` run instead.

### Retrieval

`forgelm.retrieval` has BM25, dense and fused search over chunked documents
with character offsets, recall / MRR / nDCG, numbered citations, and
`verify_quotes`, which flags any quotation in a generated answer that does not
appear word for word in the cited passages.

### BFSI and compliance

`forgelm.packs.bfsi` maps common tasks (sales-call compliance, policy
questions, document classification, risk scoring, marketing checks, complaint
classification, regulatory change monitoring) to the kind of system the
evidence supports, and screens insurance sales calls in three tiers: rules,
then a small classifier, then routing to a reviewer, each flag carrying the
text that triggered it. Its checklist, patterns and example utterances are
illustrative, not a statement of regulatory requirements, and have not been
validated on real calls.

## Hardware

| Machine | LLM fine-tuning | Classifiers and tabular |
|---|---|---|
| CPU only | models up to about 1.5B parameters; deploy larger ones as GGUF | everything trains and serves here |
| NVIDIA | Unsloth or PEFT + TRL, 4-bit QLoRA | fp16 on the GPU |
| AMD (ROCm) | PEFT + TRL, LoRA on 16-bit weights by default | ROCm or CPU |
| Intel Arc (XPU) | PEFT + TRL on PyTorch's XPU backend | OpenVINO or CPU |
| Apple Silicon | MLX-LM | MPS or CPU |

Only the CPU row has been run in this repository. The others follow each
project's documentation and are tested against simulated devices.

## The platform services

Build the base images first, then start everything:

```bash
docker compose build base ml-base
```

```bash
docker compose up -d
```

| Service | Port | Role |
|---|---|---|
| api-gateway | 8000 | routing; API docs at `/docs` |
| training-service | 8001 | LoRA / QLoRA fine-tuning of LLMs |
| chat-comparison-service | 8002 | base versus fine-tuned chat |
| data-generation-service | 8003 | instruction-data generation and preparation |
| visualization-service | 8004 | training analytics |
| model-management-service | 8005 | model download and configuration |
| factory-service | 8006 | `forgelm` over HTTP; CPU only |
| orchestrator | 8080 | service discovery and the GPU lock |

The GPU services need an NVIDIA GPU and the NVIDIA Container Toolkit; the
factory service does not. Copy `.env.example` to `.env` and set `HF_TOKEN`
for gated models; every setting is described in `.env.example`. Running
without a GPU, the dev and prod overlays, and Kubernetes are covered in
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Each service serves its OpenAPI
schema at `/openapi.json` on its own port (for example
`http://localhost:8001/openapi.json` for training).

The dashboard is in `frontend/`:

```bash
cd frontend && npm install --legacy-peer-deps && npm run dev
```

Its Factory page covers hardware, planning, classifier training and call
review, through these gateway routes under `/api/v1/factory`:

| Route | Purpose |
|---|---|
| `GET /hardware` | this machine, or a reference profile |
| `POST /plan` | a plan for a task and its constraints |
| `POST /classifiers` | train a classifier (synchronous) |
| `POST /classifiers/{name}/predict` | classify, with an abstention threshold |
| `GET /classifiers/{name}` | run record and model card |
| `POST /generative/advice` | backend, memory estimate and LoRA settings |
| `POST /review-call` | BFSI sales-call screen |
| `GET /usecases` | BFSI use-case map |

## Repository layout

| Path | What is there |
|---|---|
| `forgelm/` | the core package (planner, students, teachers, distillation, metrics, compression, tabular, fine-tuning, GGUF, retrieval, gate, provenance, BFSI pack, CLI) |
| `src/microservices/` | the FastAPI services |
| `src/shared/llmforge_common/` | code the services share |
| `frontend/` | the React dashboard |
| `experiments/` | E1-E9 scripts, result files and `RESULTS.md` |
| `tests/factory/` | the test suite |
| `deployments/kubernetes/` | Kubernetes manifests |
| `scripts/` | a load test for a running stack |
| `docs/` | [ARCHITECTURE.md](docs/ARCHITECTURE.md) (design, what was changed, what is owed), [RESEARCH.md](docs/RESEARCH.md) (evidence behind each decision), [DEPLOYMENT.md](docs/DEPLOYMENT.md) |

## Development

```bash
python -m pytest tests/factory -q
```

The tests need no GPU, network or model downloads. Two tests that download
small models are marked `slow` and run with `-m slow`.

```bash
bash experiments/run_all.sh
```

```bash
python experiments/report.py
```

To load-test a running stack: `python scripts/load_test_gpu.py basic` (needs
`aiohttp`).

`run_all.sh` runs every experiment whose result file is missing, so an
interrupted run resumes where it stopped; `FORCE=1` re-runs everything. A full
run takes about five hours on four CPU cores and downloads models and
datasets from the Hugging Face Hub. `report.py` regenerates
`experiments/RESULTS.md` from the result files.

Conventions for contributors, including the rule that `forgelm` imports with
only numpy installed, are in [CLAUDE.md](CLAUDE.md).

## Status and limits

- Everything in `forgelm`, the factory service and the experiments was built
  and run on a CPU-only machine.
- The GPU training service has unit tests but has not been run on a GPU,
  and nothing has run on AMD, Intel or Apple hardware.
- The CUDA images have not been built. The factory, base and gateway images
  were built without their Debian `apt` step, because the build sandbox
  blocked the package mirror.
- The front end's `npm run build` fails on 75 TypeScript errors in older
  files, none in the Factory page.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) lists what is still owed, in
order.

## Licence

MIT.
