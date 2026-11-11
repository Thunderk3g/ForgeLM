# ForgeLM

A model intelligence factory. Take intelligence that already exists (a large
model, documents, a handful of labels, a table of outcomes), then specialise
it, distil it, compress it, and ship a small domain model that runs on the
hardware you have.

ForgeLM began as an LLM fine-tuning platform (seven FastAPI microservices and a
React dashboard, built around Unsloth on NVIDIA GPUs). That platform is still
here. Around it there is now a hardware-agnostic core, `forgelm`, that covers
what a fine-tuning UI cannot: small classifiers that need no LLM at inference,
tabular models, teacher-student distillation, calibration and abstention,
export with a parity check, and a planner that says which of these a task
needs before any compute is spent.

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): what the platform was, what was changed, what is still owed
- [docs/RESEARCH.md](docs/RESEARCH.md): the research behind each decision, with evidence strength
- [experiments/RESULTS.md](experiments/RESULTS.md): what was measured in this repository

## Quick start: the core, on any machine

```bash
pip install -e ".[classify,tabular,train,generative,retrieval,api,dev]"
```

```bash
forgelm doctor
```

`doctor` reports the CPU's instruction sets, memory (honouring container
limits), any NVIDIA, AMD, Apple or Intel accelerator, and which backends are
installed. Nothing in the core requires a GPU.

Ask what to build:

```bash
forgelm plan --task text_classification --classes 77 --labels-per-class 8
```

```bash
forgelm plan --task generation --model-size-b 8 --hardware amd-16gb --deploy cpu-server
```

```bash
forgelm plan --task tabular_classification --rows 3000 --features 30 --explainable
```

A plan lists candidates cheapest first, what each needs, how to deploy the
result on the target machine, the guardrails that are not optional, and how
strong the evidence behind each recommendation is. `--hardware` and `--deploy`
plan for machines other than the one you are on.

Train a classifier that needs no LLM at inference:

```bash
forgelm train --data complaints.csv --text-col text --label-col category --student static --out models/complaints
```

```bash
forgelm predict --model models/complaints --threshold 0.8 "my premium was debited twice"
```

`train` holds out data it never trains on, calibrates confidence, and writes
the model together with an evaluation contract, a run record, a model card and
a CycloneDX ML-BOM. `--threshold` makes the model abstain when unsure.

Benchmark tabular models on a CSV:

```bash
forgelm tabular --data risk.csv --target default
```

Specialise a small instruct model on any PyTorch device (CUDA, ROCm, XPU,
MPS or CPU; no bitsandbytes needed) and export it for llama.cpp or Ollama:

```bash
forgelm finetune --data train.jsonl --base-model HuggingFaceTB/SmolLM2-360M-Instruct --out runs/intent --merge
```

```bash
forgelm export-gguf --model runs/intent/merged --quant Q4_K_M --fetch-converter --out runs/intent/gguf
```

The export step needs the `export` extra (`pip install -e ".[export]"`), which
pins the `gguf` package to the same llama.cpp release as the converter.

`export-gguf` checks that the chat template stored in the GGUF file renders
the same prompt as the Hugging Face tokenizer, the usual silent failure after
export. On a Mac, `forgelm finetune --backend mlx` writes the data and prints
the `mlx_lm lora` command instead of training with PyTorch.

### In Python

```python
from forgelm.students import make_student
from forgelm.teachers import LocalLLMTeacher
from forgelm.distill import distill

teacher = LocalLLMTeacher("Qwen/Qwen2.5-1.5B-Instruct", labels, descriptions)
student = make_student("static")            # lookup table + linear head, no torch at inference
report = distill(teacher, student, unlabelled_texts, anchor_texts, anchor_labels)
print(report.teacher_on_anchor["accuracy"], report.student_on_anchor["accuracy"])
```

The anchor is a small human-labelled set that the teacher never labels for
training. Teacher and student are both scored on it, so the report shows how
good the teacher is and how much the student lost.

## The student ladder

| Student | What it is | Needs at inference |
|---|---|---|
| `zero_shot` | similarity to a one-line description of each label; no training data | embedding model |
| `tfidf` | word and character n-grams, logistic regression | scikit-learn |
| `static` | static token embeddings (Model2Vec), logistic regression | numpy, model2vec |
| `encoder` | frozen sentence encoder, logistic regression | sentence-transformers |
| `finetuned` | encoder fine-tuned end to end | torch, or ONNX Runtime after export |

All take hard or soft labels, calibrate with `student.calibrate(...)`, and
save to a directory. Which one wins depends on how many labels you have;
[experiments/RESULTS.md](experiments/RESULTS.md) has the measurements.

## Before training: is it worth it?

- `forgelm.gate.decide` compares options (for example zero-shot embeddings, a
  linear head on a few labels, a prompted local LLM) on the same test set and
  recommends the cheapest one that no costlier option beats by a margin you
  set, using a paired bootstrap interval and an exact McNemar test. When the
  test set is too small to tell, it says so.
- `LocalLLMTeacher(..., demonstrations=..., k_demos=5, demo_embedder=...)`
  prompts a local LLM with the labelled examples most similar to each query.
- `forgelm.retrieval` gives BM25, dense and hybrid search with passage
  offsets, retrieval metrics, and `verify_quotes`, which flags any quotation in
  a generated answer that does not occur verbatim in the cited passages.

## Hardware

| Machine | Generative fine-tuning | Classifiers and tabular |
|---|---|---|
| CPU only | models up to about 1.5B parameters; larger ones elsewhere, deploy as GGUF | everything trains and serves here |
| NVIDIA | Unsloth or PEFT + TRL, QLoRA | fp16 on GPU |
| AMD (ROCm) | PEFT + TRL, LoRA on 16-bit weights by default | ROCm or CPU |
| Intel Arc (XPU) | PEFT + TRL on PyTorch's native XPU backend | OpenVINO or CPU |
| Apple Silicon | MLX-LM | MPS or CPU |

Only the CPU row was exercised in this repository. The others are rules drawn
from each project's documentation and tested against simulated devices. The
training service follows the same rules when it loads a model: 4-bit
bitsandbytes on NVIDIA as before, 16-bit on AMD unless quantising is asked for,
and the device's own dtype on Intel, Apple and CPU.

## BFSI and compliance

`forgelm.packs.bfsi` maps common tasks (sales-call compliance, policy Q&A,
document classification, risk categorisation, marketing checks, complaint
classification, regulatory change monitoring) to the kind of system the
evidence supports, and includes a three-tier screen for insurance sales calls:
rules, then a compact classifier, then routing to a reviewer, each flag
carrying the text span that triggered it.

The checklist, patterns and seed utterances in the pack are illustrative. They
are not a statement of regulatory requirements and have not been validated on
real calls.

## The platform services

```bash
docker-compose up -d
```

| Service | Port | Role |
|---|---|---|
| api-gateway | 8000 | routing; API docs at `/docs` |
| training-service | 8001 | LoRA / QLoRA fine-tuning of causal LMs |
| chat-comparison-service | 8002 | base versus fine-tuned chat |
| data-generation-service | 8003 | instruction-data generation and preparation |
| visualization-service | 8004 | training analytics |
| model-management-service | 8005 | model download and configuration |
| factory-service | 8006 | the `forgelm` core over HTTP (CPU only) |
| orchestrator | 8080 | service discovery, GPU lock |

The React dashboard is in `frontend/` (`npm install --legacy-peer-deps` because of an
existing peer-dependency conflict, then `npm run dev`); its
Factory page covers hardware, planning, classifier training and call review. The
GPU-backed services expect an NVIDIA GPU and the NVIDIA Container Toolkit; the
factory service does not. Set `HF_TOKEN` in `.env` for gated models (the older
`HUGGINGFACE_TOKEN` name is still accepted).

Factory endpoints, through the gateway at `/api/v1/factory`:

| Endpoint | Purpose |
|---|---|
| `GET /hardware` | probe this machine, or describe a reference profile |
| `POST /plan` | plan for a task, data regime and constraints |
| `POST /classifiers` | train an LLM-free classifier (synchronous; the gateway allows `GATEWAY_LONG_REQUEST_TIMEOUT_S`, default 30 minutes, for this route only) |
| `POST /classifiers/{name}/predict` | classify, with an abstention threshold |
| `GET /classifiers/{name}` | run record and model card |
| `POST /generative/advice` | backend, memory estimate and PEFT config for an LLM fine-tune |
| `POST /review-call` | BFSI sales-call screen |
| `GET /usecases` | BFSI use-case map |

Factory service settings (environment variables):

| Variable | Default | Purpose |
|---|---|---|
| `FACTORY_MODELS_DIR` | `/app/models/factory` | where trained classifiers are stored |
| `FACTORY_ALLOWED_MODELS` | the ladder's six default models | comma-separated Hugging Face repo ids a request may name as `model`; anything else is refused |
| `FACTORY_MAX_TRAIN_ROWS` | 50000 | rows per training request |
| `FACTORY_MAX_PREDICT_ROWS` | 10000 | texts per predict or review-call request |
| `FACTORY_MAX_TEXT_CHARS` | 20000 | characters per text on predict and review-call |

## Tests and experiments

```bash
python -m pytest tests/factory -q
```

The factory tests need no GPU, network or model downloads. Two tests marked
`slow` download small models; they are skipped by default and run with
`-m slow`.

```bash
bash experiments/run_all.sh
```

```bash
python experiments/report.py
```

`run_all.sh` runs every experiment whose result file is missing, so an
interrupted run resumes where it stopped (`FORCE=1` re-runs everything; the
full set takes about five hours on four CPU cores, and models and datasets
download from the Hugging Face Hub). `report.py` regenerates
`experiments/RESULTS.md` from the result files.

## Status

The `forgelm` core, the factory service and the experiments were built and
run on a CPU-only machine. The changes to the GPU training service compile and
have unit tests but have not been run on a GPU. The factory-service, base and
gateway images were built and run; images that need the Debian `apt` layer
were built without it, because this sandbox blocks the package mirror, and the
CUDA images were not built. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full list.

## Licence

MIT.
