# Architecture: from a fine-tuning UI to a model factory

This document records what the platform was, what was found wrong with it,
what was changed, and what is still owed. Research behind the decisions is in
[RESEARCH.md](RESEARCH.md); measurements are in
[../experiments/RESULTS.md](../experiments/RESULTS.md).

## 1. What the platform was

Seven FastAPI microservices behind a gateway, plus a React front end:

| Service | Port | Role |
|---|---|---|
| api-gateway | 8000 | routing, auth, aggregation |
| training-service | 8001 | LoRA / QLoRA fine-tuning of causal LMs with Unsloth |
| chat-comparison-service | 8002 | base versus fine-tuned chat |
| data-generation-service | 8003 | instruction-data generation and preparation |
| visualization-service | 8004 | training analytics |
| model-management-service | 8005 | model download and config |
| orchestrator | 8080 | service discovery, Redis GPU lock |

One workflow: pick a base LLM, prepare instruction data, fine-tune, chat with
the result.

## 2. Audit findings

Found by reading the code. Items marked **fixed** were changed in this pass.

### Scope

- **One model class.** Only generative causal LMs. No classifiers, no tabular
  models, no distillation, no calibration or abstention, no export step.
- **One vendor.** `hardware_detection.py` imports torch at module load, asks
  only `torch.cuda.is_available()`, and chooses settings from four hard-coded
  VRAM tiers named after NVIDIA cards. Apple, AMD and Intel accelerators fall
  through to "CPU, float32". The README requires a 16 GB NVIDIA GPU.
- **No evaluation.** Training loss is the only measured quantity. Nothing
  decides whether a tuned model is better than the base, or good enough.

### Correctness

- **`start_fine_tuning` blocked until training finished.** (fixed) The method
  was defined twice in `training_service.py`. The later definition won. It
  submitted the job inside `with ThreadPoolExecutor()`, and leaving that block
  waits for the job. The `POST /fine-tune` request, and the event loop serving
  health, status and log streaming, hung for the whole run. The earlier
  definition, which took the cross-service GPU lock, was unreachable.
- **PEFT method silently ignored.** (fixed) The Transformers path applied a
  fixed r=16 LoRA for `lora` and `qlora`. For `dora`, `pissa` or `oft`, which
  the API schema advertises, it applied no adapter and ran a full fine-tune.
  `peft_rank`, `peft_alpha`, `peft_dropout` and `target_modules` were ignored.
- **Dead re-implementations.** (fixed) `peft_methods.py` hand-rolled DoRA, OFT
  and PiSSA layers (about 200 lines) that no code path called.
- **Shadowed methods.** (fixed) `TrainingCore` defined
  `_detect_model_precision`, `_get_optimal_precision_settings` and
  `_validate_precision_settings` twice each; 150 unreachable lines.
- **`exec` is monkey-patched at import** to work around one Unsloth release
  (`training_service.py`, top of file). Left in place: it cannot be tested
  without that Unsloth version. Remove it once the pinned version is upgraded.

### Maintainability

- **Copy-paste instead of a shared package.** Byte-identical copies:

  | File | Copies | Lines each |
  |---|---|---|
  | `service_orchestrator.py` | 5 (a 6th differs) | 582 |
  | `gpu_orchestrator.py`, `gpu_client.py`, `gpu_monitoring.py`, `gpu_inference_wrapper.py` | 4 each | 303 to 489 |
  | `research_logger.py` | 3 (a 4th differs) | 485 |
  | `memory.py` | 5 | 137 |
  | `model_pairs.yaml` | 5 | 125 |
  | `schemas.py` | 3 | 247 |

  Counting every byte-identical Python file of 20 lines or more, 13 files are
  duplicated and about 11,000 of the 42,000 Python lines are redundant copies.
  `src/shared/llmforge_contracts`
  is an unfinished start on fixing this. Not changed here: each service's
  Dockerfile copies only its own directory, so de-duplication means changing
  every image, which could not be built or tested in this environment.
- `training_service.py` (2,270 lines) and `training_service_core.py` (1,576)
  implement the same training flow twice, in-process and in a worker.
- No `pyproject.toml`, no root requirements file. (fixed: `pyproject.toml`)
- `.gitignore` ignored every `test_*.py`, so no unit test could be committed.
  The existing tests are integration scripts that need a running GPU stack.
  (fixed for `tests/factory/`)
- `README.md` was UTF-16 and described a monolith (`app.py`,
  `fine_tuning.py`) that no longer exists. (fixed)
- Seven fix-log documents at the repository root (`EXIT_CODE_139_FIX.md`,
  `RTX_3050_4GB_SETUP.md` ...) and a 44 KB `issue.txt` log dump. Left in place.

## 3. What it is now

The existing services keep running. A new core package, `forgelm/`, holds the
decisions and the model classes the services lacked, and is importable
without any of them.

```
            existing intelligence                         what ships
  ┌───────────────────────────────────┐        ┌──────────────────────────────┐
  │ large model · documents · labels  │        │ small model + calibration    │
  │ unlabelled text · tables          │        │ + abstention threshold       │
  └────────────────┬──────────────────┘        │ + run record, card, ML-BOM   │
                   │                           └──────────────▲───────────────┘
                   ▼                                          │
   hardware ─► planner ─► teachers ─► distill ─► students ─► metrics ─► compress
   (probe)     (what to   (label or   (anchor-   (ladder)    (quality,  (ONNX,
               build,      generate)   checked)               calibra-   int8 behind
               and why)                                       tion)      a parity gate)
                   │
                   ├─► tabular      (baselines + in-context model, licence-gated)
                   ├─► generative   (backend, memory, PEFT for LLM fine-tuning)
                   └─► packs.bfsi   (use-case map, call screening)

   provenance: EvalContract · RunRecord · model card · CycloneDX ML-BOM
```

### Package map

| Module | Responsibility | Heavy imports |
|---|---|---|
| `forgelm.hardware` | CPU features, RAM with container limits, NVIDIA / AMD / Apple / Intel accelerators, installed backends | none (torch optional) |
| `forgelm.planner` | task + data + constraints + hardware to an ordered, evidence-cited plan | none |
| `forgelm.evidence` | the claim, source and strength behind each planner rule | none |
| `forgelm.students` | zero-shot, TF-IDF, static-embedding, frozen-encoder and fine-tuned classifiers behind one interface | per student |
| `forgelm.teachers` | local LLM (answer-letter logits), OpenAI-compatible endpoint, any callable | per teacher |
| `forgelm.distill` | teacher to student with a human-labelled anchor; synthetic-data prompts and collapse checks | none |
| `forgelm.metrics` | accuracy, F1, ECE, Brier, temperature scaling, abstention, cascade simulation | numpy |
| `forgelm.compress` | ONNX export, int8 quantisation, parity gate, torch-free ONNX serving | torch for export only |
| `forgelm.tabular` | logistic regression, GBDT, EBM, TabICL; licence registry | per model |
| `forgelm.generative` | training backend, memory estimate, PEFT config for causal LMs | none |
| `forgelm.provenance` | evaluation contract, run record, model card, ML-BOM | none |
| `forgelm.packs.bfsi` | use-case map, sales-call taxonomy, rule screens, call review | none |
| `forgelm.cli` | `forgelm doctor / plan / train / predict / tabular` | none |

### Design rules

1. **The core imports nothing heavy.** Its only hard dependency is numpy.
   Every backend is an optional extra. A broken GPU stack cannot break
   planning, classifiers or tabular work.
2. **Train hardware and deploy hardware are separate inputs.** A plan can
   train on a GPU and target a CPU without VNNI.
3. **Every recommendation cites its evidence and that evidence's strength.**
   `measured` entries must point at a result file that exists; a test enforces
   it.
4. **Evaluation is part of each stage.** Distillation needs an anchor.
   Export runs a parity gate. Training writes an evaluation contract.
5. **Integrate, do not rebuild.** PEFT, TRL, Unsloth, MLX-LM, sentence-
   transformers, Model2Vec, scikit-learn, LightGBM, InterpretML, TabICL and
   ONNX Runtime do the work. The new code is the glue, the decision logic and
   the checks between stages.

### How the existing services use it

- **training-service** imports one module, `utils/factory_bridge.py`. It
  builds the `peft` config from the request, picks device and precision for
  any accelerator vendor, and adds a vendor-neutral `factory` block to
  `GET /hardware-profile`. If `forgelm` is not installed the bridge returns
  the previous hard-coded behaviour.
- **factory-service** (new, port 8006, CPU only, no ML base image) exposes the
  core over HTTP under `/api/v1/factory`: `hardware`, `plan`, `usecases`,
  `generative/advice`, `classifiers` (train, list, card, predict) and
  `review-call`.
- **api-gateway** routes `/api/v1/factory` to it.

## 4. Verification status

| Area | Status |
|---|---|
| `forgelm` core | unit tests in `tests/factory/`, no GPU or network needed |
| Student ladder, distillation, compression, tabular, cascades | measured on CPU; see `experiments/RESULTS.md` |
| factory-service | exercised through FastAPI's test client |
| training-service changes | compile; bridge and `start_fine_tuning` have unit tests. **Not run end to end**: no GPU, Unsloth or Redis here |
| Docker images, compose, Kubernetes manifests | edited, **not built** |
| Backend selection for NVIDIA / AMD / Apple / Intel | tested against reference profiles, **not on real devices** |
| Front end | unchanged; no UI for the factory yet |

## 5. What is still owed, in order

1. **Run the changed training path on a real GPU** before merging to a
   release branch. Start a job, confirm the request returns at once, and
   confirm a `dora` request trains an adapter.
2. **De-duplicate the services.** Move the copied modules into one installed
   package (extend `forgelm` or finish `llmforge_contracts`) and delete the
   copies. Needs every image rebuilt.
3. **Collapse the two training implementations** into `TrainingCore`, with the
   in-process path calling it.
4. **Replace `AdaptiveTrainingConfig`'s VRAM tiers** with
   `forgelm.generative.select_backend`, and add a measured dry run (a few real
   steps) behind the memory estimate.
5. **Add an MLX-LM training path** for Apple Silicon and a GGUF export step
   with a chat-template round-trip test.
6. **Front end**: a planning page and a classifier page over the factory API.
7. **Prompt-optimisation and retrieval baselines** ahead of fine-tuning
   (integrate DSPy / an existing RAG stack; do not write new ones).
8. **Real BFSI evaluation data.** Label a few hundred reviewed call utterances
   and replace the illustrative seed set.
