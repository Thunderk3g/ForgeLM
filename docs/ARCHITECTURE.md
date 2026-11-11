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

  Counting every byte-identical Python file of 20 lines or more, 13 groups of
  files were duplicated, about 11,000 of the 42,000 Python lines. (fixed: one
  installable package, `src/shared/llmforge_common`, holds each module once;
  the 60 former copies are 13-line shims that keep every import path working,
  and every service image installs the package. About 14,500 lines removed. A
  test fails if two byte-identical Python files of 20+ lines reappear.)
  Near-duplicates that have genuinely diverged were left alone: the three
  `model_service.py` files (80-90% similar), the training service's and the
  orchestrator's `service_orchestrator.py`, the visualization service's
  `research_logger.py`, and the data-generation service's `schemas.py`.
  `model_pairs.yaml` still exists in five identical copies.
- `training_service.py` and `training_service_core.py` implement the same
  training flow twice, in-process and in a worker. (partly fixed: the five
  helper methods that were word-for-word identical now live once in
  `services/training_common.py`. Five same-named methods differ in behaviour
  and stay separate; merging them needs a GPU to verify.)
- The model loader in both training paths always used 4-bit bitsandbytes with
  `device_map="auto"`, whatever the hardware. (fixed: `causal_lm_load_plan` in
  the bridge keeps that for NVIDIA, and loads 16-bit on AMD unless quantising
  is asked for, and plain device dtype on Intel, Apple and CPU.)
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
| `forgelm.teachers` | local LLM (answer-letter logits, optional few-shot with retrieved demonstrations), OpenAI-compatible endpoint, any callable | per teacher |
| `forgelm.distill` | teacher to student with a human-labelled anchor; synthetic-data prompts and collapse checks | none |
| `forgelm.metrics` | accuracy, F1, ECE, Brier, temperature scaling, abstention, cascade simulation | numpy |
| `forgelm.compress` | ONNX export, int8 quantisation, parity gate, torch-free ONNX serving | torch for export only |
| `forgelm.tabular` | logistic regression, GBDT, EBM, TabICL; licence registry | per model |
| `forgelm.generative` | training backend, memory estimate, PEFT config for causal LMs | none |
| `forgelm.provenance` | evaluation contract, run record, model card, ML-BOM | none |
| `forgelm.packs.bfsi` | use-case map, sales-call taxonomy, rule screens, call review | none |
| `forgelm.finetune` | portable LoRA fine-tuning of chat models (any PyTorch device, no bitsandbytes), adapter merge, exact-match evaluation, MLX-LM command and data for Apple | per call |
| `forgelm.gguf` | llama.cpp conversion and quantisation, chat-template round-trip check, output parity | llama-cpp-python when used |
| `forgelm.dryrun` | a few real training steps on synthetic tokens to measure peak memory, set beside the estimate | torch when used |
| `forgelm.retrieval` | chunking with offsets, BM25, dense and fused (RRF) search, recall / MRR / nDCG, citation formatting, verbatim-quote check | none (scipy to build BM25) |
| `forgelm.gate` | paired bootstrap, exact McNemar and a cost-ranked "is the costlier option worth it" decision that states its uncertainty | none |
| `forgelm.cli` | `forgelm doctor / plan / train / predict / tabular / finetune / export-gguf` | none |

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
- **api-gateway** routes `/api/v1/factory` to it and forwards the full path
  (the older services receive the path with their prefix stripped).
- **frontend** has a Factory page with four tabs (hardware, plan,
  classifiers, call review) over that API.

## 4. Verification status

| Area | Status |
|---|---|
| `forgelm` core | unit tests in `tests/factory/` (about 560), no GPU or network needed |
| Student ladder, distillation, tabular, cascades, contract screening | measured on CPU; see `experiments/RESULTS.md` (E1, E2, E4, E5, E6) |
| ONNX export / int8 (E3), CPU fine-tune to GGUF (E7), prompt-or-train (E8), retrieval (E9) | scripted and smoke-tested; full runs are part of `experiments/run_all.sh` and land in `RESULTS.md` when complete |
| factory-service | FastAPI test client; image built and exercised with curl in a container (all endpoints, non-root user) |
| api-gateway factory routing | path rule unit-tested against every existing route; not run behind a live gateway |
| training-service changes | unit tests with stubbed GPU libraries. **Not run end to end**: no GPU, Unsloth or Redis here |
| Shared package and images | base and gateway images built and run; the Debian `apt` layer could not run here (package mirror blocked by this sandbox's network policy), so images that need it were built without it |
| ML images (CUDA base) | Dockerfiles linted only, **not built** |
| Kubernetes manifest for factory-service | YAML validated, **not applied** to a cluster |
| Backend selection and loading for NVIDIA / AMD / Apple / Intel | tested against simulated devices, **not on real hardware** |
| Front end | the Factory page type-checks and renders with mocked API responses in a headless browser. The project's `npm run build` still fails on 75 TypeScript errors in older files, none in the Factory code |

### Independent review

After the first version was written, a separate reviewer read the new code
and reported 14 defects. All were fixed, each fix was checked by a second
independent reviewer, and each has a regression test. The ones that mattered
most:

- A queued fine-tuning job was invisible (`/status` returned 404, cancel
  failed) until the job ahead of it finished.
- Rank-stabilised LoRA was switched on automatically at rank 32 and above,
  silently multiplying the adapter's effective step (8x at rank 64, alpha 128).
  It is now opt-in.
- Teacher answers that could not be parsed became class-0 training labels.
- A confidence threshold could split a group of tied confidences, reporting
  better accuracy and lower coverage than the threshold delivers.
- The evaluation contract passed a metric that was NaN.
- Tabular scoring assumed labels were encoded 0..k-1.
- The factory service accepted a classifier name with a trailing newline,
  could interleave two concurrent trainings of the same name, and would load
  any model id a request named.
- Two experiments compared numbers computed under different settings.

The completion work in section 5 went through the same process: each
workstream was built by one agent, checked by an independent reviewer, and
repaired and re-checked where the reviewer found a real problem (two needed
that: a timing figure that was a total rather than a per-item value, and a
results table that could show predictions from an earlier request).

Known and not fixed: cancelling a job that is already running sets a flag
but does not stop the trainer thread (stopping it cleanly needs changes to
the trainer callbacks that cannot be tested without a GPU). Log streaming now
waits off the event loop, but each open stream holds a worker thread for up
to a second at a time.

## 5. What was owed, and where it stands

| Item | Status |
|---|---|
| De-duplicate the services | done (`llmforge_common`); near-duplicates listed above remain |
| Collapse the two training implementations | partly: identical helpers shared; behaviour-different methods left for a GPU-verified pass |
| Vendor-aware hardware handling and a measured dry run | done in `hardware_detection.py`, the bridge's loading plan and `forgelm.dryrun`; simulated devices only |
| Apple Silicon training path and GGUF export | done as `forgelm.finetune` (MLX-LM command and data; the MLX run itself needs a Mac) and `forgelm.gguf` with a chat-template round-trip check |
| Front end for the factory | done (Factory page) |
| Prompting and retrieval baselines before fine-tuning | done in-house and small: few-shot prompting with retrieved demonstrations, a statistical gate, and BM25 / dense / hybrid retrieval. No DSPy-style prompt optimiser and no reranker |
| Deployment of the factory service | done: Kubernetes manifest, compose settings, `.env.example` |

## 6. Still owed, in order

1. **Run the training service on a real GPU** before merging to a release
   branch: confirm a job request returns at once, a `dora` request trains an
   adapter, and the NVIDIA loading path behaves as before. Then on an AMD,
   Apple and Intel machine if available.
2. **Rebuild and push every image**, base images first (`docker compose build
   base ml-base`, then the services). The training service image only works
   with a rebuilt image now that its shared modules come from `llmforge_common`.
3. **Merge the remaining duplicated training methods** once item 1 gives a
   way to verify them.
4. **Real BFSI evaluation data.** Label a few hundred reviewed call
   utterances, including ASR transcripts and Hinglish, and replace the
   illustrative seed set. Nothing in the BFSI pack has been validated on real
   calls.
5. **Prompt optimisation** (for example DSPy) and a **reranker** for
   retrieval, if E8 and E9 show the simple baselines fall short.
6. **Fix the 75 pre-existing TypeScript errors** so the front end builds.
