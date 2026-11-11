"""ForgeLM core: a hardware-agnostic model intelligence factory.

The package is deliberately light at import time. Nothing here imports torch,
transformers or any other heavy dependency until a function that needs it is
called, so ``forgelm doctor`` and the planner run on a bare Python install.

Pipeline stages, each usable on its own:

    hardware      what compute is here (CPU / NVIDIA / AMD / Apple / Intel)
    planner       given a task, data regime and constraints, which model
                  families to try, in what order, and why
    students      small task models that need no LLM at inference
    teachers      large models (or any callable) that label or generate data
    distill       teacher -> student, with a human-labelled anchor set
    metrics       accuracy, F1, calibration, abstention, cascades
    compress      ONNX export and int8 quantisation behind a parity gate
    tabular       classical and in-context tabular models
    generative    backend, memory and PEFT choices for LLM fine-tuning
    provenance    run records, model cards, ML-BOM
"""

__version__ = "0.1.0"

from .hardware import Accelerator, HardwareProfile, probe  # noqa: F401
from .planner import Constraints, Plan, TaskSpec, plan  # noqa: F401
