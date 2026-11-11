"""The training-argument helpers in ``training_common.py``, and log streaming.

``training_service.py`` needs the GPU stack (torch, Unsloth, Redis) to import,
so it is never imported in this process. As in ``test_services.py``, single
methods are cut out of the source with ``ast`` and run against fakes.
``training_common.py`` imports nothing heavy and is loaded directly from its
path. One group of tests does import the module whole, in a child process
where the GPU stack is replaced by stand-ins.

Nothing here trains a model. What these tests show is that the shared helpers
return what the two original copies returned (``TrainingService`` and the
since-removed ``TrainingCore``) and that log streaming no longer holds the
event loop; the training flow itself still needs a run on a GPU.
"""

import ast
import asyncio
import copy
import importlib.util
import json
import os
import queue
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SERVICES = ROOT / "src/microservices/training_service/src/services"
SERVICE_FILE = SERVICES / "training_service.py"
COMMON_FILE = SERVICES / "training_common.py"

# The methods that were word-for-word copies in both classes and now live in the mixin.
MERGED = ["_get_instruction_training_args", "_get_chat_completion_args", "_get_reasoning_training_args",
          "_apply_training_type_optimizations", "get_chat_sft_config"]

# The commit the consolidation started from. Used only by the tests that re-run the
# original methods, which skip when the commit is not in the checkout.
BASELINE = "8585879"


def _class_node(path, class_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)


def _methods(path, class_name):
    """Every def in the class body, in order, repeats included."""
    return [n for n in _class_node(path, class_name).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _method_source(path, class_name, name, text=None):
    text = path.read_text(encoding="utf-8") if text is None else text
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    defs = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    assert len(defs) == 1, f"{class_name}.{name} must be defined exactly once"
    return textwrap.dedent(ast.get_source_segment(text, defs[0]))


@pytest.fixture(scope="module")
def common():
    spec = importlib.util.spec_from_file_location("training_common_under_test", COMMON_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # no stubs: it must load without torch, Unsloth or the service's own packages
    return module


def _helpers(common, unsloth):
    """The mixin as a training class would use it, with the Unsloth answer fixed."""
    return type("Helpers", (common.TrainingArgsMixin,), {"_unsloth_available": lambda self: unsloth})()


# ---- inputs ----------------------------------------------------------------

# A hardware config that sets every optional key the helpers read, and one that sets none.
FULL = {"training_args": {"per_device_train_batch_size": 4, "gradient_accumulation_steps": 2, "logging_steps": 7,
                          "save_steps": 250, "dataloader_pin_memory": True},
        "chat_config": {"template_type": "llama3", "assistant_only_loss": False}, "enable_reasoning": True}
MINIMAL = {"training_args": {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 8}}
CONFIGS = {"full": FULL, "minimal": MINIMAL}
# The request paired with each config. The second asks for more than the 4096-token cap.
REQUESTS = {"full": SimpleNamespace(learning_rate=2e-4, max_seq_length=2048),
            "minimal": SimpleNamespace(learning_rate=5e-5, max_seq_length=8192)}
# What _apply_training_type_optimizations is handed: it must change these keys only when it should.
ARGS_BEFORE = {"learning_rate": 0.0002, "gradient_checkpointing": False, "group_by_length": True}


# ---- expected values --------------------------------------------------------
# Recorded by running the methods of TrainingService and of TrainingCore as they stood at
# commit 8585879 (before the merge) on the inputs above. Both classes gave these values.

# These three do not look at the Unsloth flag.
TYPE_ARGS = {
    ("_get_instruction_training_args", "full"): {
        "per_device_train_batch_size": 4, "gradient_accumulation_steps": 2, "warmup_steps": 5,
        "learning_rate": 0.0002, "weight_decay": 0.01, "lr_scheduler_type": "linear", "logging_steps": 7,
        "save_steps": 250, "dataloader_num_workers": 0, "dataloader_pin_memory": True,
        "remove_unused_columns": False},
    ("_get_instruction_training_args", "minimal"): {
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8, "warmup_steps": 5,
        "learning_rate": 5e-05, "weight_decay": 0.01, "lr_scheduler_type": "linear", "logging_steps": 10,
        "save_steps": 100, "dataloader_num_workers": 0, "dataloader_pin_memory": False,
        "remove_unused_columns": False},
    ("_get_chat_completion_args", "full"): {
        "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4, "warmup_steps": 10,
        "learning_rate": 0.00016, "weight_decay": 0.005, "lr_scheduler_type": "cosine", "logging_steps": 7,
        "save_steps": 250, "dataloader_num_workers": 0, "dataloader_pin_memory": False,
        "remove_unused_columns": False, "group_by_length": True, "prediction_loss_only": False},
    ("_get_chat_completion_args", "minimal"): {
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 16, "warmup_steps": 10,
        "learning_rate": 4e-05, "weight_decay": 0.005, "lr_scheduler_type": "cosine", "logging_steps": 5,
        "save_steps": 200, "dataloader_num_workers": 0, "dataloader_pin_memory": False,
        "remove_unused_columns": False, "group_by_length": True, "prediction_loss_only": False},
    ("_get_reasoning_training_args", "full"): {
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8, "warmup_steps": 20,
        "learning_rate": 0.0001, "weight_decay": 0.001, "lr_scheduler_type": "cosine", "logging_steps": 7,
        "save_steps": 250, "dataloader_num_workers": 0, "dataloader_pin_memory": False,
        "remove_unused_columns": False, "group_by_length": False, "prediction_loss_only": False},
    ("_get_reasoning_training_args", "minimal"): {
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 32, "warmup_steps": 20,
        "learning_rate": 2.5e-05, "weight_decay": 0.001, "lr_scheduler_type": "cosine", "logging_steps": 5,
        "save_steps": 500, "dataloader_num_workers": 0, "dataloader_pin_memory": False,
        "remove_unused_columns": False, "group_by_length": False, "prediction_loss_only": False},
}

SFT_BASE = {"dataset_text_field": "text", "packing": True, "dataset_num_proc": 1, "chars_per_token": 3.5}
# (unsloth available, config) -> get_chat_sft_config
SFT_CONFIG = {
    (False, "full"): {**SFT_BASE, "max_seq_length": 2048},
    (False, "minimal"): {**SFT_BASE, "max_seq_length": 4096},
    (True, "full"): {**SFT_BASE, "max_seq_length": 2048, "use_unsloth_template": True, "assistant_only_loss": False,
                     "dataset_kwargs": {"add_special_tokens": False, "chat_template": "llama3"}},
    (True, "minimal"): {**SFT_BASE, "max_seq_length": 4096, "use_unsloth_template": True, "assistant_only_loss": True,
                        "dataset_kwargs": {"add_special_tokens": False, "chat_template": "chatml"}},
}

UNCHANGED = dict(ARGS_BEFORE)
REASONING = {"learning_rate": 0.0002, "gradient_checkpointing": True, "group_by_length": False,
             "prediction_loss_only": False}
# (unsloth available, config, training type) -> the args dictionary afterwards
OPTIMIZED = {
    **{(False, config, kind): UNCHANGED for config in CONFIGS
       for kind in ("instruction", "chat", "reasoning", "something-else")},
    **{(True, config, kind): UNCHANGED for config in CONFIGS for kind in ("instruction", "something-else")},
    # chat with enable_reasoning set in the config gets the reasoning settings as well
    (True, "full", "chat"): REASONING,
    (True, "minimal", "chat"): {"learning_rate": 0.0002, "gradient_checkpointing": True, "group_by_length": True},
    (True, "full", "reasoning"): REASONING,
    (True, "minimal", "reasoning"): REASONING,
}


def _run_helpers(helpers):
    """Every merged helper on every input, in the shape of the expected tables above."""
    type_args, sft, optimized = {}, {}, {}
    for config_name, config in CONFIGS.items():
        request = REQUESTS[config_name]
        for name in MERGED[:3]:
            type_args[(name, config_name)] = getattr(helpers, name)(copy.deepcopy(config), request)
        sft[config_name] = helpers.get_chat_sft_config(request, copy.deepcopy(config))
        for kind in ("instruction", "chat", "reasoning", "something-else"):
            args = dict(ARGS_BEFORE)
            returned = helpers._apply_training_type_optimizations(args, kind, {"enabled": True}, copy.deepcopy(config))
            assert returned is args  # updated in place and handed back, which the caller relies on
            optimized[(config_name, kind)] = args
    return type_args, sft, optimized


# ---- the shared helpers -----------------------------------------------------

@pytest.mark.parametrize("unsloth", [False, True])
def test_shared_helpers_return_what_the_two_original_copies_returned(common, unsloth):
    type_args, sft, optimized = _run_helpers(_helpers(common, unsloth))
    assert type_args == TYPE_ARGS
    assert sft == {config: SFT_CONFIG[(unsloth, config)] for config in CONFIGS}
    assert optimized == {(config, kind): OPTIMIZED[(unsloth, config, kind)] for config, kind in optimized}
    # Exact float equality is intended: 0.8 * 2e-4 must stay the same product it always was.
    assert type_args[("_get_chat_completion_args", "full")]["learning_rate"] == 2e-4 * 0.8


def test_helpers_do_not_modify_the_config_they_are_given(common):
    helpers = _helpers(common, True)
    for config_name, config in CONFIGS.items():
        before = copy.deepcopy(config)
        for name in MERGED[:3]:
            getattr(helpers, name)(config, REQUESTS[config_name])
        helpers.get_chat_sft_config(REQUESTS[config_name], config)
        helpers._apply_training_type_optimizations(dict(ARGS_BEFORE), "chat", {}, config)
        assert config == before


def test_the_mixin_refuses_to_guess_whether_unsloth_is_available(common):
    bare = common.TrainingArgsMixin()
    with pytest.raises(NotImplementedError, match="_unsloth_available"):
        bare.get_chat_sft_config(REQUESTS["full"], FULL)
    with pytest.raises(NotImplementedError):
        bare._apply_training_type_optimizations({}, "chat", {}, FULL)


def test_training_common_imports_nothing_heavy():
    tree = ast.parse(COMMON_FILE.read_text(encoding="utf-8"))
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level.add(node.module)
    assert top_level == {"__future__", "typing"}

    # Anywhere in the file, including inside functions and the TYPE_CHECKING block: the one
    # other import allowed is the request schema, for annotations only.
    everywhere = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            everywhere.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            everywhere.add((node.module or "").split(".")[0])
    assert everywhere == {"__future__", "typing", "models"}
    guarded = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "TYPE_CHECKING"]
    assert len(guarded) == 1 and "models.schemas" in ast.unparse(guarded[0])


# ---- how the class uses them ------------------------------------------------

@pytest.mark.parametrize("path,class_name", [(SERVICE_FILE, "TrainingService")])
def test_each_class_inherits_the_helpers_and_defines_no_method_twice(common, path, class_name):
    cls = _class_node(path, class_name)
    assert [ast.unparse(base) for base in cls.bases] == ["TrainingArgsMixin"]
    imports = [n for n in ast.parse(path.read_text(encoding="utf-8")).body
               if isinstance(n, ast.ImportFrom) and n.module == "services.training_common"]
    assert [alias.name for n in imports for alias in n.names] == ["TrainingArgsMixin"]

    names = [n.name for n in _methods(path, class_name)]
    repeated = sorted({name for name in names if names.count(name) > 1})
    assert repeated == [], f"{class_name} defines these more than once (the later one silently wins): {repeated}"
    assert not set(MERGED) & set(names), "a merged helper is defined in the class again and shadows the shared one"
    for name in MERGED:
        assert callable(getattr(common.TrainingArgsMixin, name))


@pytest.mark.parametrize("path,class_name", [(SERVICE_FILE, "TrainingService")])
def test_each_class_answers_with_its_own_modules_unsloth_flag(common, path, class_name):
    # Run the class's real _unsloth_available in a stand-in for its module's globals.
    module_globals = {"UNSLOTH_AVAILABLE": False}
    exec(_method_source(path, class_name, "_unsloth_available"), module_globals)
    helpers = type(class_name, (common.TrainingArgsMixin,),
                   {"_unsloth_available": module_globals["_unsloth_available"]})()

    assert helpers.get_chat_sft_config(REQUESTS["full"], FULL) == SFT_CONFIG[(False, "full")]
    assert helpers._apply_training_type_optimizations(dict(ARGS_BEFORE), "reasoning", {}, FULL) == UNCHANGED
    # The originals read the module global on every call, so a later change must show.
    module_globals["UNSLOTH_AVAILABLE"] = True
    assert helpers.get_chat_sft_config(REQUESTS["full"], FULL) == SFT_CONFIG[(True, "full")]
    assert helpers._apply_training_type_optimizations(dict(ARGS_BEFORE), "reasoning", {}, FULL) == REASONING


def _baseline_source(path):
    relative = path.relative_to(ROOT).as_posix()
    try:
        shown = subprocess.run(["git", "show", f"{BASELINE}:{relative}"], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        pytest.skip(f"git is not usable here: {e}")
    if shown.returncode != 0:
        pytest.skip(f"commit {BASELINE} is not in this checkout")
    return shown.stdout


@pytest.mark.parametrize("path,class_name", [(SERVICE_FILE, "TrainingService")])
@pytest.mark.parametrize("unsloth", [False, True])
def test_shared_helpers_match_the_original_methods_run_side_by_side(common, path, class_name, unsloth):
    source = _baseline_source(path)
    module_globals = {"Dict": dict, "Any": object, "FineTuneRequest": object, "UNSLOTH_AVAILABLE": unsloth}
    for name in MERGED:
        exec(_method_source(path, class_name, name, text=source), module_globals)
    original = type("Original", (), {name: module_globals[name] for name in MERGED})()
    assert _run_helpers(original) == _run_helpers(_helpers(common, unsloth))


# ---- the real modules, imported with the GPU stack stubbed -------------------
#
# The tests above run single methods. This one imports training_service.py
# whole, in a child process whose torch, Unsloth, transformers and service-local
# packages are stand-ins, to check what cutting methods out cannot: that
# ``from services.training_common import ...`` resolves, that the class can be
# built on the mixin, and that the caller of the helpers
# (``_create_adaptive_training_args``) still passes on the same values.
# A child process, because importing the module sets environment variables.

HARNESS = r'''
import importlib, json, sys, types
from pathlib import Path
from unittest.mock import MagicMock

src, unsloth, scratch, config = sys.argv[1], sys.argv[2] == "1", sys.argv[3], json.loads(sys.argv[4])
sys.path.insert(0, src)

def module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m

torch = MagicMock(name="torch")
torch.cuda.is_available.return_value = False
sys.modules["torch"] = torch
module("datasets", load_from_disk=MagicMock())
if unsloth:
    module("unsloth", __version__="stub", FastLanguageModel=MagicMock(), is_bfloat16_supported=lambda: True)
else:
    sys.modules["unsloth"] = None  # makes "import unsloth" raise ImportError

class TrainingArguments(dict):  # keeps the keyword arguments it was built with
    def __init__(self, **kwargs):
        super().__init__(kwargs)

module("transformers", TrainingArguments=TrainingArguments, AutoTokenizer=MagicMock(),
       AutoModelForCausalLM=MagicMock(), Trainer=MagicMock(), TrainerCallback=type("TrainerCallback", (), {}))
module("trl", SFTTrainer=MagicMock(), SFTConfig=MagicMock())
module("peft", get_peft_model=MagicMock(), LoraConfig=MagicMock())
module("models"); module("models.schemas", FineTuneRequest=object, TrainingStatus=object)
module("utils"); module("utils.logging", get_logger=lambda name=None: MagicMock())
module("utils.factory_bridge", backend_advice=MagicMock(), resolve_peft_config=MagicMock(),
       torch_device_and_dtype=MagicMock(), causal_lm_load_plan=MagicMock())
module("utils.performance", performance_monitor=MagicMock(), timing_context=MagicMock(), timed=MagicMock())
module("utils.hardware_detection", hardware_detector=MagicMock(), AdaptiveTrainingConfig=MagicMock())
module("config"); module("config.settings", settings=MagicMock())
module("common"); module("common.gpu_client", GPUClient=MagicMock(), GPULockManager=MagicMock(),
                         GPUOperationType=MagicMock(), GPULockPriority=MagicMock(), get_service_id=lambda: "svc")
# A bare package, so services/__init__.py does not run.
package = module("services")
package.__path__ = [str(Path(src) / "services")]

service_module = importlib.import_module("services.training_service")
assert service_module.UNSLOTH_AVAILABLE is unsloth

request = types.SimpleNamespace(learning_rate=2e-4, max_seq_length=2048, epochs=3, output_dir=scratch)
TYPE_HELPERS = ["_get_instruction_training_args", "_get_chat_completion_args", "_get_reasoning_training_args"]

def helpers(obj):
    out = {name: getattr(obj, name)(config, request) for name in TYPE_HELPERS}
    out["get_chat_sft_config"] = obj.get_chat_sft_config(request, config)
    for kind in ("instruction", "chat", "reasoning"):
        out["optimize:" + kind] = obj._apply_training_type_optimizations({"seed": 1}, kind, {}, config)
    return out

service = service_module.TrainingService.__new__(service_module.TrainingService)  # __init__ probes hardware
service.logger = MagicMock()

result = {
    "service_helpers": helpers(service),
    "service_training_args": {kind: service._create_adaptive_training_args(request, "task-1234-abcd", config, kind)
                              for kind in ("instruction", "chat", "reasoning")},
    "service_mro": [c.__name__ for c in service_module.TrainingService.__mro__],
}
# The flag is read when a helper runs, not when the class is built.
service_module.UNSLOTH_AVAILABLE = not unsloth
result["service_sft_after_flag_flip"] = service.get_chat_sft_config(request, config)
print(json.dumps(result, sort_keys=True))
'''

# FULL plus the two keys only _create_adaptive_training_args reads.
HARNESS_CONFIG = {**FULL, "training_args": {**FULL["training_args"], "report_to": []},
                  "gpu_optimizations": {"enabled": True}}


def _import_real_modules(src_dir, unsloth, tmp_path):
    """Run HARNESS against the tree at src_dir; returns its result with output_dir values removed."""
    scratch = tmp_path / f"run-{len(list(tmp_path.iterdir()))}"
    scratch.mkdir()
    # TMPDIR: training_service.py makes a Triton cache directory at import; keep it under tmp_path.
    # PYTHONIOENCODING: the modules print emoji while importing, whatever the locale is.
    env = {**os.environ, "TMPDIR": str(scratch), "KUBERNETES_MODE": "false", "PYTHONIOENCODING": "utf-8"}
    command = [sys.executable, "-c", HARNESS, str(src_dir), "1" if unsloth else "0", str(scratch),
               json.dumps(HARNESS_CONFIG)]
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=300, env=env)
    assert done.returncode == 0, done.stderr[-3000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])  # the modules print while importing
    for built in result["service_training_args"].values():
        assert built.pop("output_dir") == str(scratch / "checkpoints")
    return result


# Recorded from the modules as they stood at commit 8585879, run through the same harness.
SERVICE_ARGS_ALWAYS = {
    False: {"bf16": False, "fp16": False, "optim": "adamw_hf"},
    True: {"bf16": True, "fp16": False, "optim": "adamw_8bit", "gradient_checkpointing": True, "max_grad_norm": 0.3},
}
SERVICE_ARGS_SHARED = {"dataloader_num_workers": 0, "eval_strategy": "no", "logging_steps": 7, "num_train_epochs": 3,
                       "remove_unused_columns": False, "report_to": [], "save_steps": 250, "save_total_limit": 2,
                       "seed": 3407}
SERVICE_ARGS_BY_TYPE = {
    "instruction": {"dataloader_pin_memory": True, "gradient_accumulation_steps": 2, "learning_rate": 0.0002,
                    "lr_scheduler_type": "linear", "per_device_train_batch_size": 4,
                    "run_name": "llmforge_instruction_task-123", "warmup_steps": 5, "weight_decay": 0.01},
    "chat": {"dataloader_pin_memory": False, "gradient_accumulation_steps": 4, "learning_rate": 0.00016,
             "lr_scheduler_type": "cosine", "per_device_train_batch_size": 2,
             "run_name": "llmforge_chat_task-123", "warmup_steps": 10, "weight_decay": 0.005},
    "reasoning": {"dataloader_pin_memory": False, "gradient_accumulation_steps": 8, "learning_rate": 0.0001,
                  "lr_scheduler_type": "cosine", "per_device_train_batch_size": 1,
                  "run_name": "llmforge_reasoning_task-123", "warmup_steps": 20, "weight_decay": 0.001},
}
# Added by _apply_training_type_optimizations for chat (reasoning enabled in the config) and reasoning.
SERVICE_ARGS_UNSLOTH_EXTRA = {"group_by_length": False, "prediction_loss_only": False}


@pytest.mark.parametrize("unsloth", [False, True])
def test_the_real_module_imports_and_builds_the_recorded_training_arguments(tmp_path, unsloth):
    result = _import_real_modules(SERVICES.parent, unsloth, tmp_path)

    assert result["service_mro"] == ["TrainingService", "TrainingArgsMixin", "object"]

    # The helpers, reached through the real class.
    helpers = result["service_helpers"]
    for name in MERGED[:3]:
        assert helpers[name] == TYPE_ARGS[(name, "full")]
    assert helpers["get_chat_sft_config"] == SFT_CONFIG[(unsloth, "full")]
    assert helpers["optimize:instruction"] == {"seed": 1}
    extra = {"gradient_checkpointing": True, **SERVICE_ARGS_UNSLOTH_EXTRA} if unsloth else {}
    assert helpers["optimize:chat"] == helpers["optimize:reasoning"] == {"seed": 1, **extra}
    assert result["service_sft_after_flag_flip"] == SFT_CONFIG[(not unsloth, "full")]

    # Their caller: every keyword argument TrainingArguments is built with.
    for kind, built in result["service_training_args"].items():
        extra = SERVICE_ARGS_UNSLOTH_EXTRA if unsloth and kind != "instruction" else {}
        assert built == {**SERVICE_ARGS_SHARED, **SERVICE_ARGS_ALWAYS[unsloth], **SERVICE_ARGS_BY_TYPE[kind], **extra}


@pytest.mark.parametrize("unsloth", [False, True])
def test_the_real_module_builds_what_the_original_module_built(tmp_path, unsloth):
    baseline = tmp_path / "baseline"
    (baseline / "services").mkdir(parents=True)
    (baseline / "services" / SERVICE_FILE.name).write_text(_baseline_source(SERVICE_FILE), encoding="utf-8")
    before = _import_real_modules(baseline, unsloth, tmp_path)
    after = _import_real_modules(SERVICES.parent, unsloth, tmp_path)
    assert before.pop("service_mro") == ["TrainingService", "object"]
    del after["service_mro"]
    assert after == before


# ---- log streaming ----------------------------------------------------------

def _stream_service():
    """stream_logs bound to a stand-in service, plus the module-level tables it reads."""
    module_globals = {"asyncio": asyncio, "json": json, "queue": queue, "time": time,
                      "task_log_queues": {}, "task_completion_flags": {}}
    exec(_method_source(SERVICE_FILE, "TrainingService", "stream_logs"), module_globals)
    errors = []

    class Service:
        stream_logs = module_globals["stream_logs"]
        logger = SimpleNamespace(error=errors.append)

    return Service(), module_globals["task_log_queues"], module_globals["task_completion_flags"], errors


ENTRIES = [
    {"timestamp": "2026-01-01 10:00:00", "type": "train_start", "message": "Starting training with 30 total steps",
     "total_steps": 30, "total_epochs": 3},
    {"timestamp": "2026-01-01 10:00:05", "type": "training", "message": "Step 1/30 (3.3%) - Epoch: 0.10",
     "step": 1, "total_steps": 30, "epoch": 0.1, "total_epochs": 3, "loss": None, "learning_rate": None,
     "progress_percentage": 3.3333333333333335},
    {"timestamp": "2026-01-01 10:00:40", "type": "training", "message": "Step 10/30 (33.3%) - Loss: 1.2345",
     "step": 10, "total_steps": 30, "loss": 1.2345, "learning_rate": 0.0002},
]


def _stream_entries_scenario(service, queues, flags):
    """Two entries already queued, a third arriving from another thread, then completion."""
    queues["job"] = queue.Queue()
    flags["job"] = False
    queues["job"].put(ENTRIES[0])
    queues["job"].put(ENTRIES[1])

    async def scenario():
        received = []
        trainer = threading.Timer(0.1, queues["job"].put, args=(ENTRIES[2],))
        trainer.start()
        async for item in service.stream_logs("job"):
            received.append(item)
            if len(received) == 3:
                flags["job"] = True  # what fine_tune does when training ends
        trainer.join()
        return received

    return asyncio.run(scenario())


def test_stream_logs_yields_queued_entries_then_new_ones_as_json_strings_in_order():
    service, queues, flags, errors = _stream_service()
    received = _stream_entries_scenario(service, queues, flags)
    # Exactly what main.py's two endpoints expect: one JSON string per entry, nothing else.
    assert received == [json.dumps(entry) for entry in ENTRIES]
    assert [json.loads(item) for item in received] == ENTRIES
    assert errors == [] and queues["job"].empty()


def test_stream_logs_keeps_the_event_loop_running_while_it_waits_for_an_entry():
    service, queues, flags, errors = _stream_service()
    queues["job"] = queue.Queue()
    flags["job"] = False

    async def scenario():
        ticks = 0

        async def other_work():  # stands for health checks, status requests and other streams
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        background = asyncio.create_task(other_work())
        stream = service.stream_logs("job")
        started = time.perf_counter()
        first = await stream.__anext__()  # nothing is queued, so this is the one-second wait
        waited = time.perf_counter() - started
        ticks_during_wait = ticks
        flags["job"] = True
        rest = [item async for item in stream]
        background.cancel()
        return first, waited, ticks_during_wait, rest

    first, waited, ticks_during_wait, rest = asyncio.run(scenario())
    heartbeat = json.loads(first)
    assert set(heartbeat) == {"type", "timestamp"} and heartbeat["type"] == "heartbeat"
    time.strptime(heartbeat["timestamp"], "%Y-%m-%d %H:%M:%S")
    assert 0.9 <= waited < 10.0  # still the one-second timeout, with slack for a busy machine
    # With the queue read made on the loop thread this is 0: nothing else ran for the whole second.
    assert ticks_during_wait >= 5
    assert rest == [] and errors == []


def test_stream_logs_makes_the_blocking_read_in_another_thread_with_the_same_arguments():
    service, queues, flags, errors = _stream_service()
    reads = []

    class RecordingQueue:
        """Nothing queued up front; hands out two entries, then reports a timeout at once."""

        def __init__(self):
            self.pending = list(ENTRIES[:2])

        def empty(self):
            return True

        def get(self, block=True, timeout=None):
            reads.append((threading.get_ident(), block, timeout))
            if self.pending:
                return self.pending.pop(0)
            raise queue.Empty

    queues["job"] = RecordingQueue()
    flags["job"] = False

    async def scenario():
        received = []
        async for item in service.stream_logs("job"):
            received.append(item)
            if len(received) == 3:
                flags["job"] = True
        return received, threading.get_ident()

    received, loop_thread = asyncio.run(scenario())
    assert received[:2] == [json.dumps(entry) for entry in ENTRIES[:2]]
    assert json.loads(received[2])["type"] == "heartbeat" and len(received) == 3  # queue.Empty still means heartbeat
    assert len(reads) == 3 and all(thread != loop_thread for thread, _, _ in reads)
    assert {(block, timeout) for _, block, timeout in reads} == {(True, 1.0)}
    assert errors == []


def test_stream_logs_reports_an_unknown_task_as_a_single_error_entry():
    service, _queues, _flags, errors = _stream_service()

    async def scenario():
        return [item async for item in service.stream_logs("nope")]

    assert asyncio.run(scenario()) == [json.dumps({"type": "error", "message": "'Task nope not found'"})]
    assert len(errors) == 1 and "nope" in errors[0]
