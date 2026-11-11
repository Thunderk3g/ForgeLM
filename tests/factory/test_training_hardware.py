"""The training service's hardware logic on NVIDIA and non-NVIDIA machines, and the dry run.

Four things are checked here.

1. NVIDIA settings did not move. ``hardware_detection.py`` is loaded twice,
   once from the working tree and once from commit 8585879 (the version before
   vendor awareness was added), and both are asked the same questions for
   simulated NVIDIA cards in every VRAM tier. The answers must be identical,
   down to the order of dictionary keys and the log lines. A digest of the
   baseline's answers is pinned as well, so the check still means something in
   a checkout where that commit is no longer reachable.
2. Apple MPS, Intel XPU and AMD ROCm profiles produce the configurations the
   module documents. torch is simulated: nothing here ran on such hardware.
3. ``forgelm.dryrun`` runs real optimisation steps on CPU with a tiny randomly
   initialised model, and its memory readers for the accelerators are driven
   with a stand-in for torch.
4. The bridge functions the service calls.

No GPU, no network and no model download are needed.
"""

import dataclasses
import hashlib
import importlib.util
import itertools
import json
import math
import os
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
DETECTION_FILE = "src/microservices/training_service/src/utils/hardware_detection.py"
BRIDGE_FILE = ROOT / "src/microservices/training_service/src/utils/factory_bridge.py"
BASELINE_COMMIT = "8585879"
GIB = 1024**3

# sha256 of the baseline's answers over the hand-built grid in _golden_answers,
# one per way Unsloth can be present. Computed from commit 8585879; the
# side-by-side test asserts the baseline still produces them whenever the
# commit is reachable.
PINNED_BASELINE_DIGESTS = {
    "unsloth absent": "0a287bea46a8e63000e824950cde1a247d6cdfd8fd5484cd763350593d7a659e",
    "unsloth with bf16": "fdfb3b934c0699f9f56f759d8c53dff5d563b1bf9b421bfdf700981dda26ee2b",
    "unsloth without bf16": "c867b3f240ffe202ddfd918e9f808b61fad8879f8dcf073544ede2a5178ee1f9",
}
UNSLOTH_VARIANTS = {"unsloth absent": None, "unsloth with bf16": True, "unsloth without bf16": False}


# ---- loading the module with its service-only imports replaced ---------------

class RecordingLogger:
    """Keeps every log line, so the tests can compare what two versions said."""

    def __init__(self):
        self.lines = []

    def _log(self, level):
        return lambda message, *args, **kwargs: self.lines.append((level, str(message)))

    def __getattr__(self, level):
        if level in ("debug", "info", "warning", "error", "critical", "exception"):
            return self._log(level)
        raise AttributeError(level)


@contextmanager
def _stand_ins(modules):
    """Put stand-in modules in sys.modules for the duration, then restore what was there."""
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, original in saved.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _load_bridge(name="factory_bridge_for_hardware_tests"):
    spec = importlib.util.spec_from_file_location(name, BRIDGE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_detection(source, label, unsloth_bf16=None, bridge=None):
    """Execute hardware_detection source as a module.

    The module imports ``utils.logging`` (needs the service's shared package),
    ``unsloth`` (NVIDIA only), ``torch`` and ``psutil``. All four are replaced
    while it loads. ``unsloth_bf16`` is None for "Unsloth not installed", or
    the answer its ``is_bfloat16_supported`` gives. torch and psutil are empty
    placeholders: each test installs the fake it needs on the loaded module.
    """
    log = RecordingLogger()
    utils = types.ModuleType("utils")
    utils.__path__ = []  # a package, so "from utils import factory_bridge" can look for a submodule
    logging_stub = types.ModuleType("utils.logging")
    logging_stub.get_logger = lambda name=None: log
    stand_ins = {"utils": utils, "utils.logging": logging_stub, "torch": types.ModuleType("torch"),
                 "psutil": types.ModuleType("psutil"), "unsloth": None}  # None makes "import unsloth" fail
    if unsloth_bf16 is not None:
        unsloth = types.ModuleType("unsloth")
        unsloth.is_bfloat16_supported = lambda: unsloth_bf16
        stand_ins["unsloth"] = unsloth
    if bridge is not None:
        stand_ins["utils.factory_bridge"] = bridge
        utils.factory_bridge = bridge

    module = types.ModuleType(f"hardware_detection_{label}")
    module.__file__ = str(ROOT / DETECTION_FILE)
    # dataclasses looks the defining module up by name while the class is being built.
    with _stand_ins({**stand_ins, module.__name__: module}):
        exec(compile(source, f"<hardware_detection {label}>", "exec"), module.__dict__)
    module.test_log = log
    return module


def _current_source():
    return (ROOT / DETECTION_FILE).read_text()


@pytest.fixture(scope="module")
def baseline_source():
    """hardware_detection.py as it was at the baseline commit, or skip if git cannot produce it."""
    try:
        shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASELINE_COMMIT}:{DETECTION_FILE}"],
                               capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git is not usable here ({exc}); the pinned digests still guard the NVIDIA settings")
    if shown.returncode != 0:
        pytest.skip(f"commit {BASELINE_COMMIT} is not reachable in this checkout; "
                    "the pinned digests still guard the NVIDIA settings")
    return shown.stdout


@pytest.fixture(scope="module")
def bridge():
    return _load_bridge()


@pytest.fixture
def detection(bridge):
    """The current module, with the real bridge behind it and Unsloth absent."""
    return _load_detection(_current_source(), "current", bridge=bridge)


# ---- simulated machines ------------------------------------------------------------

def make_torch(kind, name="", memory_gb=0.0, capability=(8, 9), count=1, bf16=True, old_torch=False,
               xpu_memory="properties"):
    """A stand-in for the parts of torch the detector reads.

    ``kind`` is "cuda" (NVIDIA), "rocm", "mps", "xpu" or "none". ``old_torch``
    leaves out everything a release from before XPU, MPS and ROCm support does
    not have, to check that the detector never trips over a missing attribute.
    """
    cuda_on = kind in ("cuda", "rocm")

    def properties(index=0):
        if not cuda_on:
            raise AssertionError("Torch not compiled with CUDA enabled")
        props = SimpleNamespace(total_memory=int(memory_gb * GIB), name=name)
        if capability is not None:  # ROCm builds may not report an NVIDIA compute capability
            props.major, props.minor = capability
        return props

    torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: cuda_on, device_count=lambda: count if cuda_on else 0,
        get_device_properties=properties, is_bf16_supported=lambda: bf16))
    if old_torch:
        return torch
    torch.version = SimpleNamespace(hip="6.2.41133" if kind == "rocm" else None)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: kind == "mps"))
    torch.xpu = SimpleNamespace(is_available=lambda: kind == "xpu", device_count=lambda: count if kind == "xpu" else 0)
    if kind == "xpu":
        torch.xpu.get_device_name = lambda index=0: name
        if xpu_memory == "properties":
            torch.xpu.get_device_properties = lambda index=0: SimpleNamespace(total_memory=int(memory_gb * GIB))
        elif xpu_memory == "mem_get_info":
            torch.xpu.mem_get_info = lambda index=0: (int(memory_gb * GIB * 0.9), int(memory_gb * GIB))
    return torch


def install_machine(module, monkeypatch, torch, ram_gb=32.0, available_gb=24.0, cores=8, threads=16):
    """Point a loaded module at a simulated machine. Returns the list of commands it runs."""
    commands = []

    def run(command, **kwargs):
        commands.append(list(command))
        if command[0] == "nvidia-smi" and "--query-gpu=driver_version" in command:
            return SimpleNamespace(returncode=0, stdout="550.54.14\n")
        if command[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="Default, 312 MiB\n")
        if command[0] == "sysctl":
            return SimpleNamespace(returncode=0, stdout="Apple M2\n")
        raise FileNotFoundError(command[0])

    module.torch = torch
    module.psutil = SimpleNamespace(
        cpu_count=lambda logical=True: threads if logical else cores,
        cpu_freq=lambda: SimpleNamespace(max=3600.0),
        virtual_memory=lambda: SimpleNamespace(total=int(ram_gb * GIB), available=int(available_gb * GIB)))
    module.subprocess = SimpleNamespace(run=run)
    # Free disk space changes while other work writes files; fix it so two detections can be compared.
    monkeypatch.setattr(os, "statvfs", lambda path: SimpleNamespace(f_frsize=4096, f_bavail=25_000_000))
    return commands


AS_DETECTED = object()


def detect(module, monkeypatch, torch, container_gb=AS_DETECTED, **machine):
    """Run the detector on a simulated machine.

    The container memory limit is the one thing still read from the real
    machine (its cgroup files). The side-by-side tests leave it as detected,
    since both versions read the same files. Tests that assert on memory
    figures pass ``container_gb`` so their numbers do not depend on where
    the suite runs.
    """
    commands = install_machine(module, monkeypatch, torch, **machine)
    profile = module.HardwareDetector().detect_full_profile()
    if container_gb is not AS_DETECTED:
        profile.container_memory_gb = container_gb
    return profile, commands


def request(**fields):
    base = dict(batch_size=2, gradient_accumulation_steps=4, epochs=1, learning_rate=2e-4, max_seq_length=1024)
    return SimpleNamespace(**{**base, **fields})


SMALL, LARGE, UNSIZED = "Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "my-org/custom-model"

REQUESTS = [
    request(model_name="unsloth/Qwen2.5-0.5B-Instruct"),
    request(base_model="meta-llama/Llama-3.2-3B-Instruct", batch_size=8, gradient_accumulation_steps=32, eval_steps=50),
    request(base_model_dir="/models/mistral-7b-instruct", batch_size=1, gradient_accumulation_steps=1),
    request(model_name="microsoft/phi-2", batch_size=4),
    request(model_name="google/gemma-2b-it"),
    request(model_name="codellama/CodeLlama-7b-hf"),
    request(model_name="gpt2"),
    request(),  # no model name at all
]
MODEL_NAMES = [None, "unsloth/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3.2-3B", "mistral-7b", "microsoft/phi-2",
               "google/gemma-2b-it", "codellama-7b", "gpt2"]
LORA_MEMORY = [0.0, 2.8, 4.0, 4.1, 5.6, 8.0, 8.1, 12.0, 16.0, 16.1, 20.4, 68.0]


def answers(module, profile):
    """Everything AdaptiveTrainingConfig says about one profile, in a JSON-friendly shape."""
    config = module.AdaptiveTrainingConfig(profile)
    out = {"summary": config.get_hardware_summary(),
           "gpu_optimizations": [config._get_gpu_optimizations(name) for name in MODEL_NAMES],
           "gpu_optimizations_default": config._get_gpu_optimizations(),
           "lora": [config._get_adaptive_lora_config(memory, name)
                    for memory, name in itertools.product(LORA_MEMORY, MODEL_NAMES)],
           "optimal": [config.generate_optimal_config(r) for r in REQUESTS],
           "enhanced": [config.generate_enhanced_adaptive_config(r, kind)
                        for r in REQUESTS[:3] for kind in ("instruction", "chat", "reasoning", "unknown")],
           "system": config._get_system_optimizations(),
           "summary_again": config.get_hardware_summary()}
    return out


def as_text(value):
    """JSON without sorting: two results are the same text only if their keys are in the same order."""
    return json.dumps(value)


def legacy_fields(module, profile):
    """The profile restricted to the fields the baseline version has."""
    return {f.name: getattr(profile, f.name) for f in dataclasses.fields(module.HardwareProfile)}


def strip_new_summary_keys(out):
    """Check and remove the two keys get_hardware_summary gained, leaving what the baseline returns."""
    for key in ("summary", "summary_again"):
        summary = dict(out[key])
        assert summary.pop("warnings") == []
        assert summary.pop("accelerator") in ("cuda", "none")
        out[key] = summary
    return out


# ---- 1. NVIDIA settings are what they were --------------------------------------------

NVIDIA_CARDS = [
    # (name, VRAM in GB, compute capability)       tier
    ("NVIDIA GeForce RTX 3050 Laptop GPU", 4.0, (8, 6)),    # <= 4
    ("NVIDIA GeForce GTX 1650", 4.0, (7, 5)),               # <= 4, Turing
    ("NVIDIA GeForce GTX 1060 3GB", 3.0, (6, 1)),           # <= 4, no bfloat16
    ("NVIDIA GeForce RTX 2060", 6.0, (7, 5)),               # <= 8
    ("NVIDIA GeForce RTX 4060", 8.0, (8, 9)),               # <= 8
    ("NVIDIA GeForce RTX 4060", 7.996, (8, 9)),             # <= 8, as a driver reports it
    ("NVIDIA GeForce RTX 3060", 12.0, (8, 6)),              # <= 16 (and the 12 GB training tier)
    ("NVIDIA GeForce RTX 4080", 16.0, (8, 9)),              # <= 16
    ("Tesla T4", 15.0, None),                               # <= 16, capability unreadable
    ("NVIDIA GeForce RTX 4090", 24.0, (8, 9)),              # > 16
    ("NVIDIA A100-SXM4-80GB", 80.0, (8, 0)),                # > 16
    ("Some Future Card", 16.01, None),                      # just over a tier edge, unknown card
]


@pytest.mark.parametrize("variant", list(UNSLOTH_VARIANTS))
@pytest.mark.parametrize("name,memory_gb,capability", NVIDIA_CARDS)
def test_nvidia_settings_are_identical_to_the_baseline_commit(baseline_source, bridge, monkeypatch, variant, name,
                                                              memory_gb, capability):
    unsloth = UNSLOTH_VARIANTS[variant]
    old = _load_detection(baseline_source, "baseline", unsloth)
    new = _load_detection(_current_source(), "current", unsloth, bridge=bridge)
    for count, (ram, available) in ((1, (32.0, 24.0)), (2, (8.0, 3.0))):
        torch = make_torch("cuda", name, memory_gb, capability, count=count)
        old_profile, old_commands = detect(old, monkeypatch, torch, ram_gb=ram, available_gb=available)
        new_profile, new_commands = detect(new, monkeypatch, torch, ram_gb=ram, available_gb=available)

        # The detector sees the same machine and asks nvidia-smi the same things.
        assert legacy_fields(old, new_profile) == legacy_fields(old, old_profile)
        assert new_commands == old_commands
        assert new_profile.accelerator == "cuda" and new_profile.accelerator_memory_gb == old_profile.gpu_memory_gb

        old_answers = answers(old, old_profile)
        new_answers = strip_new_summary_keys(answers(new, new_profile))
        for key in old_answers:
            assert new_answers[key] == old_answers[key], key
        assert as_text(new_answers) == as_text(old_answers)  # same keys in the same order, same values
        # ... and said the same things in the log while doing it.
        assert new.test_log.lines and new.test_log.lines == old.test_log.lines

        # Nothing new is attached to an NVIDIA configuration.
        config = new.AdaptiveTrainingConfig(new_profile)
        full = config.generate_optimal_config(REQUESTS[0])
        assert config.warnings == [] and "warnings" not in full and "accelerator" not in full
        assert "warnings" not in full["gpu_optimizations"]
        old.test_log.lines.clear()
        new.test_log.lines.clear()


@pytest.mark.parametrize("old_torch", [False, True])
def test_cpu_only_settings_are_identical_to_the_baseline_commit(baseline_source, bridge, monkeypatch, old_torch):
    """A machine without an accelerator keeps the original CPU fallback, on new and on old torch."""
    old = _load_detection(baseline_source, "baseline")
    new = _load_detection(_current_source(), "current", bridge=bridge)
    torch = make_torch("none", old_torch=old_torch)
    old_profile, old_commands = detect(old, monkeypatch, torch)
    new_profile, new_commands = detect(new, monkeypatch, torch)
    assert legacy_fields(old, new_profile) == legacy_fields(old, old_profile)
    assert new_profile.accelerator == "none" and new_profile.accelerator_memory_gb == 0.0
    assert new_commands == old_commands
    assert as_text(strip_new_summary_keys(answers(new, new_profile))) == as_text(answers(old, old_profile))
    assert new.test_log.lines == old.test_log.lines
    assert new.AdaptiveTrainingConfig(new_profile).get_hardware_summary()["gpu"] == "CPU Only"


def _golden_profiles(module):
    """Profiles built the way existing code builds them: from the original fields only."""
    cards = [("NVIDIA GeForce RTX 3050", (8, 6)), ("NVIDIA GeForce GTX 1650", (7, 5)),
             ("NVIDIA GeForce GTX 1060", (6, 1)), ("Tesla T4", None), ("Unknown GPU", None)]
    memories = [3.8, 4.0, 6.0, 8.0, 8.5, 11.6, 12.0, 15.6, 16.0, 16.5, 24.0, 48.0, 80.0]
    hosts = [dict(system_memory_gb=32.0, available_memory_gb=24.0, container_memory_gb=None, cpu_cores=16,
                  storage_type="NVMe"),
             dict(system_memory_gb=8.0, available_memory_gb=4.0, container_memory_gb=None, cpu_cores=4,
                  storage_type="HDD"),
             dict(system_memory_gb=64.0, available_memory_gb=50.0, container_memory_gb=12.0, cpu_cores=8,
                  storage_type="SSD")]
    fixed = dict(gpu_driver_version="550.54.14", cpu_threads=16, cpu_freq_max=3.6, cpu_architecture="x86_64",
                 available_storage_gb=100.0, os_type="Linux", python_version="3.11.9", is_container=True,
                 is_kubernetes=False)
    profiles = []
    for index, (memory, (name, capability)) in enumerate(itertools.product(memories, cards)):
        profiles.append(module.HardwareProfile(gpu_count=1, gpu_memory_gb=memory, gpu_name=name,
                                               gpu_compute_capability=capability, cuda_available=True,
                                               **hosts[index % len(hosts)], **fixed))
    for host in hosts:  # and the machine with no GPU at all
        profiles.append(module.HardwareProfile(gpu_count=0, gpu_memory_gb=0.0, gpu_name="No GPU",
                                               gpu_compute_capability=None, cuda_available=False, **host, **fixed))
    return profiles


def _golden_answers(module, strip):
    out = []
    for profile in _golden_profiles(module):
        result = answers(module, profile)
        out.append(strip_new_summary_keys(result) if strip else result)
    return out


def _digest(value):
    return hashlib.sha256(as_text(value).encode()).hexdigest()


@pytest.mark.parametrize("variant", list(UNSLOTH_VARIANTS))
def test_nvidia_settings_match_the_pinned_baseline_digest(bridge, variant):
    """The same guarantee without needing the baseline commit: 68 hand-built profiles, one digest."""
    new = _load_detection(_current_source(), "current", UNSLOTH_VARIANTS[variant], bridge=bridge)
    assert _digest(_golden_answers(new, strip=True)) == PINNED_BASELINE_DIGESTS[variant]


@pytest.mark.parametrize("variant", list(UNSLOTH_VARIANTS))
def test_the_pinned_digest_is_the_baseline_commits(baseline_source, bridge, variant):
    """The pinned digests are not just a snapshot of today's code: the baseline commit produces them."""
    old = _load_detection(baseline_source, "baseline", UNSLOTH_VARIANTS[variant])
    new = _load_detection(_current_source(), "current", UNSLOTH_VARIANTS[variant], bridge=bridge)
    old_answers, new_answers = _golden_answers(old, strip=False), _golden_answers(new, strip=True)
    for index, (was, now) in enumerate(zip(old_answers, new_answers)):
        assert now == was, f"profile {index}"
    assert _digest(old_answers) == PINNED_BASELINE_DIGESTS[variant]
    assert new.test_log.lines == old.test_log.lines


def test_profiles_built_from_the_original_fields_still_work(detection):
    nvidia, cpu = _golden_profiles(detection)[0], _golden_profiles(detection)[-1]
    assert (nvidia.accelerator, nvidia.accelerator_memory_gb, nvidia.accelerator_name) == \
        ("cuda", 3.8, "NVIDIA GeForce RTX 3050")
    assert (cpu.accelerator, cpu.accelerator_memory_gb, cpu.accelerator_name) == ("none", 0.0, "")
    assert nvidia.accelerator_notes == [] and nvidia.bf16_supported is None
    # A profile-like object from before the new fields existed is read the same way.
    duck = SimpleNamespace(**legacy_fields(detection, nvidia))
    for name in ("accelerator", "accelerator_memory_gb", "accelerator_name", "bf16_supported", "accelerator_notes"):
        delattr(duck, name)
    config = detection.AdaptiveTrainingConfig(duck)
    assert config._get_gpu_optimizations("qwen") == detection.AdaptiveTrainingConfig(nvidia)._get_gpu_optimizations("qwen")
    assert config.get_hardware_summary()["accelerator"] == "cuda" and config.warnings == []


# ---- 2a. Apple MPS (simulated) --------------------------------------------------------

def mps_machine(detection, monkeypatch, ram_gb=16.0, available_gb=12.0):
    return detect(detection, monkeypatch, make_torch("mps"), container_gb=None, ram_gb=ram_gb,
                  available_gb=available_gb)


def test_mps_is_detected_with_a_stated_share_of_unified_memory(detection, monkeypatch):
    monkeypatch.delenv("FORGELM_MPS_MEMORY_FRACTION", raising=False)
    profile, commands = mps_machine(detection, monkeypatch)
    assert profile.accelerator == "mps" and profile.cuda_available is False
    assert profile.accelerator_memory_gb == pytest.approx(16.0 * 0.65)
    assert profile.accelerator_name == "Apple M2"
    # The fields the rest of the service reports are filled too.
    assert (profile.gpu_name, profile.gpu_count) == ("Apple M2", 1)
    assert profile.gpu_memory_gb == pytest.approx(10.4)
    note = profile.accelerator_notes[0]
    assert "65%" in note and "16.0GB" in note and "10.4GB" in note and "FORGELM_MPS_MEMORY_FRACTION" in note
    assert not any(c[0] == "nvidia-smi" for c in commands)  # there is none on a Mac
    assert not any(level == "warning" for level, _ in detection.test_log.lines)


def test_mps_memory_share_can_be_overridden_and_bad_values_are_ignored(detection, monkeypatch):
    monkeypatch.setenv("FORGELM_MPS_MEMORY_FRACTION", "0.5")
    assert mps_machine(detection, monkeypatch)[0].accelerator_memory_gb == pytest.approx(8.0)
    for bad in ("0", "1.5", "lots", "-0.2"):
        monkeypatch.setenv("FORGELM_MPS_MEMORY_FRACTION", bad)
        assert mps_machine(detection, monkeypatch)[0].accelerator_memory_gb == pytest.approx(10.4)
        assert any("FORGELM_MPS_MEMORY_FRACTION" in line for level, line in detection.test_log.lines if level == "warning")


def test_mps_gets_an_accelerator_configuration_not_the_cpu_fallback(detection, monkeypatch):
    monkeypatch.delenv("FORGELM_MPS_MEMORY_FRACTION", raising=False)
    profile, _ = mps_machine(detection, monkeypatch)
    config = detection.AdaptiveTrainingConfig(profile)
    opts = config._get_gpu_optimizations(SMALL)
    assert opts == {"enabled": True, "device_map": "mps", "torch_dtype": "float16", "load_in_4bit": False,
                    "load_in_8bit": False, "flash_attention": False, "gradient_checkpointing": True,
                    "accelerator": "mps", "device": "mps", "load_bits": 16, "warnings": []}

    full = config.generate_optimal_config(request(model_name=SMALL, batch_size=8))
    assert full["accelerator"] == "mps" and full["gpu_optimizations"] == opts
    # 10.4 GB budgeted, 12 GB of RAM free: the budget is the limit, and it is not discounted twice.
    assert full["effective_memory_gb"] == pytest.approx(10.4)
    # Sized by the same tiers as a GPU with that much memory: the 12 GB training tier ...
    args = full["training_args"]
    assert (args["per_device_train_batch_size"], args["gradient_accumulation_steps"], args["max_seq_length"]) == \
        (2, 12, 1536)
    assert args["dataloader_pin_memory"] is False
    # ... and the LoRA and sequence-length settings the tiers give for 10.4 GB.
    assert full["lora_config"] == config._get_adaptive_lora_config(10.4, SMALL) and full["lora_config"]["r"] == 32
    assert full["model_loading"]["max_seq_length"] == 2048
    precision = full["precision_info"]
    assert precision["optimal_precision"] == "float16" and precision["fallback_precision"] == "float16"
    assert precision["hardware_support"] == {"bfloat16": False, "float16": True, "float32": True}
    assert any("65%" in w for w in full["warnings"]) and any("bitsandbytes" in w for w in full["warnings"])
    # No log line talks about an NVIDIA card on a Mac.
    assert not any("RTX" in line for _, line in detection.test_log.lines)
    enhanced = config.generate_enhanced_adaptive_config(request(model_name=SMALL), "chat")
    assert enhanced["advanced_optimizations"]["use_flash_attention"] is False


def test_mps_free_memory_limits_training_because_the_pool_is_shared(detection, monkeypatch):
    profile, _ = mps_machine(detection, monkeypatch, ram_gb=16.0, available_gb=5.0)
    full = detection.AdaptiveTrainingConfig(profile).generate_optimal_config(request(model_name=SMALL))
    assert full["effective_memory_gb"] == pytest.approx(5.0)
    assert full["lora_config"]["r"] == 20  # the 8 GB tier for a Qwen model


def test_mps_warns_when_the_model_cannot_fit_since_there_is_no_quantised_fallback(detection, monkeypatch):
    profile, _ = mps_machine(detection, monkeypatch)
    config = detection.AdaptiveTrainingConfig(profile)
    opts = config._get_gpu_optimizations(LARGE)
    assert opts["load_in_4bit"] is False and opts["load_in_8bit"] is False
    assert len(opts["warnings"]) == 1 and "out-of-memory" in opts["warnings"][0] and "8B" in opts["warnings"][0]
    assert opts["warnings"][0] in config.get_hardware_summary()["warnings"]
    # Asking about a model that fits afterwards clears the model-specific warning.
    assert config._get_gpu_optimizations(SMALL)["warnings"] == []
    assert not any("out-of-memory" in w for w in config.get_hardware_summary()["warnings"])
    # A size the caller knows wins over the name.
    assert config._get_gpu_optimizations(UNSIZED)["warnings"] == []
    assert config._get_gpu_optimizations(UNSIZED, params_billion=8.0)["warnings"]


def test_mps_summary_describes_the_device(detection, monkeypatch):
    monkeypatch.delenv("FORGELM_MPS_MEMORY_FRACTION", raising=False)
    profile, _ = mps_machine(detection, monkeypatch)
    summary = detection.AdaptiveTrainingConfig(profile).get_hardware_summary()
    assert summary["gpu"] == "Apple M2 (Apple MPS, 10.4GB budgeted of 16.0GB unified memory)"
    assert summary["accelerator"] == "mps"
    assert summary["precision_support"] == {"bfloat16": False, "float16": True}
    assert len(summary["warnings"]) == 2 and summary["warnings"][0] == profile.accelerator_notes[0]
    for key in ("memory", "container_limit", "cpu", "storage", "environment"):  # every original key is still there
        assert key in summary


# ---- 2b. Intel XPU (simulated) --------------------------------------------------------

ARC = "Intel(R) Arc(TM) A770 Graphics"


def test_xpu_gets_an_accelerator_configuration_not_the_cpu_fallback(detection, monkeypatch):
    profile, commands = detect(detection, monkeypatch, make_torch("xpu", ARC, 16.0), container_gb=None)
    assert profile.accelerator == "xpu" and profile.cuda_available is False
    assert (profile.accelerator_memory_gb, profile.accelerator_name) == (16.0, ARC)
    assert (profile.gpu_memory_gb, profile.gpu_name, profile.gpu_count) == (16.0, ARC, 1)
    assert not any(c[0] == "nvidia-smi" for c in commands)

    config = detection.AdaptiveTrainingConfig(profile)
    opts = config._get_gpu_optimizations(SMALL)
    assert opts == {"enabled": True, "device_map": "xpu", "torch_dtype": "bfloat16", "load_in_4bit": False,
                    "load_in_8bit": False, "flash_attention": False, "gradient_checkpointing": True,
                    "accelerator": "xpu", "device": "xpu", "load_bits": 16, "warnings": []}
    full = config.generate_optimal_config(request(model_name=SMALL, batch_size=8))
    # Dedicated memory: the same 85% share and the same tiers as a 16 GB NVIDIA card.
    assert full["effective_memory_gb"] == pytest.approx(16.0 * 0.85)
    args = full["training_args"]
    assert (args["per_device_train_batch_size"], args["gradient_accumulation_steps"], args["max_seq_length"]) == \
        (4, 8, 2048)
    assert args["dataloader_pin_memory"] is False
    assert full["lora_config"] == config._get_adaptive_lora_config(13.6, SMALL)
    assert full["precision_info"]["optimal_precision"] == "bfloat16"
    assert full["precision_info"]["hardware_support"] == {"bfloat16": True, "float16": True, "float32": True}

    summary = config.get_hardware_summary()
    assert summary["gpu"] == f"{ARC} (Intel XPU, 16.0GB)" and summary["accelerator"] == "xpu"
    assert any("float64" in w for w in summary["warnings"]) and any("bitsandbytes" in w for w in summary["warnings"])
    assert config._get_gpu_optimizations(LARGE)["warnings"]  # 8B at 16 bits does not fit in 16 GB


def test_xpu_memory_is_read_however_the_torch_build_offers_it(detection, monkeypatch):
    profile, _ = detect(detection, monkeypatch, make_torch("xpu", ARC, 8.0, xpu_memory="mem_get_info"))
    assert profile.accelerator_memory_gb == 8.0 and profile.accelerator_notes == []

    # A build that reports no memory at all: nothing raises, the smallest tier is used, and it says so.
    profile, _ = detect(detection, monkeypatch, make_torch("xpu", ARC, 8.0, xpu_memory="none"), container_gb=None)
    assert profile.accelerator == "xpu" and profile.accelerator_memory_gb == 0.0
    assert "does not report" in profile.accelerator_notes[0]
    config = detection.AdaptiveTrainingConfig(profile)
    full = config.generate_optimal_config(request(model_name=SMALL))
    assert full["gpu_optimizations"]["enabled"] is True and full["effective_memory_gb"] == 0.0
    assert full["training_args"]["per_device_train_batch_size"] == 1 and full["lora_config"]["r"] == 8
    summary = config.get_hardware_summary()
    assert profile.accelerator_notes[0] in summary["warnings"]
    assert summary["gpu"] == f"{ARC} (Intel XPU, memory not reported)"


def test_detection_survives_a_torch_without_the_newer_attributes(detection, monkeypatch):
    """No torch.version.hip, no torch.xpu, no torch.backends.mps: an NVIDIA card is still an NVIDIA card."""
    profile, _ = detect(detection, monkeypatch, make_torch("cuda", "NVIDIA GeForce RTX 3060", 12.0, old_torch=True))
    assert profile.accelerator == "cuda" and profile.gpu_memory_gb == 12.0
    profile, _ = detect(detection, monkeypatch, make_torch("none", old_torch=True))
    assert profile.accelerator == "none"
    # And one where asking raises instead of answering.
    broken = make_torch("none")
    broken.xpu.is_available = lambda: 1 / 0
    broken.backends.mps.is_available = lambda: 1 / 0
    profile, _ = detect(detection, monkeypatch, broken)
    assert profile.accelerator == "none"


# ---- 2c. AMD ROCm (simulated) ---------------------------------------------------------

RADEON = "AMD Radeon RX 7800 XT"


def rocm_config(detection, monkeypatch, memory_gb=16.0, bf16=True, allow_4bit=None):
    if allow_4bit is None:
        monkeypatch.delenv("FORGELM_ROCM_ALLOW_4BIT", raising=False)
    else:
        monkeypatch.setenv("FORGELM_ROCM_ALLOW_4BIT", allow_4bit)
    # capability=None: a ROCm build has no NVIDIA compute capability to report.
    profile, commands = detect(detection, monkeypatch, make_torch("rocm", RADEON, memory_gb, None, bf16=bf16),
                               container_gb=None)
    return detection.AdaptiveTrainingConfig(profile), profile, commands


def nvidia_config(detection, monkeypatch, memory_gb):
    profile, _ = detect(detection, monkeypatch, make_torch("cuda", "NVIDIA GeForce RTX 4080", memory_gb, (8, 9)),
                        container_gb=None)
    return detection.AdaptiveTrainingConfig(profile)


def test_rocm_is_told_apart_from_nvidia(detection, monkeypatch):
    config, profile, commands = rocm_config(detection, monkeypatch)
    assert profile.accelerator == "rocm" and profile.cuda_available is True  # it is reached through torch.cuda
    assert (profile.gpu_name, profile.gpu_memory_gb, profile.bf16_supported) == (RADEON, 16.0, True)
    assert not any(c[0] == "nvidia-smi" for c in commands) and profile.gpu_driver_version == "Unknown"
    summary = config.get_hardware_summary()
    assert summary["gpu"] == f"{RADEON} (AMD ROCm, 16.0GB)" and summary["accelerator"] == "rocm"
    assert summary["precision_support"] == {"bfloat16": True, "float16": True}
    assert len(summary["warnings"]) == 2
    assert "FORGELM_ROCM_ALLOW_4BIT=1" in summary["warnings"][0] and "NaN" in summary["warnings"][0]
    assert "flash attention" in summary["warnings"][1]


@pytest.mark.parametrize("memory_gb", [4.0, 8.0, 16.0])
def test_rocm_trains_a_model_that_fits_on_16_bit_weights(detection, monkeypatch, memory_gb):
    config, _, _ = rocm_config(detection, monkeypatch, memory_gb)
    tier = nvidia_config(detection, monkeypatch, memory_gb)._get_gpu_optimizations(SMALL)
    assert tier["load_in_4bit"] or tier["load_in_8bit"]  # the NVIDIA tier would quantise here
    opts = config._get_gpu_optimizations(SMALL)
    assert opts["load_in_4bit"] is False and opts["load_in_8bit"] is False and opts["load_bits"] == 16
    assert opts["gradient_checkpointing"] is True and opts["flash_attention"] is False
    assert (opts["accelerator"], opts["device"], opts["enabled"]) == ("rocm", "cuda", True)
    assert opts["torch_dtype"] == tier["torch_dtype"] and opts["device_map"] == tier["device_map"]
    assert opts["warnings"] == []
    for key in tier:  # every key the NVIDIA settings have is still there
        assert key in opts


def test_rocm_keeps_quantised_loading_with_a_warning_when_the_model_is_too_large(detection, monkeypatch):
    config, _, _ = rocm_config(detection, monkeypatch, 16.0)
    opts = config._get_gpu_optimizations(LARGE)  # 8B at 16 bits needs about 20 GB
    assert opts["load_in_8bit"] is True and opts["load_in_4bit"] is False and opts["load_bits"] == 8
    warning, = opts["warnings"]
    assert "unreliable on ROCm" in warning and "Llama-3.1-8B" in warning and "FORGELM_ROCM_ALLOW_4BIT=1" in warning
    assert warning in config.warnings and warning in config.get_hardware_summary()["warnings"]

    config, _, _ = rocm_config(detection, monkeypatch, 8.0)
    opts = config._get_gpu_optimizations("mistralai/Mistral-7B-Instruct-v0.3")
    assert opts["load_in_4bit"] is True and opts["load_bits"] == 4 and "4-bit" in opts["warnings"][0]

    full = config.generate_optimal_config(request(model_name="mistralai/Mistral-7B-Instruct-v0.3"))
    assert full["accelerator"] == "rocm" and opts["warnings"][0] in full["warnings"]
    assert any(level == "warning" and line == opts["warnings"][0] for level, line in detection.test_log.lines)


def test_rocm_keeps_quantised_loading_with_a_warning_when_the_size_is_unknown(detection, monkeypatch):
    config, _, _ = rocm_config(detection, monkeypatch, 16.0)
    opts = config._get_gpu_optimizations(UNSIZED)
    assert opts["load_in_8bit"] is True and "could not be read from its name" in opts["warnings"][0]
    assert config._get_gpu_optimizations()["warnings"]  # no name at all
    # The caller can supply the size, and the request can carry it.
    assert config._get_gpu_optimizations(UNSIZED, params_billion=0.5)["load_bits"] == 16
    full = config.generate_optimal_config(request(model_name=UNSIZED, model_params_billion=1.0))
    assert full["gpu_optimizations"]["load_bits"] == 16 and len(full["warnings"]) == 2  # hardware warnings only


def test_rocm_environment_override_restores_the_nvidia_quantisation_tiers(detection, monkeypatch):
    for memory_gb in (4.0, 8.0, 16.0, 24.0):
        tier = nvidia_config(detection, monkeypatch, memory_gb)._get_gpu_optimizations(SMALL)
        config, _, _ = rocm_config(detection, monkeypatch, memory_gb, allow_4bit="1")
        opts = config._get_gpu_optimizations(SMALL)
        assert (opts["load_in_4bit"], opts["load_in_8bit"]) == (tier["load_in_4bit"], tier["load_in_8bit"])
        assert opts["gradient_checkpointing"] == tier["gradient_checkpointing"]
        assert opts["flash_attention"] is False and opts["warnings"] == []
        assert "FORGELM_ROCM_ALLOW_4BIT is set" in config.get_hardware_summary()["warnings"][0]
    for spelling in ("true", "YES", " on "):
        config, _, _ = rocm_config(detection, monkeypatch, 8.0, allow_4bit=spelling)
        assert config._get_gpu_optimizations(SMALL)["load_in_4bit"] is True
    for spelling in ("0", "false", ""):
        config, _, _ = rocm_config(detection, monkeypatch, 8.0, allow_4bit=spelling)
        assert config._get_gpu_optimizations(SMALL)["load_in_4bit"] is False


def test_rocm_on_a_large_card_matches_the_unquantised_nvidia_tier(detection, monkeypatch):
    medium = "meta-llama/Llama-3.2-3B-Instruct"  # about 11 GB as 16-bit weights at batch 4, length 2048
    config, _, _ = rocm_config(detection, monkeypatch, 24.0)
    tier = nvidia_config(detection, monkeypatch, 24.0)._get_gpu_optimizations(medium)
    opts = config._get_gpu_optimizations(medium)
    assert {k: opts[k] for k in tier if k != "flash_attention"} == {k: v for k, v in tier.items() if k != "flash_attention"}
    assert opts["load_bits"] == 16 and opts["warnings"] == []
    too_big = config._get_gpu_optimizations("codellama/CodeLlama-13b-hf")
    assert too_big["load_in_4bit"] is False and "20.4GB of 24.0GB" in too_big["warnings"][0]


@pytest.mark.parametrize("memory_gb", [4.0, 8.0, 12.0, 16.0, 24.0])
def test_rocm_keeps_the_tier_logic_for_everything_but_loading(detection, monkeypatch, memory_gb):
    """Batch size, accumulation, sequence length and LoRA come from the same tiers as an NVIDIA card."""
    r = request(model_name=SMALL, batch_size=8)
    nvidia = nvidia_config(detection, monkeypatch, memory_gb).generate_optimal_config(r)
    config, _, _ = rocm_config(detection, monkeypatch, memory_gb)
    detection.test_log.lines.clear()
    rocm = config.generate_optimal_config(r)
    for key in ("hardware_profile", "effective_memory_gb", "training_args", "model_loading", "lora_config",
                "system_optimizations"):
        assert rocm[key] == nvidia[key], key
    assert list(rocm)[:len(nvidia)] == list(nvidia) and list(rocm)[len(nvidia):] == ["accelerator", "warnings"]
    # The log lines that name NVIDIA cards are not written for a Radeon.
    assert detection.test_log.lines and not any("RTX" in line for _, line in detection.test_log.lines)


def test_rocm_precision_follows_what_torch_reports(detection, monkeypatch):
    config, _, _ = rocm_config(detection, monkeypatch, 16.0, bf16=True)
    assert config._get_gpu_optimizations(SMALL)["torch_dtype"] == "bfloat16"
    config, _, _ = rocm_config(detection, monkeypatch, 16.0, bf16=False)
    assert config._get_gpu_optimizations(SMALL)["torch_dtype"] == "float16"
    assert config.get_hardware_summary()["precision_support"] == {"bfloat16": False, "float16": True}


def test_rocm_decision_does_not_need_forgelm(detection, bridge, monkeypatch):
    """Without forgelm the bridge has no estimate, and a coarser rule in the module decides."""
    config, _, _ = rocm_config(detection, monkeypatch, 16.0)
    with_forgelm = config._lora_16bit_memory_gb(3.0, 2, 1536)
    monkeypatch.setattr(bridge, "FORGELM_AVAILABLE", False)
    assert bridge.estimate_memory_gb(3.0) is None
    without = config._lora_16bit_memory_gb(3.0, 2, 1536)
    assert without == pytest.approx(3.0 * 2.0 * 1.5 + 0.8) and without >= with_forgelm  # the cautious side
    assert config._get_gpu_optimizations(SMALL)["load_bits"] == 16
    assert config._get_gpu_optimizations(LARGE)["load_bits"] == 8


def test_unsloth_enhancements_do_not_claim_flash_attention_off_nvidia(bridge, monkeypatch):
    """With Unsloth installed, the enhancement block agrees with flash_attention being off; NVIDIA keeps it."""
    module = _load_detection(_current_source(), "current_unsloth", unsloth_bf16=True, bridge=bridge)
    monkeypatch.delenv("FORGELM_ROCM_ALLOW_4BIT", raising=False)
    for torch, expected in ((make_torch("cuda", "NVIDIA GeForce RTX 4090", 24.0), "flash_attention_2"),
                            (make_torch("rocm", RADEON, 16.0, None), "sdpa"), (make_torch("mps"), "sdpa"),
                            (make_torch("xpu", ARC, 16.0), "sdpa")):
        profile, _ = detect(module, monkeypatch, torch, container_gb=None)
        config = module.AdaptiveTrainingConfig(profile)
        for kind in ("instruction", "chat", "reasoning"):
            enhanced = config.generate_enhanced_adaptive_config(request(model_name=SMALL), kind)
            assert enhanced["unsloth_enhancements"]["unsloth_attention_implementation"] == expected
            assert enhanced["advanced_optimizations"]["use_flash_attention"] is (expected == "flash_attention_2")


def test_module_works_when_the_bridge_cannot_be_imported(monkeypatch):
    """The bridge is optional: without it float32 is reported, which every device supports."""
    module = _load_detection(_current_source(), "no_bridge")  # "from utils import factory_bridge" fails
    assert module._bridge is None
    profile, _ = detect(module, monkeypatch, make_torch("mps"), ram_gb=16.0, available_gb=12.0)
    opts = module.AdaptiveTrainingConfig(profile)._get_gpu_optimizations(SMALL)
    assert (opts["enabled"], opts["torch_dtype"], opts["load_bits"]) == (True, "float32", 32)


@pytest.mark.parametrize("name,expected", [
    ("Qwen/Qwen2.5-0.5B-Instruct", 0.5), ("unsloth/Llama-3.2-3B-Instruct-bnb-4bit", 3.0),
    ("HuggingFaceTB/SmolLM2-135M", 0.135), ("google/gemma-2-9b-it", 9.0), ("TinyLlama-1.1B-Chat", 1.1),
    ("codellama/CodeLlama-13b-hf", 13.0), ("/models/mistral-7b-instruct", 7.0), ("llama3.2:3b", 3.0),
    ("mistralai/Mixtral-8x7B-v0.1", None), ("microsoft/phi-3-mini-4k-instruct", None), ("gpt2", None),
    ("model-4bit", None), ("", None), (None, None),
])
def test_model_size_is_read_from_the_name_only_when_it_is_unambiguous(detection, name, expected):
    assert detection.AdaptiveTrainingConfig._params_billion_from_name(name) == expected


def test_warnings_for_the_running_machine_reuse_the_last_detection(detection, monkeypatch):
    install_machine(detection, monkeypatch, make_torch("rocm", RADEON, 16.0, None))
    monkeypatch.delenv("FORGELM_ROCM_ALLOW_4BIT", raising=False)
    detector = detection.hardware_detector
    assert detector.last_profile is None
    warnings = detection.current_hardware_warnings()  # no profile yet: detects once
    assert len(warnings) == 2 and detector.last_profile.accelerator == "rocm"
    detector.detect_full_profile = lambda: pytest.fail("detection ran again")
    assert detection.current_hardware_warnings() == warnings


# ---- 4. the bridge ---------------------------------------------------------------------

def test_bridge_dtype_rule_is_unchanged_and_matches_the_dry_runs_copy(bridge, monkeypatch):
    from forgelm import dryrun

    for accelerator, bf16 in itertools.product(("cuda", "rocm", "xpu", "mps", "cpu", "none", "tpu"), (False, True)):
        assert bridge.dtype_name_for(accelerator, bf16) == dryrun.dtype_name_for(accelerator, bf16)

    # torch_device_and_dtype gives what it gave before the rule was pulled out of it.
    def fake(kind, bf16=True):
        torch = make_torch(kind, bf16=bf16)
        torch.bfloat16, torch.float16, torch.float32 = "bf16", "fp16", "fp32"
        return torch

    expected = {("cuda", True): ("cuda", "bf16"), ("cuda", False): ("cuda", "fp16"), ("rocm", True): ("cuda", "bf16"),
                ("xpu", True): ("xpu", "bf16"), ("mps", True): ("mps", "fp16"), ("none", True): ("cpu", "fp32")}
    for (kind, bf16), answer in expected.items():
        monkeypatch.setitem(sys.modules, "torch", fake(kind, bf16))
        assert bridge.torch_device_and_dtype() == answer


def test_bridge_dtype_and_detector_agree_on_every_accelerator(detection, bridge, monkeypatch):
    """The precision the detector reports is the precision the loader will use."""
    for kind, name, memory in (("mps", "", 0.0), ("xpu", ARC, 16.0)):
        torch = make_torch(kind, name, memory)
        torch.bfloat16, torch.float16, torch.float32 = "bfloat16", "float16", "float32"
        profile, _ = detect(detection, monkeypatch, torch, container_gb=None)
        monkeypatch.setitem(sys.modules, "torch", torch)
        device, dtype = bridge.torch_device_and_dtype()
        opts = detection.AdaptiveTrainingConfig(profile)._get_gpu_optimizations(SMALL)
        assert (opts["device"], opts["torch_dtype"]) == (device, dtype) == (profile.accelerator, dtype)


def test_bridge_backend_advice_carries_the_detectors_warnings(bridge, monkeypatch):
    # In this process there is no service "utils" package, so there is nothing to ask.
    assert bridge.backend_advice()["detector_warnings"] == []

    utils = types.ModuleType("utils")
    utils.__path__ = []
    stub = types.ModuleType("utils.hardware_detection")
    stub.current_hardware_warnings = lambda: ["AMD ROCm: something to know"]
    utils.hardware_detection = stub
    with _stand_ins({"utils": utils, "utils.hardware_detection": stub}):
        assert bridge.backend_advice()["detector_warnings"] == ["AMD ROCm: something to know"]
        stub.current_hardware_warnings = lambda: object()  # a stubbed-out detector must not leak into the API
        assert bridge.backend_advice()["detector_warnings"] == []
        stub.current_hardware_warnings = lambda: 1 / 0
        assert bridge.backend_advice()["detector_warnings"] == []

    monkeypatch.setattr(bridge, "FORGELM_AVAILABLE", False)
    assert bridge.backend_advice() == {"available": False, "reason": "forgelm package not installed in this image"}
    assert bridge.training_dry_run(object()) == {"available": False,
                                                 "reason": "forgelm package not installed in this image"}


def test_bridge_backend_advice_with_the_real_detector(detection, bridge, monkeypatch):
    """The bridge and the real module together, on a simulated ROCm machine."""
    install_machine(detection, monkeypatch, make_torch("rocm", RADEON, 16.0, None))
    monkeypatch.delenv("FORGELM_ROCM_ALLOW_4BIT", raising=False)
    utils = types.ModuleType("utils")
    utils.__path__ = []
    utils.hardware_detection = detection
    with _stand_ins({"utils": utils, "utils.hardware_detection": detection}):
        warnings = bridge.backend_advice()["detector_warnings"]
    assert len(warnings) == 2 and all(w.startswith("AMD ROCm") for w in warnings)


# ---- 3. the dry run (real steps, on CPU) ---------------------------------------------------

@pytest.fixture
def tiny_gpt2():
    """A two-layer GPT-2 with random weights, built from a config: nothing is downloaded."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=500, n_positions=64,
                                     bos_token_id=0, eos_token_id=0)
    return transformers.GPT2LMHeadModel(config)


def state_of(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def same_state(model, state):
    import torch

    now = model.state_dict()
    return list(now) == list(state) and all(torch.equal(now[name], state[name]) for name in state)


GPT2_LORA = {"r": 4, "lora_alpha": 8, "target_modules": ["c_attn"], "fan_in_fan_out": True}


def check_measurement(result, steps):
    assert result["completed"] is True and result["out_of_memory"] is False and result["error"] is None
    assert result["seconds_per_step"] > 0 and len(result["step_seconds"]) == steps == result["steps"]
    assert all(s > 0 for s in result["step_seconds"])
    for key in ("peak_gb", "total_gb", "peak_increase_gb", "baseline_gb", "model_weights_gb"):
        assert math.isfinite(result[key]) and result[key] >= 0, key
    assert result["device"] == "cpu" and result["accelerator"] == "cpu"
    assert result["measure"].startswith("process RSS growth")
    # On a CPU the total is the weights, which were resident before the baseline, plus the growth.
    assert result["total_gb"] == pytest.approx(result["model_weights_gb"] + result["peak_gb"], abs=2e-4)
    assert math.isfinite(result["loss_first"]) and math.isfinite(result["loss_last"])


def test_dry_run_takes_real_steps_on_cpu(tiny_gpt2):
    from forgelm import dryrun

    before = state_of(tiny_gpt2)
    tiny_gpt2.eval()
    result = dryrun.measure_training_memory(tiny_gpt2, 500, batch_size=2, seq_len=32, steps=3)
    check_measurement(result, 3)
    assert result["weights_changed"] is True and not same_state(tiny_gpt2, before)  # AdamW really stepped
    assert result["restored"] is False and result["lora"] is False
    assert result["trainable_params"] == result["total_params"] == sum(p.numel() for p in tiny_gpt2.parameters())
    assert (result["batch_size"], result["seq_len"], result["weight_bits"]) == (2, 32, 32)
    assert tiny_gpt2.training is False  # left in the mode it was found in
    assert all(p.grad is None for p in tiny_gpt2.parameters())
    assert json.dumps(result)  # plain data, ready for an API response


def test_dry_run_with_lora_trains_the_adapter_and_leaves_the_base_model_alone(tiny_gpt2):
    pytest.importorskip("peft")
    from forgelm import dryrun

    before = state_of(tiny_gpt2)
    trainable_before = [p.requires_grad for p in tiny_gpt2.parameters()]
    result = dryrun.measure_training_memory(tiny_gpt2, None, batch_size=2, seq_len=32, steps=3, lora=GPT2_LORA)
    check_measurement(result, 3)
    assert result["lora"] is True and result["weights_changed"] is True and result["restored"] is True
    assert 0 < result["trainable_params"] < result["total_params"]
    # The adapter is gone, the base weights are bit-for-bit what they were, and they are trainable again.
    assert not [name for name, _ in tiny_gpt2.named_modules() if "lora" in name.lower()]
    assert same_state(tiny_gpt2, before)
    assert [p.requires_grad for p in tiny_gpt2.parameters()] == trainable_before


def test_dry_run_on_a_model_that_already_carries_an_adapter_changes_the_adapter(tiny_gpt2):
    """What the service has in hand: a PEFT model. The steps must move its adapter weights."""
    peft = pytest.importorskip("peft")
    import torch
    from forgelm import dryrun

    base_before = state_of(tiny_gpt2)
    model = peft.get_peft_model(tiny_gpt2, peft.LoraConfig(task_type="CAUSAL_LM", **GPT2_LORA))
    adapter_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    assert adapter_before and all("lora" in n for n in adapter_before)
    result = dryrun.measure_training_memory(model, 500, batch_size=1, seq_len=16, steps=2)
    check_measurement(result, 2)
    adapter_after = dict(model.named_parameters())
    assert any(not torch.equal(adapter_after[n], v) for n, v in adapter_before.items())
    assert all(torch.equal(model.state_dict()[f"base_model.model.{n}"], v) for n, v in base_before.items()
               if f"base_model.model.{n}" in model.state_dict())

    # With restore=True the steps are still real, and the adapter is put back afterwards.
    adapter_now = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    result = dryrun.measure_training_memory(model, 500, batch_size=1, seq_len=16, steps=2, restore=True)
    assert result["weights_changed"] is True and result["restored"] is True
    assert all(torch.equal(dict(model.named_parameters())[n], v) for n, v in adapter_now.items())


def test_dry_run_loads_a_model_by_id_and_frees_it(tiny_gpt2, tmp_path):
    from forgelm import dryrun

    tiny_gpt2.save_pretrained(tmp_path / "tiny")  # a local path stands in for a hub id: no network
    result = dryrun.measure_training_memory(str(tmp_path / "tiny"), None, batch_size=2, seq_len=16, steps=2)
    check_measurement(result, 2)
    assert result["model"] == str(tmp_path / "tiny") and result["model_resident"] is False
    assert result["weights_changed"] is True


def test_dry_run_under_autocast_and_with_a_plain_module(tiny_gpt2):
    import torch
    from forgelm import dryrun

    result = dryrun.measure_training_memory(tiny_gpt2, 500, batch_size=1, seq_len=16, steps=2, amp_dtype="bf16")
    check_measurement(result, 2)
    assert result["amp_dtype"] == "bf16" and result["weights_changed"] is True

    class IdsToLogits(torch.nn.Module):  # not a Hugging Face model: no labels argument, returns logits
        def __init__(self):
            super().__init__()
            self.embed, self.out = torch.nn.Embedding(50, 8), torch.nn.Linear(8, 50)

        def forward(self, ids):
            return self.out(self.embed(ids))

    result = dryrun.measure_training_memory(IdsToLogits(), 50, batch_size=2, seq_len=8, steps=2)
    check_measurement(result, 2)
    assert result["weights_changed"] is True
    # A tokenizer is accepted in place of the number; the embedding size caps it either way.
    assert dryrun._vocab_size(tiny_gpt2, SimpleNamespace(vocab_size=400)) == 400
    assert dryrun._vocab_size(tiny_gpt2, [0] * 9000) == 500


def test_dry_run_rejects_inputs_it_cannot_run(tiny_gpt2):
    from forgelm import dryrun

    with pytest.raises(ValueError, match="longer than the model's 64 positions"):
        dryrun.measure_training_memory(tiny_gpt2, 500, batch_size=1, seq_len=128)
    for bad in (dict(batch_size=0, seq_len=8), dict(batch_size=1, seq_len=1), dict(batch_size=1, seq_len=8, steps=0),
                dict(batch_size=True, seq_len=8), dict(batch_size=1, seq_len=8, amp_dtype="fp8")):
        with pytest.raises(ValueError):
            dryrun.measure_training_memory(tiny_gpt2, 500, **bad)
    for p in tiny_gpt2.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="nothing in the model is trainable"):
        dryrun.measure_training_memory(tiny_gpt2, 500, batch_size=1, seq_len=8)


def test_dry_run_reports_out_of_memory_and_raises_anything_else():
    torch = pytest.importorskip("torch")
    from forgelm import dryrun

    class Fails(torch.nn.Module):
        def __init__(self, message, on_call):
            super().__init__()
            self.embed, self.out = torch.nn.Embedding(50, 8), torch.nn.Linear(8, 50)
            self.message, self.on_call, self.calls = message, on_call, 0

        def forward(self, ids):
            self.calls += 1
            if self.calls == self.on_call:
                raise RuntimeError(self.message)
            return self.out(self.embed(ids))

    model = Fails("CUDA out of memory. Tried to allocate 2.00 GiB", on_call=2)
    result = dryrun.measure_training_memory(model, 50, batch_size=2, seq_len=8, steps=3)
    assert result["completed"] is False and result["out_of_memory"] is True
    assert "out of memory" in result["error"] and result["steps"] == 1 and result["steps_requested"] == 3
    assert result["weights_changed"] is True  # the first step went through
    report = dryrun.compare_with_estimate(result, available_gb=1000.0)
    assert report["verdict"] == "does_not_fit" and report["measured_fits"] is False
    assert "ran out of memory" in report["summary"]

    with pytest.raises(RuntimeError, match="shape mismatch"):
        dryrun.measure_training_memory(Fails("shape mismatch", on_call=1), 50, batch_size=2, seq_len=8)


def test_fits():
    from forgelm.dryrun import fits

    assert fits(8.9, 10.0) and fits(9.0, 10.0) and not fits(9.1, 10.0)  # 90% headroom by default
    assert fits(9.5, 10.0, headroom=1.0) and not fits(5.1, 10.0, headroom=0.5)
    assert fits(0.0, 0.0) and not fits(0.1, 0.0)
    for bad in ((float("nan"), 10.0), (-1.0, 10.0), (1.0, float("inf")), (1.0, -1.0), (None, 10.0)):
        with pytest.raises(ValueError):
            fits(*bad)
    for headroom in (0.0, 1.1, -0.5):
        with pytest.raises(ValueError):
            fits(1.0, 10.0, headroom=headroom)


def test_estimate_measurement_and_verdict_are_reported_together(tiny_gpt2):
    from forgelm import dryrun
    from forgelm.generative import estimate_memory_gb

    result = dryrun.measure_training_memory(tiny_gpt2, 500, batch_size=2, seq_len=32, steps=2)
    report = dryrun.compare_with_estimate(result, available_gb=4.0)
    # The estimate is forgelm.generative's, asked about this model at this batch and length.
    assert report["estimate"] == estimate_memory_gb(result["total_params"] / 1e9, 32, 2, 32, True, True)
    assert report["measurement"] is result and report["available_gb"] == 4.0 and report["headroom"] == 0.9
    assert report["verdict"] == "fits" and report["measured_fits"] is True and report["estimate_fits"] is True
    assert report["estimate_minus_measured_gb"] == pytest.approx(report["estimate"]["total_gb"] - result["total_gb"],
                                                                 abs=0.01)
    assert "estimate 0.80 GiB" in report["summary"] and report["summary"].endswith(": fits")

    # The verdict follows the measurement, not the estimate.
    tight = dryrun.compare_with_estimate(result, available_gb=result["total_gb"] / 2)
    assert tight["verdict"] == "does_not_fit" and tight["estimate_fits"] is False
    between = dryrun.compare_with_estimate(result, available_gb=0.5)
    assert between["verdict"] == "fits" and between["estimate_fits"] is False  # 0.8 GiB estimated, far less used

    # Without a figure from the caller, what the machine has now is used, and the report says which.
    automatic = dryrun.compare_with_estimate(result)
    assert automatic["available_gb"] > 0 and "RAM available now" in automatic["available_basis"]
    assert "plus the loaded weights" in automatic["available_basis"]

    # An adapter-only run is estimated as LoRA, a caller's parameter count and bit width win.
    lora_like = dict(result, trainable_params=10)
    assert dryrun.compare_with_estimate(lora_like, available_gb=4.0, params_billion=3.0, bits=4)["estimate"] == \
        estimate_memory_gb(3.0, 4, 2, 32, True, False)
    unmeasured = dict(result, total_gb=None)
    unknown = dryrun.compare_with_estimate(unmeasured, available_gb=4.0)
    assert unknown["verdict"] == "unknown" and unknown["measured_fits"] is None
    assert unknown["estimate_minus_measured_gb"] is None


def test_bridge_dry_run_returns_the_report_and_puts_the_weights_back(bridge, tiny_gpt2):
    before = state_of(tiny_gpt2)
    report = bridge.training_dry_run(tiny_gpt2, batch_size=2, seq_len=32, steps=2, available_gb=4.0)
    assert report["available"] is True and report["verdict"] == "fits"
    assert {"estimate", "measurement", "available_gb", "measured_fits", "estimate_fits", "summary"} <= set(report)
    check_measurement(report["measurement"], 2)
    assert report["measurement"]["weights_changed"] is True and report["measurement"]["restored"] is True
    assert same_state(tiny_gpt2, before)  # the service goes on to train the model it loaded
    assert json.dumps(report)


# ---- 3b. the accelerator memory readers, against a stand-in for torch ----------------------

class FakeDeviceCounters:
    """torch.cuda / torch.xpu memory counters for two devices."""

    def __init__(self):
        self.allocated = {0: 4 * GIB, 1: 2 * GIB}
        self.peak = dict(self.allocated)
        self.calls = []

    def synchronize(self, index=None):
        self.calls.append(("synchronize", index))

    def reset_peak_memory_stats(self, index=None):
        self.calls.append(("reset", index))
        self.peak[index] = self.allocated[index]

    def memory_allocated(self, index=None):
        return self.allocated[index]

    def max_memory_allocated(self, index=None):
        return self.peak[index]

    def max_memory_reserved(self, index=None):
        return self.peak[index] + GIB

    def use(self, index, extra):  # what a training step does to the allocator
        self.peak[index] = max(self.peak[index], self.allocated[index] + extra)


def test_cuda_and_rocm_peak_comes_from_the_allocators_own_counter():
    from forgelm.dryrun import _MemoryMeter

    for kind in ("cuda", "rocm"):
        counters = FakeDeviceCounters()
        counters.peak = {0: 9 * GIB, 1: 9 * GIB}  # an earlier peak that must not be reported
        meter = _MemoryMeter(SimpleNamespace(cuda=counters), kind, [0, 1])
        assert meter.source == "torch.cuda.max_memory_allocated"
        meter.start()
        assert ("reset", 0) in counters.calls and ("reset", 1) in counters.calls
        counters.use(0, 3 * GIB)
        meter.sample()
        counters.use(1, GIB)
        assert meter.stop() == (6 * GIB, 10 * GIB)  # both devices, summed
        assert meter.reserved_peak == 12 * GIB and meter.notes == []


def test_xpu_peak_uses_the_counter_when_the_build_has_one_and_samples_when_not():
    from forgelm.dryrun import _MemoryMeter

    counters = FakeDeviceCounters()
    meter = _MemoryMeter(SimpleNamespace(xpu=counters), "xpu", [0])
    assert meter.source == "torch.xpu.max_memory_allocated"
    meter.start()
    counters.use(0, 2 * GIB)
    assert meter.stop() == (4 * GIB, 6 * GIB)

    readings = iter([4 * GIB, 7 * GIB, 5 * GIB, 5 * GIB])
    sampled = _MemoryMeter(SimpleNamespace(xpu=SimpleNamespace(memory_allocated=lambda index: next(readings))), "xpu")
    assert sampled.source == "torch.xpu.memory_allocated (sampled)" and "can be missed" in sampled.notes[0]
    sampled.start()
    sampled.sample()
    sampled.sample()
    assert sampled.stop() == (4 * GIB, 7 * GIB)


def test_mps_peak_is_sampled_from_the_metal_driver():
    from forgelm.dryrun import _MemoryMeter

    readings = iter([3 * GIB, 5 * GIB, 8 * GIB, 6 * GIB, 6 * GIB])
    mps = SimpleNamespace(driver_allocated_memory=lambda: next(readings))
    meter = _MemoryMeter(SimpleNamespace(mps=mps), "mps")
    assert meter.source == "torch.mps.driver_allocated_memory (sampled)" and "no peak counter" in meter.notes[0]
    meter.start()
    for _ in range(3):
        meter.sample()
    assert meter.stop() == (3 * GIB, 8 * GIB)


def test_devices_without_a_memory_api_fall_back_to_process_memory_and_say_so():
    from forgelm.dryrun import _MemoryMeter

    for kind, torch in (("mps", SimpleNamespace(mps=SimpleNamespace())), ("xpu", SimpleNamespace()),
                        ("mps", SimpleNamespace())):
        meter = _MemoryMeter(torch, kind)
        assert meter.source.startswith("process RSS growth") and "process memory was measured instead" in meter.notes[0]
        meter.start()
        baseline, peak = meter.stop()
        assert peak >= baseline > 0


def test_cpu_peak_without_psutil_uses_the_resource_high_water_mark(monkeypatch):
    from forgelm import dryrun

    marks = iter([10 * GIB, 13 * GIB])
    monkeypatch.setattr(dryrun, "_process_memory_readers",
                        lambda: (None, lambda: next(marks), "process RSS growth (resource high-water mark)"))
    meter = dryrun._MemoryMeter(SimpleNamespace(), "cpu")
    assert "psutil is not installed" in meter.notes[0]
    meter.start()
    meter.sample()  # nothing to sample without psutil
    assert meter.stop() == (10 * GIB, 13 * GIB)

    # With psutil, a new high-water mark set between two samples is still caught.
    current, marks = iter([10 * GIB, 11 * GIB, 11 * GIB]), iter([12 * GIB, 15 * GIB])
    monkeypatch.setattr(dryrun, "_process_memory_readers",
                        lambda: (lambda: next(current), lambda: next(marks), "process RSS growth (psutil)"))
    meter = dryrun._MemoryMeter(SimpleNamespace(), "cpu")
    meter.start()
    meter.sample()
    assert meter.stop() == (10 * GIB, 15 * GIB) and meter.notes == []

    # Neither source (Windows without psutil): nothing is measured, and nothing is invented.
    monkeypatch.setattr(dryrun, "_process_memory_readers", lambda: (None, None, "unavailable"))
    meter = dryrun._MemoryMeter(SimpleNamespace(), "cpu")
    meter.start()
    assert meter.stop() == (None, None) and "memory was not measured" in meter.notes[0]


def test_out_of_memory_is_recognised_across_torch_versions():
    from forgelm.dryrun import _accelerator_of, _best_device, _is_out_of_memory

    class OutOfMemoryError(RuntimeError):
        pass

    new_torch = SimpleNamespace(OutOfMemoryError=OutOfMemoryError, cuda=SimpleNamespace(OutOfMemoryError=OutOfMemoryError))
    old_torch = SimpleNamespace(cuda=SimpleNamespace())
    assert _is_out_of_memory(new_torch, OutOfMemoryError("anything"))
    assert _is_out_of_memory(old_torch, RuntimeError("MPS backend out of memory (MPS allocated: 9 GB)"))
    assert not _is_out_of_memory(new_torch, RuntimeError("size mismatch"))
    assert not _is_out_of_memory(new_torch, ValueError("out of memory"))  # only runtime errors count

    assert _accelerator_of(make_torch("rocm"), "cuda") == "rocm" and _accelerator_of(make_torch("cuda"), "cuda") == "cuda"
    assert _accelerator_of(make_torch("cuda", old_torch=True), "cuda") == "cuda"
    assert [_accelerator_of(make_torch("none"), t) for t in ("mps", "xpu", "cpu", "meta")] == ["mps", "xpu", "cpu", "cpu"]
    assert [_best_device(make_torch(k)) for k in ("cuda", "rocm", "xpu", "mps", "none")] == \
        ["cuda", "cuda", "xpu", "mps", "cpu"]
    assert _best_device(make_torch("none", old_torch=True)) == "cpu"


def test_dryrun_imports_without_torch_or_any_other_heavy_package():
    """forgelm must import with only numpy installed: a child process where the rest cannot be imported."""
    code = ("import sys\n"
            "for name in ('torch', 'transformers', 'peft', 'psutil', 'sklearn', 'pandas'):\n"
            "    sys.modules[name] = None\n"
            "import forgelm.dryrun as d\n"
            "assert d.fits(1.0, 2.0) and d.dtype_name_for('mps') == 'float16'\n"
            "print('ok')\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr
