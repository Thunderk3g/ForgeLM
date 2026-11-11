"""How the training service loads a base model on each kind of hardware.

The service modules need the GPU stack to import, so the bridge is loaded by
path and torch's device queries are simulated with monkeypatch.
"""

import ast
import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[2]
SERVICES = ROOT / "src/microservices/training_service/src/services"


@pytest.fixture
def bridge():
    path = ROOT / "src/microservices/training_service/src/utils/factory_bridge.py"
    spec = importlib.util.spec_from_file_location("factory_bridge_load_plan", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _simulate(monkeypatch, *, cuda=False, hip=None, bf16=True, xpu=False, mps=False):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: bf16, raising=False)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    fake_xpu = type("XPU", (), {"is_available": staticmethod(lambda: xpu)})
    monkeypatch.setattr(torch, "xpu", fake_xpu, raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_nvidia_keeps_the_existing_loader(bridge, monkeypatch):
    _simulate(monkeypatch, cuda=True)
    plan = bridge.causal_lm_load_plan({"load_in_4bit": False})
    assert plan["accelerator"] == "cuda" and plan["kwargs"] == {} and plan["kbit"] is True
    assert plan["training_precision"] == "bf16"


def test_rocm_defaults_to_16_bit_weights(bridge, monkeypatch):
    _simulate(monkeypatch, cuda=True, hip="6.2", bf16=True)
    plan = bridge.causal_lm_load_plan({"load_in_4bit": False, "load_in_8bit": False})
    assert plan["accelerator"] == "rocm" and plan["kbit"] is False
    assert "quantization_config" not in plan["kwargs"]
    assert plan["kwargs"]["torch_dtype"] is torch.bfloat16 and plan["kwargs"]["device_map"] == "auto"
    assert plan["training_precision"] == "bf16"
    # No hardware configuration at all is treated the same way: never quantise by default.
    assert bridge.causal_lm_load_plan(None)["kbit"] is False


def test_rocm_quantises_only_when_the_configuration_asks(bridge, monkeypatch):
    _simulate(monkeypatch, cuda=True, hip="6.2", bf16=False)
    four = bridge.causal_lm_load_plan({"load_in_4bit": True})
    assert four["kbit"] is True and four["kwargs"]["quantization_config"].load_in_4bit
    assert four["training_precision"] == "fp16"
    eight = bridge.causal_lm_load_plan({"load_in_8bit": True})
    assert eight["kbit"] is True and eight["kwargs"]["quantization_config"].load_in_8bit


@pytest.mark.parametrize("device,flags,dtype,precision", [
    ("xpu", {"xpu": True}, torch.bfloat16, "bf16"),
    ("mps", {"mps": True}, torch.float16, None),
])
def test_intel_and_apple_load_on_the_device_without_bitsandbytes(bridge, monkeypatch, device, flags, dtype,
                                                                  precision):
    _simulate(monkeypatch, **flags)
    plan = bridge.causal_lm_load_plan({"load_in_4bit": True})  # ignored: no bitsandbytes here
    assert plan["accelerator"] == device and plan["move_to"] == device and plan["kbit"] is False
    assert plan["kwargs"] == {"torch_dtype": dtype}
    assert plan["training_precision"] == precision


def test_cpu_loads_float32(bridge, monkeypatch):
    _simulate(monkeypatch)
    plan = bridge.causal_lm_load_plan({"load_in_4bit": True})
    assert plan == {**plan, "accelerator": "cpu", "kbit": False, "move_to": None, "training_precision": None}
    assert plan["kwargs"] == {"torch_dtype": torch.float32}


@pytest.mark.parametrize("filename,method", [
    ("training_service.py", "_fine_tune_with_unsloth"),
    ("training_service_core.py", "_train_with_unsloth"),
])
def test_both_loaders_use_the_plan_and_guard_kbit_preparation(filename, method):
    source = (SERVICES / filename).read_text()
    func = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == method)
    body = ast.get_source_segment(source, func)
    assert "causal_lm_load_plan(" in body
    # Every call to prepare_model_for_kbit_training sits under `if load_plan["kbit"]:`.
    for node in ast.walk(func):
        if isinstance(node, ast.If) and 'load_plan["kbit"]' in ast.get_source_segment(source, node.test):
            guarded = {id(n) for n in ast.walk(node)}
            break
    else:
        pytest.fail("no `if load_plan['kbit']` guard")
    calls = [n for n in ast.walk(func) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", "")) == "prepare_model_for_kbit_training"]
    assert calls and all(id(c) in guarded for c in calls)
    # BitsAndBytesConfig is built only inside the NVIDIA branch.
    for node in ast.walk(func):
        if isinstance(node, ast.If) and 'load_plan["accelerator"] == "cuda"' in ast.get_source_segment(source,
                                                                                                    node.test):
            nvidia = {id(n) for n in ast.walk(node)}
            break
    else:
        pytest.fail("no NVIDIA branch")
    bnb = [n for n in ast.walk(func) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "BitsAndBytesConfig"]
    assert bnb and all(id(c) in nvidia for c in bnb)
