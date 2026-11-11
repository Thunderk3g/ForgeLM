import pytest

from forgelm.generative import (PEFT_METHODS, capacity, estimate_memory_gb, largest_model_that_fits,
                                lora_config, select_backend)
from forgelm.hardware import reference_profile

# Published minimum VRAM for 4-bit QLoRA (GiB): (Unsloth requirements table, LLaMA-Factory README).
PUBLISHED = {3: (3.5, 1.5), 8: (6.0, 6.0), 14: (8.5, 12.0), 32: (26.0, 24.0), 70: (41.0, 48.0)}


@pytest.mark.parametrize("params,published", PUBLISHED.items())
def test_estimate_is_inside_the_published_range_with_slack(params, published):
    est = estimate_memory_gb(params, bits=4, batch_size=1, seq_len=1024)["total_gb"]
    lo, hi = min(published), max(published)
    assert lo * 0.65 <= est <= hi * 1.35, (params, est, published)


def test_estimate_grows_with_everything_that_should_grow_it():
    base = estimate_memory_gb(8, 4, 2, 1024)["total_gb"]
    assert estimate_memory_gb(8, 16, 2, 1024)["total_gb"] > base
    assert estimate_memory_gb(8, 4, 8, 1024)["total_gb"] > base
    assert estimate_memory_gb(8, 4, 2, 4096)["total_gb"] > base
    assert estimate_memory_gb(8, 4, 2, 1024, gradient_checkpointing=False)["total_gb"] > base
    assert estimate_memory_gb(8, 16, 2, 1024, full_finetune=True)["total_gb"] > 4 * base


def test_largest_model_that_fits_is_consistent_with_the_estimate():
    fit = largest_model_that_fits(24.0)
    assert estimate_memory_gb(fit)["total_gb"] <= 24.0 * 0.85 + 0.1
    assert estimate_memory_gb(fit + 2)["total_gb"] > 24.0 * 0.85


def test_backend_per_vendor():
    assert select_backend(reference_profile("nvidia-24gb"), 8).backend == "unsloth"
    assert select_backend(reference_profile("nvidia-24gb"), 8, prefer="peft_trl").backend == "peft_trl"
    amd = select_backend(reference_profile("amd-16gb"), 3)
    assert (amd.backend, amd.device, amd.load_bits) == ("peft_trl", "rocm", 16)
    assert any("NaN" in w for w in amd.warnings)
    assert select_backend(reference_profile("amd-16gb"), 12).load_bits == 4  # does not fit at 16-bit
    apple = select_backend(reference_profile("apple-16gb"), 3)
    assert apple.backend == "mlx_lm" and apple.available_gb < 16
    assert "MLX" in apple.serve_format
    assert select_backend(reference_profile("intel-arc-16gb"), 3).device == "xpu"


def test_apple_without_mlx_suggests_installing_it():
    hw = reference_profile("apple-16gb")
    hw.packages.pop("mlx_lm")
    choice = select_backend(hw, 3)
    assert choice.backend == "peft_trl" and "mlx-lm" in choice.install_hint


def test_cpu_is_feasible_only_for_tiny_models():
    cpu = reference_profile("cpu-server")
    assert select_backend(cpu, 0.5).feasible
    big = select_backend(cpu, 8)
    assert not big.feasible and "llama.cpp" in big.serve_runtime
    assert capacity(cpu)["max_params_billion"] <= 1.5


def test_too_large_for_the_gpu_reports_what_would_fit():
    choice = select_backend(reference_profile("nvidia-4gb"), 14)
    assert not choice.feasible
    assert any("largest model that fits" in w for w in choice.warnings)


def test_multi_gpu_is_mentioned():
    choice = select_backend(reference_profile("nvidia-2x80gb"), 70)
    assert any("2 GPUs" in r for r in choice.reasons)


def test_lora_config_mapping():
    name, cfg, notes = lora_config("dora", rank=16)
    assert name == "LoraConfig" and cfg["use_dora"] is True and "use_rslora" not in cfg and notes
    _, cfg, notes = lora_config("lora", rank=64)
    assert cfg["use_rslora"] is True and cfg["lora_alpha"] == 128
    _, cfg, _ = lora_config("lora", rank=64, auto_rslora=False)
    assert "use_rslora" not in cfg
    _, cfg, _ = lora_config("pissa", rank=8, alpha=8, target_modules=["q_proj"])
    assert cfg["init_lora_weights"].startswith("pissa") and cfg["target_modules"] == ["q_proj"]
    assert lora_config("oft")[0] == "OFTConfig"
    with pytest.raises(ValueError):
        lora_config("vera")
    assert {"lora", "qlora", "dora", "pissa", "oft"} <= set(PEFT_METHODS)


@pytest.mark.parametrize("method", ["lora", "qlora", "rslora", "dora", "pissa", "olora", "oft"])
def test_configs_are_accepted_by_the_installed_peft(method):
    peft = pytest.importorskip("peft")
    name, cfg, _ = lora_config(method, rank=8)
    assert getattr(peft, name)(**cfg) is not None
