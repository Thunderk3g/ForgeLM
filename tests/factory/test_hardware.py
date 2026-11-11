import pytest

from forgelm.hardware import (REFERENCE_PROFILES, HardwareProfile, parse_cgroup_limit, parse_cpu_flags,
                              parse_cpu_model, parse_nvidia_smi, probe, reference_profile)

CPUINFO = """processor : 0
model name : Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
flags : fpu vme sse sse2 avx avx2 fma f16c avx512f avx512bw avx512_vnni
"""


def test_cpu_parsers():
    assert parse_cpu_model(CPUINFO).startswith("Intel(R) Xeon")
    flags = parse_cpu_flags(CPUINFO)
    assert "avx512_vnni" in flags and "avx2" in flags and "sse" not in flags
    assert parse_cpu_flags("nothing useful") == []


def test_arm_features_line():
    assert "asimddp" in parse_cpu_flags("Features : fp asimd asimddp sve")


def test_nvidia_smi_merges_identical_cards():
    acc = parse_nvidia_smi("NVIDIA H100, 81559\nNVIDIA H100, 81559\nGeForce RTX 3050, 4096\n")
    assert [(a.name, a.count) for a in acc] == [("NVIDIA H100", 2), ("GeForce RTX 3050", 1)]
    assert acc[1].memory_gb == pytest.approx(4.0)
    assert parse_nvidia_smi("garbage line\n") == []


def test_cgroup_limit():
    assert parse_cgroup_limit("max") is None
    assert parse_cgroup_limit("9223372036854771712") is None  # cgroup v1 "unlimited"
    assert parse_cgroup_limit("8589934592") == 8 * 1024**3


def test_probe_runs_without_torch():
    hw = probe(use_torch=False)
    assert isinstance(hw, HardwareProfile)
    assert hw.logical_cores >= 1 and hw.ram_total_gb > 0
    assert hw.device_class in ("cpu", "nvidia", "amd", "apple", "intel_gpu")
    assert "Device class" in hw.summary()
    assert hw.to_dict()["device_class"] == hw.device_class


@pytest.mark.parametrize("name,expected", [
    ("cpu-laptop", "cpu"), ("cpu-server", "cpu"), ("nvidia-4gb", "nvidia"), ("nvidia-24gb", "nvidia"),
    ("amd-16gb", "amd"), ("apple-16gb", "apple"), ("intel-arc-16gb", "intel_gpu")])
def test_reference_profiles_classify(name, expected):
    assert reference_profile(name).device_class == expected


def test_reference_profile_names_are_complete():
    for name in REFERENCE_PROFILES:
        reference_profile(name)
    with pytest.raises(KeyError):
        reference_profile("quantum-laptop")


def test_gpu_unusable_by_torch_is_not_a_training_device():
    hw = reference_profile("nvidia-24gb")
    hw.accelerators[0].torch_usable = False
    assert hw.primary is None and hw.device_class == "cpu"


def test_vnni_detection():
    assert reference_profile("cpu-server").has_vnni
    assert not reference_profile("cpu-laptop").has_vnni
    assert reference_profile("apple-16gb").has_vnni  # arm dot-product
