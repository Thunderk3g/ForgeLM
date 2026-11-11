import pytest

from forgelm.evidence import EVIDENCE, LEVELS, weakest_level
from forgelm.hardware import REFERENCE_PROFILES, reference_profile
from forgelm.planner import TASK_KINDS, Constraints, TaskSpec, plan

CPU = reference_profile("cpu-laptop")


def names(p):
    return " | ".join(s.name for s in p.steps)


def test_every_evidence_entry_is_well_formed_and_filled():
    for key, entry in EVIDENCE.items():
        assert entry["level"] in LEVELS, key
        assert entry["source"], key
        assert entry["claim"] and entry["claim"] != "PENDING", f"{key} has no measured result yet"


def test_measured_evidence_points_at_a_result_file_that_exists():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for key, entry in EVIDENCE.items():
        if entry["level"] == "measured":
            for source in entry["source"].split(";"):
                assert (root / source.strip()).is_file(), f"{key}: {source.strip()} missing"


def test_weakest_level_wins():
    assert weakest_level(["peft-benchmark", "vram-heuristic"]) == "assumption"
    assert weakest_level([]) == "assumption"


@pytest.mark.parametrize("hardware", REFERENCE_PROFILES)
@pytest.mark.parametrize("kind", TASK_KINDS)
def test_every_task_plans_on_every_hardware(hardware, kind):
    task = TaskSpec(kind=kind, n_classes=5, labelled_per_class=40, n_rows=5000, n_features=20)
    result = plan(task, reference_profile(hardware))
    assert result.steps and result.guardrails
    for step in result.steps:
        for key in step.evidence:
            assert key in EVIDENCE
    assert "Candidates, cheapest first" in result.render()
    assert result.to_dict()["steps"][0]["name"]


def test_few_shot_text_plan_avoids_fine_tuning():
    result = plan(TaskSpec(n_classes=77, labelled_per_class=8), CPU)
    assert "frozen sentence encoder" in names(result) and "fine-tune" not in names(result)
    assert any("fine-tuning an encoder" in n["what"] for n in result.not_recommended)
    assert "SetFit" not in names(result)  # 77 classes: pair count explodes


def test_setfit_only_for_few_classes_and_few_labels():
    assert "SetFit" in names(plan(TaskSpec(n_classes=4, labelled_per_class=8), CPU))
    assert "SetFit" not in names(plan(TaskSpec(n_classes=4, labelled_per_class=64), CPU))


def test_plenty_of_labels_adds_fine_tuning_sized_for_the_hardware():
    cpu = plan(TaskSpec(n_classes=10, labelled_per_class=200), CPU)
    gpu = plan(TaskSpec(n_classes=10, labelled_per_class=200), reference_profile("nvidia-24gb"))
    assert "MiniLM" in names(cpu) and "deberta" in names(gpu)
    long_docs = plan(TaskSpec(n_classes=10, labelled_per_class=200, avg_tokens=2000), CPU)
    assert "ModernBERT" in names(long_docs)
    multi = plan(TaskSpec(n_classes=10, labelled_per_class=200, multilingual=True), CPU)
    assert "multilingual-e5" in names(multi) and any("Hinglish" in w for w in multi.warnings)


def test_zero_labels_paths():
    bare = plan(TaskSpec(n_classes=6), CPU)
    # With no labels the plan is: look at zero-shot, then hand-label a few per class (the cheapest
    # large gain measured in E1 / E2).
    assert [s.name for s in bare.steps] == ["zero-shot by label description",
                                            "label about 8 examples per class by hand"]
    assert any("anchor" in w for w in bare.warnings)
    with_pool = plan(TaskSpec(n_classes=6, n_unlabelled=5000), CPU, Constraints(teacher_available=True))
    assert "teacher-label the unlabelled pool" in names(with_pool)
    no_pool = plan(TaskSpec(n_classes=6), CPU, Constraints(teacher_available=True))
    assert "synthetic" in names(no_pool)


def test_deployment_differs_from_training_hardware():
    c = Constraints(deploy=reference_profile("cpu-laptop"), max_latency_ms=1)
    result = plan(TaskSpec(n_classes=10, labelled_per_class=200), reference_profile("nvidia-24gb"), c)
    assert result.hardware_class == "nvidia" and result.deploy_class == "cpu"
    text = " ".join(result.deployment)
    assert "reduce_range" in text  # laptop CPU has no VNNI
    assert "only static embeddings met it" in text and "TF-IDF took" in text  # E1: TF-IDF is 4.7-9.6 ms per text


def test_on_prem_and_cascade_constraints():
    result = plan(TaskSpec(n_classes=5, labelled_per_class=50, needs_world_knowledge=True), CPU,
                  Constraints(llm_at_inference=True, data_must_stay_on_prem=True))
    assert result.approach == "hybrid cascade"
    assert "escalate low-confidence" in names(result)
    assert any("on-premises" in g for g in result.guardrails)


def test_tabular_respects_licence_and_explainability():
    task = TaskSpec(kind="tabular_classification", n_rows=3000, n_features=30)
    commercial = plan(task, CPU)
    assert "tabicl" in names(commercial) and "tabpfn-2.5+" not in names(commercial)
    assert any("non-commercial" in n["why"] for n in commercial.not_recommended)
    research = plan(task, CPU, Constraints(commercial_use=False))
    assert "tabpfn-2.5+" in names(research)
    explain = plan(task, CPU, Constraints(needs_inherent_explainability=True))
    assert explain.steps[0].name.startswith("ebm (champion)")
    temporal = plan(TaskSpec(kind="tabular_classification", n_rows=3000, n_features=30, temporal_or_grouped=True),
                    CPU)
    assert "tabicl" not in names(temporal)


@pytest.mark.parametrize("hardware,backend,device", [
    ("nvidia-24gb", "unsloth", "cuda"), ("nvidia-4gb", "peft_trl", "cuda"), ("amd-16gb", "peft_trl", "rocm"),
    ("apple-16gb", "mlx_lm", "mps"), ("intel-arc-16gb", "peft_trl", "xpu")])
def test_generation_backend_follows_hardware(hardware, backend, device):
    result = plan(TaskSpec(kind="generation", model_params_billion=3), reference_profile(hardware))
    step = next(s for s in result.steps if "LoRA fine-tune" in s.name)
    assert step.detail["backend"]["backend"] == backend and step.detail["backend"]["device"] == device
    # plain LoRA: the plan must not switch on rank-stabilised scaling, nor say that it did
    lora = step.detail["lora"]
    assert (lora["r"], lora["lora_alpha"]) == (32, 64) and "use_rslora" not in lora
    assert "plain LoRA r=32" in step.action and "with rank-stabilised scaling" not in step.action
    assert "learning-rate sweep" in step.action
    assert not any("LoftQ" in n["what"] and "available" in n["why"] for n in result.not_recommended)
    assert result.steps[0].name == "freeze an evaluation contract"


def test_generation_on_cpu_is_flagged_infeasible_for_large_models():
    result = plan(TaskSpec(kind="generation", model_params_billion=8), CPU)
    step = next(s for s in result.steps if "LoRA fine-tune" in s.name)
    assert not step.feasible and "[not runnable here]" in result.render()


def test_document_qa_recommends_retrieval_not_fine_tuning():
    result = plan(TaskSpec(kind="document_qa"), CPU)
    assert result.approach.startswith("retrieval")
    assert any("fine-tuning an LLM on the regulation text" in n["what"] for n in result.not_recommended)


def test_unknown_task_kind_rejected():
    with pytest.raises(ValueError):
        TaskSpec(kind="image_generation")


def test_every_forgelm_module_imports_with_numpy_only():
    """CLAUDE.md: forgelm/ must import with only numpy installed. A child process imports every module
    while any import of a heavy package is recorded and refused, so both an unconditional import and a
    guarded ``try: import scipy`` at module level fail the test."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys\n"
        "HEAVY = {'torch', 'transformers', 'peft', 'psutil', 'sklearn', 'scipy', 'pandas', 'sentence_transformers',\n"
        "         'model2vec', 'llama_cpp', 'gguf', 'jinja2', 'tokenizers', 'huggingface_hub', 'onnxruntime',\n"
        "         'lightgbm', 'interpret', 'tabicl'}\n"
        "tried = []\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in HEAVY:\n"
        "            tried.append(name.split('.')[0])\n"
        "            raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import importlib, pkgutil, forgelm\n"
        "for m in pkgutil.walk_packages(forgelm.__path__, 'forgelm.'):\n"
        "    if not m.name.endswith('__main__'):\n"
        "        importlib.import_module(m.name)\n"
        "import forgelm.cli, forgelm.dryrun as d\n"
        "assert d.fits(1.0, 2.0) and d.dtype_name_for('mps') == 'float16'\n"
        "forgelm.cli.build_parser().parse_args(['export-gguf', '--model', 'm', '--out', 'o'])\n"
        "assert not tried, tried\n"
        "print('ok')\n")
    root = Path(__file__).resolve().parents[2]
    done = subprocess.run([sys.executable, "-c", code], cwd=str(root), capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
