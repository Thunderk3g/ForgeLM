"""Tests for forgelm.finetune, forgelm.gguf and the finetune / export-gguf commands.

Everything here runs without a download. The tiny end-to-end fine-tune builds
a randomly initialised two-layer model from a config. Tests that download a
real model or need llama-cpp-python are marked slow.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from forgelm import gguf
from forgelm.cli import build_parser, main, recipe_from_args
from forgelm.finetune import (IGNORE_INDEX, FineTuneRecipe, batch_schedule, check_record, collate, encode_chat,
                              evaluate_chat, load_chat_records, lr_multiplier, mlx_iterations, mlx_lora_command,
                              normalise_answer, prepare_mlx_run, warmup_steps, write_mlx_config, write_mlx_dataset)

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# A tokenizer small enough to check by hand
# --------------------------------------------------------------------------

class WordTokenizer:
    """Whitespace tokenizer with a ChatML-like template, in the shape finetune uses.

    ``joiner`` is what the generation prompt ends with. With the default " "
    the answer is a separate word; with "" the answer is glued to the prompt's
    last word, so one token straddles the prompt / answer boundary.
    """

    SPECIAL = ["<pad>", "<|im_start|>", "<|im_end|>"]
    pad_token_id, eos_token_id = 0, 2

    def __init__(self, joiner=" "):
        self.joiner = joiner
        self.vocab = {t: i for i, t in enumerate(self.SPECIAL)}
        self.chat_template = "word-template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        text = ""
        for m in messages:
            if m["role"] == "assistant":
                text += f"<|im_start|> assistant:{self.joiner}{m['content']} <|im_end|> "
            else:
                text += f"<|im_start|> {m['role']} {m['content']} <|im_end|> "
        if add_generation_prompt:
            text += f"<|im_start|> assistant:{self.joiner}"
        return text

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False  # the template has written them already
        return {"input_ids": [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]}

    def decode(self, ids):
        inverse = {i: w for w, i in self.vocab.items()}
        return " ".join(inverse[i] for i in ids)


def conversation(answer="blue sky", question="what colour is the sky"):
    return [{"role": "system", "content": "answer briefly"}, {"role": "user", "content": question},
            {"role": "assistant", "content": answer}]


def supervised(tok, encoded):
    return tok.decode([t for t, l in zip(encoded["input_ids"], encoded["labels"]) if l != IGNORE_INDEX])


# --------------------------------------------------------------------------
# Label masking, truncation and padding
# --------------------------------------------------------------------------

def test_only_the_final_answer_and_its_end_of_turn_are_supervised():
    tok = WordTokenizer()
    e = encode_chat(tok, conversation(), max_seq_len=100)
    assert supervised(tok, e) == "blue sky <|im_end|>"
    assert e["n_target"] == 3 and e["truncated"] is False
    n_prompt = len(e["input_ids"]) - 3
    assert e["labels"][:n_prompt] == [IGNORE_INDEX] * n_prompt
    assert e["labels"][n_prompt:] == e["input_ids"][n_prompt:]


def test_earlier_assistant_turns_are_not_supervised():
    tok = WordTokenizer()
    messages = conversation()[:2] + [{"role": "assistant", "content": "first reply"},
                                     {"role": "user", "content": "and the grass"},
                                     {"role": "assistant", "content": "green"}]
    assert supervised(tok, encode_chat(tok, messages, max_seq_len=100)) == "green <|im_end|>"


def test_training_on_all_tokens_masks_nothing():
    tok = WordTokenizer()
    e = encode_chat(tok, conversation(), max_seq_len=100, train_on_responses_only=False)
    assert e["labels"] == e["input_ids"] and e["n_target"] == len(e["input_ids"])


def test_a_token_straddling_the_prompt_boundary_is_supervised():
    # "assistant:" + "blue" tokenise as one word "assistant:blue" in the full
    # conversation, but "assistant:" alone in the prompt. Counting the prompt's
    # own length would mask the answer's first word; counting the shared prefix
    # does not.
    tok = WordTokenizer(joiner="")
    e = encode_chat(tok, conversation(), max_seq_len=100)
    assert supervised(tok, e) == "assistant:blue sky <|im_end|>"


def test_truncation_cuts_on_the_right_and_reports_what_is_left():
    tok = WordTokenizer()
    full = encode_chat(tok, conversation(), max_seq_len=100)
    n = len(full["input_ids"])
    cut = encode_chat(tok, conversation(), max_seq_len=n - 1)
    assert cut["truncated"] and len(cut["input_ids"]) == n - 1
    assert supervised(tok, cut) == "blue sky" and cut["n_target"] == 2
    prompt_only = encode_chat(tok, conversation(), max_seq_len=n - 3)
    assert prompt_only["n_target"] == 0 and all(l == IGNORE_INDEX for l in prompt_only["labels"])


def test_a_template_whose_prompt_is_not_a_prefix_is_refused():
    class Reordering(WordTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            text = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            # A template that, like some reasoning models', rewrites earlier
            # turns once an answer follows them.
            return text.replace("system", "SYSTEM") if messages[-1]["role"] == "assistant" else text

    with pytest.raises(ValueError, match="train_on_responses_only=False"):
        encode_chat(Reordering(), conversation(), max_seq_len=100)


def test_collate_pads_on_the_right_with_masked_labels_and_attention():
    a = {"input_ids": [5, 6, 7], "labels": [IGNORE_INDEX, 6, 7]}
    b = {"input_ids": [8], "labels": [8]}
    batch = collate([a, b], pad_token_id=0)
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 6, 7], [8, IGNORE_INDEX, IGNORE_INDEX]]
    assert batch["input_ids"].dtype == np.int64


def test_batch_schedule_sees_every_example_once_per_epoch_and_is_seeded():
    batches = batch_schedule(10, 4, 2, seed=3)
    assert len(batches) == 6  # 3 batches per epoch, the last one of 2
    for epoch in (batches[:3], batches[3:]):
        assert sorted(np.concatenate(epoch).tolist()) == list(range(10))
    assert [b.tolist() for b in batches] == [b.tolist() for b in batch_schedule(10, 4, 2, seed=3)]
    assert len(batch_schedule(10, 4, 0.5, seed=0)) == 2  # fractional epochs round to whole batches


def test_learning_rate_warms_up_then_decays_without_a_zero_step():
    total = 10
    n_warmup = warmup_steps(0.2, total)
    assert n_warmup == 2 and warmup_steps(3, total) == 3 and warmup_steps(50, total) == total - 1
    rates = [lr_multiplier(s, total, n_warmup) for s in range(total)]
    assert rates[0] == 0.5 and rates[1] == 1.0 and rates[2] == 1.0
    assert all(a >= b for a, b in zip(rates[2:], rates[3:]))
    assert min(rates) > 0


def test_recipe_validates_and_round_trips(tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        FineTuneRecipe(base_model="m", batch_size=0)
    with pytest.raises(ValueError, match="epochs"):
        FineTuneRecipe(base_model="m", epochs=0)
    r = FineTuneRecipe(base_model="m", rank=8, target_modules=["q_proj"])
    assert r.effective_alpha == 16
    assert FineTuneRecipe.load(r.save(str(tmp_path / "recipe.json"))) == r


def test_records_are_checked_with_a_useful_message(tmp_path):
    with pytest.raises(ValueError, match="last message must be the assistant"):
        check_record({"messages": conversation()[:2]})
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"messages": conversation()}) + "\n\nnot json\n")
    with pytest.raises(ValueError, match="line 3: not valid JSON"):
        load_chat_records(str(path))


def test_normalise_answer_ignores_case_quotes_space_and_final_stop():
    assert normalise_answer('  "Card_Arrival." ') == normalise_answer("card_arrival")
    assert normalise_answer("card arrival") != normalise_answer("card_arrival")


def test_evaluate_chat_scores_exact_match_and_keeps_raw_outputs():
    records = [{"messages": conversation("blue")}, {"messages": conversation("green", "grass")}]
    replies = iter([{"text": "Blue.", "new_tokens": 2, "seconds": 0.5}, "yellow"])
    scored = evaluate_chat(lambda messages, limit: next(replies), records)
    assert scored["accuracy"] == 0.5 and scored["correct"] == [True, False]
    assert scored["outputs"] == ["Blue.", "yellow"] and scored["references"] == ["blue", "green"]


# --------------------------------------------------------------------------
# mlx-lm (argument list and files only: no Apple hardware here)
# --------------------------------------------------------------------------

def test_mlx_dataset_is_one_chat_record_per_line(tmp_path):
    records = [{"messages": conversation(f"answer {i}\nwith a newline"), "extra": i} for i in range(10)]
    written = write_mlx_dataset(records, str(tmp_path), valid_fraction=0.2, seed=0)
    assert (written["n_train"], written["n_valid"]) == (8, 2)
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 8 and all(set(r) == {"messages"} for r in rows)
    assert all(set(m) == {"role", "content"} for r in rows for m in r["messages"])
    assert len((tmp_path / "valid.jsonl").read_text().splitlines()) == 2
    # Without a validation share, a stale valid.jsonl must not be left behind.
    write_mlx_dataset(records, str(tmp_path), valid_fraction=0)
    assert not (tmp_path / "valid.jsonl").exists()


def test_mlx_config_carries_rank_scale_and_schedule(tmp_path):
    recipe = FineTuneRecipe(base_model="m", rank=8, alpha=16, dropout=0.05, learning_rate=2e-5, epochs=2,
                            batch_size=4, warmup=0.1)
    text = Path(write_mlx_config(recipe, str(tmp_path), n_train=40)).read_text()
    assert "rank: 8" in text
    assert "scale: 2.0" in text  # alpha / rank, not alpha
    assert "dropout: 0.05" in text
    assert "arguments: [0.00002, 0.0, 18]" in text and "warmup: 2" in text  # 20 updates, 2 of warm-up
    with pytest.raises(ValueError, match="path inside a block"):
        write_mlx_config(FineTuneRecipe(base_model="m", target_modules=["q_proj"]), str(tmp_path), 40)
    dotted = FineTuneRecipe(base_model="m", target_modules=["self_attn.q_proj"])
    assert 'keys: ["self_attn.q_proj"]' in Path(write_mlx_config(dotted, str(tmp_path), 40)).read_text()


def test_mlx_command_uses_the_documented_flags(tmp_path):
    recipe = FineTuneRecipe(base_model="mlx-community/SmolLM2-360M-Instruct", method="qlora", rank=8,
                            learning_rate=1e-4, epochs=2, batch_size=4, grad_accumulation=2, max_seq_len=256, seed=7)
    records = [{"messages": conversation()} for _ in range(10)]
    argv = prepare_mlx_run(recipe, records, str(tmp_path / "data"), str(tmp_path / "adapters"))
    assert argv[:4] == ["python", "-m", "mlx_lm", "lora"]
    flags = dict(zip(argv[4::], argv[5::]))
    assert flags["--model"] == recipe.base_model
    assert flags["--fine-tune-type"] == "lora"  # QLoRA is LoRA on a quantised --model in mlx-lm
    assert flags["--iters"] == str(mlx_iterations(recipe, 9)) == "6"  # 9 training records, 3 batches x 2 epochs
    assert flags["--batch-size"] == "4" and flags["--grad-accumulation-steps"] == "2"
    assert flags["--max-seq-length"] == "256" and flags["--seed"] == "7" and flags["--num-layers"] == "-1"
    assert Path(flags["--config"]).name == "lora_config.yaml" and Path(flags["--config"]).exists()
    assert "--train" in argv and "--mask-prompt" in argv
    documented = {"--model", "--train", "--data", "--fine-tune-type", "--optimizer", "--mask-prompt", "--num-layers",
                  "--batch-size", "--iters", "--learning-rate", "--grad-accumulation-steps", "--adapter-path",
                  "--max-seq-length", "--config", "--seed"}
    assert {a for a in argv if a.startswith("--")} <= documented
    whole = FineTuneRecipe(base_model="m", train_on_responses_only=False)
    assert "--mask-prompt" not in mlx_lora_command(whole, str(tmp_path / "data"), "a", n_train=4)
    with pytest.raises(ValueError, match="mlx-lm cannot train"):
        mlx_lora_command(FineTuneRecipe(base_model="m", method="oft"), str(tmp_path), "a", n_train=4)
    with pytest.raises(FileNotFoundError, match="write_mlx_dataset"):
        mlx_lora_command(recipe, str(tmp_path / "empty"), "a")


# --------------------------------------------------------------------------
# Output parity and the template checks
# --------------------------------------------------------------------------

def test_output_parity_passes_identical_answers_after_normalising():
    result = gguf.output_parity(["card_arrival", "Fee."], ["card_arrival.", "fee"], labels=["card_arrival", "fee"])
    assert result.passed and result.agreement == 1.0 and result.accuracy_drop == 0.0


def test_output_parity_reports_disagreements_and_accuracy_drop():
    ref = ["a", "b", "c", "d"]
    result = gguf.output_parity(ref, ["a", "b", "x", "d"], labels=["a", "b", "c", "x"])
    assert not result.passed and result.agreement == 0.75 and result.disagreements == [2]
    assert result.reference_accuracy == 0.75 and result.candidate_accuracy == 0.5
    assert result.accuracy_drop == pytest.approx(0.25)
    assert any("agreement" in r for r in result.reasons) and any("accuracy dropped" in r for r in result.reasons)
    # A candidate that changes answers but only ever fixes them fails on agreement alone.
    better = gguf.output_parity(ref, ["a", "b", "c", "x"], labels=["a", "b", "c", "x"])
    assert better.accuracy_drop < 0 and better.reasons == ["answer agreement 0.7500 below 0.98"]
    with pytest.raises(ValueError, match="length mismatch"):
        gguf.output_parity(["a"], ["a", "b"])
    with pytest.raises(ValueError, match="no outputs"):
        gguf.output_parity([], [])


def test_compare_prompts_points_at_the_first_difference():
    assert gguf.compare_prompts("abc", "abc") == {"identical": True, "first_difference": None}
    diff = gguf.compare_prompts("<|im_start|>user\nhi", "<|im_start|>user hi")
    assert diff["identical"] is False and diff["first_difference"] == 16


CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


class TemplateTokenizer:
    """Renders a jinja chat template the way transformers does."""

    def __init__(self, template):
        self.chat_template = template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return gguf.render_stored_template(self.chat_template, messages,
                                           add_generation_prompt=add_generation_prompt)["prompt"]


def write_metadata_gguf(path, template):
    """A GGUF file holding only tokenizer metadata, which is all the round trip reads."""
    writer = pytest.importorskip("gguf").GGUFWriter(str(path), arch="llama")
    if template is not None:
        writer.add_chat_template(template)
    writer.add_token_list(["<pad>", "<|im_start|>", "<|im_end|>"])
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(path)


@pytest.fixture
def without_llama_cpp(monkeypatch):
    """Make llama_cpp look uninstalled, so the jinja2 renderer and the gguf reader are used."""
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "llama_cpp" else real(name, *a))
    monkeypatch.setitem(sys.modules, "llama_cpp", None)


def test_round_trip_passes_when_the_stored_template_matches(tmp_path, without_llama_cpp):
    path = write_metadata_gguf(tmp_path / "m.gguf", CHATML)
    result = gguf.chat_template_round_trip(TemplateTokenizer(CHATML), path)
    assert result["passed"] and result["identical"] and result["template_identical"]
    assert result["tokens_identical"] is None  # tokens need llama-cpp-python
    assert result["rendered_with"] == "jinja2" and result["read_with"] == "gguf"


def test_round_trip_catches_a_template_changed_by_export(tmp_path, without_llama_cpp):
    # The classic silent failure: a runtime template that drops the newline
    # after the role. Nothing errors; the model just sees a new prompt format.
    exported = CHATML.replace("{{ m['role'] }}\n", "{{ m['role'] }} ")
    path = write_metadata_gguf(tmp_path / "m.gguf", exported)
    result = gguf.chat_template_round_trip(TemplateTokenizer(CHATML), path)
    assert not result["passed"] and not result["identical"]
    assert result["first_difference"] == len("<|im_start|>system")
    assert "differs from character" in result["reasons"][0]


def test_round_trip_fails_when_no_template_was_exported(tmp_path, without_llama_cpp):
    path = write_metadata_gguf(tmp_path / "m.gguf", None)
    result = gguf.chat_template_round_trip(TemplateTokenizer(CHATML), path)
    assert not result["passed"] and "stores no chat template" in result["reasons"][0]


# --------------------------------------------------------------------------
# Clear errors when the GGUF stack is missing
# --------------------------------------------------------------------------

def test_running_a_gguf_without_llama_cpp_says_how_to_install_it(tmp_path, without_llama_cpp):
    path = tmp_path / "m.gguf"
    path.write_bytes(b"GGUF")
    with pytest.raises(ImportError, match="pip install llama-cpp-python"):
        gguf.GGUFChat(str(path))


def test_quantising_without_llama_cpp_or_the_binary_says_how_to_get_either(tmp_path, monkeypatch, without_llama_cpp):
    monkeypatch.setattr(gguf.shutil, "which", lambda name: None)
    src = tmp_path / "m.f16.gguf"
    src.write_bytes(b"GGUF")
    with pytest.raises(ImportError, match="llama-quantize"):
        gguf.quantize_gguf(str(src), str(tmp_path / "m.q4.gguf"))
    with pytest.raises(ValueError, match="unknown quantisation"):
        gguf.quantize_gguf(str(src), str(tmp_path / "x.gguf"), "Q3_FAKE")


def test_reading_metadata_with_neither_package_says_what_to_install(tmp_path, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name in ("gguf", "llama_cpp") else real(name, *a))
    path = tmp_path / "m.gguf"
    path.write_bytes(b"GGUF")
    with pytest.raises(ImportError, match=f"gguf=={gguf.GGUF_PIP_VERSION}"):
        gguf.read_chat_template(str(path))


def test_converting_explains_a_missing_script_package_or_model(tmp_path, monkeypatch):
    monkeypatch.delenv(gguf.CONVERT_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="fetch_convert_script"):
        gguf.convert_to_gguf(str(tmp_path), str(tmp_path / "m.gguf"))
    with pytest.raises(ValueError, match="quantize_gguf"):
        gguf.convert_to_gguf(str(tmp_path), str(tmp_path / "m.gguf"), "x.py", outtype="Q4_K_M")
    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text("raise SystemExit(1)\n")
    with pytest.raises(FileNotFoundError, match="merge an adapter first"):
        gguf.convert_to_gguf(str(tmp_path), str(tmp_path / "m.gguf"), str(script))
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "gguf", None)
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "gguf" else real(name, *a))
    with pytest.raises(ImportError, match=f"pip install gguf=={gguf.GGUF_PIP_VERSION}"):
        gguf.convert_to_gguf(str(tmp_path), str(tmp_path / "m.gguf"), str(script))


def test_a_failing_converter_reports_its_own_error(tmp_path):
    pytest.importorskip("gguf")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text("import sys\nprint('unsupported architecture FooForCausalLM', file=sys.stderr)\nsys.exit(3)\n")
    with pytest.raises(RuntimeError, match=r"exit 3\).*not the pinned one[\s\S]*unsupported architecture"):
        gguf.convert_to_gguf(str(tmp_path / "model"), str(tmp_path / "m.gguf"), str(script))


def test_the_pinned_converter_and_package_versions_agree():
    assert gguf.LLAMA_CPP_TAG == f"gguf-v{gguf.GGUF_PIP_VERSION}"
    assert gguf.CONVERT_SCRIPT_URL.format(tag=gguf.LLAMA_CPP_TAG).startswith(
        "https://raw.githubusercontent.com/ggml-org/llama.cpp/gguf-v")
    assert len(gguf.CONVERT_SCRIPT_SHA256) == 64


def test_new_modules_import_with_numpy_only():
    # Block every heavy package, then import: the factory must stay usable
    # when the GGUF or training stack is broken or absent.
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'torch', 'transformers', 'peft', 'llama_cpp', 'gguf', 'jinja2',\n"
        "                                  'pandas', 'sklearn', 'tokenizers', 'huggingface_hub'}:\n"
        "            raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import forgelm.finetune, forgelm.gguf, forgelm.cli\n"
        "forgelm.cli.build_parser().parse_args(['export-gguf', '--model', 'm', '--out', 'o'])\n"
        "print('ok')\n")
    done = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def test_finetune_options_become_a_recipe():
    args = build_parser().parse_args([
        "finetune", "--data", "d.jsonl", "--base-model", "HuggingFaceTB/SmolLM2-360M-Instruct", "--out", "o",
        "--rank", "8", "--alpha", "32", "--epochs", "1.5", "--batch-size", "2", "--grad-accumulation", "4",
        "--target-modules", "q_proj", "v_proj", "--all-tokens", "--learning-rate", "1e-4", "--warmup", "5"])
    recipe = recipe_from_args(args)
    assert (recipe.rank, recipe.alpha, recipe.epochs, recipe.batch_size, recipe.grad_accumulation) == (8, 32, 1.5, 2, 4)
    assert recipe.target_modules == ["q_proj", "v_proj"] and recipe.train_on_responses_only is False
    assert recipe.learning_rate == 1e-4 and recipe.warmup == 5
    defaults = recipe_from_args(build_parser().parse_args(["finetune", "--data", "d", "--base-model", "m",
                                                           "--out", "o"]))
    assert defaults == FineTuneRecipe(base_model="m")  # the CLI defaults are the recipe defaults
    assert args.backend == "torch" and args.merge is False


def test_export_gguf_needs_exactly_one_source_and_a_known_quantisation(capsys):
    parser = build_parser()
    args = parser.parse_args(["export-gguf", "--adapter", "a", "--out", "o"])
    assert args.quant == "Q4_K_M" and args.outtype == "f16" and args.model is None
    for bad in (["export-gguf", "--out", "o"], ["export-gguf", "--model", "m", "--adapter", "a", "--out", "o"],
                ["export-gguf", "--model", "m", "--out", "o", "--quant", "Q4_FAKE"]):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)
    assert parser.parse_args(["export-gguf", "--model", "m", "--out", "o", "--quant", "none"]).quant == "none"


def test_finetune_mlx_backend_writes_inputs_and_prints_the_command(tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps({"messages": conversation(f"a{i}")}) + "\n" for i in range(5)))
    code = main(["finetune", "--data", str(data), "--base-model", "mlx-community/SmolLM2-360M-Instruct",
                 "--out", str(tmp_path / "run"), "--backend", "mlx", "--rank", "4"])
    assert code == 0
    assert (tmp_path / "run" / "mlx_data" / "train.jsonl").exists()
    assert "rank: 4" in (tmp_path / "run" / "mlx_data" / "lora_config.yaml").read_text()
    assert "python -m mlx_lm lora --model mlx-community/SmolLM2-360M-Instruct" in capsys.readouterr().out


def test_finetune_rejects_bad_data_before_loading_a_model(tmp_path, capsys):
    data = tmp_path / "bad.jsonl"
    data.write_text(json.dumps({"messages": conversation()[:2]}) + "\n")
    assert main(["finetune", "--data", str(data), "--base-model", "m", "--out", str(tmp_path / "o")]) == 2
    assert "last message must be the assistant" in capsys.readouterr().err


def test_finetune_eval_exact_without_eval_data_is_refused_up_front(tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"messages": conversation()}) + "\n")
    assert main(["finetune", "--data", str(data), "--base-model", "m", "--out", str(tmp_path / "o"),
                 "--eval-exact"]) == 2
    assert "--eval-exact needs --eval-data" in capsys.readouterr().err


def test_export_gguf_reports_an_adapter_directory_that_is_not_one(tmp_path, capsys):
    pytest.importorskip("peft")
    assert main(["export-gguf", "--adapter", str(tmp_path), "--out", str(tmp_path / "out")]) == 2
    assert "adapter_config.json" in capsys.readouterr().err


def test_export_gguf_reports_a_missing_converter(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(gguf.CONVERT_ENV, raising=False)
    model = tmp_path / "merged"
    model.mkdir()
    (model / "config.json").write_text("{}")
    assert main(["export-gguf", "--model", str(model), "--out", str(tmp_path / "out")]) == 2
    assert "fetch_convert_script" in capsys.readouterr().err


# --------------------------------------------------------------------------
# A real fine-tune of a tiny random model (no download)
# --------------------------------------------------------------------------

CHATML_NEWLINE = ("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + "
                  "'<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
                  "{% endif %}")


def build_tiny_model(directory):
    """A two-layer Llama with a word-level tokenizer, saved like a Hub model."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Regex, models, pre_tokenizers

    words = "system user assistant what colour is the sky grass sun blue green yellow answer briefly".split()
    vocab = {w: i for i, w in enumerate(["<pad>", "<unk>", "<|im_start|>", "<|im_end|>", "\n"] + words)}
    raw = tokenizers.Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(r"\n"), "isolated"),
                                                 pre_tokenizers.Split(" ", "removed")])
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>",
                                               eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>"])
    tok.chat_template = CHATML_NEWLINE
    torch.manual_seed(0)
    config = transformers.LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                      max_position_embeddings=64, pad_token_id=0, eos_token_id=3,
                                      bos_token_id=None, initializer_range=0.2)
    transformers.LlamaForCausalLM(config).save_pretrained(str(directory))
    tok.save_pretrained(str(directory))
    return str(directory)


def colour_records():
    pairs = [("sky", "blue"), ("grass", "green"), ("sun", "yellow")]
    return [{"messages": [{"role": "system", "content": "answer briefly"},
                          {"role": "user", "content": f"what colour is the {thing}"},
                          {"role": "assistant", "content": colour}]} for thing, colour in pairs]


def test_tiny_finetune_lowers_the_loss_and_merges_to_loadable_weights(tmp_path):
    from forgelm.finetune import evaluate_exact, finetune_lora, generate, load_model, merge_adapter

    import torch
    from peft import PeftModel

    base = build_tiny_model(tmp_path / "base")
    records = colour_records() * 4
    recipe = FineTuneRecipe(base_model=base, rank=8, learning_rate=1e-2, epochs=30, batch_size=4,
                            grad_accumulation=2, warmup=0.1, max_seq_len=64, seed=0)
    steps_seen = []
    result = finetune_lora(recipe, records, str(tmp_path / "adapter"), eval_records=colour_records(), device="cpu",
                           on_progress=lambda step, total, loss: steps_seen.append((step, total)))
    losses = result["loss_history"]
    assert result["device"] == "cpu" and result["dtype"] == "float32"
    assert result["steps"] == len(losses) == 45 and steps_seen[-1] == (45, 45)  # 3 batches x 30 epochs / 2
    assert np.mean(losses[-5:]) < 0.5 * np.mean(losses[:5])
    assert result["eval_loss_history"][-1]["loss"] < result["eval_loss_history"][0]["loss"]
    assert 0 < result["trainable_parameters"] < result["total_parameters"]
    saved = {p.name for p in (tmp_path / "adapter").iterdir()}
    assert {"adapter_config.json", "adapter_model.safetensors", "recipe.json", "training_log.json"} <= saved
    assert FineTuneRecipe.load(str(tmp_path / "adapter" / "recipe.json")) == recipe

    # base_model=None: the merge finds the base model in recipe.json.
    merged = merge_adapter(None, str(tmp_path / "adapter"), str(tmp_path / "merged"))
    assert merged["base_model"] == base and merged["weights_bytes"] > 0
    assert not (tmp_path / "merged" / "adapter_config.json").exists()
    model, tokenizer = load_model(str(tmp_path / "merged"), device="cpu")
    adapted = PeftModel.from_pretrained(load_model(base, device="cpu")[0], str(tmp_path / "adapter")).eval()
    ids = torch.tensor([tokenizer(tokenizer.apply_chat_template(records[0]["messages"], tokenize=False),
                                  add_special_tokens=False)["input_ids"]])
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, adapted(ids).logits, atol=1e-4)

    reply = generate(model, tokenizer, records[0]["messages"][:-1], max_new_tokens=4, return_details=True)
    assert reply["new_tokens"] <= 4 and reply["prompt_tokens"] > 0
    scored = evaluate_exact(model, tokenizer, colour_records(), max_new_tokens=4)
    assert scored["n"] == 3 and len(scored["outputs"]) == 3 and scored["accuracy"] == 1.0


# --------------------------------------------------------------------------
# Slow: a real model through llama.cpp
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_real_model_exports_to_gguf_and_keeps_its_prompt_and_answers(tmp_path):
    pytest.importorskip("llama_cpp")
    pytest.importorskip("gguf")
    from huggingface_hub import snapshot_download

    from forgelm.finetune import generate, load_model

    model_dir = snapshot_download("HuggingFaceTB/SmolLM2-135M-Instruct",
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    script = os.environ.get(gguf.CONVERT_ENV) or gguf.fetch_convert_script(str(tmp_path / "llama_cpp"))
    f16 = gguf.convert_to_gguf(model_dir, str(tmp_path / "m.f16.gguf"), script)
    q4 = gguf.quantize_gguf(f16, str(tmp_path / "m.Q4_K_M.gguf"), "Q4_K_M", n_threads=1)
    assert Path(q4).stat().st_size < 0.6 * Path(f16).stat().st_size

    model, tokenizer = load_model(model_dir, device="cpu")
    trip = gguf.chat_template_round_trip(tokenizer, q4)
    assert trip["passed"] and trip["identical"] and trip["tokens_identical"], trip["reasons"]

    messages = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    reference = generate(model, tokenizer, messages, 12)
    chat = gguf.GGUFChat(f16, n_threads=1)
    try:
        assert gguf.output_parity([reference], [chat.chat(messages, 12)]).passed
    finally:
        chat.close()
