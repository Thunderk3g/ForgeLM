"""GGUF export of a fine-tuned generative model, with checks on the artefact.

A fine-tune is evaluated in PyTorch but served, on most machines without a
large GPU, as a quantised GGUF file under llama.cpp. Three things can change
between the two without any error being raised: the chat template the runtime
applies, the tokenisation of that prompt, and the weights themselves after
quantisation. This module does the export and checks each of the three:

    fetch_convert_script       llama.cpp's converter, pinned and checksummed
    convert_to_gguf            Hugging Face model directory -> GGUF (f16 / f32 / ...)
    quantize_gguf              GGUF -> Q4_K_M and the other llama.cpp types
    GGUFChat                   greedy chat with a GGUF file (llama-cpp-python)
    chat_template_round_trip   is the prompt the same before and after export?
    output_parity              does the exported model still give the same answers?

Nothing heavy is imported at module level. ``gguf`` (the converter's library)
and ``llama_cpp`` (the runtime) are optional; each function that needs one
says which and how to install it.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------
# The pinned converter
# --------------------------------------------------------------------------
#
# llama.cpp's convert_hf_to_gguf.py and the ``gguf`` Python package are
# developed in one repository and change together: a converter from one
# release run against the package of another typically dies on a missing
# attribute, or writes metadata the runtime does not expect. The package is
# published to PyPI from tags named ``gguf-v<version>``, so the tag below is
# the converter that belongs to exactly this package version.
#
# Tested here: gguf 0.19.0 from PyPI, the converter at tag gguf-v0.19.0, and
# llama-cpp-python 0.3.36 reading the result (see experiments/e7).
LLAMA_CPP_TAG = "gguf-v0.19.0"
GGUF_PIP_VERSION = "0.19.0"
CONVERT_SCRIPT_NAME = "convert_hf_to_gguf.py"
CONVERT_SCRIPT_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/{tag}/" + CONVERT_SCRIPT_NAME
# The converter is code that will be executed, so the download is verified.
CONVERT_SCRIPT_SHA256 = "eeb4549316ef6d7666971e32f7b86a1107bcd2f741dc3538c80fff319b67af98"
CONVERT_ENV = "LLAMA_CPP_CONVERT"

# What the converter itself can write. Finer quantisation is a second step
# (quantize_gguf) because llama.cpp implements it in C++, not in the script.
CONVERT_OUTTYPES = ("f32", "f16", "bf16", "q8_0", "auto")

# llama.cpp quantisation names accepted by quantize_gguf, smallest first.
QUANT_TYPES = ("Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M",
               "Q6_K", "Q8_0", "F16", "BF16")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_versions() -> Dict[str, Optional[str]]:
    """Versions of the export stack, for run records. ``None`` means not installed."""
    return {"gguf": _version("gguf"), "llama_cpp_python": _version("llama-cpp-python"),
            "pinned_llama_cpp_tag": LLAMA_CPP_TAG, "pinned_gguf": GGUF_PIP_VERSION}


def _require_gguf():
    """Import the ``gguf`` package or explain how to get the right one."""
    try:
        return importlib.import_module("gguf")
    except ImportError:
        raise ImportError(
            f"the 'gguf' package is not installed. Install the version that matches the pinned converter: "
            f"pip install gguf=={GGUF_PIP_VERSION}") from None


def _require_llama_cpp():
    """Import ``llama_cpp`` or explain how to get it."""
    try:
        return importlib.import_module("llama_cpp")
    except ImportError:
        raise ImportError(
            "llama-cpp-python is not installed, and it is needed to run or quantise a GGUF file from Python. "
            "Install it with: pip install llama-cpp-python (this compiles llama.cpp and takes several minutes; "
            "set CMAKE_BUILD_PARALLEL_LEVEL to limit the cores it uses). Quantisation alone also works with "
            "the llama-quantize binary from a llama.cpp build on PATH.") from None


def fetch_convert_script(dest: str, tag: str = LLAMA_CPP_TAG, sha256: Optional[str] = None) -> str:
    """Download llama.cpp's ``convert_hf_to_gguf.py`` for ``tag`` and return its path.

    ``dest`` is a directory or a file path. For the pinned tag the download is
    checked against :data:`CONVERT_SCRIPT_SHA256`; for another tag pass the
    ``sha256`` you expect, or the file is taken as it comes. A file already at
    ``dest`` with the right checksum is reused without a download.
    """
    import urllib.error
    import urllib.request

    expected = sha256 or (CONVERT_SCRIPT_SHA256 if tag == LLAMA_CPP_TAG else None)
    target = Path(dest)
    if target.is_dir() or target.suffix != ".py":
        target = target / CONVERT_SCRIPT_NAME
    if target.exists() and expected and _sha256(target) == expected:
        return str(target)
    url = CONVERT_SCRIPT_URL.format(tag=tag)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            f"could not download {url} ({exc}). Download it by hand, or point {CONVERT_ENV} at "
            f"{CONVERT_SCRIPT_NAME} in a llama.cpp checkout of tag {tag}") from None
    got = hashlib.sha256(data).hexdigest()
    if expected and got != expected:
        raise RuntimeError(f"{url} has sha256 {got}, expected {expected}; refusing to save a converter that is "
                           "not the one that was tested")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(data)
    partial.replace(target)  # never leave a half-written script where a later call would reuse it
    return str(target)


# Put on the converter's import path only when the real package is absent and
# the model has no SentencePiece file; see convert_to_gguf.
_SENTENCEPIECE_STAND_IN = '''"""Stand-in written by forgelm.gguf for a model that has no SentencePiece tokenizer."""


class SentencePieceProcessor:
    def __init__(self, *args, **kwargs):
        raise ImportError("this model needs the real 'sentencepiece' package: pip install sentencepiece")
'''


def _tail(text: str, lines: int = 15) -> str:
    return "\n".join((text or "").strip().splitlines()[-lines:])


def convert_to_gguf(model_dir: str, out_path: str, convert_script: Optional[str] = None,
                    outtype: str = "f16", timeout: Optional[float] = None) -> str:
    """Convert a Hugging Face model directory to one GGUF file.

    ``model_dir`` must hold full weights and the tokenizer, as written by
    :func:`forgelm.finetune.merge_adapter`; an adapter directory will not do.
    ``convert_script`` is the path to llama.cpp's ``convert_hf_to_gguf.py``;
    when omitted the ``LLAMA_CPP_CONVERT`` environment variable is used. The
    script runs as a subprocess of this interpreter, so it needs ``gguf``,
    ``torch`` and ``transformers`` importable here, and ``sentencepiece`` for
    a model whose tokenizer is a SentencePiece ``tokenizer.model`` file.

    ``outtype`` "f16" is the usual intermediate: half the size of the float32
    weights, and what :func:`quantize_gguf` quantises from.
    """
    if outtype not in CONVERT_OUTTYPES:
        raise ValueError(f"outtype must be one of {CONVERT_OUTTYPES}, got {outtype!r}; for Q4_K_M and the other "
                         "k-quants, convert to f16 and then call quantize_gguf")
    script = convert_script or os.environ.get(CONVERT_ENV)
    if not script:
        raise FileNotFoundError(
            f"no converter given. Pass convert_script, set {CONVERT_ENV}, or download the pinned one with "
            f"forgelm.gguf.fetch_convert_script(dest) (llama.cpp tag {LLAMA_CPP_TAG})")
    script_path = Path(script)
    if not script_path.is_file():
        raise FileNotFoundError(f"converter not found at {script_path}")
    model = Path(model_dir)
    if not (model / "config.json").exists():
        raise FileNotFoundError(f"{model} holds no config.json; it must be a full model directory "
                                "(merge an adapter first with forgelm.finetune.merge_adapter)")
    if (model / "adapter_config.json").exists() and not list(model.glob("model*.safetensors")):
        raise ValueError(f"{model} is an adapter, not a model; merge it first with forgelm.finetune.merge_adapter")
    if importlib.util.find_spec("gguf") is None:
        _require_gguf()
    have = _version("gguf")
    pinned_script = _sha256(script_path) == CONVERT_SCRIPT_SHA256
    if pinned_script and have != GGUF_PIP_VERSION:
        # Caught before the subprocess because the failure it causes there is
        # an AttributeError deep inside the converter that names neither side.
        raise RuntimeError(f"the converter is the one pinned at {LLAMA_CPP_TAG}, which needs gguf=="
                           f"{GGUF_PIP_VERSION}, but gguf {have} is installed: pip install gguf=={GGUF_PIP_VERSION}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script_path), str(model), "--outfile", str(out), "--outtype", outtype]
    environment = dict(os.environ)
    with tempfile.TemporaryDirectory(prefix="forgelm-gguf-") as scratch:
        if importlib.util.find_spec("sentencepiece") is None:
            if (model / "tokenizer.model").exists():
                raise ImportError(f"{model} has a SentencePiece tokenizer (tokenizer.model) and the converter needs "
                                  "the 'sentencepiece' package to read it: pip install sentencepiece")
            # The converter imports sentencepiece before it looks for a
            # tokenizer.model, and only falls back to the BPE vocabulary in
            # tokenizer.json when that file is missing. A model without one
            # never uses the package, so give the subprocess an importable
            # stand-in rather than demand an install that would do nothing.
            (Path(scratch) / "sentencepiece.py").write_text(_SENTENCEPIECE_STAND_IN)
            environment["PYTHONPATH"] = scratch + os.pathsep + environment.get("PYTHONPATH", "")
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=environment)
    if done.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        hint = "" if pinned_script else (
            f" The converter is not the pinned one ({LLAMA_CPP_TAG}); it must come from the llama.cpp release "
            f"that matches the installed gguf package ({have}).")
        raise RuntimeError(f"GGUF conversion failed (exit {done.returncode}).{hint}\n{_tail(done.stderr or done.stdout)}")
    return str(out)


def quantize_gguf(src: str, dst: str, qtype: str = "Q4_K_M", n_threads: Optional[int] = None) -> str:
    """Quantise a GGUF file (normally f16) to ``qtype`` and return ``dst``.

    Uses the ``llama-quantize`` binary when one is on PATH, otherwise the same
    routine through ``llama_cpp.llama_model_quantize``. The result must still
    be checked against the unquantised model: see :func:`output_parity`.
    """
    qtype = qtype.upper()
    if qtype not in QUANT_TYPES:
        raise ValueError(f"unknown quantisation {qtype!r}; choose from {QUANT_TYPES}")
    if not Path(src).is_file():
        raise FileNotFoundError(f"{src} not found; convert the model first with convert_to_gguf")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    binary = shutil.which("llama-quantize")
    if binary:
        command = [binary, str(src), str(dst), qtype] + ([str(n_threads)] if n_threads else [])
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"llama-quantize failed (exit {done.returncode}):\n{_tail(done.stderr or done.stdout)}")
    else:
        llama_cpp = _require_llama_cpp()
        import ctypes

        ftype = getattr(llama_cpp, f"LLAMA_FTYPE_MOSTLY_{qtype}", None)
        if ftype is None:
            raise RuntimeError(f"the installed llama-cpp-python does not know quantisation {qtype}; upgrade it, "
                               "or put llama-quantize on PATH")
        params = llama_cpp.llama_model_quantize_default_params()
        params.ftype = ftype
        if n_threads:
            params.nthread = int(n_threads)
        code = llama_cpp.llama_model_quantize(str(src).encode("utf-8"), str(dst).encode("utf-8"),
                                              ctypes.byref(params))
        if code != 0:
            raise RuntimeError(f"llama_model_quantize returned {code} for {src} -> {qtype}")
    if not Path(dst).is_file() or Path(dst).stat().st_size == 0:
        raise RuntimeError(f"quantisation reported success but {dst} is missing or empty")
    return str(dst)


# --------------------------------------------------------------------------
# Running a GGUF file
# --------------------------------------------------------------------------

class GGUFChat:
    """Greedy chat with a GGUF file through llama-cpp-python.

    The prompt is built from the chat template stored in the file, which is
    what a deployed llama.cpp runtime does; nothing from the Hugging Face
    tokenizer is used. Decoding is greedy with no repetition penalty so the
    outputs can be compared with :func:`forgelm.finetune.generate`.

    Consecutive calls that share a prompt prefix (a fixed system prompt, say)
    reuse llama.cpp's cache for that prefix. That is how the runtime behaves
    in service, and it makes throughput measured here depend on the prompts.
    """

    def __init__(self, path: str, n_threads: Optional[int] = None, n_ctx: int = 2048, n_gpu_layers: int = 0):
        llama_cpp = _require_llama_cpp()
        if not Path(path).is_file():
            raise FileNotFoundError(f"GGUF file not found: {path}")
        self.path = str(path)
        # verbose=False also silences llama.cpp's load log on stderr.
        self.llm = llama_cpp.Llama(model_path=self.path, n_ctx=n_ctx, n_threads=n_threads,
                                   n_threads_batch=n_threads, n_gpu_layers=n_gpu_layers, verbose=False)

    def chat_details(self, messages: Sequence[dict], max_tokens: int = 64) -> Dict[str, object]:
        """The reply with token counts and seconds, in the shape ``generate(return_details=True)`` uses."""
        started = time.time()
        response = self.llm.create_chat_completion(
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            max_tokens=max_tokens, temperature=0.0, top_k=1, top_p=1.0, min_p=0.0, repeat_penalty=1.0,
            presence_penalty=0.0, frequency_penalty=0.0)
        seconds = time.time() - started
        usage = response.get("usage") or {}
        text = response["choices"][0]["message"].get("content") or ""
        return {"text": text.strip(), "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "new_tokens": int(usage.get("completion_tokens", 0)), "seconds": seconds}

    def chat(self, messages: Sequence[dict], max_tokens: int = 64) -> str:
        """The assistant's reply to ``messages``, greedy-decoded."""
        return self.chat_details(messages, max_tokens)["text"]

    def close(self) -> None:
        """Free the model now rather than at garbage collection (it can be gigabytes)."""
        if self.llm is not None:
            self.llm.close()
            self.llm = None


# --------------------------------------------------------------------------
# Chat-template round trip
# --------------------------------------------------------------------------

def read_chat_template(gguf_path: str) -> Dict[str, Optional[str]]:
    """The chat template and the BOS / EOS token text stored in a GGUF file.

    Read with the ``gguf`` package when it is installed, otherwise with
    ``llama_cpp``. ``template`` is ``None`` when the file stores none.
    """
    if not Path(gguf_path).is_file():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
    if importlib.util.find_spec("gguf") is not None:
        reader = _require_gguf().GGUFReader(str(gguf_path))

        def value(key):
            item = reader.get_field(key)
            return None if item is None else item.contents()

        def token(key):
            index = value(key)
            return None if index is None else reader.get_field("tokenizer.ggml.tokens").contents(int(index))

        return {"template": value("tokenizer.chat_template"), "bos_token": token("tokenizer.ggml.bos_token_id"),
                "eos_token": token("tokenizer.ggml.eos_token_id"), "read_with": "gguf"}
    if importlib.util.find_spec("llama_cpp") is None:
        raise ImportError(f"reading GGUF metadata needs the 'gguf' package (pip install gguf=={GGUF_PIP_VERSION}) "
                          "or llama-cpp-python (pip install llama-cpp-python); neither is installed")
    llm = _require_llama_cpp().Llama(model_path=str(gguf_path), vocab_only=True, verbose=False)
    try:
        def text(token_id):
            return llm.detokenize([token_id], special=True).decode("utf-8", "replace") if token_id >= 0 else None

        return {"template": llm.metadata.get("tokenizer.chat_template"), "bos_token": text(llm.token_bos()),
                "eos_token": text(llm.token_eos()), "read_with": "llama_cpp"}
    finally:
        llm.close()


def render_stored_template(template: str, messages: Sequence[dict], bos_token: Optional[str] = None,
                           eos_token: Optional[str] = None, add_generation_prompt: bool = True) -> Dict[str, str]:
    """Render a chat template the way the GGUF runtime will.

    With llama-cpp-python installed its own formatter is used, so the string
    is the one :class:`GGUFChat` feeds the model. Without it the template is
    rendered with jinja2 under the same settings, which tests the stored
    template but not the runtime's handling of it. ``rendered_with`` says
    which. llama.cpp's own server renders templates with a C++ engine that is
    not reachable from Python; this does not test that engine.
    """
    messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    if importlib.util.find_spec("llama_cpp") is not None:
        formatter = importlib.import_module("llama_cpp.llama_chat_format").Jinja2ChatFormatter(
            template=template, eos_token=eos_token or "", bos_token=bos_token or "",
            add_generation_prompt=add_generation_prompt)
        return {"prompt": formatter(messages=messages).prompt, "rendered_with": "llama_cpp"}
    try:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        raise ImportError("rendering the stored chat template needs llama-cpp-python (pip install "
                          "llama-cpp-python) or jinja2 (pip install jinja2)") from None

    def raise_exception(message):
        raise ValueError(message)

    # trim_blocks and lstrip_blocks are what both transformers and
    # llama-cpp-python set; without them a template written for either
    # renders with extra newlines.
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    prompt = environment.from_string(template).render(
        messages=messages, add_generation_prompt=add_generation_prompt, bos_token=bos_token or "",
        eos_token=eos_token or "", raise_exception=raise_exception)
    return {"prompt": prompt, "rendered_with": "jinja2"}


def compare_prompts(hf_prompt: str, gguf_prompt: str, context: int = 40) -> Dict[str, object]:
    """Whether two rendered prompts are the same string, and where they first differ."""
    if hf_prompt == gguf_prompt:
        return {"identical": True, "first_difference": None}
    at = next((i for i, (a, b) in enumerate(zip(hf_prompt, gguf_prompt)) if a != b),
              min(len(hf_prompt), len(gguf_prompt)))
    lo = max(0, at - context)
    return {"identical": False, "first_difference": at,
            "hf_excerpt": hf_prompt[lo:at + context], "gguf_excerpt": gguf_prompt[lo:at + context]}


# A conversation with a system message and a second user turn: templates most
# often go wrong on the system role and on what separates one turn from the next.
ROUND_TRIP_MESSAGES = [
    {"role": "system", "content": "You are a careful assistant."},
    {"role": "user", "content": "Which account is this about?"},
    {"role": "assistant", "content": "The savings account ending 4821."},
    {"role": "user", "content": "Close it, please."},
]


def chat_template_round_trip(hf_tokenizer, gguf_path: str, messages: Optional[Sequence[dict]] = None) -> Dict[str, object]:
    """Check that export did not change the prompt the model will be given.

    The same conversation is rendered twice, with ``add_generation_prompt`` on:
    by the Hugging Face tokenizer's chat template, which is what the model was
    fine-tuned and evaluated with, and by the template stored in the GGUF
    metadata, which is what the runtime will apply. A difference is the classic
    silent failure after export: nothing errors, the model simply answers a
    prompt format it was never trained on, and quality drops for no visible
    reason.

    Returns a dict with ``passed``, ``identical`` (the two prompt strings),
    both strings, where they first differ, and ``reasons``. When
    llama-cpp-python is installed the tokens are compared as well
    (``tokens_identical``): the Hugging Face tokenisation of the prompt against
    the GGUF vocabulary's, which catches a special token that the exported
    vocabulary splits into pieces. Without it ``tokens_identical`` is ``None``
    and the check covers the strings only.
    """
    messages = list(messages) if messages is not None else list(ROUND_TRIP_MESSAGES)
    hf_prompt = hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    stored = read_chat_template(gguf_path)
    result: Dict[str, object] = {"passed": False, "identical": False, "tokens_identical": None,
                                 "hf_prompt": hf_prompt, "gguf_prompt": None, "first_difference": None,
                                 "template_identical": None, "read_with": stored["read_with"],
                                 "rendered_with": None, "reasons": []}
    reasons: List[str] = result["reasons"]
    if not stored["template"]:
        reasons.append("the GGUF file stores no chat template; llama.cpp will guess one from the model name, "
                       "which may not be the format the model was trained with")
        return result
    hf_template = getattr(hf_tokenizer, "chat_template", None)
    # Informational: two different template texts can render the same prompt,
    # so only the rendered strings decide the verdict.
    result["template_identical"] = hf_template == stored["template"] if isinstance(hf_template, str) else None
    rendered = render_stored_template(stored["template"], messages, stored["bos_token"], stored["eos_token"])
    result.update(gguf_prompt=rendered["prompt"], rendered_with=rendered["rendered_with"])
    result.update(compare_prompts(hf_prompt, rendered["prompt"]))
    if not result["identical"]:
        reasons.append(f"the prompt differs from character {result['first_difference']}: Hugging Face renders "
                       f"{result['hf_excerpt']!r}, the GGUF template renders {result['gguf_excerpt']!r}")
    if importlib.util.find_spec("llama_cpp") is not None:
        llm = _require_llama_cpp().Llama(model_path=str(gguf_path), vocab_only=True, verbose=False)
        try:
            # special=True parses "<|im_start|>" and the like as single
            # tokens; add_bos=False because the template writes its own.
            gguf_ids = list(llm.tokenize(rendered["prompt"].encode("utf-8"), add_bos=False, special=True))
        finally:
            llm.close()
        hf_ids = list(hf_tokenizer(hf_prompt, add_special_tokens=False)["input_ids"])
        result["tokens_identical"] = hf_ids == gguf_ids
        result["n_tokens"] = {"hf": len(hf_ids), "gguf": len(gguf_ids)}
        if result["identical"] and not result["tokens_identical"]:
            at = next((i for i, (a, b) in enumerate(zip(hf_ids, gguf_ids)) if a != b), min(len(hf_ids), len(gguf_ids)))
            reasons.append(f"the prompt strings match but their tokens differ from position {at} "
                           f"({len(hf_ids)} Hugging Face tokens, {len(gguf_ids)} GGUF tokens): the exported "
                           "vocabulary does not tokenise the prompt the way the training tokenizer does")
    result["passed"] = bool(result["identical"]) and result["tokens_identical"] is not False
    return result


# --------------------------------------------------------------------------
# Output parity
# --------------------------------------------------------------------------

@dataclass
class OutputParity:
    passed: bool
    agreement: float  # fraction of items where both models give the same (normalised) answer
    reference_accuracy: Optional[float]
    candidate_accuracy: Optional[float]
    accuracy_drop: Optional[float]
    n: int
    disagreements: List[int] = field(default_factory=list)  # item indices, for reading the raw outputs
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def output_parity(reference_outputs: Sequence[str], candidate_outputs: Sequence[str],
                  labels: Optional[Sequence[str]] = None, min_agreement: float = 0.98,
                  max_accuracy_drop: float = 0.01,
                  normalise: Optional[Callable[[str], str]] = None) -> OutputParity:
    """Decide whether an exported model may replace the one it was made from.

    The generative counterpart of :func:`forgelm.compress.parity_gate`, with
    the same two checks and the same default limits. Agreement asks whether
    the candidate gives the same answer as the reference on each item;
    accuracy drop, when ``labels`` are given, asks whether the answers that
    changed got worse. Answers are compared after ``normalise`` (default
    :func:`forgelm.finetune.normalise_answer`), so this suits short answers
    such as a label or an extracted value, not free prose.
    """
    if normalise is None:
        from .finetune import normalise_answer as normalise
    if len(reference_outputs) != len(candidate_outputs):
        raise ValueError(f"length mismatch: {len(reference_outputs)} reference outputs, "
                         f"{len(candidate_outputs)} candidate outputs")
    if labels is not None and len(labels) != len(reference_outputs):
        raise ValueError(f"length mismatch: {len(reference_outputs)} outputs, {len(labels)} labels")
    n = len(reference_outputs)
    if n == 0:
        raise ValueError("no outputs to compare")
    ref = [normalise(t) for t in reference_outputs]
    cand = [normalise(t) for t in candidate_outputs]
    disagreements = [i for i in range(n) if ref[i] != cand[i]]
    agreement = 1.0 - len(disagreements) / n
    reasons: List[str] = []
    ref_acc = cand_acc = drop = None
    if labels is not None:
        gold = [normalise(t) for t in labels]
        ref_acc = sum(r == g for r, g in zip(ref, gold)) / n
        cand_acc = sum(c == g for c, g in zip(cand, gold)) / n
        drop = ref_acc - cand_acc
        if drop > max_accuracy_drop:
            reasons.append(f"accuracy dropped {drop:.4f} (limit {max_accuracy_drop})")
    if agreement < min_agreement:
        reasons.append(f"answer agreement {agreement:.4f} below {min_agreement}")
    return OutputParity(not reasons, agreement, ref_acc, cand_acc, drop, n, disagreements, reasons)
