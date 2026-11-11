"""Compile a student to a portable artefact, and prove it still works.

The step most pipelines skip is re-running the evaluation on the converted
model. Quantisation failures are silent: the file exports, loads and returns
plausible-looking numbers. So :func:`parity_gate` is part of the export, not
an optional extra, and :func:`export_and_gate` refuses to bless an int8 model
that drifts from the fp32 original.

ONNX Runtime is the common denominator here: one ``.onnx`` file serves on
x86 and arm64 CPUs and on GPUs through execution providers, with torch not
installed at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .metrics import classification_report


@dataclass
class ParityResult:
    passed: bool
    agreement: float  # fraction of items where both models pick the same class
    reference_accuracy: Optional[float]
    candidate_accuracy: Optional[float]
    accuracy_drop: Optional[float]
    max_prob_delta: float
    reasons: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def parity_gate(reference_probs, candidate_probs, y_true: Optional[Sequence[int]] = None,
                min_agreement: float = 0.98, max_accuracy_drop: float = 0.01) -> ParityResult:
    """Decide whether a converted model may replace the original.

    Two independent checks, because each catches what the other misses:
    agreement (does it make the same decisions?) and, when labels are given,
    accuracy drop (are the changed decisions worse?).
    """
    ref = np.asarray(reference_probs, dtype=np.float64)
    cand = np.asarray(candidate_probs, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {cand.shape}")
    agreement = float(np.mean(ref.argmax(1) == cand.argmax(1)))
    reasons: List[str] = []
    ref_acc = cand_acc = drop = None
    if y_true is not None:
        y = np.asarray(y_true)
        ref_acc = float(np.mean(ref.argmax(1) == y))
        cand_acc = float(np.mean(cand.argmax(1) == y))
        drop = ref_acc - cand_acc
        if drop > max_accuracy_drop:
            reasons.append(f"accuracy dropped {drop:.4f} (limit {max_accuracy_drop})")
    if agreement < min_agreement:
        reasons.append(f"decision agreement {agreement:.4f} below {min_agreement}")
    return ParityResult(not reasons, agreement, ref_acc, cand_acc, drop,
                        float(np.max(np.abs(ref - cand))), reasons)


# --------------------------------------------------------------------------
# ONNX export of a fine-tuned sequence classifier
# --------------------------------------------------------------------------

def export_classifier_onnx(model_dir: str, out_path: str, max_length: int = 128, opset: int = 17) -> str:
    """Export a saved ``FineTunedStudent`` directory to a single fp32 ONNX file.

    Attention is forced to the eager implementation: ModernBERT-family models
    fail to export through their compiled / flash-attention paths.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="eager")
    except (TypeError, ValueError):
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    if hasattr(model.config, "reference_compile"):
        model.config.reference_compile = False
    model.eval()

    sample = tok(["export sample text", "a second, somewhat longer sample sentence"], padding=True,
                 truncation=True, max_length=max_length, return_tensors="pt")
    names = [n for n in tok.model_input_names if n in sample]

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *args):
            return self.inner(**dict(zip(names, args))).logits

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dynamic = {n: {0: "batch", 1: "sequence"} for n in names}
    dynamic["logits"] = {0: "batch"}
    with torch.inference_mode():
        torch.onnx.export(_Wrapper(model), tuple(sample[n] for n in names), out_path, input_names=names,
                          output_names=["logits"], dynamic_axes=dynamic, opset_version=opset, dynamo=False)
    return out_path


def quantize_dynamic_int8(fp32_path: str, int8_path: str, has_vnni: bool = True, per_channel: bool = False) -> str:
    """Dynamic int8 quantisation of an ONNX file.

    ``has_vnni=False`` enables ``reduce_range``: on AVX2/AVX-512 CPUs without
    VNNI the u8*s8 multiply can saturate and corrupt results, and ONNX Runtime
    documents 7-bit weights as the fix. Pass ``HardwareProfile.has_vnni`` of
    the *deployment* machine, not the build machine.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8, per_channel=per_channel,
                     reduce_range=not has_vnni)
    return int8_path


class OnnxClassifier:
    """Serve an exported classifier with onnxruntime + tokenizers only (no torch)."""

    def __init__(self, onnx_path: str, tokenizer_dir: str, labels: Sequence[str], max_length: int = 128,
                 temperature: float = 1.0, providers: Optional[Sequence[str]] = None, threads: Optional[int] = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(onnx_path, sess_options=opts,
                                            providers=list(providers) if providers else ["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        pad_id = self.tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            pad_id = self.tokenizer.token_to_id("<pad>") or 0
        self.tokenizer.enable_padding(pad_id=pad_id)
        self.labels, self.temperature = list(labels), temperature

    @classmethod
    def from_student_dir(cls, onnx_path: str, student_dir: str, **kwargs) -> "OnnxClassifier":
        meta = json.loads(Path(student_dir, "student.json").read_text())
        return cls(onnx_path, student_dir, meta["labels"], meta.get("max_length", 128),
                   meta.get("temperature", 1.0), **kwargs)

    def predict_proba(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        out = []
        texts = list(texts)
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer.encode_batch(texts[start:start + batch_size])
            feeds = {}
            if "input_ids" in self.input_names:
                feeds["input_ids"] = np.asarray([e.ids for e in enc], dtype=np.int64)
            if "attention_mask" in self.input_names:
                feeds["attention_mask"] = np.asarray([e.attention_mask for e in enc], dtype=np.int64)
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.asarray([e.type_ids for e in enc], dtype=np.int64)
            logits = self.session.run(["logits"], feeds)[0].astype(np.float64) / max(self.temperature, 1e-6)
            logits -= logits.max(1, keepdims=True)
            probs = np.exp(logits)
            out.append(probs / probs.sum(1, keepdims=True))
        return np.concatenate(out)

    def predict(self, texts: Sequence[str]) -> List[str]:
        return [self.labels[i] for i in self.predict_proba(texts).argmax(1)]


def export_and_gate(student_dir: str, out_dir: str, texts: Sequence[str], y_true: Optional[Sequence[int]],
                    reference_probs: np.ndarray, has_vnni: bool = True, min_agreement: float = 0.98,
                    max_accuracy_drop: float = 0.01, temperature: Optional[float] = None) -> Dict[str, dict]:
    """Export fp32 and int8 ONNX, gate each against the PyTorch reference.

    ``reference_probs`` must come from the PyTorch student at the same
    temperature the ONNX classifiers will use: the one stored in
    ``student.json`` by default, or ``temperature`` when given. Mixing the two
    leaves accuracy and agreement intact but makes probability deltas and
    calibration figures incomparable.

    Returns a dict with a ``recommended`` key naming the smallest artefact that
    passed. If int8 fails the gate the fp32 file is recommended instead, and
    the failure is recorded rather than hidden.
    """
    meta = json.loads(Path(student_dir, "student.json").read_text())
    max_length = meta.get("max_length", 128)
    os.makedirs(out_dir, exist_ok=True)
    fp32 = export_classifier_onnx(student_dir, os.path.join(out_dir, "model.fp32.onnx"), max_length)
    int8 = quantize_dynamic_int8(fp32, os.path.join(out_dir, "model.int8.onnx"), has_vnni=has_vnni)

    report: Dict[str, dict] = {}
    for name, path in (("onnx_fp32", fp32), ("onnx_int8", int8)):
        clf = OnnxClassifier.from_student_dir(path, student_dir)
        if temperature is not None:
            clf.temperature = temperature
        probs = clf.predict_proba(texts)
        gate = parity_gate(reference_probs, probs, y_true, min_agreement, max_accuracy_drop)
        report[name] = {"path": path, "size_mb": os.path.getsize(path) / 1e6, "parity": gate.to_dict()}
        if y_true is not None:
            report[name]["metrics"] = classification_report(y_true, probs)
    passed = [n for n in ("onnx_int8", "onnx_fp32") if report[n]["parity"]["passed"]]
    report["recommended"] = {"artefact": passed[0] if passed else "pytorch",
                             "reason": "smallest artefact that passed the parity gate" if passed
                             else "no ONNX artefact passed the parity gate; keep the PyTorch model"}
    return report
