"""Evaluation contracts, run records, model cards and an ML bill of materials.

For a regulated deployer the trained weights are the least of what must be
kept. An auditor asks: which data, which recipe, which base model at which
revision, on what hardware, evaluated how, and is this file the one that was
evaluated? Everything here is plain JSON / Markdown written next to the
artefact, with no tracking server required.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dataset(texts: Sequence[str], labels: Optional[Iterable] = None) -> str:
    """Order-sensitive hash of a labelled dataset."""
    h = hashlib.sha256()
    labels = list(labels) if labels is not None else [None] * len(texts)
    for text, label in zip(texts, labels):
        h.update(str(text).encode("utf-8"))
        h.update(b"\x1f")
        h.update(str(label).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def hash_directory(directory: str) -> Dict[str, str]:
    base = Path(directory)
    return {str(p.relative_to(base)): sha256_file(str(p)) for p in sorted(base.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------
# Evaluation contract
# --------------------------------------------------------------------------

@dataclass
class EvalContract:
    """A frozen acceptance test: this data, this metric, this threshold.

    Create it once, before training. Every later stage (a new student, a
    calibration step, a quantised export, a re-train on a new base model) is
    checked against the same contract, so "it still works" has one meaning.
    """

    name: str
    dataset_sha256: str
    n_examples: int
    thresholds: Dict[str, float]  # metric -> minimum (or maximum, see lower_is_better)
    lower_is_better: List[str] = field(default_factory=lambda: ["ece", "brier", "nll", "latency_ms_p95"])
    created_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    @classmethod
    def create(cls, name: str, texts: Sequence[str], labels: Iterable, thresholds: Dict[str, float]):
        return cls(name, sha256_dataset(texts, labels), len(texts), dict(thresholds))

    def check(self, metrics: Dict[str, float], texts: Optional[Sequence[str]] = None,
              labels: Optional[Iterable] = None) -> Dict[str, object]:
        """Compare ``metrics`` with the thresholds; verify the data if given."""
        failures: List[str] = []
        if texts is not None:
            if sha256_dataset(texts, labels) != self.dataset_sha256:
                failures.append("evaluation data does not match the contract's dataset hash")
        for metric, limit in self.thresholds.items():
            if metric not in metrics:
                failures.append(f"metric {metric!r} missing")
                continue
            value = metrics[metric]
            if metric in self.lower_is_better:
                if value > limit:
                    failures.append(f"{metric} {value:.4f} above limit {limit}")
            elif value < limit:
                failures.append(f"{metric} {value:.4f} below minimum {limit}")
        return {"contract": self.name, "passed": not failures, "failures": failures}

    def save(self, path: str) -> str:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: str) -> "EvalContract":
        return cls(**json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------
# Run record
# --------------------------------------------------------------------------

@dataclass
class RunRecord:
    name: str
    task: str
    recipe: Dict[str, object]
    data: Dict[str, object]  # split name -> {"sha256":..., "n":...}
    metrics: Dict[str, object]
    hardware: Dict[str, object] = field(default_factory=dict)
    base_models: List[Dict[str, str]] = field(default_factory=list)  # {"name", "revision", "licence"}
    artefacts: Dict[str, str] = field(default_factory=dict)  # relative path -> sha256
    contract: Optional[Dict[str, object]] = None
    notes: List[str] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def save(self, directory: str) -> str:
        path = Path(directory) / "run_record.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return str(path)

    def model_card(self) -> str:
        """A Markdown model card generated from the record."""
        lines = [f"# {self.name}", "", f"Run `{self.run_id}`, created {self.created_utc}.", "",
                 "## Intended use", "", f"Task: {self.task}.", "",
                 "This card is generated from the run record. It states what was measured; it does not "
                 "establish fitness for any regulated purpose.", "", "## Base models", ""]
        lines += [f"- {m.get('name')} (revision {m.get('revision', 'unpinned')}, licence {m.get('licence', 'unknown')})"
                  for m in self.base_models] or ["- none (trained from scratch or classical model)"]
        lines += ["", "## Data", ""]
        lines += [f"- {split}: {info.get('n')} examples, sha256 `{str(info.get('sha256'))[:16]}...`"
                  for split, info in self.data.items()]
        lines += ["", "## Recipe", "", "```json", json.dumps(self.recipe, indent=2, default=str), "```", "",
                  "## Results", ""]
        for key, value in self.metrics.items():
            lines.append(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}")
        if self.contract:
            verdict = "passed" if self.contract.get("passed") else "FAILED"
            lines += ["", f"Evaluation contract `{self.contract.get('contract')}`: {verdict}."]
            lines += [f"- {f}" for f in self.contract.get("failures", [])]
        lines += ["", "## Hardware", "",
                  f"- trained on: {self.hardware.get('device_class', 'unknown')} "
                  f"({self.hardware.get('cpu_model', '')}, {self.hardware.get('ram_total_gb', '?')} GiB RAM)"]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        lines += ["", "## Artefacts", ""] + [f"- `{p}` sha256 `{h[:16]}...`" for p, h in self.artefacts.items()]
        return "\n".join(lines) + "\n"

    def ml_bom(self) -> dict:
        """A minimal CycloneDX 1.6 ML-BOM for a locally produced model.

        Existing AI-BOM generators assume the model lives on a public hub;
        this emits one from the local run record instead.
        """
        components = [{
            "type": "machine-learning-model", "bom-ref": f"model:{self.run_id}", "name": self.name,
            "version": self.run_id,
            "hashes": [{"alg": "SHA-256", "content": h} for h in self.artefacts.values()],
            "modelCard": {
                "modelParameters": {"task": self.task,
                                    "datasets": [{"type": "dataset", "name": split,
                                                  "contents": {"properties": [
                                                      {"name": "sha256", "value": str(info.get("sha256"))},
                                                      {"name": "examples", "value": str(info.get("n"))}]}}
                                                 for split, info in self.data.items()]},
                "quantitativeAnalysis": {"performanceMetrics": [
                    {"type": k, "value": str(v)} for k, v in self.metrics.items() if isinstance(v, (int, float))]},
            },
        }]
        for m in self.base_models:
            components.append({"type": "machine-learning-model", "bom-ref": f"base:{m.get('name')}",
                               "name": m.get("name"), "version": m.get("revision", "unpinned"),
                               "licenses": [{"license": {"name": m.get("licence", "unknown")}}]})
        return {"bomFormat": "CycloneDX", "specVersion": "1.6", "serialNumber": f"urn:uuid:{uuid.uuid4()}",
                "version": 1, "metadata": {"timestamp": self.created_utc,
                                           "tools": [{"name": "forgelm", "version": "0.1.0"}]},
                "components": components,
                "dependencies": [{"ref": f"model:{self.run_id}",
                                  "dependsOn": [f"base:{m.get('name')}" for m in self.base_models]}]}

    def write_all(self, directory: str) -> List[str]:
        out = [self.save(directory)]
        card = Path(directory) / "MODEL_CARD.md"
        card.write_text(self.model_card())
        bom = Path(directory) / "ml-bom.cdx.json"
        bom.write_text(json.dumps(self.ml_bom(), indent=2))
        return out + [str(card), str(bom)]
