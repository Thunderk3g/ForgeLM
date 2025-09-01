import os
import json
import re
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from pydantic import BaseModel

# Optional heavy deps (lazy usage)
try:
    import spacy
except Exception:
    spacy = None

try:
    import nltk  # noqa: F401  # kept for users who may extend with NLTK; not required here
except Exception:
    nltk = None

from datasets import Dataset
from imblearn.over_sampling import SMOTE  # only used when balance=True


# =========================
# Pydantic request schema
# =========================
class PrepareDataRequest(BaseModel):
    input_file: str = "/app/data/input/dataset.csv"
    output_dir: str = "/app/data/prepared"
    task: str = "unsupervised"  # one of: unsupervised, supervised, instruction
    augment: bool = False
    balance: bool = False         # supervised only
    test_size: float = 0.2
    max_length: Optional[int] = 512


# =========================
# Utilities
# =========================
def load_spacy_model():
    """
    Load a small spaCy English model if available; otherwise return None.
    We keep this optional so CPU-only environments don't crash.
    """
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def clean_text(text: str, nlp=None) -> str:
    """
    Clean and normalize text: collapse whitespace, strip, and remove non-word
    symbols except common punctuation.
    """
    if not isinstance(text, str):
        text = str(text)

    # Basic cleanup
    text = re.sub(r"\s+", " ", text).strip()
    # Keep word chars, spaces and selected punctuation
    text = re.sub(r"[^\w\s\.\!\?\,\;\:\-\(\)]", "", text)
    return text


def augment_text(text: str, nlp=None) -> List[str]:
    """
    Simple augmentation: word shuffle on non-stopword tokens.
    Returns [original, augmented?].
    """
    augmented = [text]
    if nlp is None or not isinstance(text, str) or len(text) < 10:
        return augmented

    doc = nlp(text)
    words = [t.text for t in doc if not t.is_stop and not t.is_punct]
    if len(words) > 3:
        shuffled = words.copy()
        np.random.shuffle(shuffled)
        augmented.append(" ".join(shuffled))
    return augmented


def format_for_instruction_tuning(df: pd.DataFrame) -> List[Dict]:
    """
    Convert a pandas DataFrame into instruction-tuning records.
    Supports:
      - Alpaca-like: instruction / input? / output
      - OpenAI-like: prompt / completion
      - Fallback single-column text -> completion with generic instruction
    """
    out: List[Dict] = []
    has_instruction = "instruction" in df.columns
    has_output = "output" in df.columns
    has_input = "input" in df.columns
    has_prompt = "prompt" in df.columns
    has_completion = "completion" in df.columns
    has_text = "text" in df.columns

    for _, row in df.iterrows():
        if has_instruction and has_output:
            out.append({
                "instruction": str(row.get("instruction", "")),
                "input": str(row.get("input", "")) if has_input else "",
                "output": str(row.get("output", "")),
            })
        elif has_prompt and has_completion:
            out.append({
                "instruction": str(row.get("prompt", "")),
                "input": "",
                "output": str(row.get("completion", "")),
            })
        elif has_text:
            out.append({
                "instruction": "Continue the following text:",
                "input": "",
                "output": str(row.get("text", "")),
            })
    return out


# =========================
# Main entry
# =========================
def prepare_data(request: PrepareDataRequest) -> Dict[str, str]:
    """
    Prepare and clean data for fine-tuning.

    Steps:
      - Load CSV/JSON/Parquet
      - Drop dups/nulls
      - Clean object/text columns
      - Optional: length filter, augmentation, balancing (supervised)
      - Format into HF Datasets for task type
      - Train/val split, save to disk + Parquet + metadata
    """
    try:
        # --- Setup
        nlp = load_spacy_model()
        output_path = Path(request.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # --- Load
        src = Path(request.input_file)
        if not src.exists():
            raise FileNotFoundError(f"Input file not found: {request.input_file}")

        suffix = src.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(src)
        elif suffix == ".json":
            df = pd.read_json(src, lines=False)
        elif suffix == ".parquet":
            df = pd.read_parquet(src)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        print(f"[DataPrep] Original rows: {len(df)}")

        # --- Dedup / Nulls
        df = df.drop_duplicates()
        print(f"[DataPrep] After dedup: {len(df)}")
        df = df.dropna(how="all").dropna()
        print(f"[DataPrep] After dropna: {len(df)}")

        if len(df) == 0:
            raise ValueError("No data after cleaning (duplicates/nulls).")

        # --- Identify text-like columns
        text_cols = [c for c in df.columns if df[c].dtype == object]

        # --- Clean text columns (vectorized apply can be memory-heavy; do chunked)
        for col in text_cols:
            df[col] = df[col].astype(str).apply(lambda x: clean_text(x, nlp))

        # --- Optional max length filter (based on char length)
        if request.max_length:
            if "text" in df.columns:
                before = len(df)
                df = df[df["text"].str.len() <= request.max_length]
                print(f"[DataPrep] Length filter on 'text': {before} -> {len(df)}")
            elif "output" in df.columns:
                before = len(df)
                df = df[df["output"].str.len() <= request.max_length]
                print(f"[DataPrep] Length filter on 'output': {before} -> {len(df)}")

        if len(df) == 0:
            raise ValueError("No data left after length filtering.")

        # --- Optional balancing (supervised only, placeholder informative run)
        if request.balance and request.task == "supervised":
            if "label" in df.columns:
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer

                    # choose a text column (prefer 'text', else first object column)
                    text_col = "text" if "text" in df.columns else (text_cols[0] if text_cols else None)
                    if text_col is None:
                        raise ValueError("No text column available for SMOTE vectorization.")

                    vec = TfidfVectorizer(max_features=2000)
                    X = vec.fit_transform(df[text_col].astype(str)).toarray()
                    y = df["label"].values

                    smote = SMOTE(random_state=42)
                    X_bal, y_bal = smote.fit_resample(X, y)

                    # Note: Mapping synthetic vectors back to natural text is non-trivial.
                    # We log info and keep original df to avoid incoherent synthetic texts.
                    print(f"[DataPrep] SMOTE info (for reference): {len(X)} -> {len(X_bal)} vectors.")
                    print("[DataPrep] WARNING: Retaining original texts (no inverse vector mapping).")
                except Exception as e:
                    print(f"[DataPrep] SMOTE failed: {e}. Skipping balancing.")

        # --- Optional augmentation on text columns
        if request.augment and text_cols:
            aug_rows: List[dict] = []
            for _, row in df.iterrows():
                base = row.to_dict()
                aug_rows.append(base)
                for col in text_cols:
                    txt = base.get(col)
                    if isinstance(txt, str) and len(txt) > 50:
                        for aug in augment_text(txt, nlp)[1:]:
                            new_row = dict(base)
                            new_row[col] = aug
                            aug_rows.append(new_row)
            df = pd.DataFrame(aug_rows)
            print(f"[DataPrep] After augmentation: {len(df)} rows")

        if len(df) == 0:
            raise ValueError("No data to format after augmentation/cleaning.")

        # --- Task-specific formatting into datasets.Dataset
        if request.task == "instruction":
            # Create instruction triples
            formatted = format_for_instruction_tuning(df)

            # Try Unsloth ShareGPT path
            dataset = None
            try:
                from unsloth.chat_templates import standardize_sharegpt

                sharegpt_records = []
                for item in formatted:
                    user_content = item["instruction"]
                    if item.get("input", "").strip():
                        user_content += f"\n\n{item['input']}"
                    sharegpt_records.append({
                        "conversations": [
                            {"from": "human", "value": user_content},
                            {"from": "gpt", "value": item["output"]},
                        ]
                    })

                dataset = Dataset.from_list(sharegpt_records)
                dataset = standardize_sharegpt(dataset)
                print(f"[DataPrep] Unsloth ShareGPT formatting applied to {len(dataset)} samples")

            except Exception as e:
                print(f"[DataPrep] Unsloth ShareGPT unavailable/failed ({e}); using fallback format.")
                enhanced = []
                for item in formatted:
                    user_content = item["instruction"]
                    if item.get("input", "").strip():
                        user_content += f"\n\n{item['input']}"
                    text = (
                        "<|im_start|>user\n"
                        f"{user_content}"
                        "<|im_end|>\n"
                        "<|im_start|>assistant\n"
                        f"{item['output']}"
                        "<|im_end|>"
                    )
                    enhanced.append({"text": text})
                dataset = Dataset.from_list(enhanced)

        elif request.task == "supervised":
            # Expect at least: text + label
            if "text" not in df.columns:
                # if no explicit 'text', concatenate object columns
                if text_cols:
                    df["text"] = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
                else:
                    raise ValueError("Supervised requires a 'text' column or textual columns to combine.")
            dataset = Dataset.from_pandas(df.reset_index(drop=True))

        else:
            # Unsupervised
            if "text" not in df.columns:
                if text_cols:
                    df["text"] = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
                else:
                    raise ValueError("Unsupervised requires at least one textual column.")
            dataset = Dataset.from_pandas(df[["text"]].reset_index(drop=True))

        # --- Split (multiprocessing-enabled map example if needed later)
        if len(dataset) > 100:
            split = dataset.train_test_split(test_size=request.test_size, seed=42)
            train_ds, val_ds = split["train"], split["test"]
        else:
            train_ds = dataset
            val_ds = dataset.select(range(min(20, len(dataset))))

        # --- Save (HF dataset + Parquet)
        train_path = output_path / "train"
        val_path = output_path / "val"
        train_ds.save_to_disk(str(train_path))
        val_ds.save_to_disk(str(val_path))

        # Save also as parquet for quick inspection (requires pyarrow, which datasets pulls in)
        train_parquet = output_path / "train.parquet"
        val_parquet = output_path / "val.parquet"
        try:
            train_ds.to_parquet(str(train_parquet))
            val_ds.to_parquet(str(val_parquet))
        except Exception as e:
            print(f"[DataPrep] to_parquet failed ({e}); skipping parquet export.")

        # --- Metadata
        has_sharegpt = "conversations" in train_ds.column_names
        fmt = "sharegpt" if has_sharegpt else ("instruction" if request.task == "instruction" else "text")
        dataset_info = {
            "task_type": request.task,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "format": fmt,
            "has_unsloth_formatting": has_sharegpt,
            "recommended_packing": True,
            "estimated_tokens": (len(train_ds) * request.max_length) if request.max_length else "unknown",
        }
        with open(output_path / "dataset_info.json", "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2)

        return {
            "status": "success",
            "message": f"Data prepared successfully. Train: {len(train_ds)}, Val: {len(val_ds)}",
            "train_path": str(train_path),
            "val_path": str(val_path),
            "optimization_info": {
                "format_applied": fmt,
                "unsloth_optimized": has_sharegpt,
                "packing_recommended": True,
                "total_samples": len(train_ds) + len(val_ds),
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Data preparation failed: {str(e)}",
            "train_path": "",
            "val_path": "",
        }


# Optional: quick CLI hook for local testing
if __name__ == "__main__":
    req = PrepareDataRequest()
    print(prepare_data(req))
