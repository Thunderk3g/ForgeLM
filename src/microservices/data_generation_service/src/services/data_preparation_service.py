"""
Data Preparation Service for LLMForge

Handles data cleaning, processing, and formatting for fine-tuning tasks.
Port of the complete logic from utils/legacy/data_preparation.py
"""

import os
import json
import re
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import pandas as pd
from datasets import Dataset

# Optional heavy deps (lazy usage)
try:
    import spacy
except Exception:
    spacy = None

try:
    import nltk  # noqa: F401  # kept for users who may extend with NLTK; not required here
except Exception:
    nltk = None

# Optional heavy deps (lazy usage)
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

try:
    from sentence_transformers import SentenceTransformer
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.utils import PlotlyJSONEncoder
except Exception as e:
    SentenceTransformer = None
    torch = None
    cosine_similarity = None
    PCA = None
    sns = None
    plt = None
    px = None
    go = None
    PlotlyJSONEncoder = None

from models.schemas import (
    PrepareDataRequest, PrepareDataResponse, EmbeddingModelConfig, 
    PreprocessingConfig, PreprocessingAnalytics, TensorVisualizationData,
    EmbeddingModelInfo, EmbeddingModelListResponse
)
from utils.logging import get_logger

logger = get_logger(__name__)


class DataPreparationService:
    """Service for data preparation and cleaning"""
    
    def __init__(self):
        self.nlp = self._load_spacy_model()
        self.embedding_models = {}
        self.available_models = self._get_available_embedding_models()
    
    def _load_spacy_model(self):
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
    
    def _get_available_embedding_models(self) -> List[EmbeddingModelInfo]:
        """Get list of available embedding models"""
        models = [
            EmbeddingModelInfo(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_type="sentence_transformer",
                description="High-quality general-purpose embedding model",
                max_sequence_length=512,
                embedding_dimension=768,
                supported_tasks=["similarity", "classification", "clustering"],
                performance_metrics={"accuracy": 0.89, "speed": 0.85},
                memory_requirements="~1.2GB",
                is_available=SentenceTransformer is not None
            ),
            EmbeddingModelInfo(
                model_name="BAAI/bge-base-en-v1.5",
                model_type="sentence_transformer", 
                description="State-of-the-art general embedding model",
                max_sequence_length=512,
                embedding_dimension=768,
                supported_tasks=["similarity", "retrieval", "classification"],
                performance_metrics={"accuracy": 0.92, "speed": 0.82},
                memory_requirements="~1.1GB",
                is_available=SentenceTransformer is not None
            ),
            EmbeddingModelInfo(
                model_name="jinaai/jina-embeddings-v2-small-en",
                model_type="sentence_transformer",
                description="Efficient small embedding model for fast processing",
                max_sequence_length=8192,
                embedding_dimension=512,
                supported_tasks=["similarity", "retrieval"],
                performance_metrics={"accuracy": 0.85, "speed": 0.95},
                memory_requirements="~400MB",
                is_available=SentenceTransformer is not None
            )
        ]
        return models
    
    def get_embedding_models(self) -> EmbeddingModelListResponse:
        """Get available embedding models"""
        categories = list(set(model.model_type for model in self.available_models))
        return EmbeddingModelListResponse(
            models=self.available_models,
            total_count=len(self.available_models),
            categories=categories
        )
    
    def _load_embedding_model(self, config: EmbeddingModelConfig) -> Optional[Any]:
        """Load and cache embedding model"""
        if SentenceTransformer is None:
            logger.warning("sentence-transformers not available")
            return None
        
        model_key = f"{config.model_name}_{config.max_sequence_length}"
        
        if model_key not in self.embedding_models:
            try:
                model = SentenceTransformer(config.model_name)
                if config.device:
                    model = model.to(config.device)
                self.embedding_models[model_key] = model
                logger.info(f"Loaded embedding model: {config.model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model {config.model_name}: {e}")
                return None
        
        return self.embedding_models[model_key]
    
    def _generate_embeddings(self, texts: List[str], config: EmbeddingModelConfig) -> Optional[np.ndarray]:
        """Generate embeddings for texts"""
        model = self._load_embedding_model(config)
        if model is None:
            return None
        
        try:
            embeddings = model.encode(
                texts,
                batch_size=config.batch_size,
                normalize_embeddings=config.normalize_embeddings,
                show_progress_bar=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return None
    
    def _remove_duplicates_by_embedding(self, df: pd.DataFrame, config: EmbeddingModelConfig, 
                                      preprocessing_config: PreprocessingConfig) -> tuple[pd.DataFrame, Dict[str, int]]:
        """Remove duplicates using embedding similarity"""
        if not preprocessing_config.remove_duplicates_by_embedding:
            return df, {"duplicates_removed": 0, "original_count": len(df)}
        
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols:
            return df, {"duplicates_removed": 0, "original_count": len(df)}
        
        # Combine text columns for embedding
        texts = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
        
        embeddings = self._generate_embeddings(texts, config)
        if embeddings is None:
            return df, {"duplicates_removed": 0, "original_count": len(df), "error": "embedding_failed"}
        
        try:
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find duplicates above threshold
            duplicates_to_remove = set()
            original_count = len(df)
            
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i][j] >= preprocessing_config.similarity_threshold:
                        duplicates_to_remove.add(j)
            
            # Remove duplicates
            indices_to_keep = [i for i in range(len(df)) if i not in duplicates_to_remove]
            filtered_df = df.iloc[indices_to_keep].reset_index(drop=True)
            
            stats = {
                "duplicates_removed": len(duplicates_to_remove),
                "original_count": original_count,
                "final_count": len(filtered_df),
                "similarity_threshold": preprocessing_config.similarity_threshold
            }
            
            logger.info(f"Removed {len(duplicates_to_remove)} duplicates using embedding similarity")
            return filtered_df, stats
            
        except Exception as e:
            logger.error(f"Error in embedding-based deduplication: {e}")
            return df, {"duplicates_removed": 0, "original_count": len(df), "error": str(e)}
    
    def _generate_correlation_analysis(self, df: pd.DataFrame, embeddings: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Generate correlation analysis for visualization"""
        if sns is None or plt is None:
            return {"error": "visualization libraries not available"}
        
        try:
            correlation_data = {}
            
            # Text-based correlations
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlation_matrix = df[numeric_cols].corr()
                correlation_data["numeric_correlation"] = correlation_matrix.to_dict()
            
            # Embedding-based correlations if available
            if embeddings is not None and len(embeddings) > 1:
                embedding_corr = np.corrcoef(embeddings)
                correlation_data["embedding_correlation"] = {
                    "shape": embedding_corr.shape,
                    "mean_correlation": float(np.mean(embedding_corr)),
                    "std_correlation": float(np.std(embedding_corr)),
                    "max_correlation": float(np.max(embedding_corr[embedding_corr < 1.0])),
                    "min_correlation": float(np.min(embedding_corr))
                }
            
            return correlation_data
            
        except Exception as e:
            logger.error(f"Error generating correlation analysis: {e}")
            return {"error": str(e)}
    
    def _generate_tensor_visualizations(self, embeddings: np.ndarray, texts: List[str]) -> Optional[TensorVisualizationData]:
        """Generate tensor visualizations"""
        if embeddings is None or PCA is None:
            return None
        
        try:
            # Basic tensor stats
            tensor_stats = {
                "shape": list(embeddings.shape),
                "mean": float(np.mean(embeddings)),
                "std": float(np.std(embeddings)),
                "min": float(np.min(embeddings)),
                "max": float(np.max(embeddings)),
                "norm": float(np.linalg.norm(embeddings))
            }
            
            # PCA for dimensionality reduction
            pca = PCA(n_components=min(3, embeddings.shape[1]))
            pca_embeddings = pca.fit_transform(embeddings)
            
            visualization_data = {
                "pca_components": pca_embeddings.tolist(),
                "explained_variance": pca.explained_variance_ratio_.tolist(),
                "sample_texts": texts[:10] if len(texts) > 10 else texts
            }
            
            # Generate correlation matrix if embeddings are not too large
            correlation_matrix = None
            if embeddings.shape[0] <= 1000:
                correlation_matrix = np.corrcoef(embeddings).tolist()
            
            return TensorVisualizationData(
                tensor_shape=list(embeddings.shape),
                tensor_stats=tensor_stats,
                visualization_data=visualization_data,
                correlation_matrix=correlation_matrix,
                embedding_projections={
                    "pca_2d": pca_embeddings[:, :2].tolist() if pca_embeddings.shape[1] >= 2 else None,
                    "pca_3d": pca_embeddings[:, :3].tolist() if pca_embeddings.shape[1] >= 3 else None
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating tensor visualizations: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
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
    
    def _clean_text_aggressive(self, text: str) -> str:
        """
        Aggressive text cleaning: remove special characters, normalize cases, etc.
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove extra punctuation and special characters
        text = re.sub(r'[^\w\s\.\!\?\,]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short words (less than 2 characters)
        text = ' '.join([word for word in text.split() if len(word) >= 2])
        
        return text

    def _augment_text(self, text: str) -> List[str]:
        """
        Simple augmentation: word shuffle on non-stopword tokens.
        Returns [original, augmented?].
        """
        augmented = [text]
        if self.nlp is None or not isinstance(text, str) or len(text) < 10:
            return augmented

        doc = self.nlp(text)
        words = [t.text for t in doc if not t.is_stop and not t.is_punct]
        if len(words) > 3:
            shuffled = words.copy()
            np.random.shuffle(shuffled)
            augmented.append(" ".join(shuffled))
        return augmented

    def _format_for_instruction_tuning(self, df: pd.DataFrame) -> List[Dict]:
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

    async def prepare_data(self, request: PrepareDataRequest) -> PrepareDataResponse:
        """
        Prepare and clean data for fine-tuning with enhanced preprocessing capabilities.

        Steps:
          - Load CSV/JSON/Parquet
          - Drop dups/nulls
          - Clean object/text columns
          - Optional: embedding-based processing, visualization generation
          - Optional: length filter, augmentation, balancing (supervised)
          - Format into HF Datasets for task type
          - Train/val split, save to disk + Parquet + metadata
        """
        start_time = time.time()
        preprocessing_analytics = PreprocessingAnalytics()
        
        try:
            # --- Setup
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

            logger.info(f"[DataPrep] Original rows: {len(df)}")

            # --- Dedup / Nulls
            df = df.drop_duplicates()
            logger.info(f"[DataPrep] After dedup: {len(df)}")
            df = df.dropna(how="all").dropna()
            logger.info(f"[DataPrep] After dropna: {len(df)}")

            if len(df) == 0:
                raise ValueError("No data after cleaning (duplicates/nulls).")

            # --- Identify text-like columns
            text_cols = [c for c in df.columns if df[c].dtype == object]

            # --- Clean text columns (vectorized apply can be memory-heavy; do chunked)
            preprocessing_config = request.preprocessing_config or PreprocessingConfig()
            
            # Apply text cleaning based on level
            for col in text_cols:
                if preprocessing_config.text_cleaning_level == "minimal":
                    df[col] = df[col].astype(str).str.strip()
                elif preprocessing_config.text_cleaning_level == "aggressive":
                    df[col] = df[col].astype(str).apply(lambda x: self._clean_text_aggressive(x))
                else:  # standard
                    df[col] = df[col].astype(str).apply(lambda x: self._clean_text(x))
            
            # --- Generate embeddings if requested
            embeddings = None
            duplicate_removal_stats = {"duplicates_removed": 0, "original_count": len(df)}
            
            if preprocessing_config.enable_embeddings and preprocessing_config.embedding_config:
                logger.info("Generating embeddings for preprocessing")
                
                # Combine text columns for embedding
                texts = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
                embeddings = self._generate_embeddings(texts, preprocessing_config.embedding_config)
                
                if embeddings is not None:
                    preprocessing_analytics.embedding_stats = {
                        "model_name": preprocessing_config.embedding_config.model_name,
                        "embedding_shape": list(embeddings.shape),
                        "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
                        "embedding_dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else 0
                    }
                    
                    # Embedding-based duplicate removal
                    df, duplicate_removal_stats = self._remove_duplicates_by_embedding(
                        df, preprocessing_config.embedding_config, preprocessing_config
                    )
                    logger.info(f"After embedding-based dedup: {len(df)} rows")
                    
                    # Regenerate embeddings for remaining data if duplicates were removed
                    if duplicate_removal_stats["duplicates_removed"] > 0:
                        texts = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
                        embeddings = self._generate_embeddings(texts, preprocessing_config.embedding_config)

            # --- Optional max length filter (based on char length)
            if request.max_length:
                if "text" in df.columns:
                    before = len(df)
                    df = df[df["text"].str.len() <= request.max_length]
                    logger.info(f"[DataPrep] Length filter on 'text': {before} -> {len(df)}")
                elif "output" in df.columns:
                    before = len(df)
                    df = df[df["output"].str.len() <= request.max_length]
                    logger.info(f"[DataPrep] Length filter on 'output': {before} -> {len(df)}")

            if len(df) == 0:
                raise ValueError("No data left after length filtering.")

            # --- Optional balancing (supervised only, placeholder informative run)
            if request.balance and request.task == "supervised":
                if "label" in df.columns:
                    try:
                        if SMOTE is None:
                            logger.warning("[DataPrep] imblearn not available. Skipping SMOTE balancing.")
                        else:
                            try:
                                from sklearn.feature_extraction.text import TfidfVectorizer
                            except ImportError:
                                logger.warning("[DataPrep] sklearn not available. Skipping SMOTE balancing.")
                                raise ValueError("sklearn not available for SMOTE balancing")

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
                            logger.info(f"[DataPrep] SMOTE info (for reference): {len(X)} -> {len(X_bal)} vectors.")
                            logger.warning("[DataPrep] WARNING: Retaining original texts (no inverse vector mapping).")
                    except Exception as e:
                        logger.warning(f"[DataPrep] SMOTE failed: {e}. Skipping balancing.")

            # --- Optional augmentation on text columns
            if request.augment and text_cols:
                aug_rows: List[dict] = []
                for _, row in df.iterrows():
                    base = row.to_dict()
                    aug_rows.append(base)
                    for col in text_cols:
                        txt = base.get(col)
                        if isinstance(txt, str) and len(txt) > 50:
                            for aug in self._augment_text(txt)[1:]:
                                new_row = dict(base)
                                new_row[col] = aug
                                aug_rows.append(new_row)
                df = pd.DataFrame(aug_rows)
                logger.info(f"[DataPrep] After augmentation: {len(df)} rows")

            if len(df) == 0:
                raise ValueError("No data to format after augmentation/cleaning.")

            # --- Task-specific formatting into datasets.Dataset
            if request.task == "instruction":
                # Create instruction triples
                formatted = self._format_for_instruction_tuning(df)

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
                    logger.info(f"[DataPrep] Unsloth ShareGPT formatting applied to {len(dataset)} samples")

                except Exception as e:
                    logger.warning(f"[DataPrep] Unsloth ShareGPT unavailable/failed ({e}); using fallback format.")
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

            # Create dataset_dict.json for proper DatasetDict loading
            dataset_dict_info = {
                "splits": {
                    "train": {
                        "name": "train",
                        "num_bytes": train_ds.data.nbytes if hasattr(train_ds.data, 'nbytes') else None,
                        "num_examples": len(train_ds),
                        "shard_lengths": None,
                        "dataset_name": "prepared_dataset"
                    },
                    "val": {
                        "name": "val", 
                        "num_bytes": val_ds.data.nbytes if hasattr(val_ds.data, 'nbytes') else None,
                        "num_examples": len(val_ds),
                        "shard_lengths": None,
                        "dataset_name": "prepared_dataset"
                    }
                }
            }
            with open(output_path / "dataset_dict.json", "w", encoding="utf-8") as f:
                json.dump(dataset_dict_info, f, indent=2)
            logger.info("[DataPrep] Created dataset_dict.json for DatasetDict loading")

            # Save also as parquet for quick inspection (requires pyarrow, which datasets pulls in)
            train_parquet = output_path / "train.parquet"
            val_parquet = output_path / "val.parquet"
            try:
                train_ds.to_parquet(str(train_parquet))
                val_ds.to_parquet(str(val_parquet))
            except Exception as e:
                logger.warning(f"[DataPrep] to_parquet failed ({e}); skipping parquet export.")

            # --- Generate analytics and visualizations
            if preprocessing_config.generate_correlations and embeddings is not None:
                correlation_analysis = self._generate_correlation_analysis(df, embeddings)
                preprocessing_analytics.correlation_analysis = correlation_analysis
                
            if preprocessing_config.create_tensor_visualizations and embeddings is not None:
                texts = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
                tensor_visualizations = self._generate_tensor_visualizations(embeddings, texts)
                preprocessing_analytics.tensor_visualizations = tensor_visualizations
                
            # Quality analysis
            if preprocessing_config.quality_analysis:
                quality_metrics = await self._analyze_dataset_quality_enhanced(df)
                preprocessing_analytics.quality_metrics = quality_metrics
                
            # Token analysis
            if preprocessing_config.token_analysis:
                token_distribution = await self._analyze_token_distribution(df, text_cols)
                preprocessing_analytics.token_distribution = token_distribution
            
            # Set processing time and duplicate removal stats
            preprocessing_analytics.preprocessing_time_ms = (time.time() - start_time) * 1000
            preprocessing_analytics.duplicate_removal_stats = duplicate_removal_stats

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
                "preprocessing_config": preprocessing_config.dict() if preprocessing_config else None,
                "has_embeddings": embeddings is not None,
                "embedding_dimension": embeddings.shape[1] if embeddings is not None else None,
            }
            with open(output_path / "dataset_info.json", "w", encoding="utf-8") as f:
                json.dump(dataset_info, f, indent=2)
                
            # Save preprocessing analytics
            if preprocessing_analytics:
                analytics_path = output_path / "preprocessing_analytics.json"
                with open(analytics_path, "w", encoding="utf-8") as f:
                    json.dump(preprocessing_analytics.dict(), f, indent=2, default=str)
                    
            # Save embeddings if generated
            if embeddings is not None:
                embeddings_path = output_path / "embeddings.npy"
                np.save(str(embeddings_path), embeddings)
                logger.info(f"Saved embeddings to {embeddings_path}")
                
            # Generate visualization URLs
            visualization_urls = {}
            if preprocessing_config.create_tensor_visualizations and tensor_visualizations:
                visualization_urls["tensor_analysis"] = f"/visualizations/tensor/{output_path.name}"
            if preprocessing_config.generate_correlations and correlation_analysis:
                visualization_urls["correlation_matrix"] = f"/visualizations/correlation/{output_path.name}"

            return PrepareDataResponse(
                status="success",
                message=f"Data prepared successfully. Train: {len(train_ds)}, Val: {len(val_ds)}",
                train_path=str(train_path),
                val_path=str(val_path),
                optimization_info={
                    "format_applied": fmt,
                    "unsloth_optimized": has_sharegpt,
                    "packing_recommended": True,
                    "total_samples": len(train_ds) + len(val_ds),
                    "has_embeddings": embeddings is not None,
                    "embedding_dimension": embeddings.shape[1] if embeddings is not None else None,
                },
                preprocessing_analytics=preprocessing_analytics,
                visualization_urls=visualization_urls
            )

        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            return PrepareDataResponse(
                status="error",
                message=f"Data preparation failed: {str(e)}",
                train_path="",
                val_path="",
                preprocessing_analytics=preprocessing_analytics if 'preprocessing_analytics' in locals() else None
            )
    
    async def _analyze_dataset_quality_enhanced(self, df: pd.DataFrame) -> Dict[str, float]:
        """Enhanced dataset quality analysis"""
        try:
            text_cols = [c for c in df.columns if df[c].dtype == object]
            if not text_cols:
                return {"error": "No text columns found"}
            
            metrics = {}
            
            # Text length analysis
            all_texts = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
            text_lengths = [len(text) for text in all_texts]
            
            metrics.update({
                "avg_text_length": np.mean(text_lengths),
                "median_text_length": np.median(text_lengths),
                "text_length_std": np.std(text_lengths),
                "min_text_length": min(text_lengths),
                "max_text_length": max(text_lengths)
            })
            
            # Vocabulary richness
            word_counts = []
            unique_words = set()
            for text in all_texts:
                words = text.split()
                word_counts.append(len(words))
                unique_words.update(words)
            
            total_words = sum(word_counts)
            vocabulary_richness = len(unique_words) / max(total_words, 1)
            
            metrics.update({
                "avg_word_count": np.mean(word_counts),
                "vocabulary_size": len(unique_words),
                "vocabulary_richness": vocabulary_richness,
                "avg_words_per_sample": total_words / len(df)
            })
            
            # Empty content analysis
            empty_rows = sum(1 for text in all_texts if len(text.strip()) == 0)
            
            metrics.update({
                "empty_rows_ratio": empty_rows / len(df),
                "data_completeness": 1 - (empty_rows / len(df))
            })
            
            # Overall quality score (0-1)
            quality_score = (
                min(vocabulary_richness * 2, 1.0) * 0.3 +  # Vocabulary richness
                min(metrics["data_completeness"], 1.0) * 0.4 +  # Data completeness
                min(metrics["avg_text_length"] / 100, 1.0) * 0.3  # Text length adequacy
            )
            
            metrics["overall_quality_score"] = quality_score
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in enhanced quality analysis: {e}")
            return {"error": str(e)}
    
    async def _analyze_token_distribution(self, df: pd.DataFrame, text_cols: List[str]) -> Dict[str, Any]:
        """Analyze token distribution across the dataset"""
        try:
            if not text_cols:
                return {"error": "No text columns provided"}
            
            # Combine all text
            all_texts = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
            
            # Word-level analysis (approximate tokens)
            all_words = []
            text_token_counts = []
            
            for text in all_texts:
                words = text.split()
                all_words.extend(words)
                # Approximate token count (words * 1.3 for subword tokenization)
                token_count = int(len(words) * 1.3)
                text_token_counts.append(token_count)
            
            # Token distribution statistics
            token_stats = {
                "mean_tokens_per_sample": np.mean(text_token_counts),
                "median_tokens_per_sample": np.median(text_token_counts),
                "std_tokens_per_sample": np.std(text_token_counts),
                "min_tokens": min(text_token_counts),
                "max_tokens": max(text_token_counts),
                "total_estimated_tokens": sum(text_token_counts)
            }
            
            # Token distribution histogram
            histogram_data = self._create_distribution_histogram(text_token_counts, bins=20)
            
            # Word frequency analysis
            word_freq = {}
            for word in all_words:
                word_lower = word.lower()
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
            
            # Top words
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50]
            
            return {
                "token_statistics": token_stats,
                "token_distribution_histogram": histogram_data,
                "vocabulary_stats": {
                    "total_unique_words": len(word_freq),
                    "total_word_occurrences": len(all_words),
                    "most_frequent_words": top_words[:20]
                },
                "estimated_training_tokens": sum(text_token_counts)
            }
            
        except Exception as e:
            logger.error(f"Error in token distribution analysis: {e}")
            return {"error": str(e)}
    
    def _create_distribution_histogram(self, data: List[float], bins: int = 20) -> List[Dict[str, Any]]:
        """Create histogram data for distribution visualization"""
        try:
            import numpy as np
            
            if not data:
                return []
            
            # Calculate histogram
            hist, bin_edges = np.histogram(data, bins=bins)
            
            # Create histogram data
            histogram_data = []
            for i in range(len(hist)):
                bin_start = bin_edges[i]
                bin_end = bin_edges[i + 1]
                histogram_data.append({
                    "range": f"{bin_start:.0f}-{bin_end:.0f}",
                    "count": int(hist[i]),
                    "percentage": float(hist[i] / len(data) * 100)
                })
            
            return histogram_data
            
        except Exception as e:
            logger.error(f"Error creating distribution histogram: {e}")
            return []