# LLMForge Data Preparation Best Practices

## 1. Raw Data Cleaning
- **Deduplication**: Remove duplicate entries using hashing or similarity detection algorithms
- **Whitespace Normalization**: Standardize whitespace (e.g. replace multiple spaces with single space)
- **Language Detection**: Use langdetect or fasttext for multilingual datasets
- **UTF-8 Sanitization**: Remove non-UTF-8 characters with regex [^\\x00-\\x7F]

## 2. Tokenisation & Chunking

Tokenization breaks text into tokens suitable for the model's vocabulary, while chunking divides long texts into manageable pieces. For LLM fine-tuning, especially with LlamaFactory, choose methods that align with the base model's tokenizer.

### Tokenization Methods
- **Byte-Pair Encoding (BPE)**: Default for many models like GPT series. Efficient for English but may struggle with morphologically rich languages. Use Hugging Face's [`tiktoken`](https://github.com/openai/tiktoken) for OpenAI models or [`tokenizers`](https://huggingface.co/docs/tokenizers) library.
- **SentencePiece (Unigram or BPE)**: Recommended for multilingual and code-mixed data. Llama models use SentencePiece BPE. Train custom tokenizers if domain-specific (e.g., medical terms) using `sentencepiece` library with vocabulary size 32k-128k.
  - Pros: Handles unknown words via subwords; space-agnostic.
  - Cons: Larger vocabularies increase model size.

### Handling Max Length
- Set `max_length` to 80-90% of the model's context window (e.g., 2048 for Llama-7B) to leave room for prompts/instructions.
- Truncate dynamically: Use `truncation=True` in Hugging Face datasets with padding strategy `max_length`.
- For LlamaFactory: Configure in `dataset_info.json` under `max_seq_length`. Avoid exceeding to prevent OOM errors during training.

### Chunking Strategies for Long Texts
- **Fixed-size chunks**: Split into equal token lengths (e.g., 512 tokens). Simple but may break semantic units.
- **Sliding window with overlap**: Use 20-50% overlap (e.g., chunk size 1024, stride 512) to preserve context across chunks. Implement with `sliding_window` in LangChain or custom Python loops.
  - Example code:
    ```python
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    def chunk_text(text, max_length=1024, overlap=256):
        tokens = tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_length - overlap):
            chunk = tokens[i:i + max_length]
            chunks.append(tokenizer.decode(chunk))
        return chunks
    ```
- **Semantic chunking**: Use sentence boundaries (NLTK/SpaCy) or embedding-based splits for better coherence. Ideal for RAG but compute-intensive.
- Best practice: For fine-tuning, aim for chunks that fit single examples; overlap helps in instruction-tuning datasets.

In LlamaFactory, enable `packing` for efficient batching of short sequences, reducing padding waste.

## 3. Data Augmentation

Data augmentation increases dataset diversity, helping LLMs generalize better, especially for low-resource tasks. Apply judiciously (10-30% of data) to avoid introducing noise. In LlamaFactory, integrate via custom dataset preprocessors.

### Techniques
- **Synonym Replacement**: Replace words with synonyms using WordNet (NLTK) or contextual embeddings (BERT). Limit to 10-20% of words; avoid function words.
  - Example: `nlpaug.augmenter.word.synonym.SynonymAug(aug_src='wordnet')`
  - When to use: Instruction-tuning for varied phrasing.
- **Back-Translation**: Translate to another language (e.g., English → French → English) using APIs like Google Translate or Helsinki-NLP models. Preserves meaning while altering structure.
  - Pros: Effective for non-English data; improves robustness.
  - Cons: API costs; potential translation errors.
  - Implementation: Use `transformers` pipeline with MarianMT models.
- **Random Deletion/Insertion/Swap**: Delete 10-15% tokens, insert noise, or swap adjacent words. Simple and fast.
  - Example: `nlpaug.augmenter.word.delete.RandomWordDeletionAug()`
- **EDA (Easy Data Augmentation)**: Combine deletion, synonym, swap, and insertion. Rule-based, no ML needed.
- **Advanced**: Paraphrasing with T5/BART models or counterfactual augmentation for bias mitigation.

Best practices: Validate augmented data with human review or perplexity scores. For fine-tuning, augment only inputs (not labels) in supervised tasks. In LlamaFactory, use `augmentation_ratio` in config if supported.

## 4. Class Balancing / Sampling

Imbalanced classes lead to biased models. Balance during preprocessing to ensure fair representation, especially in classification or preference-tuning datasets.

### Techniques
- **Stratified Splitting**: Use `train_test_split` with `stratify=y` in scikit-learn to maintain class ratios in train/validation sets.
- **Up-sampling**: Duplicate minority class samples or use bootstrapping. Simple but risks overfitting.
- **Down-sampling**: Randomly remove majority class samples. Fast but loses data.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Generate synthetic text samples via embeddings. Use TextSMOTE or adaptations like `imbalanced-learn` with TF-IDF.
  - Steps: Embed texts → Apply SMOTE on vectors → Decode back to text using nearest neighbors.
  - Limitation: May create nonsensical text; better for short texts.
- **Class Weights**: Instead of resampling, adjust loss weights during training (e.g., `class_weight='balanced'` in Hugging Face Trainer).
- **Advanced**: Focal loss or adaptive sampling in LlamaFactory configs.

For LLM fine-tuning: Aim for 1:1 ratios in binary tasks; use weighted sampling for multi-class. Monitor with metrics like F1-score per class.

## 5. Embedding Selection

Embeddings represent text semantically for tasks like similarity search or clustering in data prep. Choose based on trade-offs: quality vs. speed/cost.

### Model Trade-offs
- **Cheap Dense Models (e.g., MiniLM, all-MiniLM-L6-v2)**: 384-dim, fast inference (~10ms/sentence). Good for batch processing large datasets.
  - Pros: Low latency, runs on CPU; sufficient for dedup/clustering.
  - Cons: Lower quality on nuanced semantics.
  - Use: Initial data cleaning; batch size 128-512.
- **Higher-Quality Models (e.g., BGE-large-en-v1.5, E5-mistral)**: 768-1024 dim, better multilingual support and instruction-tuned.
  - Pros: Superior retrieval accuracy; handles long contexts.
  - Cons: Higher latency (50-200ms); GPU preferred.
  - Use: When quality > speed, e.g., semantic chunking.
- **Proprietary (e.g., OpenAI text-embedding-ada-002)**: State-of-the-art, 1536 dim.
  - Pros: Excellent performance; easy API.
  - Cons: Cost (~$0.0001/1k tokens); privacy concerns.
  - Batch size: Up to 2048 via API.

### Considerations
- **Latency vs. Quality**: Benchmark with your data (e.g., using `sentence-transformers`). Start with MiniLM, upgrade if downstream metrics suffer.
- **Batch Size**: Larger batches (256+) for throughput; monitor VRAM.
- In LlamaFactory: Embeddings used in RAG pipelines; configure via `embedding_model` in dataset config.

Select based on use case: MiniLM for prototyping, BGE for production fine-tuning.

## 6. Evaluation of Embeddings

Evaluate embeddings to ensure they capture meaningful semantics before fine-tuning. This step validates preprocessing quality and guides model selection.

### Intrinsic Evaluation
Intrinsic metrics assess embedding quality independently of downstream tasks, focusing on semantic consistency.

- **Cosine Similarity Heatmaps**: Compute pairwise similarities on a representative sample (e.g., 50-100 documents). Visualize with libraries like seaborn or plotly. Look for high similarity within semantically related pairs and low for unrelated ones.
  - Threshold: >0.7 for positives, <0.3 for negatives.
  - Code snippet:
    ```python
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import seaborn as sns
    import matplotlib.pyplot as plt

    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = ["Sample texts..."]
    embeddings = model.encode(sentences)
    similarities = cosine_similarity(embeddings)
    sns.heatmap(similarities, annot=True)
    plt.show()
    ```
- **Dimensionality Reduction Visualizations**: Use t-SNE or UMAP to project embeddings into 2D/3D space. Color by labels or topics to check clustering.
  - Parameters: For UMAP, n_neighbors=15, min_dist=0.1; for t-SNE, perplexity=30.
  - Libraries: `umap-learn` or `sklearn.manifold.TSNE`.
  - Ideal: Tight clusters for similar items, separation for dissimilar.
- **Analogy and Intrinsic Tasks**: Test vector arithmetic (e.g., vec("Paris") - vec("France") + vec("Italy") ≈ vec("Rome")). Use datasets like WordSim-353 for correlation scores (Spearman >0.6 is good).

### Downstream Evaluation
Measure impact on fine-tuning performance.

- **Fine-Tuning Metrics**: After preprocessing, train a small model and monitor validation loss, perplexity, or accuracy. Compare pre/post-embedding filtering (e.g., remove low-quality chunks if loss decreases).
  - In LlamaFactory: Use built-in logging for `eval_loss` and `perplexity` during SFT/RLHF stages.
- **Task-Specific Benchmarks**: For classification, compute F1/precision/recall on a validation set; for generation, use ROUGE, BLEU, or BERTScore. Embeddings should improve these by 5-10% over baselines.
- **Efficiency and Robustness**: Evaluate deduplication recall (e.g., Jaccard similarity >0.9 catches duplicates) or chunking coherence (average intra-chunk similarity > inter-chunk).

### Best Practices
- **Benchmarking**: Run evaluations on a subset (10%) to save compute. Use tools like `sentence-transformers` evaluation module.
- **In LlamaFactory**: Integrate via custom callbacks in training scripts; log to Weights & Biases or TensorBoard for visualization.
- **Thresholds and Iteration**: If intrinsic scores <0.5, revisit embedding model or data cleaning. Re-evaluate after each pipeline change.
- **Tools**: FAISS for scalable nearest-neighbor search; scikit-learn for metrics.

Regular evaluation ensures high-quality data feeds into fine-tuning, reducing training iterations and improving model performance.