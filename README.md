# 🛡️ BEN-Chat — SOP Chatbot (V5: Accuracy-First)

> Deterministic SOP retrieval • Zero hallucination • Calibrated confidence

## Architecture (V5)

```
[ USER QUERY ]
      │
      ▼
┌───────────────────────┐
│  Query Normalize      │
│  lowercase + punct    │
└───────────────────────┘
      │
      ▼
┌───────────────────────┐
│  BGE-M3 Embeddings    │
│  Dense + Sparse       │
└───────────────────────┘
      │
  ┌───┴────────────┐
  ▼                ▼
[FAISS Dense]  [BM25 Lexical]
  │                │
  └───┬────────────┘
      ▼
┌───────────────────────┐
│  Learned Fusion       │
│  (query-adaptive)     │
└───────────────────────┘
      │
      ▼
┌───────────────────────┐
│  Cross-Encoder        │
│  Reranker (Top-K → 1) │
└───────────────────────┘
      │
      ▼
┌───────────────────────┐
│  Confidence           │
│  Calibration (Platt)  │
└───────────────────────┘
      │
  ┌───┴───┐
  ▼       ▼
[ANSWER] [REJECT]
```

## What's New in V5

| Feature | V4 | V5 |
|---|---|---|
| Encoder | Dual (BGE + MiniLM) | Single BGE-M3 (dense + sparse) |
| Lexical | None | BM25 |
| Fusion | Fixed weights (0.7/0.3) | Learned (logistic regression) |
| Retrieval | Top-5 | Top-20 |
| Confidence | Raw cosine | Platt-calibrated probability |
| Threshold | Fixed (0.40) | Adaptive (learned on validation) |
| Eval Exports | None | ROC curve + confusion matrix |

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Build index (FAISS + BM25)
python build.py
```

## Usage

```bash
# Run evaluation
python evaluate.py

# Launch web UI
python app.py
# Open http://localhost:7860
```

## Training (Optional)

```bash
# Mine hard negatives
python training/build_hard_negs.py

# Fine-tune BGE-M3
python training/finetune_bge_m3.py
```

## Project Structure

```
ben-chat/
├── app.py                # Gradio web UI
├── build.py              # Full build script
├── config.py             # Central configuration
├── evaluate.py           # Evaluation pipeline
├── requirements.txt      # Python dependencies
│
├── core/
│   ├── bge_m3_embed.py   # BGE-M3 dense + sparse embedder
│   ├── bm25_index.py     # BM25 lexical retrieval
│   ├── calibrate.py      # Platt scaling confidence calibration
│   ├── fusion.py         # Learned score fusion
│   ├── hard_negatives.py # Hard negative mining
│   ├── index.py          # FAISS vector index
│   ├── normalizer.py     # Query normalization
│   ├── pipeline.py       # V5 retrieval pipeline
│   ├── reranker.py       # Cross-encoder reranker
│   └── rewriter.py       # Optional LLM rewrite (Ollama)
│
├── eval/
│   ├── roc_curve.py      # ROC curve generation
│   └── confusion_matrix.py
│
├── training/
│   ├── finetune_bge_m3.py
│   └── build_hard_negs.py
│
├── runtime/
│   └── semantic_cache.py # Optional query cache
│
├── scripts/
│   └── parse_data.py     # SOP data parser
│
└── data/
    ├── sop_data.json
    ├── sop_data_augmented.json
    ├── sop_index.faiss
    ├── bm25_index.pkl
    └── id_map.json
```

## Expected Accuracy

| Stage | Accuracy |
|---|---|
| V4 baseline | ~92% |
| + BM25 | ~94–95% |
| + BGE-M3 | ~95–96% |
| + Adaptive threshold | ~96–97% |
| + Hard-negative tuning | ~97–99% |

## License

MIT
