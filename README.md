# 🛡️ SOP Chatbot - Zero Hallucination

A **deterministic, high-accuracy** SOP chatbot that answers **only** from a pre-loaded Q&A knowledge base. Uses FAISS vector search with a hard similarity threshold — the LLM is **never** used to generate answers, only to optionally rewrite retrieved answers.

## Architecture

```
User Question → Normalize → Embed (bge-base) → FAISS Top-1 → Threshold Check → Verbatim Answer / Reject
                                                                                      ↓
                                                                          Optional LLM Rewrite (Ollama)
```

### Hallucination Prevention Rules
1. LLM is **NOT** allowed to answer questions
2. LLM can **only** rewrite retrieved answers
3. Top-1 retrieval only
4. Hard similarity threshold (0.75)
5. Reject unknown questions
6. Return verbatim SOP answer

## Requirements

- **Python** 3.10+
- **GPU**: NVIDIA GPU with CUDA support (e.g., RTX 5090)
- **Optional**: [Ollama](https://ollama.ai) for answer rewriting

## Quick Start

### 1. Install Dependencies

```powershell
cd d:\2026_testing\chatbot2
pip install -r requirements.txt
```

### 2. Build Index

```powershell
python build.py
```

This will:
- Parse `data/data.txt` into structured JSON
- Generate paraphrased variants for better matching
- Embed all questions using `bge-base-en-v1.5`
- Build and save FAISS index

### 3. Run Evaluation

```powershell
python evaluate.py
```

### 4. Launch Web UI

```powershell
python app.py
```

Open `http://localhost:7860` in your browser.

## Tech Stack

| Component | Technology | License |
|-----------|-----------|---------|
| Embeddings | `BAAI/bge-base-en-v1.5` | Apache 2.0 |
| Vector Search | FAISS (GPU) | MIT |
| Web UI | Gradio | Apache 2.0 |
| LLM Rewrite | Ollama + Mistral | Apache 2.0 |
| Framework | Python + sentence-transformers | Apache 2.0 |

All components are **open-source** and **corporate-use compatible**.

## Project Structure

```
chatbot2/
├── app.py              # Gradio web interface
├── build.py            # One-step build script
├── config.py           # Central configuration
├── evaluate.py         # Accuracy evaluation pipeline
├── requirements.txt    # Python dependencies
├── core/
│   ├── embedder.py     # Embedding model wrapper
│   ├── index.py        # FAISS index management
│   ├── normalizer.py   # Input text normalization
│   ├── pipeline.py     # Full retrieval pipeline
│   └── rewriter.py     # Optional LLM rewriter
├── scripts/
│   └── parse_data.py   # Data parser & augmentor
└── data/
    ├── data.txt        # Raw SOP Q&A pairs
    ├── sop_data.json   # Parsed base dataset
    └── sop_data_augmented.json  # Augmented with paraphrases
```

## Configuration

Edit `config.py` to tune:
- `SIMILARITY_THRESHOLD` — Adjust strictness (default: 0.75)
- `EMBEDDING_MODEL` — Switch between bge/e5 models
- `USE_LLM_REWRITE` — Enable/disable Ollama rewriting
- `OLLAMA_MODEL` — Choose Ollama model for rewriting

## Optional: Ollama Setup

If you want answer rewriting (purely cosmetic — no new information):

```powershell
# Install Ollama from https://ollama.ai
ollama pull mistral
ollama serve
```

Then set `USE_LLM_REWRITE = True` in `config.py`.
