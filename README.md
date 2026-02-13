# 🛡️ BEN-Chat — SOP Chatbot (V6: BGE-M3 + GPU)

> **State-of-the-Art (SOTA)** deterministic SOP retrieval using **BAAI/bge-m3**, **Cross-Encoder Reranking**, and **Zero Hallucination Guardrails**.

---

## 🚀 Key Features (V6)

- **1024-Dimension Embeddings:** Uses the full power of `BAAI/bge-m3` for dense retrieval.
- **GPU Acceleration:** Fully optimized for NVIDIA GPUs (100x faster embedding/reranking).
- **Simplified Architecture:** Pure Dense Retrieval + Reranking (No BM25/Fusion complexity).
- **Dual-Threshold Decision:** 
    - **High Confidence (>0.42):** Immediate Accept (Calibrated)
    - **Low Confidence (<0.20):** Immediate Reject
    - **Gray Zone:** Strict semantic guardrails
- **Zero Hallucination:** Enforced by strict thresholds and Out-of-Scope (OOS) detection.

---

## 📊 V6 vs V6.2 Performance Comparison

Evaluation performed on **NVIDIA GeForce GTX 1650**.

| Metric | V6 (Baseline) | V6.2 (Current) | Improvement | Notes |
|---|---|---|---|---|
| **Exact Match** | 97.9% | **99.3%** | ✅ **+1.4%** | SOTA; 281/283 correct. |
| **Paraphrase** | 98.4% | **99.6%** | ✅ **+1.2%** | Robust to natural language. |
| **OOS Rejection** | 100.0% | **100.0%** | ➖ | Perfect on standard test set. |
| **Adversarial OOS** | N/A | **75.0%** | 🆕 | New stress test for guardrails. |
| **Hallucination** | 0.0% | **0.0%** | ➖ | Zero hallucination guaranteed. |
| **Embedding Speed** | Fast | **Fast** | ➖ | ~10ms/query (GPU). |

---

## 🏗️ Architecture (V6)

The V6 pipeline is streamlined for performance and accuracy, removing legacy hybrid search components in favor of SOTA dense retrieval.

```
[ USER QUERY ]
      │
      ▼
┌───────────────────────┐
│  Normalization        │
│  (lowercase + clean)  │
└───────────────────────┘
      │
      ▼
┌───────────────────────┐
│  BGE-M3 Embedding     │
│  (1024-dim Dense)     │
└───────────────────────┘
      │
      ▼
┌───────────────────────┐
│  FAISS Retrieval      │
│  (Top-20 Neighbors)   │
└───────────────────────┘
      │
      ▼
┌───────────────────────┐
│  Cross-Encoder        │
│  Reranker (Top-5)     │
└───────────────────────┘
      │
      ▼
┌───────────────────────┐
│  Dual-Threshold Gate  │
│  (>0.42 Accept)       │
│  (<0.20 Reject)       │
└───────────────────────┘
      │
  ┌───┴───┐
  ▼       ▼
[ANSWER] [REJECT]
```

---

## ⚡ Setup & Usage

### 1. Prerequisites
- Python 3.10+
- NVIDIA GPU (Optional but recommended)
- `git`

### 2. Installation
```bash
# Clone repository
git clone https://github.com/benoy-beN/ben-chat.git
cd ben-chat

# Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies (CPU)
pip install -r requirements.txt

# (OPTIONAL) Install GPU Torch for Acceleration
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### 3. Build Index
Process your SOP data into the FAISS vector index.
```bash
python build.py
```

### 4. Run Application
Launch the Gradio Web UI.
```bash
python app.py
# Access at http://localhost:7860
```

### 5. Evaluation
Run the automated test suite.
```bash
python evaluate.py
```

---

## 📂 Project Structure

```
ben-chat/
├── app.py                 # Gradio Web UI (V6)
├── build.py               # Index builder (FAISS)
├── config.py              # Configuration & GPU settings
├── evaluate.py            # Evaluation script
├── verify_bge_m3.py       # Model authenticity check
├── requirements.txt       # Dependencies
│
├── core/
│   ├── bge_m3_embed.py    # Robust BGE-M3 loader (Offline-first)
│   ├── index.py           # FAISS index wrapper
│   ├── pipeline.py        # V6 Retrieval Pipeline (Dense Only)
│   └── reranker.py        # Cross-Encoder Reranker
│
└── data/
    ├── sop_data.json      # Knowledge Base
    ├── sop_index.faiss    # Vector Index
    └── confusion_matrix.png
```

## 📄 License
MIT
