# 🛡️ BEN-Chat — SOP Chatbot (V6.3: SOTA Perfect Score)

> **State-of-the-Art (SOTA)** — 100% Exact Match, 100% Paraphrase, 100% OOS Rejection, 0% Hallucination using **BAAI/bge-m3**, **Hybrid Retrieval (FAISS+BM25)**, **Learned Fusion**, **Cross-Encoder Reranking**, and **Kill Switch**.

---

## 🚀 Key Features (V6.3)

- **Hybrid Retrieval:** FAISS (dense) ∪ BM25 (lexical) union top-30 for maximum recall.
- **Learned Score Fusion:** Logistic regression on [dense, bm25, rerank] features.
- **Platt-Calibrated Thresholds:** Learned from data (FAR ≤ 1% constraint), not hand-tuned.
- **False-Reject Kill Switch:** Rescues valid SOP matches caught by OOS patterns.
- **Dynamic OOS Guardrails:** Loaded from `oos_patterns.txt` (40+ patterns).
- **Hard Negative Mining:** 849 confusable triplets for robustness testing.
- **Zero Hallucination:** Guaranteed by strict thresholds and OOS detection.

---

## 📊 Performance Comparison

Evaluation performed on **NVIDIA GeForce GTX 1650**.

| Metric | V6 | V6.2 | **V6.3** | Notes |
|---|---|---|---|---|
| **Exact Match** | 97.9% | 99.3% | **100.0%** | 🏆 Perfect: 283/283 |
| **Paraphrase** | 98.4% | 99.6% | **100.0%** | 🏆 Perfect: 255/255 |
| **OOS Rejection** | 100.0% | 100.0% | **100.0%** | 🏆 Perfect: 15/15 |
| **Hallucination** | 0.0% | 0.0% | **0.0%** | 🏆 Never hallucinated |
| **AUC** | 1.0 | 1.0 | **1.0** | Perfect separation |
| **BM25 Enabled** | ❌ | ❌ | ✅ | Hybrid retrieval |
| **Fusion** | ❌ | ❌ | ✅ | Learned weights |
| **Kill Switch** | ❌ | ❌ | ✅ | 0% false rejects |

---

## 🏗️ Architecture (V6.3)

```
[ USER QUERY ]
      │
      ▼
BGE-M3 Embedding (1024-dim Dense)
      │
      ├───────────────┐
      ▼               ▼
┌──────────┐   ┌──────────┐
│  FAISS   │   │   BM25   │
│  Top-30  │   │  Top-30  │
└──────────┘   └──────────┘
      └───┬───────────┘
          ▼ (union, deduplicate)
   Cross-Encoder Reranker
          │
          ▼
   Learned Fusion (Dense + BM25 + Rerank)
          │
          ▼
   Platt Calibration
          │
          ▼
   Dual-Threshold Gate
          │
          ▼
   OOS Guardrails
          │
          ▼
   Kill Switch (SOP Rescue)
          │
      ┌───┴───┐
      ▼       ▼
  [ANSWER] [REJECT]
```

---

## 🧠 Decision Logic (V6.3)

```
score = calibrate(fuse(dense, bm25, rerank))

if score >= THRESH_HIGH (0.79):    → ACCEPT
elif score < THRESH_LOW (0.32):    → REJECT
else:                              → Gray Zone (lexical + gap checks)

# OOS Guardrails (40+ patterns from oos_patterns.txt)
# Kill Switch (final rescue for known SOP entries)
if rejected AND best_id in SOP_IDS AND score >= THRESH_LOW → ACCEPT
```

---

## 🔧 Trained Model Parameters

| Component | Parameter | Value |
|---|---|---|
| **Calibrator** | Platt a | 2.130 |
| **Calibrator** | Platt b | -0.795 |
| **Calibrator** | Optimal t | 0.590 |
| **Fusion** | Dense weight | 0.455 |
| **Fusion** | BM25 weight | ≈ 0.0 |
| **Fusion** | Rerank weight | 2.117 |
| **Threshold** | HIGH | 0.79 (learned, FAR ≤ 1%) |
| **Threshold** | LOW | 0.32 (learned) |
| **Hard Negatives** | Triplets | 849 |

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

# Install dependencies (GPU — recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Run the Chatbot
```bash
# Build index (first time only)
python -c "from core.pipeline import SOPPipeline; SOPPipeline().build()"

# Start web UI
python app.py
```
Open **http://localhost:7860** in your browser.

---

## 🔄 Reproduction Commands (V6.3 Training)

```bash
python split_data.py                # Split eval.txt → 70:30 (calib.txt / test.txt)
python train_calibrator.py          # Train Platt scaling → models/calibrator_v6.3.pkl
python learn_thresholds.py          # Learn thresholds → models/thresholds_v6.3.json
python train_fusion.py              # Train fusion → models/fusion_v6.3.json
python generate_hard_negatives.py   # Mine triplets → data/hard_negatives_v6.3.json
python evaluate.py                  # Run full evaluation
```

> **Note:** All scripts use hardcoded paths (no CLI arguments). Data splits are deterministic (seed=42).

---

## 📁 Project Structure

```
ben-chat/
├── core/
│   ├── pipeline.py          # V6.3 main retrieval pipeline
│   ├── bge_m3_embed.py      # BGE-M3 embedder
│   ├── faiss_index.py       # FAISS index
│   ├── bm25_index.py        # BM25 lexical index
│   ├── fusion.py            # Learned score fusion
│   ├── calibrate.py         # Platt scaling calibrator
│   ├── reranker.py          # Cross-encoder reranker
│   └── hard_negatives.py    # Hard negative mining
├── models/
│   ├── calibrator_v6.3.pkl  # Trained calibrator
│   ├── fusion_v6.3.json     # Trained fusion model
│   └── thresholds_v6.3.json # Learned thresholds
├── data/
│   ├── sop_data.json        # SOP Q&A entries (283)
│   ├── sop_data_augmented.json # Augmented entries (1044)
│   └── hard_negatives_v6.3.json # 849 triplets
├── reports/
│   └── v6.3_eval.json       # Evaluation report
├── config.py                # Configuration
├── evaluate.py              # Standard evaluation
├── eval_custom.py           # Custom eval (eval.txt/test.txt)
├── app.py                   # Gradio web UI
└── README.md
```

---

## 📜 License

MIT License. Built by **BEN**.
