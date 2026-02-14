"""
Central configuration for SOP Chatbot (V5 — Accuracy-First).
All tunable parameters in one place.
"""
import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SOP_DATA_FILE = os.path.join(DATA_DIR, "sop_data.json")
RAW_DATA_FILE = os.path.join(DATA_DIR, "data.txt")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "sop_index.faiss")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")

# V6.3 model/index files
MODELS_DIR = os.path.join(BASE_DIR, "models")
BM25_INDEX_FILE = os.path.join(DATA_DIR, "bm25_index.pkl")
FUSION_MODEL_FILE = os.path.join(MODELS_DIR, "fusion_v6.3.json")
CALIBRATOR_FILE = os.path.join(MODELS_DIR, "calibrator_v6.3.pkl")

# ── Hardware Settings ──────────────────────────────────
# ── Hardware Settings ──────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {DEVICE}")

# ── Embedding Model (V5: Single Strong Retriever) ─────
# Check for local manual download first
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "bge-m3")
if os.path.exists(LOCAL_MODEL_PATH):
    EMBEDDING_MODEL = LOCAL_MODEL_PATH
    print(f"📂 Using local embedding model: {EMBEDDING_MODEL}")
else:
    # BGE-M3: dense + sparse in one model (downloads from HF)
    EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 1024  # BGE-M3 dense dimension

# Prefix for BGE queries (used by fallback mode)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Retrieval (V6.3 — Learned Thresholds) ─────────────
_THRESH_FILE = os.path.join(MODELS_DIR, "thresholds_v6.3.json")
if os.path.exists(_THRESH_FILE):
    import json as _json
    with open(_THRESH_FILE, "r") as _f:
        _t = _json.load(_f)
    THRESHOLD_HIGH = _t.get("thresh_high", 0.42)
    THRESHOLD_LOW  = _t.get("thresh_low", 0.20)
    LEX_MIN = _t.get("lex_min", 0.20)
    GAP_MIN = _t.get("gap_min", 0.02)
else:
    THRESHOLD_HIGH = 0.42         # Fallback: V6.2 calibrated optimal
    THRESHOLD_LOW  = 0.20
    LEX_MIN = 0.20
    GAP_MIN = 0.02

SIMILARITY_THRESHOLD = 0.45   # Legacy fallback
TOP_K_RETRIEVAL = 20          # Candidates for reranking
TOP_K_FINAL = 1               # Final output

# Fusion default weights: [dense, sparse, bm25]
FUSION_WEIGHTS = [0.5, 0.2, 0.3]

# ── LLM Rewrite (Optional) ───────────────────────────
USE_LLM_REWRITE = True
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:1.5b"

# ── Rejection Message ────────────────────────────────
REJECTION_MESSAGE = (
    "This question is not covered in the SOP.\n"
    "Please contact HR."
)

# ── Web UI ────────────────────────────────────────────
APP_HOST = "0.0.0.0"
APP_PORT = 7860
