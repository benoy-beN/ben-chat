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

# V5 new index/model files
BM25_INDEX_FILE = os.path.join(DATA_DIR, "bm25_index.pkl")
FUSION_MODEL_FILE = os.path.join(DATA_DIR, "fusion_model.pkl")
CALIBRATOR_FILE = os.path.join(DATA_DIR, "calibrator.pkl")

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

# ── Retrieval (V6) ────────────────────────────────────
THRESHOLD_HIGH = 0.42         # Set to calibrated optimal (F1 max)
THRESHOLD_LOW  = 0.20         # Wide gray zone for safety
LEX_MIN = 0.20                # Minimum token overlap for gray zone
GAP_MIN = 0.02                # Minimum reranker score gap for gray zone

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
