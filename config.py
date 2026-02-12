"""
Central configuration for SOP Chatbot.
All tunable parameters in one place.
"""
import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SOP_DATA_FILE = os.path.join(DATA_DIR, "sop_data.json")
RAW_DATA_FILE = os.path.join(DATA_DIR, "data.txt")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "sop_index.faiss")
FAISS_INDEX_FILE_B = os.path.join(DATA_DIR, "sop_index_b.faiss")
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")

# ── Hardware Settings ──────────────────────────────────
DEVICE = "cpu"  # Force CPU for stability

# ── Embedding Model ───────────────────────────────────
# Apache 2.0 license — corporate-safe
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
MODEL_B_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 768
# Prefix required by bge models for queries
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Retrieval ─────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.40  # Calibrated for Ensemble (Hybrid + Reranker)
HYBRID_WEIGHT_A = 0.7       # BGE Weight
HYBRID_WEIGHT_B = 0.3       # E5 Weight
TOP_K_RETRIEVAL = 5         # Candidates for reranking
TOP_K_FINAL = 1             # Final output

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
