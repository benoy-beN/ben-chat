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
ID_MAP_FILE = os.path.join(DATA_DIR, "id_map.json")

# ── Embedding Model ───────────────────────────────────
# Apache 2.0 license — corporate-safe
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768
# Prefix required by bge models for queries
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Retrieval ─────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.75
TOP_K = 1

# ── LLM Rewrite (Optional) ───────────────────────────
USE_LLM_REWRITE = False
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"

# ── Rejection Message ────────────────────────────────
REJECTION_MESSAGE = (
    "This question is not covered in the SOP.\n"
    "Please contact HR."
)

# ── Web UI ────────────────────────────────────────────
APP_HOST = "0.0.0.0"
APP_PORT = 7860
