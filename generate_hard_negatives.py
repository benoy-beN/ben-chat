"""
V6.3 Hard Negative Generation Script.
Mines confusable pairs from sop_data.json using existing core/hard_negatives.py.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.hard_negatives import mine_hard_negatives, save_triplets
from core.bge_m3_embed import BGEM3Embedder
import config


def generate():
    print("=" * 60)
    print("  V6.3 HARD NEGATIVE GENERATION")
    print("=" * 60)

    with open(config.SOP_DATA_FILE, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"📊 Loaded {len(entries)} SOP entries")

    # Embed all questions
    embedder = BGEM3Embedder(config.EMBEDDING_MODEL)
    embedder.load()

    questions = [e["question"] for e in entries]
    embeddings = embedder.embed_batch(questions, is_query=False)
    dense_vecs = embeddings["dense"]

    # Normalize
    norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    dense_vecs = (dense_vecs / norms).astype(np.float32)

    # Mine hard negatives
    triplets = mine_hard_negatives(
        entries=entries,
        dense_vectors=dense_vecs,
        top_k=10,
        max_negatives=3,
    )

    # Save
    out_path = os.path.join(config.DATA_DIR, "hard_negatives_v6.3.json")
    save_triplets(triplets, out_path)

    print(f"\n✅ Generated {len(triplets)} hard negative triplets")


if __name__ == "__main__":
    generate()
