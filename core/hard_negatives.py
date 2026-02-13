"""
Hard Negative Mining (V5).
For each SOP question, finds near-miss candidates with different answers.
Used to generate training triplets for fine-tuning BGE-M3.
"""
import json
import os
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def mine_hard_negatives(
    entries: list[dict],
    dense_vectors: np.ndarray,
    top_k: int = 10,
    max_negatives: int = 3,
) -> list[dict]:
    """
    Mine hard negatives from the SOP dataset.
    
    For each entry, finds the top-K nearest neighbors by dense similarity
    that have a DIFFERENT source_id (i.e. different answer).
    
    Args:
        entries: SOP entries with 'question', 'answer', 'id', 'source_id'
        dense_vectors: np.ndarray of shape (N, dim) - pre-computed embeddings
        top_k: Number of neighbors to search
        max_negatives: Max hard negatives per anchor
        
    Returns:
        List of triplet dicts:
        {
            "anchor": str,      # Original question
            "positive": str,    # Same answer / paraphrase
            "negative": str,    # Hard negative (similar but wrong)
            "anchor_id": str,
            "negative_id": str,
        }
    """
    import faiss

    n, dim = dense_vectors.shape
    
    # Build temp FAISS index
    index = faiss.IndexFlatIP(dim)
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = (dense_vectors / norms).astype(np.float32)
    
    index.add(normalized)
    
    # Search for each entry
    scores, indices = index.search(normalized, top_k + 1)  # +1 because self is included
    
    triplets = []
    for i, entry in enumerate(entries):
        anchor_source = entry.get("source_id", entry.get("id"))
        anchor_q = entry["question"]
        anchor_answer = entry["answer"]
        
        neg_count = 0
        for j_pos in range(len(indices[i])):
            j = int(indices[i][j_pos])
            if j == i:
                continue  # Skip self
            
            neighbor = entries[j]
            neighbor_source = neighbor.get("source_id", neighbor.get("id"))
            
            # Hard negative: high similarity but different answer
            if neighbor_source != anchor_source:
                triplets.append({
                    "anchor": anchor_q,
                    "positive": anchor_answer,
                    "negative": neighbor["question"],
                    "anchor_id": anchor_source,
                    "negative_id": neighbor_source,
                    "similarity": float(scores[i][j_pos]),
                })
                neg_count += 1
                if neg_count >= max_negatives:
                    break

    print(f"✅ Mined {len(triplets)} hard negative triplets from {n} entries")
    return triplets


def save_triplets(triplets: list[dict], path: str = None):
    """Save triplets to JSON file."""
    path = path or os.path.join(config.DATA_DIR, "hard_negatives.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(triplets, f, indent=2, ensure_ascii=False)
    print(f"💾 Hard negatives saved → {path} ({len(triplets)} triplets)")


def load_triplets(path: str = None) -> list[dict]:
    """Load triplets from JSON file."""
    path = path or os.path.join(config.DATA_DIR, "hard_negatives.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
