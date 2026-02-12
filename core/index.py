"""
FAISS index management — build, search, save, load.
Uses IndexFlatIP (inner product) on L2-normalized vectors for cosine similarity.
"""
import json
import os

import faiss
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FAISSIndex:
    """
    Manages FAISS index for SOP question vectors.
    Uses inner product (dot product) on normalized vectors = cosine similarity.
    """

    def __init__(self):
        self.index = None
        self.id_map: list[dict] = []  # Maps FAISS index position → SOP entry

    def build(self, vectors: np.ndarray, entries: list[dict]):
        """
        Build FAISS index from embedding vectors.

        Args:
            vectors: np.ndarray of shape (n, dim), L2-normalized
            entries: List of SOP entries with id, question, answer, source_id
        """
        dim = vectors.shape[1]
        n = vectors.shape[0]

        # Use GPU for FAISS if available
        try:
            res = faiss.StandardGpuResources()
            cpu_index = faiss.IndexFlatIP(dim)
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            print(f"🚀 FAISS index on GPU")
        except Exception:
            self.index = faiss.IndexFlatIP(dim)
            print(f"💻 FAISS index on CPU")

        self.index.add(vectors)
        self.id_map = entries

        print(f"✅ Built FAISS index: {n} vectors, {dim} dimensions")

    def search(self, query_vector: np.ndarray, top_k: int = 1) -> list[dict]:
        """
        Search the index for the closest match(es).

        Args:
            query_vector: np.ndarray of shape (dim,)
            top_k: Number of results to return

        Returns:
            List of {"score": float, "entry": dict} sorted by score descending.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        # Reshape for FAISS: (1, dim)
        query_vector = query_vector.reshape(1, -1).astype(np.float32)

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            results.append({
                "score": float(score),
                "entry": self.id_map[idx],
            })

        return results

    def save(self, index_path: str = None, map_path: str = None):
        """Save FAISS index and ID map to disk."""
        index_path = index_path or config.FAISS_INDEX_FILE
        map_path = map_path or config.ID_MAP_FILE

        # If GPU index, convert to CPU for saving
        try:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        except Exception:
            cpu_index = self.index

        faiss.write_index(cpu_index, index_path)

        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, indent=2, ensure_ascii=False)

        print(f"💾 Index saved → {index_path}")
        print(f"💾 ID map saved → {map_path}")

    def load(self, index_path: str = None, map_path: str = None):
        """Load FAISS index and ID map from disk."""
        index_path = index_path or config.FAISS_INDEX_FILE
        map_path = map_path or config.ID_MAP_FILE

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"ID map file not found: {map_path}")

        cpu_index = faiss.read_index(index_path)

        # Try GPU
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            print(f"🚀 FAISS index loaded on GPU")
        except Exception:
            self.index = cpu_index
            print(f"💻 FAISS index loaded on CPU")

        with open(map_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)

        print(f"✅ Loaded FAISS index: {self.index.ntotal} vectors")
        return self
