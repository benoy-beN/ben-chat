"""
Embedding model wrapper using sentence-transformers.
Uses BAAI/bge-base-en-v1.5 (Apache 2.0) for GPU-accelerated embeddings.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Embedder:
    """
    Wraps a sentence-transformers model for encoding text to vectors.
    Supports GPU acceleration and batch encoding.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = None

    def load(self, device: str = "cpu"):
        """Load the embedding model onto the specified device."""
        print(f"🔄 Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=device)

        print(f"✅ Model loaded on device: {self.model.device}")
        return self

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        For bge models, prepends the query instruction prefix.

        Returns:
            np.ndarray of shape (dim,) — L2-normalized vector.
        """
        if self.model is None:
            self.load()

        # bge models need instruction prefix for queries
        if "bge" in self.model_name.lower():
            text = config.QUERY_PREFIX + text

        vector = self.model.encode(
            text,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
            show_progress_bar=False,
        )
        return np.array(vector, dtype=np.float32)

    def embed_batch(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """
        Embed a batch of texts.

        Args:
            texts: List of strings to embed.
            is_query: If True and using bge model, prepend query prefix.

        Returns:
            np.ndarray of shape (n, dim) — L2-normalized vectors.
        """
        if self.model is None:
            self.load()

        # For bge models, add prefix to queries but NOT to documents
        if is_query and "bge" in self.model_name.lower():
            texts = [config.QUERY_PREFIX + t for t in texts]

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
        )
        return np.array(vectors, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        if self.model is None:
            self.load()
        return self.model.get_sentence_embedding_dimension()
