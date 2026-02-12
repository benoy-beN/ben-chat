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

    def _preprocess(self, text: str, is_query: bool) -> str:
        """Add model-specific prefix."""
        model_lower = self.model_name.lower()
        
        if "bge" in model_lower:
            if is_query:
                return config.QUERY_PREFIX + text
            return text  # BGE docs have no prefix
            
        if "e5" in model_lower:
            if is_query:
                return "query: " + text
            return "passage: " + text
            
        return text

    def embed(self, text: str) -> np.ndarray:
        """Embed a single query string."""
        if self.model is None:
            self.load()

        text = self._preprocess(text, is_query=True)

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vector, dtype=np.float32)

    def embed_batch(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Embed a batch of texts."""
        if self.model is None:
            self.load()

        processed_texts = [self._preprocess(t, is_query=is_query) for t in texts]

        vectors = self.model.encode(
            processed_texts,
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
