"""
BGE-M3 Embedding model wrapper.
Uses BAAI/bge-m3 for dense + sparse vectors in a single forward pass.
Replaces dual-encoder setup (V4) with single strong retriever (V5).
"""
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BGEM3Embedder:
    """
    Wraps FlagEmbedding's BGEM3FlagModel for dense + sparse encoding.
    Single model produces both vector types — no need for dual encoders.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = None
        self._dimension = None

    def load(self, device: str = "cpu"):
        """Load the BGE-M3 model."""
        print(f"🔄 Loading BGE-M3 model: {self.model_name}")
        
        try:
            from FlagEmbedding import BGEM3FlagModel
            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=(device != "cpu"),
            )
            self._dimension = 1024  # BGE-M3 dense dimension
            print(f"✅ BGE-M3 loaded (dense dim: {self._dimension})")
        except ImportError:
            print("⚠️  FlagEmbedding not installed. Falling back to sentence-transformers BGE.")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
            self._dimension = self.model.get_sentence_embedding_dimension()
            print(f"✅ Fallback model loaded (dim: {self._dimension})")

        return self

    def _is_flag_model(self) -> bool:
        """Check if we're using the FlagEmbedding model."""
        return hasattr(self.model, 'encode') and not hasattr(self.model, 'get_sentence_embedding_dimension')

    def embed(self, text: str) -> dict:
        """
        Embed a single query string.
        
        Returns:
            dict with:
              - "dense": np.ndarray of shape (dim,)
              - "sparse": dict of {token_id: weight} (empty if fallback)
        """
        if self.model is None:
            self.load()

        if self._is_flag_model():
            output = self.model.encode(
                [text],
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense = np.array(output['dense_vecs'][0], dtype=np.float32)
            # Sparse: list of dicts [{token_id: weight}, ...]
            sparse = output['lexical_weights'][0] if 'lexical_weights' in output else {}
            # Convert sparse keys to strings for JSON serialization
            if hasattr(sparse, 'items'):
                sparse = {str(k): float(v) for k, v in sparse.items()}
            else:
                sparse = {}
        else:
            # Fallback: sentence-transformers (dense only)
            prefix = config.QUERY_PREFIX if hasattr(config, 'QUERY_PREFIX') else ""
            vec = self.model.encode(
                prefix + text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            dense = np.array(vec, dtype=np.float32)
            sparse = {}

        return {"dense": dense, "sparse": sparse}

    def embed_batch(self, texts: list[str], is_query: bool = False) -> dict:
        """
        Embed a batch of texts.
        
        Returns:
            dict with:
              - "dense": np.ndarray of shape (N, dim)
              - "sparse": list of dicts
        """
        if self.model is None:
            self.load()

        if self._is_flag_model():
            output = self.model.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
                batch_size=32,
            )
            dense = np.array(output['dense_vecs'], dtype=np.float32)
            sparse_list = []
            if 'lexical_weights' in output:
                for sp in output['lexical_weights']:
                    if hasattr(sp, 'items'):
                        sparse_list.append({str(k): float(v) for k, v in sp.items()})
                    else:
                        sparse_list.append({})
            else:
                sparse_list = [{} for _ in texts]
        else:
            # Fallback: sentence-transformers
            prefix = config.QUERY_PREFIX if (is_query and hasattr(config, 'QUERY_PREFIX')) else ""
            processed = [prefix + t for t in texts] if prefix else texts
            vecs = self.model.encode(
                processed,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=64,
            )
            dense = np.array(vecs, dtype=np.float32)
            sparse_list = [{} for _ in texts]

        return {"dense": dense, "sparse": sparse_list}

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        if self._dimension is None:
            self.load()
        return self._dimension
