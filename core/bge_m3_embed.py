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

    def load(self, device: str = None):
        """Load the BGE-M3 model."""
        device = device or config.DEVICE
        print(f"🔄 Loading BGE-M3 model: {self.model_name} (device: {device})")
        
        # Try offline first to utilize local cache/files (e.g. .bin)
        # This prevents unnecessary downloads/checks for .safetensors
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self._load_model(device)
            print("✅ BGE-M3 loaded locally (offline mode)")
        except Exception as e_offline:
            # Fallback to online if missing
            print(f"⚠️  Local load failed ({type(e_offline).__name__}). Switching to online mode.")
            if "HF_HUB_OFFLINE" in os.environ:
                del os.environ["HF_HUB_OFFLINE"]
            
            try:
                self._load_model(device)
            except Exception as e:
                print(f"❌ Failed to load BGE-M3: {e}")
                raise e
        finally:
            if "HF_HUB_OFFLINE" in os.environ:
                del os.environ["HF_HUB_OFFLINE"]
        
        return self

    def _load_model(self, device: str):
        try:
            # Check if we can import without error
            # (transformers 5.x breaks FlagEmbedding 1.3.5)
            from FlagEmbedding import BGEM3FlagModel
            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=(device != "cpu"),
                device=device
            )
            self._dimension = 1024  # BGE-M3 dense dimension
        except Exception as e:
            # Fallback for when FlagEmbedding is incompatible or missing
            # print(f"⚠️  FlagEmbedding issue ({type(e).__name__}). Using sentence-transformers with BGE-M3.")
            from sentence_transformers import SentenceTransformer
            # Force valid BGE-M3 model name if config has something else
            st_model_name = "BAAI/bge-m3" if "bge-m3" in self.model_name.lower() else "BAAI/bge-base-en-v1.5"
            self.model = SentenceTransformer(st_model_name, device=device)
            self._dimension = self.model.get_sentence_embedding_dimension()

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
