"""
Cross-Encoder Reranker wrapper.
Uses BAAI/bge-reranker-base to score query-document pairs.
"""
from sentence_transformers import CrossEncoder
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class Reranker:
    """
    Wraps a CrossEncoder model for reranking retrieval results.
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.RERANKER_NAME
        self.model = None

    def load(self, device: str = None):
        """Load the reranker model."""
        device = device or config.DEVICE
        print(f"🔄 Loading reranker: {self.model_name}")
        self.model = CrossEncoder(self.model_name, device=device)
        print(f"✅ Reranker loaded on {self.model.device}")
        return self

    def compute_scores(self, query: str, documents: list[str]) -> list[float]:
        """
        Compute similarity scores for (query, doc) pairs.
        
        Args:
            query: The user question.
            documents: List of candidate answers/documents.
            
        Returns:
            List of float scores (higher is better).
        """
        if self.model is None:
            self.load()

        if not documents:
            return []

        # bge-reranker input [[query, document], ...]
        pairs = [[query, doc] for doc in documents]
        
        # Calculate scores
        scores = self.model.predict(pairs)
        
        if isinstance(scores, (int, float)):
             scores = [scores]
        elif hasattr(scores, "tolist"):
             scores = scores.tolist()
             
        # Normalize logits to [0,1] using Sigmoid
        # 1 / (1 + exp(-x))
        import math
        return [1 / (1 + math.exp(-s)) for s in scores]
