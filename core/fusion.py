"""
Learned Score Fusion (V5).
Query-adaptive fusion of dense, sparse, and BM25 retrieval scores.
Falls back to weighted average when no trained model is available.
"""
import os
import pickle
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ScoreFusion:
    """
    Fuses scores from multiple retrieval sources:
      - Dense (FAISS cosine similarity)
      - Sparse (BGE-M3 lexical weights)
      - BM25 (lexical token matching)
    
    Uses logistic regression when trained, else weighted average.
    """

    def __init__(self):
        self.model = None
        self.is_trained = False
        # Default weights: [dense, sparse, bm25]
        self.default_weights = getattr(
            config, 'FUSION_WEIGHTS', [0.5, 0.2, 0.3]
        )

    def fuse(self, dense_score: float, sparse_score: float, bm25_score: float) -> float:
        """
        Combine scores from all retrieval sources.
        
        Args:
            dense_score: Cosine similarity from FAISS (0-1)
            sparse_score: Sparse lexical score from BGE-M3 (normalized)
            bm25_score: BM25 score (normalized)
            
        Returns:
            Combined score (float)
        """
        if self.is_trained and self.model is not None:
            features = np.array([[dense_score, sparse_score, bm25_score]])
            return float(self.model.predict_proba(features)[0][1])
        
        # Fallback: weighted average
        w = self.default_weights
        return (w[0] * dense_score + w[1] * sparse_score + w[2] * bm25_score)

    def fuse_batch(self, scores_matrix: np.ndarray) -> np.ndarray:
        """
        Fuse a batch of score vectors.
        
        Args:
            scores_matrix: np.ndarray of shape (N, 3) — [dense, sparse, bm25]
            
        Returns:
            np.ndarray of shape (N,) — fused scores
        """
        if self.is_trained and self.model is not None:
            return self.model.predict_proba(scores_matrix)[:, 1]
        
        w = np.array(self.default_weights)
        return scores_matrix @ w

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the fusion model on labeled data.
        
        Args:
            X: Feature matrix of shape (N, 3) — [dense, sparse, bm25] scores
            y: Labels of shape (N,) — 1 for correct match, 0 for incorrect
        """
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True

        accuracy = self.model.score(X, y)
        print(f"✅ Fusion model trained — Training accuracy: {accuracy:.3f}")
        print(f"   Learned weights: {self.model.coef_[0]}")

    def save(self, path: str = None):
        """Save trained fusion model."""
        path = path or config.FUSION_MODEL_FILE
        data = {
            "model": self.model,
            "is_trained": self.is_trained,
            "default_weights": self.default_weights,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 Fusion model saved → {path}")

    def load(self, path: str = None):
        """Load trained fusion model."""
        path = path or config.FUSION_MODEL_FILE
        if not os.path.exists(path):
            print("ℹ️  No trained fusion model found. Using default weights.")
            return self

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data.get("model")
        self.is_trained = data.get("is_trained", False)
        self.default_weights = data.get("default_weights", self.default_weights)

        status = "trained model" if self.is_trained else "default weights"
        print(f"✅ Fusion model loaded ({status})")
        return self
