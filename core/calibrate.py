"""
Confidence Calibration (V5).
Platt scaling / temperature scaling to convert raw reranker scores
into calibrated probabilities for reliable threshold decisions.
"""
import os
import pickle
import math
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ConfidenceCalibrator:
    """
    Calibrates raw scores into true probabilities using Platt scaling.
    σ(a * score + b) where a, b are fit on validation data.
    
    Falls back to raw sigmoid when untrained.
    """

    def __init__(self):
        self.a = 1.0   # Scale (identity by default)
        self.b = 0.0   # Bias
        self.is_trained = False
        self.threshold = getattr(config, 'SIMILARITY_THRESHOLD', 0.45)

    def calibrate(self, score: float) -> float:
        """
        Convert raw score to calibrated confidence.
        
        Args:
            score: Raw score (e.g., reranker output or fused score)
            
        Returns:
            Calibrated probability in [0, 1]
        """
        logit = self.a * score + self.b
        # Clip to avoid overflow
        logit = max(-20.0, min(20.0, logit))
        return 1.0 / (1.0 + math.exp(-logit))

    def calibrate_batch(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate a batch of scores."""
        logits = self.a * scores + self.b
        logits = np.clip(logits, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def should_accept(self, calibrated_score: float) -> bool:
        """
        Decision function: accept or reject based on calibrated threshold.
        
        Args:
            calibrated_score: Output of calibrate()
            
        Returns:
            True if score is above threshold (accept answer)
        """
        return calibrated_score >= self.threshold

    def train(self, raw_scores: np.ndarray, labels: np.ndarray):
        """
        Fit Platt scaling parameters on validation data.
        
        Args:
            raw_scores: np.ndarray of shape (N,) — raw scores
            labels: np.ndarray of shape (N,) — 1 for correct, 0 for incorrect
        """
        from sklearn.linear_model import LogisticRegression

        X = raw_scores.reshape(-1, 1)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, labels)

        # Extract Platt parameters: σ(a*x + b)
        self.a = float(model.coef_[0][0])
        self.b = float(model.intercept_[0])
        self.is_trained = True

        # Find optimal threshold via F1 on training data
        calibrated = self.calibrate_batch(raw_scores)
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.1, 0.9, 0.01):
            preds = (calibrated >= t).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        self.threshold = float(best_t)

        print(f"✅ Calibrator trained — a={self.a:.4f}, b={self.b:.4f}")
        print(f"   Optimal threshold: {self.threshold:.3f} (F1={best_f1:.3f})")

    def save(self, path: str = None):
        """Save calibrator to disk."""
        path = path or config.CALIBRATOR_FILE
        data = {
            "a": self.a,
            "b": self.b,
            "threshold": self.threshold,
            "is_trained": self.is_trained,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 Calibrator saved → {path}")

    def load(self, path: str = None):
        """Load calibrator from disk."""
        path = path or config.CALIBRATOR_FILE
        if not os.path.exists(path):
            print("ℹ️  No trained calibrator found. Using default sigmoid.")
            return self

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.a = data.get("a", 1.0)
        self.b = data.get("b", 0.0)
        self.threshold = data.get("threshold", 0.45)
        self.is_trained = data.get("is_trained", False)

        status = f"trained (a={self.a:.3f}, b={self.b:.3f}, t={self.threshold:.3f})" if self.is_trained else "default sigmoid"
        print(f"✅ Calibrator loaded ({status})")
        return self
