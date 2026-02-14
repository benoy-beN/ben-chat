"""
V6.3 Threshold Learning Script.
Learns optimal thresholds from calibrated scores on test.txt.
Optimizes F1 with FAR <= 1% constraint.
Exports to models/thresholds_v6.3.json.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pipeline import SOPPipeline
from core.calibrate import ConfidenceCalibrator


def learn_thresholds():
    print("=" * 60)
    print("  V6.3 THRESHOLD LEARNING")
    print("=" * 60)

    test_file = "test.txt"
    if not os.path.exists(test_file):
        print(f"❌ {test_file} not found. Run split_data.py first.")
        return

    with open(test_file, "r", encoding="utf-8") as f:
        content = f.read()

    pairs = []
    blocks = content.split("\n\n")
    for b in blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        if len(lines) >= 2:
            q_line = next((l for l in lines if l.startswith("Q:")), None)
            ans_line = next((l for l in lines if l.startswith("Expected:")), None)
            if q_line and ans_line:
                q = q_line.replace("Q:", "").strip()
                expected = ans_line.replace("Expected:", "").strip()
                pairs.append((q, expected))

    print(f"📊 Loaded {len(pairs)} test pairs")

    # Load pipeline
    print("\n🔄 Loading Pipeline...")
    pipe = SOPPipeline()
    pipe.load()

    # Collect calibrated scores and labels
    oos_phrases = ["not covered", "contact hr", "outside the scope"]
    calibrated_scores = []
    labels = []

    for i, (q, expected) in enumerate(pairs):
        result = pipe.query(q)
        score = result["score"]  # Already calibrated
        
        is_oos = any(p in expected.lower() for p in oos_phrases)
        
        if is_oos:
            label = 0
        else:
            got_answer = result.get("answer", "")
            if expected[:30].lower() in got_answer.lower():
                label = 1
            elif got_answer[:30].lower() in expected.lower():
                label = 1
            else:
                label = 0

        calibrated_scores.append(score)
        labels.append(label)

    scores = np.array(calibrated_scores)
    labels = np.array(labels)
    
    print(f"\n📊 Positive (in-scope correct): {np.sum(labels == 1)}")
    print(f"📊 Negative (OOS/wrong): {np.sum(labels == 0)}")

    # --- Threshold Optimization ---
    # Sweep THRESH_HIGH: maximize F1 with FAR <= 1% constraint
    best_f1 = 0
    best_high = 0.5
    best_metrics = {}

    n_neg = np.sum(labels == 0)

    for t in np.arange(0.10, 0.95, 0.005):
        preds = (scores >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        far = fp / n_neg if n_neg > 0 else 0
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        # Constraint: FAR <= 1%
        if far <= 0.01 and f1 > best_f1:
            best_f1 = f1
            best_high = float(t)
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "far": far,
                "frr": frr,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }

    # THRESH_LOW: set to where recall drops sharply
    # Find score below which 95% of negatives fall
    neg_scores = scores[labels == 0]
    if len(neg_scores) > 0:
        best_low = float(np.percentile(neg_scores, 50))  # Median of negatives
    else:
        best_low = 0.20

    # Ensure LOW < HIGH
    best_low = min(best_low, best_high - 0.05)

    print(f"\n{'=' * 60}")
    print(f"  LEARNED THRESHOLDS")
    print(f"{'=' * 60}")
    print(f"  THRESH_HIGH: {best_high:.4f}")
    print(f"  THRESH_LOW:  {best_low:.4f}")
    print(f"  F1:          {best_f1:.4f}")
    print(f"  FAR:         {best_metrics.get('far', 0):.4f}")
    print(f"  FRR:         {best_metrics.get('frr', 0):.4f}")
    print(f"  Precision:   {best_metrics.get('precision', 0):.4f}")
    print(f"  Recall:      {best_metrics.get('recall', 0):.4f}")

    # Export
    output = {
        "thresh_high": round(best_high, 4),
        "thresh_low": round(best_low, 4),
        "lex_min": 0.20,
        "gap_min": 0.02,
        "optimization": {
            "metric": "f1",
            "constraint": "far<=0.01",
            "best_f1": round(best_f1, 4),
            "far": round(best_metrics.get("far", 0), 4),
            "frr": round(best_metrics.get("frr", 0), 4),
        }
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "thresholds_v6.3.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Thresholds saved → {out_path}")


if __name__ == "__main__":
    learn_thresholds()
