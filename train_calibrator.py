
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from core.pipeline import SOPPipeline
from core.calibrate import ConfidenceCalibrator

def train_calibrator():
    print("="*60)
    print("  TRAINING CALIBRATOR (Platt Scaling)")
    print("="*60)

    # 1. Load Calibration Data
    calib_file = "calib.txt"
    if not os.path.exists(calib_file):
        print(f"❌ {calib_file} not found. Run split_data.py first.")
        return

    with open(calib_file, "r", encoding="utf-8") as f:
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

    print(f"📊 Loaded {len(pairs)} calibration pairs.")

    # 2. Initialize Pipeline
    print("\n🔄 Loading Pipeline...")
    pipe = SOPPipeline()
    pipe.load()

    # 3. Collect Scores & Labels
    raw_scores = []
    labels = []
    
    print("\n🔄 Running Inference & Collecting Scores...")
    oos_phrases = ["not covered", "contact hr", "outside the scope"]

    for i, (q, expected) in enumerate(pairs):
        result = pipe.query(q)
        
        # Get top match
        if not result['top_k_results']:
            # No results at all? Should not happen if FAISS has data
            continue
            
        top_match = result['top_k_results'][0]
        # Use RAW score (before default calibration)
        # pipeline.query maps final_score = calibrate(rerank_score)
        # But we want rerank_score.
        score = top_match.get('rerank_score', 0.0)
        got_answer = top_match.get('answer', "")

        # Determine Ground Truth Label
        # Case A: OOS
        is_expected_oos = any(p in expected.lower() for p in oos_phrases)
        
        if is_expected_oos:
            # Expected is OOS.
            # If pipeline Retrieved an OOS snippet? (SOP usually doesn't have "Not covered" entries?)
            # Usually OOS queries should NOT match anything relevant.
            # So any High Score is a False Positive.
            # Label = 0 (Correctness of "Match").
            label = 0
            # Note: We are calibrating "Probability of Match being Correct/Relevant".
            # For OOS query, NO match is Correct.
        else:
            # In-Scope.
            # Expected is specific answer.
            # Check if retrieved answer matches Expected.
            # Approximate check
            if expected[:30].lower() in got_answer.lower():
                label = 1
            elif got_answer[:30].lower() in expected.lower():
                label = 1
            else:
                # Wrong retrieval
                label = 0
        
        raw_scores.append(score)
        labels.append(label)

        if i % 10 == 0:
            print(f"  [{i}/{len(pairs)}] Q: {q[:30]}... | Raw: {score:.4f} | Label: {label}")

    # 4. Train
    print("\n🔄 Fitting Calibration Model...")
    raw_scores = np.array(raw_scores)
    labels = np.array(labels)
    
    # Check if we have both classes
    if len(np.unique(labels)) < 2:
        print("⚠️  Warning: Only one class found in labels. Cannot train effectively.")
        print(f"    Labels: {np.unique(labels)}")
        # Force default?
    else:
        # Create fresh calibrator isolated from pipeline's
        trainer = ConfidenceCalibrator()
        trainer.train(raw_scores, labels)
        # Save to V6.3 path
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "calibrator_v6.3.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        trainer.save(save_path)
        
    print("\n✅ Training Complete.")

if __name__ == "__main__":
    train_calibrator()
