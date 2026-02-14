"""
V6.3 Fusion Weight Training Script.
Trains ScoreFusion on calib.txt using [dense, bm25, rerank] features.
Saves to models/fusion_v6.3.json.
"""
import os
import sys
import json
import numpy as np
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pipeline import SOPPipeline
from core.fusion import ScoreFusion
from core.bm25_index import BM25Index
import config


def train_fusion():
    print("=" * 60)
    print("  V6.3 FUSION WEIGHT TRAINING")
    print("=" * 60)

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
        q_line = next((l for l in lines if l.startswith("Q:")), None)
        ans_line = next((l for l in lines if l.startswith("Expected:")), None)
        if q_line and ans_line:
            q = q_line.replace("Q:", "").strip()
            expected = ans_line.replace("Expected:", "").strip()
            pairs.append((q, expected))

    print(f"📊 Loaded {len(pairs)} calibration pairs")

    # Load pipeline
    pipe = SOPPipeline()
    pipe.load()

    oos_phrases = ["not covered", "contact hr", "outside the scope"]

    # Collect features: [dense_score, bm25_score, rerank_score] and labels
    features = []
    labels = []

    for i, (q, expected) in enumerate(pairs):
        normalized = q.lower().strip()
        
        # Dense retrieval
        query_emb = pipe.embedder.embed(normalized)
        query_dense = query_emb["dense"]
        norm = np.linalg.norm(query_dense)
        if norm > 0:
            query_dense = query_dense / norm
        
        faiss_results = pipe.faiss_index.search(query_dense, top_k=5)
        if not faiss_results:
            continue
        
        dense_score = faiss_results[0]["score"]

        # BM25
        bm25_score = 0.0
        if pipe.bm25_index.bm25 is not None:
            bm25_results = pipe.bm25_index.search(normalized, top_k=5)
            if bm25_results:
                bm25_score = bm25_results[0]["score"]

        # Reranker
        texts = [faiss_results[0]["entry"]["question"]]
        try:
            rerank_scores = pipe.reranker.compute_scores(normalized, texts)
            rerank_score = rerank_scores[0]
        except:
            rerank_score = dense_score

        # Label
        is_oos = any(p in expected.lower() for p in oos_phrases)
        if is_oos:
            label = 0
        else:
            got_answer = faiss_results[0]["entry"]["answer"]
            if expected[:30].lower() in got_answer.lower():
                label = 1
            elif got_answer[:30].lower() in expected.lower():
                label = 1
            else:
                label = 0

        features.append([dense_score, bm25_score, rerank_score])
        labels.append(label)

        if i % 10 == 0:
            print(f"  [{i}/{len(pairs)}] D={dense_score:.4f} B={bm25_score:.4f} R={rerank_score:.4f} L={label}")

    X = np.array(features)
    y = np.array(labels)

    print(f"\n📊 Features shape: {X.shape}")
    print(f"📊 Positive: {np.sum(y==1)}, Negative: {np.sum(y==0)}")

    # Train
    fusion = ScoreFusion()
    fusion.train(X, y)

    # Save as pickle (ScoreFusion.save uses pickle)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fusion_v6.3.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as pickle despite .json extension (ScoreFusion expects pickle format)
    fusion.save(save_path)
    
    # Also export human-readable weights
    if fusion.is_trained and fusion.model is not None:
        weights_info = {
            "weights": fusion.model.coef_[0].tolist(),
            "intercept": fusion.model.intercept_[0],
            "feature_names": ["dense_score", "bm25_score", "rerank_score"],
        }
        readable_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fusion_weights_readable.json")
        with open(readable_path, "w") as f:
            json.dump(weights_info, f, indent=2)
        print(f"📊 Readable weights saved → {readable_path}")


if __name__ == "__main__":
    train_fusion()
