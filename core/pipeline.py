"""
Full SOP retrieval pipeline (v4 SOTA).
Handles: Hybrid Search (Dual Encoder) → Rerank → Threshold.

Architecture:
1. Normalize Query
2. Embed (bge) & Embed (e5)
3. Parallel FAISS Search
4. Weighted Score Fusion
5. Cross-Encoder Reranking (Top-K)
6. Calibrated Threshold
"""
import json
import os
import time
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from core.normalizer import normalize
from core.embedder import Embedder
from core.index import FAISSIndex
from core.rewriter import rewrite_answer
from core.reranker import Reranker

class SOPPipeline:
    """
    State-of-the-Art SOP retrieval pipeline.
    Combines two embedding models and a cross-encoder reranker.
    """
    def __init__(self):
        # Dual Encoders
        self.embedder_a = Embedder(config.EMBEDDING_MODEL)
        self.embedder_b = Embedder(config.MODEL_B_NAME)
        
        # Dual Indices
        self.index_a = FAISSIndex()
        self.index_b = FAISSIndex()
        
        # Reranker
        self.reranker = Reranker()
        
        self.sop_data: list[dict] = []
        self.is_loaded = False

    def build(self):
        """Build both indices from SOP data."""
        print("=" * 60)
        print("  Building SOTA Pipeline (Dual Encoder)")
        print("=" * 60)

        # Load data
        aug_file = os.path.join(config.DATA_DIR, "sop_data_augmented.json")
        source_file = aug_file if os.path.exists(aug_file) else config.SOP_DATA_FILE
        
        with open(source_file, "r", encoding="utf-8") as f:
            self.sop_data = json.load(f)
        
        # Ensure source_id
        for entry in self.sop_data:
            if "source_id" not in entry:
                entry["source_id"] = entry["id"]

        print(f"📂 Loaded dataset: {len(self.sop_data)} entries")
        
        questions = [entry["question"] for entry in self.sop_data]
        
        # --- Build Index A (BGE) ---
        print(f"\n🔄 [Model A] Embedding with {self.embedder_a.model_name}...")
        self.embedder_a.load()
        vecs_a = self.embedder_a.embed_batch(questions, is_query=False)
        self.index_a.build(vecs_a, self.sop_data)
        self.index_a.save(index_path=config.FAISS_INDEX_FILE)

        # --- Build Index B (E5) ---
        print(f"\n🔄 [Model B] Embedding with {self.embedder_b.model_name}...")
        self.embedder_b.load()
        vecs_b = self.embedder_b.embed_batch(questions, is_query=False)
        self.index_b.build(vecs_b, self.sop_data)
        self.index_b.save(index_path=config.FAISS_INDEX_FILE_B)

        self.is_loaded = True
        print(f"\n✅ Dual Pipeline built and saved!")
        return self

    def load(self):
        """Load all models and indices."""
        print("🔄 Loading SOTA Pipeline...")
        
        # Load indices (Model loading is lazy/on-demand usually, but we can pre-load)
        # We only really need to load indices eagerly.
        self.index_a.load(index_path=config.FAISS_INDEX_FILE)
        if hasattr(config, "FAISS_INDEX_FILE_B") and os.path.exists(config.FAISS_INDEX_FILE_B):
            self.index_b.load(index_path=config.FAISS_INDEX_FILE_B)
        else:
            print("⚠️ Index B not found. Dual encoder features disabled.")

        # Load ID map (same for both)
        self.sop_data = self.index_a.id_map
        
        self.is_loaded = True
        print("✅ Pipeline loaded!")
        return self

    def query(self, user_question: str) -> dict:
        """
        Hybrid retrieval + Reranking.
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded.")

        normalized = normalize(user_question)
        
        # --- Stage 1: Hybrid Retrieval ---
        # Search Index A
        vec_a = self.embedder_a.embed(normalized)
        results_a = self.index_a.search(vec_a, top_k=config.TOP_K_RETRIEVAL)
        
        # Search Index B
        vec_b = self.embedder_b.embed(normalized)
        results_b = self.index_b.search(vec_b, top_k=config.TOP_K_RETRIEVAL)

        # Fusion
        combined_scores = {}
        entries = {}
        
        # Helper to process results
        def process_results(results, weight):
            for r in results:
                eid = r["entry"].get("source_id", r["entry"].get("id")) # Use source ID to identify unique q
                # Use question content as unique key if IDs overlap for different paraphrases
                # Actually, our ID map stores each paraphrase as separate entry.
                # We want to retrieve specific paraphrases.
                key = r["entry"]["question"] 
                entries[key] = r["entry"]
                combined_scores[key] = combined_scores.get(key, 0.0) + (r["score"] * weight)

        process_results(results_a, config.HYBRID_WEIGHT_A)
        process_results(results_b, config.HYBRID_WEIGHT_B)
        
        # Sort candidates
        candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:config.TOP_K_RETRIEVAL] # e.g. Top 5
        
        if not top_candidates:
             return self._reject(user_question, normalized, 0.0)

        # --- Stage 2: Reranking ---
        # Construct candidate texts: Just Answer? Or Q+A?
        # Using Q+A works best generally.
        candidate_texts = []
        for q_text, score in top_candidates:
            entry = entries[q_text]
            combined_text = f"{entry['question']} {entry['answer']}"
            candidate_texts.append(combined_text)
            
        rerank_scores = self.reranker.compute_scores(normalized, candidate_texts)
        
        final_candidates = []
        for i, (q_text, old_score) in enumerate(top_candidates):
            r_score = rerank_scores[i]
            # Ensemble Score: 50% Hybrid (Recall) + 50% Reranker (Precision)
            # This handles cases where Reranker is incorrectly confident (0.001)
            # while Hybrid is confident (0.8). Result ~0.4.
            ensemble_score = 0.5 * old_score + 0.5 * r_score
            
            final_candidates.append({
                "entry": entries[q_text],
                "score": ensemble_score,
                "initial_score": old_score,
                "rerank_score": r_score
            })
            
        # Resort by ENSEMBLE score
        final_candidates.sort(key=lambda x: x["score"], reverse=True)
        best_match = final_candidates[0]
        
        # --- Stage 3: Threshold ---
        threshold = config.SIMILARITY_THRESHOLD
        
        if best_match["score"] < threshold:
             return self._reject(user_question, normalized, best_match["score"])

        # --- Stage 4: Output ---
        answer = best_match["entry"]["answer"]
        if config.USE_LLM_REWRITE:
            answer = rewrite_answer(answer)
            
        return {
            "question": user_question,
            "normalized": normalized,
            "matched_question": best_match["entry"]["question"],
            "answer": answer,
            "score": best_match["score"],
            "source_id": best_match["entry"].get("source_id"),
            "rejected": False,
            "threshold": threshold,
            "top_k_results": [
                {
                    "question": c["entry"]["question"],
                    "answer": c["entry"]["answer"],
                    "score": c["score"],
                    "initial_score": c["initial_score"],
                    "source_id": c["entry"].get("source_id")
                }
                for c in final_candidates
            ]
        }

    def query_detailed(self, user_question: str, top_k: int = 5) -> dict:
        """Alias for query to maintain compatibility with app.py."""
        # Note: top_k argument is currently ignored as config.TOP_K_RETRIEVAL is used
        return self.query(user_question)

    def _reject(self, question, normalized, score):
        return {
            "question": question,
            "normalized": normalized,
            "matched_question": None,
            "answer": config.REJECTION_MESSAGE,
            "score": score,
            "source_id": None,
            "rejected": True,
            "threshold": config.SIMILARITY_THRESHOLD,
            "top_k_results": []
        }
