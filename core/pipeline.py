"""
Full SOP retrieval pipeline (V5 — Accuracy-First).
Handles: BGE-M3 → FAISS + BM25 → Learned Fusion → Rerank → Calibrate.

Architecture:
1. Normalize Query
2. Embed with BGE-M3 (dense + sparse)
3. Parallel FAISS (dense) + BM25 (lexical) retrieval
4. Learned Fusion (query-adaptive score combination)
5. Cross-Encoder Reranking (Top-K → Top-1)
6. Confidence Calibration (Platt scaling)
7. Adaptive Threshold → Accept / Reject
"""
import json
import os
import time
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from core.normalizer import normalize
from core.bge_m3_embed import BGEM3Embedder
from core.index import FAISSIndex
from core.bm25_index import BM25Index
from core.fusion import ScoreFusion
from core.rewriter import rewrite_answer
from core.reranker import Reranker
from core.calibrate import ConfidenceCalibrator


class SOPPipeline:
    """
    V5 Accuracy-First SOP retrieval pipeline.
    Single BGE-M3 encoder + BM25 + learned fusion + calibrated reranking.
    """
    def __init__(self):
        # Single strong encoder
        self.embedder = BGEM3Embedder(config.EMBEDDING_MODEL)
        
        # Dual retrieval: vector + lexical
        self.faiss_index = FAISSIndex()
        self.bm25_index = BM25Index()
        
        # Fusion & Reranking
        self.fusion = ScoreFusion()
        self.reranker = Reranker()
        self.calibrator = ConfidenceCalibrator()
        
        self.sop_data: list[dict] = []
        self.is_loaded = False

    def build(self):
        """Build all indices from SOP data."""
        print("=" * 60)
        print("  Building V5 Pipeline (BGE-M3 + BM25)")
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
        
        # --- Build Dense + Sparse Index (BGE-M3) ---
        print(f"\n🔄 Embedding with {self.embedder.model_name}...")
        self.embedder.load()
        embeddings = self.embedder.embed_batch(questions, is_query=False)
        
        # Build FAISS index (dense)
        dense_vecs = embeddings["dense"]
        # Normalize dense vectors for cosine similarity
        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        dense_vecs = (dense_vecs / norms).astype(np.float32)
        
        self.faiss_index.build(dense_vecs, self.sop_data)
        self.faiss_index.save(index_path=config.FAISS_INDEX_FILE)
        
        # --- Build BM25 Index ---
        print(f"\n🔄 Building BM25 lexical index...")
        self.bm25_index.build(self.sop_data)
        self.bm25_index.save()
        
        # --- Load fusion & calibrator (if pre-trained) ---
        self.fusion.load()
        self.calibrator.load()

        self.is_loaded = True
        print(f"\n✅ V5 Pipeline built and saved!")
        return self

    def load(self):
        """Load all models and indices."""
        print("🔄 Loading V5 Pipeline...")
        
        # Load FAISS index
        self.faiss_index.load(index_path=config.FAISS_INDEX_FILE)
        
        # Load BM25 index
        try:
            self.bm25_index.load()
        except FileNotFoundError:
            print("⚠️  BM25 index not found. Lexical retrieval disabled.")
        
        # Load SOP data from id_map
        self.sop_data = self.faiss_index.id_map
        
        # Load fusion model (optional trained)
        self.fusion.load()
        
        # Load calibrator (optional trained)
        self.calibrator.load()
        
        self.is_loaded = True
        print("✅ V5 Pipeline loaded!")
        return self

    def query(self, user_question: str) -> dict:
        """
        V5 retrieval pipeline:
        Normalize → Embed → FAISS + BM25 → Fusion → Rerank → Calibrate → Decide
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded.")

        normalized = normalize(user_question)
        
        # --- Stage 1: Embed query with BGE-M3 ---
        query_emb = self.embedder.embed(normalized)
        query_dense = query_emb["dense"]
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(query_dense)
        if norm > 0:
            query_dense = query_dense / norm
        
        # --- Stage 2: Parallel Retrieval ---
        top_k = config.TOP_K_RETRIEVAL  # 20
        
        # Dense retrieval (FAISS)
        faiss_results = self.faiss_index.search(query_dense, top_k=top_k)
        
        # BM25 lexical retrieval
        bm25_results = []
        if self.bm25_index.bm25 is not None:
            bm25_results = self.bm25_index.search(normalized, top_k=top_k)
        
        # --- Stage 3: Learned Fusion ---
        # Collect all unique candidates
        candidates = {}  # key: question text → {entry, dense_score, bm25_score}
        
        # Normalize BM25 scores to [0, 1]
        bm25_max = max((r["score"] for r in bm25_results), default=1.0) if bm25_results else 1.0
        if bm25_max <= 0:
            bm25_max = 1.0
        
        for r in faiss_results:
            key = r["entry"]["question"]
            if key not in candidates:
                candidates[key] = {
                    "entry": r["entry"],
                    "dense_score": r["score"],
                    "sparse_score": 0.0,
                    "bm25_score": 0.0,
                }
            else:
                candidates[key]["dense_score"] = max(candidates[key]["dense_score"], r["score"])
        
        for r in bm25_results:
            key = r["entry"]["question"]
            normalized_bm25 = r["score"] / bm25_max
            if key not in candidates:
                candidates[key] = {
                    "entry": r["entry"],
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "bm25_score": normalized_bm25,
                }
            else:
                candidates[key]["bm25_score"] = max(candidates[key]["bm25_score"], normalized_bm25)
        
        if not candidates:
            return self._reject(user_question, normalized, 0.0)
        
        # Fuse scores
        candidate_list = list(candidates.values())
        for c in candidate_list:
            c["fused_score"] = self.fusion.fuse(
                c["dense_score"], c["sparse_score"], c["bm25_score"]
            )
        
        # Sort by fused score and take top candidates for reranking
        candidate_list.sort(key=lambda x: x["fused_score"], reverse=True)
        top_candidates = candidate_list[:min(len(candidate_list), config.TOP_K_RETRIEVAL)]
        
        # --- Stage 4: Cross-Encoder Reranking ---
        candidate_texts = []
        for c in top_candidates:
            combined_text = f"{c['entry']['question']} {c['entry']['answer']}"
            candidate_texts.append(combined_text)
        
        rerank_scores = self.reranker.compute_scores(normalized, candidate_texts)
        
        for i, c in enumerate(top_candidates):
            c["rerank_score"] = rerank_scores[i]
        
        # Re-sort by reranker score (reranker is precision-focused)
        top_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        best = top_candidates[0]
        
        # --- Stage 5: Confidence Calibration ---
        # When calibrator is trained, apply Platt scaling to raw reranker score.
        # When untrained, use reranker score directly (already sigmoid-normalized).
        if self.calibrator.is_trained:
            calibrated_confidence = self.calibrator.calibrate(best["rerank_score"])
            threshold = self.calibrator.threshold
        else:
            calibrated_confidence = best["rerank_score"]
            threshold = config.SIMILARITY_THRESHOLD
        
        # --- Stage 6: Threshold Decision ---
        if calibrated_confidence < threshold:
            return self._reject(user_question, normalized, calibrated_confidence, threshold)

        # --- Stage 7: Output ---
        answer = best["entry"]["answer"]
        if config.USE_LLM_REWRITE:
            answer = rewrite_answer(answer)
            
        return {
            "question": user_question,
            "normalized": normalized,
            "matched_question": best["entry"]["question"],
            "answer": answer,
            "score": calibrated_confidence,
            "source_id": best["entry"].get("source_id"),
            "rejected": False,
            "threshold": threshold,
            "top_k_results": [
                {
                    "question": c["entry"]["question"],
                    "answer": c["entry"]["answer"],
                    "score": c.get("rerank_score", c.get("fused_score", 0.0)),
                    "dense_score": c.get("dense_score", 0.0),
                    "bm25_score": c.get("bm25_score", 0.0),
                    "fused_score": c.get("fused_score", 0.0),
                    "source_id": c["entry"].get("source_id"),
                }
                for c in top_candidates[:10]  # Return top 10 for debug
            ],
        }

    def query_detailed(self, user_question: str, top_k: int = 5) -> dict:
        """Alias for query to maintain compatibility with app.py."""
        return self.query(user_question)

    def _reject(self, question, normalized, score, threshold=None):
        return {
            "question": question,
            "normalized": normalized,
            "matched_question": None,
            "answer": config.REJECTION_MESSAGE,
            "score": score,
            "source_id": None,
            "rejected": True,
            "threshold": threshold or config.SIMILARITY_THRESHOLD,
            "top_k_results": [],
        }
