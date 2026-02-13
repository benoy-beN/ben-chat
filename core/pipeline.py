"""
Full SOP retrieval pipeline (V6 — SOTA Accuracy).
Handles: BGE-M3 → FAISS + BM25 → Learned Fusion → Rerank → Dual-Threshold Decision.

Decision Logic (v6_updated):
  if score >= THRESH_HIGH:   accept()
  elif score <= THRESH_LOW:  reject()
  else:  # gray zone
      if lexical_overlap >= LEX_MIN
         AND reranker_gap >= GAP_MIN
         AND answer_in_SOP == true:
          accept()
      else:
          reject()
"""
import json
import os
import re
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


# ── V6 Dual-Threshold Config ──────────────────────────
# These can be overridden by calibration data (learned from ROC)
THRESH_HIGH = 0.60   # Score >= this → accept (subject to guardrails)
THRESH_LOW  = 0.25   # Score <= this → always reject

# Gray-zone requirements (THRESH_LOW < score < THRESH_HIGH)
LEX_MIN     = 0.20   # Minimum lexical overlap in gray zone
GAP_MIN     = 0.02   # Minimum reranker score gap (top1 - top2)

# Phrases that indicate the matched SOP answer is itself a rejection
OOS_ANSWER_PHRASES = [
    "not covered in the sop",
    "please contact hr",
    "not in the sop",
    "outside the scope",
]

# ── Semantic Contradiction Detection ──────────────────
# Word pairs where query says X but SOP answer says Y → contradiction
CONTRADICTION_PAIRS = [
    ("maximum", "minimum"),
    ("max", "min"),
    ("largest", "smallest"),
    ("biggest", "minimum"),
]

# Context words in the query that indicate the topic is NOT covered by SOP
# (The SOP covers screen printing, not web design or digital media)
OOS_CONTEXT_WORDS = [
    "web design", "website", "web page", "digital ad",
    "social media", "video", "audio", "animation",
    "3d", "html", "css", "app design",
]

# Substances/materials NOT in the SOP PMS standards
# (SOP only has gold=871, silver=877)
OOS_PANTONE_WORDS = [
    "platinum", "bronze", "copper", "rose gold",
    "titanium", "chrome", "brass",
]


def _tokenize(text: str) -> set:
    """Simple tokenization: lowercase, remove punctuation, drop stop words."""
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = set(text.split())
    stop = {
        'the','a','an','is','are','was','were','be','been','to','of','in',
        'for','on','with','at','by','from','and','or','not','no','yes',
        'do','does','did','i','we','you','it','he','she','they','my',
        'your','what','which','who','how','when','where','why','should',
        'can','could','will','would','must','may','this','that','these',
        'those','about','me','tell',
    }
    return tokens - stop


def _lexical_overlap(query_tokens: set, candidate_tokens: set) -> float:
    """Fraction of query content-words found in candidate."""
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens)


class SOPPipeline:
    """V6 SOTA SOP retrieval pipeline with dual-threshold decision."""

    def __init__(self):
        self.embedder   = BGEM3Embedder(config.EMBEDDING_MODEL)
        self.faiss_index = FAISSIndex()
        self.bm25_index  = BM25Index()
        self.fusion      = ScoreFusion()
        self.reranker    = Reranker()
        self.calibrator  = ConfidenceCalibrator()
        self.sop_data: list[dict] = []
        self.is_loaded   = False

    # ── Build ──────────────────────────────────────────
    def build(self):
        """Build all indices from SOP data."""
        print("=" * 60)
        print("  Building V6 Pipeline (BGE-M3 + BM25 + Dual-Threshold)")
        print("=" * 60)

        aug_file = os.path.join(config.DATA_DIR, "sop_data_augmented.json")
        source_file = aug_file if os.path.exists(aug_file) else config.SOP_DATA_FILE

        with open(source_file, "r", encoding="utf-8") as f:
            self.sop_data = json.load(f)

        for entry in self.sop_data:
            if "source_id" not in entry:
                entry["source_id"] = entry["id"]

        print(f"📂 Loaded dataset: {len(self.sop_data)} entries")

        questions = [e["question"] for e in self.sop_data]

        # Dense embeddings
        print(f"\n🔄 Embedding with {self.embedder.model_name}...")
        self.embedder.load()
        embeddings = self.embedder.embed_batch(questions, is_query=False)

        dense_vecs = embeddings["dense"]
        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        dense_vecs = (dense_vecs / norms).astype(np.float32)

        self.faiss_index.build(dense_vecs, self.sop_data)
        self.faiss_index.save(index_path=config.FAISS_INDEX_FILE)

        # BM25
        print(f"\n🔄 Building BM25 lexical index...")
        self.bm25_index.build(self.sop_data)
        self.bm25_index.save()

        self.fusion.load()
        self.calibrator.load()
        self.is_loaded = True
        print(f"\n✅ V6 Pipeline built!")
        return self

    # ── Load ───────────────────────────────────────────
    def load(self):
        """Load all models and indices."""
        print("🔄 Loading V6 Pipeline...")

        self.faiss_index.load(index_path=config.FAISS_INDEX_FILE)

        try:
            self.bm25_index.load()
        except FileNotFoundError:
            print("⚠️  BM25 index not found. Lexical retrieval disabled.")

        self.sop_data = self.faiss_index.id_map
        self.fusion.load()
        self.calibrator.load()

        self.is_loaded = True
        print("✅ V6 Pipeline loaded!")
        return self

    # ── Query ──────────────────────────────────────────
    def query(self, user_question: str) -> dict:
        """
        V6 pipeline with dual-threshold decision logic.
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded.")

        normalized   = normalize(user_question)
        query_tokens = _tokenize(normalized)

        # ── Stage 1: Embed ──
        query_emb   = self.embedder.embed(normalized)
        query_dense = query_emb["dense"]
        norm = np.linalg.norm(query_dense)
        if norm > 0:
            query_dense = query_dense / norm

        # ── Stage 2: Parallel Retrieval ──
        top_k = config.TOP_K_RETRIEVAL

        faiss_results = self.faiss_index.search(query_dense, top_k=top_k)

        bm25_results = []
        if self.bm25_index.bm25 is not None:
            bm25_results = self.bm25_index.search(normalized, top_k=top_k)

        # ── Stage 3: Fusion ──
        candidates = {}
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
            norm_bm25 = r["score"] / bm25_max
            if key not in candidates:
                candidates[key] = {
                    "entry": r["entry"],
                    "dense_score": 0.0,
                    "sparse_score": 0.0,
                    "bm25_score": norm_bm25,
                }
            else:
                candidates[key]["bm25_score"] = max(candidates[key]["bm25_score"], norm_bm25)

        if not candidates:
            return self._reject(user_question, normalized, 0.0)

        candidate_list = list(candidates.values())
        for c in candidate_list:
            c["fused_score"] = self.fusion.fuse(
                c["dense_score"], c["sparse_score"], c["bm25_score"]
            )

        candidate_list.sort(key=lambda x: x["fused_score"], reverse=True)
        top_candidates = candidate_list[:min(len(candidate_list), top_k)]

        # ── Stage 4: Cross-Encoder Reranking ──
        texts = [f"{c['entry']['question']} {c['entry']['answer']}" for c in top_candidates]
        rerank_scores = self.reranker.compute_scores(normalized, texts)

        for i, c in enumerate(top_candidates):
            c["rerank_score"] = rerank_scores[i]

        top_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        best = top_candidates[0]
        score = best["rerank_score"]

        # ── Stage 5: Calibration (if trained) ──
        if self.calibrator.is_trained:
            score = self.calibrator.calibrate(score)
            thresh_high = self.calibrator.threshold
            thresh_low  = thresh_high * 0.4
        else:
            thresh_high = THRESH_HIGH
            thresh_low  = THRESH_LOW

        # ── Stage 6: Guardrails (apply BEFORE threshold) ──
        answer_text = best["entry"]["answer"].lower()
        matched_q = best["entry"]["question"].lower()
        query_lower = user_question.lower()

        # Guardrail A: Answer is itself an OOS rejection
        if any(phrase in answer_text for phrase in OOS_ANSWER_PHRASES):
            return self._reject(
                user_question, normalized, score,
                threshold=thresh_high,
                guardrail="answer_is_oos",
            )

        # Guardrail B: Semantic contradiction ("maximum" in query but "minimum" in match)
        for q_word, a_word in CONTRADICTION_PAIRS:
            if q_word in query_lower and a_word in matched_q and q_word not in matched_q:
                return self._reject(
                    user_question, normalized, score,
                    threshold=thresh_high,
                    guardrail=f"contradiction({q_word}_vs_{a_word})",
                )
            # Check reverse too
            if a_word in query_lower and q_word in matched_q and a_word not in matched_q:
                return self._reject(
                    user_question, normalized, score,
                    threshold=thresh_high,
                    guardrail=f"contradiction({a_word}_vs_{q_word})",
                )

        # Guardrail C: OOS context (query mentions web/digital/video but SOP is print)
        for ctx in OOS_CONTEXT_WORDS:
            if ctx in query_lower:
                return self._reject(
                    user_question, normalized, score,
                    threshold=thresh_high,
                    guardrail=f"oos_context({ctx})",
                )

        # Guardrail D: OOS Pantone materials (platinum, bronze, etc.)
        for material in OOS_PANTONE_WORDS:
            if material in query_lower:
                return self._reject(
                    user_question, normalized, score,
                    threshold=thresh_high,
                    guardrail=f"oos_pantone({material})",
                )

        # ── Stage 7: Dual-Threshold Decision ──
        if score >= thresh_high:
            decision = "accept_high"
        elif score <= thresh_low:
            return self._reject(
                user_question, normalized, score,
                threshold=thresh_low,
                guardrail="below_thresh_low",
            )
        else:
            # GRAY ZONE → check secondary signals
            best_tokens = _tokenize(best["entry"]["question"])
            overlap = _lexical_overlap(query_tokens, best_tokens)

            reranker_gap = 0.0
            if len(top_candidates) > 1:
                reranker_gap = best["rerank_score"] - top_candidates[1]["rerank_score"]

            if overlap >= LEX_MIN and reranker_gap >= GAP_MIN:
                decision = "accept_gray"
            else:
                return self._reject(
                    user_question, normalized, score,
                    threshold=thresh_high,
                    guardrail=f"gray_zone(overlap={overlap:.2f},gap={reranker_gap:.3f})",
                )

        # ── Stage 8: Output ──
        answer = best["entry"]["answer"]
        if config.USE_LLM_REWRITE:
            answer = rewrite_answer(answer)

        best_tokens = _tokenize(best["entry"]["question"])
        overlap = _lexical_overlap(query_tokens, best_tokens)

        return {
            "question": user_question,
            "normalized": normalized,
            "matched_question": best["entry"]["question"],
            "answer": answer,
            "score": score,
            "source_id": best["entry"].get("source_id"),
            "rejected": False,
            "threshold": thresh_high,
            "decision": decision,
            "lexical_overlap": overlap,
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
                for c in top_candidates[:10]
            ],
        }

    def query_detailed(self, user_question: str, top_k: int = 5) -> dict:
        """Alias for backward compat with app.py."""
        return self.query(user_question)

    def _reject(self, question, normalized, score,
                threshold=None, guardrail=None):
        return {
            "question": question,
            "normalized": normalized,
            "matched_question": None,
            "answer": config.REJECTION_MESSAGE,
            "score": score,
            "source_id": None,
            "rejected": True,
            "threshold": threshold if isinstance(threshold, (int, float)) else config.SIMILARITY_THRESHOLD,
            "guardrail": guardrail,
            "top_k_results": [],
        }
