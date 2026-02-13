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
        self.reranker    = Reranker()
        self.calibrator  = ConfidenceCalibrator() # Restored for V6
        
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
        print("🔄 Loading V6 Pipeline (BGE-M3 Only)...")

        self.faiss_index.load(index_path=config.FAISS_INDEX_FILE)

        # BM25, Fusion removed. Calibrator restored.
        self.calibrator.load()
        
        self.sop_data = self.faiss_index.id_map
        
        self.is_loaded = True
        print("✅ V6 Pipeline (BGE-M3 + Calibrated) loaded!")
        return self

    # ── Query ──────────────────────────────────────────
    # ── Query ──────────────────────────────────────────
    def query(self, user_question: str) -> dict:
        """
        V6 pipeline (BGE-M3 + Reranker + Dual-Threshold).
        Logic:
          1. Dense Retrieval (Top-K)
          2. Reranking (Top-K)
          3. Decision Gate:
             - If score >= THRESH_HIGH (0.988) -> Accept
             - If score < THRESH_LOW (0.95) -> Reject
             - Else (Gray Zone):
                 Check Lexical Overlap >= LEX_MIN (0.20)
                 AND Reranker Gap >= GAP_MIN (0.02)
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded.")

        normalized = user_question.lower().strip()
        query_emb  = self.embedder.embed(normalized)
        query_dense = query_emb["dense"]
        
        norm = np.linalg.norm(query_dense)
        if norm > 0:
            query_dense = query_dense / norm

        # ── Stage 1: Dense Retrieval ──
        top_k = config.TOP_K_RETRIEVAL
        faiss_results = self.faiss_index.search(query_dense, top_k=top_k)

        if not faiss_results:
             return self._reject(user_question, normalized, 0.0)

        # ── Stage 2: Cross-Encoder Reranking ──
        candidates = faiss_results
        texts = [c['entry']['question'] for c in candidates]
        
        try:
            rerank_scores = self.reranker.compute_scores(normalized, texts)
        except Exception as e:
            print(f"⚠️ Reranker failed: {e}. Using dense scores.")
            rerank_scores = [c['score'] for c in candidates]

        # Attach scores & Calculate Metrics
        for i, c in enumerate(candidates):
            c["rerank_score"] = rerank_scores[i]
            c["final_score"] = c["rerank_score"] 
            
            # Pre-calculate Lexical Overlap (Jaccard on sets)
            q_set = set(normalized.split())
            d_text = c['entry']['question'].lower()
            d_set = set(d_text.split())
            if q_set:
                c["lexical_overlap"] = len(q_set.intersection(d_set)) / len(q_set)
            else:
                c["lexical_overlap"] = 0.0

            # Calibrate Score
            # Map raw reranker score (logit) to probability [0, 1]
            c["final_score"] = self.calibrator.calibrate(c["rerank_score"]) 

        best_match = candidates[0]
        final_score = best_match["final_score"]

        # Calculate Reranker Gap (Diff between #1 and #2)
        reranker_gap = 0.0
        if len(candidates) > 1:
            reranker_gap = candidates[0]["final_score"] - candidates[1]["final_score"]

        # ── Stage 3: Dual-Threshold Decision (V6) ──
        is_rejected = False
        decision_reason = "Score above threshold"
        
        # [1] High Confidence Acceptance
        if final_score >= config.THRESHOLD_HIGH:
            is_rejected = False
            decision_reason = "High confidence (>= 0.988)"
            
        # [2] Low Confidence Rejection
        elif final_score < config.THRESHOLD_LOW:
            # [5] Last-Chance Accept (Safe Guard for Exact Matches in SOP)
            # If the best match ID is a known valid SOP and score is decent? 
            # (Simplification: Just use threshold for now, or maybe check exact string match?)
            # Plan says: if best_id in SOP_IDS and score >= THRESH_LOW. 
            # But we are already < THRESH_LOW here.
            # "Target: if best_id in SOP_IDS and score >= THRESH_LOW: ACCEPT" -> This implies < THRESH_HIGH but >= THRESH_LOW.
            # Wait, the logic is: >= HIGH (Accept), < LOW (Reject).
            # So the "Last Chance" must apply to the GRAY ZONE (HIGH > score >= LOW).
            # My logic:
            # - If score >= HIGH: Accept
            # - If score < LOW: Reject
            # - Else (Gray Zone): Semantic Checks
            
            is_rejected = True
            decision_reason = "Low confidence (< 0.95)"

        # [3] Gray Zone (0.95 <= score < 0.988)
        else:
            lex_check = best_match["lexical_overlap"] >= config.LEX_MIN
            gap_check = reranker_gap >= config.GAP_MIN
            
            if lex_check and gap_check:
                is_rejected = False
                decision_reason = "Gray Zone: Validated by Lexical+Gap"
            else:
                is_rejected = True
                decision_reason = f"Gray Zone: Failed checks (Lex={best_match['lexical_overlap']:.2f}, Gap={reranker_gap:.3f})"

        # ── Apply Guardrails (Overrides Score) ──
        # Check for absolute OOS terms (e.g. "copper", "web design")
        guardrail_triggered = False
        
        # 1. Check Pantone/Material OOS
        for word in OOS_PANTONE_WORDS:
            if word in normalized:
                is_rejected = True
                decision_reason = f"Guardrail: OOS Material '{word}'"
                guardrail_triggered = True
                break
        
        # 2. Check Context OOS
        if not guardrail_triggered:
            for phrase in OOS_CONTEXT_WORDS:
                if phrase in normalized:
                    is_rejected = True
                    decision_reason = f"Guardrail: OOS Topic '{phrase}'"
                    guardrail_triggered = True
                    break

        # Flatten results for App (Fixing KeyError)
        top_results_for_app = []
        for c in candidates[:5]:
            safe_c = c.copy()
            safe_c['entry'] = c['entry'] 
            # Ensure 'question' and 'answer' are accessible if app uses them directly
            safe_c['question'] = c['entry']['question']
            safe_c['answer'] = c['entry']['answer']
            top_results_for_app.append(safe_c)

        return {
            "question": user_question,
            "answer": best_match["entry"]["answer"] if not is_rejected else config.REJECTION_MESSAGE,
            "score": final_score,
            "threshold": config.THRESHOLD_HIGH if not is_rejected else config.THRESHOLD_LOW,
            "rejected": is_rejected,
            "decision_reason": decision_reason,
            "matched_question": best_match["entry"]["question"],
            "top_k_results": top_results_for_app,
            "guardrail": decision_reason if guardrail_triggered else None,
            "source_id": best_match["entry"].get("source_id")
        }


    def _reject(self, question, raw, score):
        return {
            "question": question,
            "answer": "I cannot answer this question based on the SOP.",
            "score": score,
            "threshold": 0.0,
            "rejected": True,
            "matched_question": None,
            "top_k_results": []
        }

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
