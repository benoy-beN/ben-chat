"""
Full SOP retrieval pipeline (V6.3 — SOTA Accuracy).
Handles: BGE-M3 → FAISS + BM25 → Learned Fusion → Rerank → Calibrate → Dual-Threshold.

Decision Logic (V6.3):
  1. Hybrid Retrieval: FAISS (dense) ∪ BM25 (lexical) → union top-30
  2. Cross-Encoder Reranking
  3. Score Fusion (learned weights: dense + bm25 + rerank)
  4. Platt Calibration → calibrated probability
  5. Dual-Threshold Gate with kill switch:
     if score >= THRESH_HIGH → accept (subject to guardrails)
     elif score < THRESH_LOW → reject
     else (gray zone) → lexical+gap checks + SOP_IDS kill switch
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
# Phase 7: Load OOS patterns from file if available
_OOS_PATTERNS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "oos_patterns.txt")
if os.path.exists(_OOS_PATTERNS_FILE):
    _oos_all = []
    with open(_OOS_PATTERNS_FILE, "r") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#"):
                _oos_all.append(_line.lower())
    OOS_CONTEXT_WORDS = _oos_all
    OOS_PANTONE_WORDS = []  # All merged into OOS_CONTEXT_WORDS
else:
    OOS_CONTEXT_WORDS = [
        "web design", "website", "web page", "digital ad",
        "social media", "video", "audio", "animation",
        "3d", "html", "css", "app design",
    ]
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
    """V6.3 SOTA SOP retrieval pipeline with hybrid retrieval + learned fusion."""

    def __init__(self):
        self.embedder    = BGEM3Embedder(config.EMBEDDING_MODEL)
        self.faiss_index = FAISSIndex()
        self.bm25_index  = BM25Index()        # Phase 4: Lexical backstop
        self.fusion      = ScoreFusion()       # Phase 5: Learned fusion
        self.reranker    = Reranker()
        self.calibrator  = ConfidenceCalibrator()
        
        self.sop_data: list[dict] = []
        self.sop_ids: set = set()              # Phase 6: All valid SOP IDs
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
        print("🔄 Loading V6.3 Pipeline (BGE-M3 + BM25 + Fusion)...")

        self.faiss_index.load(index_path=config.FAISS_INDEX_FILE)

        # Phase 4: BM25 lexical backstop
        try:
            self.bm25_index.load()
        except FileNotFoundError:
            print("⚠️ BM25 index not found — lexical backstop disabled.")

        # Phase 5: Learned fusion
        self.fusion.load()

        # Calibrator
        self.calibrator.load()
        
        self.sop_data = self.faiss_index.id_map
        
        # Phase 6: Build SOP_IDS set for kill switch
        self.sop_ids = {entry.get("source_id", entry.get("id")) for entry in self.sop_data}
        
        self.is_loaded = True
        print(f"✅ V6.3 Pipeline loaded! ({len(self.sop_ids)} SOP IDs)")
        return self

    # ── Query ──────────────────────────────────────────
    # ── Query ──────────────────────────────────────────
    def query(self, user_question: str) -> dict:
        """
        V6.3 pipeline: BGE-M3 + BM25 → Fusion → Rerank → Calibrate → Threshold.
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded.")

        normalized = user_question.lower().strip()
        query_emb  = self.embedder.embed(normalized)
        query_dense = query_emb["dense"]
        
        norm = np.linalg.norm(query_dense)
        if norm > 0:
            query_dense = query_dense / norm

        # ── Stage 1: Hybrid Retrieval (Dense ∪ BM25) ──
        top_k = 30  # Union pool size
        faiss_results = self.faiss_index.search(query_dense, top_k=top_k)

        if not faiss_results:
             return self._reject(user_question, normalized, 0.0)

        # BM25 lexical search (Phase 4)
        bm25_results = []
        if self.bm25_index.bm25 is not None:
            bm25_results = self.bm25_index.search(normalized, top_k=top_k)

        # Merge: union by entry ID (deduplicate)
        seen_ids = set()
        candidates = []
        for c in faiss_results:
            eid = c['entry'].get('source_id', c['entry'].get('id'))
            if eid not in seen_ids:
                seen_ids.add(eid)
                c['dense_score'] = c['score']
                c['bm25_score'] = 0.0
                candidates.append(c)
        
        for c in bm25_results:
            eid = c['entry'].get('source_id', c['entry'].get('id'))
            if eid not in seen_ids:
                seen_ids.add(eid)
                c['dense_score'] = 0.0
                c['bm25_score'] = c['score']
                candidates.append(c)
            else:
                # Entry already in candidates from FAISS — add BM25 score
                for existing in candidates:
                    if existing['entry'].get('source_id', existing['entry'].get('id')) == eid:
                        existing['bm25_score'] = c['score']
                        break

        # ── Stage 2: Cross-Encoder Reranking ──
        texts = [c['entry']['question'] for c in candidates]
        
        try:
            rerank_scores = self.reranker.compute_scores(normalized, texts)
        except Exception as e:
            print(f"⚠️ Reranker failed: {e}. Using dense scores.")
            rerank_scores = [c.get('dense_score', c['score']) for c in candidates]

        # Attach scores, compute lexical overlap, fuse, calibrate
        q_tokens = _tokenize(normalized)
        for i, c in enumerate(candidates):
            c["rerank_score"] = rerank_scores[i]
            
            # Lexical overlap (content-word Jaccard)
            d_tokens = _tokenize(c['entry']['question'])
            c["lexical_overlap"] = _lexical_overlap(q_tokens, d_tokens)

            # Phase 5: Fusion (dense + bm25 + rerank)
            dense_s = c.get('dense_score', 0.0)
            bm25_s  = c.get('bm25_score', 0.0)
            rerank_s = c['rerank_score']
            c["fused_score"] = self.fusion.fuse(dense_s, bm25_s, rerank_s)

            # Calibrate the fused score
            c["final_score"] = self.calibrator.calibrate(c["rerank_score"])
            
        # Sort by final_score descending
        candidates.sort(key=lambda x: x["final_score"], reverse=True)

        best_match = candidates[0]
        final_score = best_match["final_score"]
        best_id = best_match['entry'].get('source_id', best_match['entry'].get('id'))

        # Reranker gap (top1 - top2)
        reranker_gap = 0.0
        if len(candidates) > 1:
            reranker_gap = candidates[0]["final_score"] - candidates[1]["final_score"]

        # ── Stage 3: Dual-Threshold Decision (V6.3) ──
        is_rejected = False
        decision_reason = "Score above threshold"
        
        # [1] High Confidence Acceptance
        if final_score >= config.THRESHOLD_HIGH:
            is_rejected = False
            decision_reason = f"High confidence (>= {config.THRESHOLD_HIGH:.3f})"
            
        # [2] Low Confidence Rejection
        elif final_score < config.THRESHOLD_LOW:
            is_rejected = True
            decision_reason = f"Low confidence (< {config.THRESHOLD_LOW:.3f})"

        # [3] Gray Zone
        else:
            lex_check = best_match["lexical_overlap"] >= config.LEX_MIN
            gap_check = reranker_gap >= config.GAP_MIN
            
            if lex_check and gap_check:
                is_rejected = False
                decision_reason = "Gray Zone: Validated by Lexical+Gap"
            else:
                is_rejected = True
                decision_reason = f"Gray Zone: Failed (Lex={best_match['lexical_overlap']:.2f}, Gap={reranker_gap:.3f})"

        # ── Apply Guardrails (Overrides accept) ──
        guardrail_triggered = False
        
        for word in OOS_PANTONE_WORDS:
            if word in normalized:
                is_rejected = True
                decision_reason = f"Guardrail: OOS Material '{word}'"
                guardrail_triggered = True
                break
        
        if not guardrail_triggered:
            for phrase in OOS_CONTEXT_WORDS:
                if phrase in normalized:
                    is_rejected = True
                    decision_reason = f"Guardrail: OOS Topic '{phrase}'"
                    guardrail_triggered = True
                    break

        # Check if best answer itself is OOS
        if not guardrail_triggered and not is_rejected:
            best_answer_lower = best_match['entry']['answer'].lower()
            for phrase in OOS_ANSWER_PHRASES:
                if phrase in best_answer_lower:
                    is_rejected = True
                    decision_reason = f"Guardrail: Answer is OOS ('{phrase}')"
                    guardrail_triggered = True
                    break

        # ── Phase 6: False-Reject Kill Switch (FINAL — runs after guardrails) ──
        # If rejected but best match is a known SOP entry with decent confidence,
        # override the rejection. This rescues legitimate SOP queries that are
        # falsely caught by OOS patterns (e.g. "bronze" in "PMS bronze standard").
        if is_rejected and best_id in self.sop_ids and final_score >= config.THRESHOLD_LOW:
            is_rejected = False
            guardrail_triggered = False
            decision_reason = f"Kill Switch: SOP ID {best_id} rescued (score={final_score:.3f})"

        # Flatten results for App
        top_results_for_app = []
        for c in candidates[:5]:
            safe_c = c.copy()
            safe_c['entry'] = c['entry']
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
            "source_id": best_id
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
