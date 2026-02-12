"""
Full SOP retrieval pipeline.
Handles: normalize → embed → FAISS search → threshold → answer/reject → optional rewrite.

This is the main entry point for answering user questions.
The LLM is NEVER used to answer questions — only to optionally rewrite retrieved answers.
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
from core.rewriter import rewrite_answer, check_ollama_status


class SOPPipeline:
    """
    Deterministic SOP retrieval pipeline.

    Flow:
        1. Normalize user input (lowercase, strip punctuation)
        2. Embed with same model used for indexing
        3. FAISS top-1 search
        4. Threshold check (≥ 0.75 similarity)
        5. Return verbatim SOP answer OR rejection message
        6. Optional: LLM rewrite of answer (no new content generation)
    """

    def __init__(self):
        self.embedder = Embedder()
        self.index = FAISSIndex()
        self.sop_data: list[dict] = []
        self.is_loaded = False

    def build(self):
        """
        Build the FAISS index from SOP data.
        Loads augmented data, embeds all questions, builds and saves the index.
        """
        print("=" * 60)
        print("  Building SOP Pipeline")
        print("=" * 60)

        # Load augmented data (with paraphrases)
        aug_file = os.path.join(config.DATA_DIR, "sop_data_augmented.json")
        if os.path.exists(aug_file):
            with open(aug_file, "r", encoding="utf-8") as f:
                self.sop_data = json.load(f)
            print(f"📂 Loaded augmented dataset: {len(self.sop_data)} entries")
        else:
            # Fall back to base data
            with open(config.SOP_DATA_FILE, "r", encoding="utf-8") as f:
                self.sop_data = json.load(f)
            # Add source_id if missing
            for entry in self.sop_data:
                if "source_id" not in entry:
                    entry["source_id"] = entry["id"]
            print(f"📂 Loaded base dataset: {len(self.sop_data)} entries")

        # Load embedder
        self.embedder.load()

        # Embed all questions (documents — no query prefix)
        questions = [entry["question"] for entry in self.sop_data]
        print(f"\n🔄 Embedding {len(questions)} questions...")
        start = time.time()
        vectors = self.embedder.embed_batch(questions, is_query=False)
        elapsed = time.time() - start
        print(f"✅ Embedded in {elapsed:.2f}s")

        # Build FAISS index
        self.index.build(vectors, self.sop_data)

        # Save to disk
        self.index.save()

        self.is_loaded = True
        print(f"\n✅ Pipeline built and saved!")
        return self

    def load(self):
        """Load a previously built index from disk."""
        print("🔄 Loading SOP Pipeline...")
        self.embedder.load()
        self.index.load()

        # Load SOP data
        aug_file = os.path.join(config.DATA_DIR, "sop_data_augmented.json")
        if os.path.exists(aug_file):
            with open(aug_file, "r", encoding="utf-8") as f:
                self.sop_data = json.load(f)
        else:
            with open(config.SOP_DATA_FILE, "r", encoding="utf-8") as f:
                self.sop_data = json.load(f)

        self.is_loaded = True
        print("✅ Pipeline loaded!")
        return self

    def query(self, user_question: str) -> dict:
        """
        Answer a user question using deterministic SOP retrieval.

        Args:
            user_question: Raw user input text.

        Returns:
            dict with keys:
                - question: Original user question
                - normalized: Normalized form
                - matched_question: Best matching SOP question (or None)
                - answer: SOP answer or rejection message
                - score: Similarity score (0-1)
                - source_id: ID of the matched SOP entry (or None)
                - rejected: True if below threshold
                - threshold: The threshold used
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call build() or load() first.")

        # Step 1: Normalize
        normalized = normalize(user_question)

        # Step 2: Embed (as query — with prefix for bge)
        query_vector = self.embedder.embed(normalized)

        # Step 3: FAISS search
        results = self.index.search(query_vector, top_k=config.TOP_K)

        if not results:
            return {
                "question": user_question,
                "normalized": normalized,
                "matched_question": None,
                "answer": config.REJECTION_MESSAGE,
                "score": 0.0,
                "source_id": None,
                "rejected": True,
                "threshold": config.SIMILARITY_THRESHOLD,
            }

        top_result = results[0]
        score = top_result["score"]
        entry = top_result["entry"]

        # Step 4: Threshold check
        if score < config.SIMILARITY_THRESHOLD:
            return {
                "question": user_question,
                "normalized": normalized,
                "matched_question": entry["question"],
                "answer": config.REJECTION_MESSAGE,
                "score": score,
                "source_id": entry.get("source_id", entry.get("id")),
                "rejected": True,
                "threshold": config.SIMILARITY_THRESHOLD,
            }

        # Step 5: Return verbatim answer
        answer = entry["answer"]

        # Step 6: Optional LLM rewrite
        if config.USE_LLM_REWRITE:
            answer = rewrite_answer(answer)

        return {
            "question": user_question,
            "normalized": normalized,
            "matched_question": entry["question"],
            "answer": answer,
            "score": score,
            "source_id": entry.get("source_id", entry.get("id")),
            "rejected": False,
            "threshold": config.SIMILARITY_THRESHOLD,
        }

    def query_detailed(self, user_question: str, top_k: int = 5) -> dict:
        """
        Like query(), but returns top-K results for debugging/analysis.
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call build() or load() first.")

        normalized = normalize(user_question)
        query_vector = self.embedder.embed(normalized)
        results = self.index.search(query_vector, top_k=top_k)

        top_results = []
        for r in results:
            top_results.append({
                "question": r["entry"]["question"],
                "answer": r["entry"]["answer"],
                "score": r["score"],
                "source_id": r["entry"].get("source_id", r["entry"].get("id")),
            })

        # Primary result
        primary = self.query(user_question)
        primary["top_k_results"] = top_results

        return primary
