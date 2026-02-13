"""
BM25 Lexical Retrieval Index (V5).
Exact token matching for keyword-level recall that embeddings miss.
Complements BGE-M3 dense/sparse retrieval.
"""
import json
import os
import pickle
import re

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BM25Index:
    """
    BM25 lexical index over SOP questions.
    Handles tokenization, scoring, save/load.
    """

    def __init__(self):
        self.bm25 = None
        self.entries: list[dict] = []
        self.tokenized_corpus: list[list[str]] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Simple whitespace tokenizer with lowercasing."""
        text = text.lower().strip()
        # Remove punctuation except hyphens within words
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.split()

    def build(self, entries: list[dict]):
        """
        Build BM25 index from SOP entries.
        
        Args:
            entries: List of SOP entries with 'question', 'answer', 'id', etc.
        """
        from rank_bm25 import BM25Okapi

        self.entries = entries
        
        # Tokenize questions for BM25
        self.tokenized_corpus = [
            self.tokenize(entry["question"]) for entry in entries
        ]
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"✅ Built BM25 index: {len(entries)} documents")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Search BM25 index for best lexical matches.
        
        Args:
            query: User question text
            top_k: Number of results to return
            
        Returns:
            List of {"score": float, "entry": dict} sorted by score descending.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() or load() first.")

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-K indices
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append({
                "score": float(scores[idx]),
                "entry": self.entries[idx],
            })

        return results

    def save(self, path: str = None):
        """Save BM25 index to disk."""
        path = path or config.BM25_INDEX_FILE
        data = {
            "entries": self.entries,
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 BM25 index saved → {path}")

    def load(self, path: str = None):
        """Load BM25 index from disk."""
        from rank_bm25 import BM25Okapi

        path = path or config.BM25_INDEX_FILE
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 index not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.entries = data["entries"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"✅ Loaded BM25 index: {len(self.entries)} documents")
        return self
