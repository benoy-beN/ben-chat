"""
Semantic Cache (V5 — Optional).
LRU-based cache for normalized queries to avoid redundant retrieval.
Provides speed + stability for repeated/similar queries.
"""
import time
from collections import OrderedDict
from core.normalizer import normalize


class SemanticCache:
    """
    Simple LRU cache keyed by normalized query text.
    TTL-based expiration to avoid stale results.
    """

    def __init__(self, max_size: int = 256, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, query: str) -> dict | None:
        """
        Look up a cached result.
        
        Args:
            query: Raw user question (will be normalized)
            
        Returns:
            Cached result dict or None if miss
        """
        key = normalize(query)
        
        if key in self.cache:
            entry = self.cache[key]
            # Check TTL
            if time.time() - entry["timestamp"] < self.ttl:
                self.hits += 1
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return entry["result"]
            else:
                # Expired
                del self.cache[key]
        
        self.misses += 1
        return None

    def put(self, query: str, result: dict):
        """
        Cache a result.
        
        Args:
            query: Raw user question
            result: Pipeline result dict
        """
        key = normalize(query)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = {
            "result": result,
            "timestamp": time.time(),
        }

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        """Cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1f}%",
        }
