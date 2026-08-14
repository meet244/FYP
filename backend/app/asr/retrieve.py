"""Two-pass retrieval: first-pass hypothesis -> most relevant syllabus units.

Character n-gram TF-IDF (n in [3,5]), per Section III-C. Word tokens fail here for
two reasons specific to code-mixed text: transliteration is inconsistent, so word
matching misses orthographic variants; and the first-pass hypothesis is noisy by
construction, so the representation has to degrade gracefully. A sparse index also
costs nothing against a neural encoder — no download, ~1 ms/query.
"""
from __future__ import annotations

import threading

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class SyllabusIndex:
    def __init__(self, units, k: int = 3):
        self.units = list(units)
        self.k = k
        docs = [
            f"{u.title} {u.prose} {' '.join(u.keywords or [])}" for u in self.units
        ]
        self.vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
        )
        self.matrix = self.vec.fit_transform(docs)

    def scores(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(len(self.units))
        return (self.matrix @ self.vec.transform([text]).T).toarray().ravel()

    def query(self, text: str):
        """Return up to k units in ASCENDING relevance, so the best renders LAST."""
        if not self.units:
            return []
        if not text or not text.strip():
            return list(reversed(self.units[: self.k]))
        top = np.argsort(-self.scores(text))[: self.k]
        return [self.units[i] for i in reversed(top)]


_cache: dict[tuple[str, int, int], SyllabusIndex] = {}
_lock = threading.Lock()


def get_index(syllabus_id: str, units, k: int = 3) -> SyllabusIndex | None:
    """Cache one index per (syllabus, k, unit-count).

    Refitting per span is pure waste; the unit count is part of the key so an
    edited syllabus invalidates the cache without an explicit bust.
    """
    units = list(units)
    if not units:
        return None
    key = (syllabus_id, k, len(units))
    with _lock:
        idx = _cache.get(key)
        if idx is None:
            idx = SyllabusIndex(units, k=k)
            _cache[key] = idx
        return idx


def invalidate(syllabus_id: str) -> None:
    with _lock:
        for key in [k for k in _cache if k[0] == syllabus_id]:
            _cache.pop(key, None)
