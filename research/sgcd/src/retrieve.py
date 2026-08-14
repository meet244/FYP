"""Two-pass retrieval: first-pass hypothesis -> most relevant syllabus units.

Character n-gram TF-IDF: no model download, handles Devanagari+Latin natively,
~1 ms/query. With 6-12 units per course a neural encoder buys nothing measurable
and costs a download plus minutes of runtime — we say so in the paper.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class SyllabusIndex:
    def __init__(self, units, k=2):
        self.units, self.k = units, k
        docs = [f"{u['title']} {u['prose']} {' '.join(u['keywords'])}" for u in units]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.M = self.vec.fit_transform(docs)

    def scores(self, text):
        if not text or not text.strip():
            return np.zeros(len(self.units))
        return (self.M @ self.vec.transform([text]).T).toarray().ravel()

    def query(self, text):
        """Return up to k units in ASCENDING relevance, so the best is rendered LAST."""
        if not text or not text.strip():
            return self.units[: self.k]
        top = np.argsort(-self.scores(text))[: self.k]
        return [self.units[i] for i in reversed(top)]

    def top_ids(self, text):
        if not text or not text.strip():
            return [u["unit_id"] for u in self.units[: self.k]]
        return [self.units[i]["unit_id"] for i in np.argsort(-self.scores(text))[: self.k]]


_indices = {}


def get_index(course, k=2):
    """Cache one index per (course_id, k) — refitting per utterance is pure waste."""
    key = (course["course_id"], k)
    if key not in _indices:
        _indices[key] = SyllabusIndex(course["units"], k=k)
    return _indices[key]
