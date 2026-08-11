"""Syllabus retrieval: pass-1 transcript -> top-k chunks -> <=200-token decode prompt."""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

IDX = Path("syllabus/index")
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SyllabusRetriever:
    def __init__(self, model_name=MODEL):
        self.docs = [json.loads(l) for l in open(IDX / "docs.jsonl", encoding="utf-8")]
        self.emb = np.load(IDX / "emb.npy")
        self.model = SentenceTransformer(model_name)
        self.terms_by_topic = json.loads(
            (IDX / "terms_by_topic.json").read_text(encoding="utf-8"))
        p = IDX / "rec2topic.json"
        self.rec2topic = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    # --- retrieval -------------------------------------------------------------
    def topk(self, query: str, k: int = 3):
        if not query.strip():
            return []
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        idx = np.argsort(-scores)[:k]
        return [(self.docs[i], float(scores[i])) for i in idx]

    def topic_for(self, query: str, k: int = 3):
        """Majority topic over the top-k chunks (score-weighted)."""
        hits = self.topk(query, k)
        if not hits:
            return None
        agg = {}
        for d, s in hits:
            agg[d["topic"]] = agg.get(d["topic"], 0.0) + s
        return max(agg, key=agg.get)

    # --- prompt construction ---------------------------------------------------
    @staticmethod
    def _render(topics: str, terms: list[str], english_only: bool, max_words: int):
        body = " ".join(terms[:max_words])
        if english_only:
            return f"Topic: {topics}. Technical terms: {body}"
        # The Hindi framing word signals the code-switched register to the decoder
        # instead of pushing it toward English-only output.
        return f"विषय: {topics}. Technical terms: {body}"

    def prompt_for(self, query: str, k: int = 3, max_words: int = 120,
                   english_only: bool = False) -> str | None:
        hits = self.topk(query, k)
        if not hits:
            return None
        topics = ", ".join(sorted({d["topic"].replace("_", " ") for d, _ in hits}))
        # Terms in chunk order, filtered through the syllabus lexicon so the limited
        # prompt budget carries technical vocabulary and not document prose.
        terms, seen = [], set()
        for d, _ in hits:
            lex = set(self.terms_by_topic.get(d["topic"], []))
            for w in d["text"].split():
                wl = w.strip(".,:;()").lower()
                if wl in lex and len(wl) > 3 and wl not in seen:
                    seen.add(wl)
                    terms.append(wl)
        return self._render(topics, terms, english_only, max_words)

    def prompt_for_topic(self, topic: str, max_words: int = 120,
                         english_only: bool = False) -> str | None:
        if topic not in self.terms_by_topic:
            return None
        terms = [t for t in self.terms_by_topic[topic] if len(t) > 3]
        return self._render(topic.replace("_", " "), terms, english_only, max_words)

    def oracle_topic(self, row) -> str | None:
        """Gold lecture topic (S6 upper bound only)."""
        rec = row.get("rec") or row["utt_id"].split("_")[1]
        return self.rec2topic.get(rec)

    # --- candidate terms for post-hoc correction -------------------------------
    def candidate_terms(self, query: str, k: int = 3, min_len: int = 4) -> list[str]:
        hits = self.topk(query, k)
        out, seen = [], set()
        for d, _ in hits:
            for t in self.terms_by_topic.get(d["topic"], []):
                if len(t) >= min_len and t not in seen:
                    seen.add(t)
                    out.append(t)
        return out
