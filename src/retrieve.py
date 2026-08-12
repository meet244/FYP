"""Retrieval and context assembly (§6, §7.1).

The circular dependency (§6.1) — biasing needs the topic, the topic needs a transcript —
is resolved by an explicit two-pass design:

    pass 1   unbiased decode                        (B0, always cached)
    pass 2   retrieve against the pass-1 transcript, assemble a bounded context,
             re-decode the same audio with that context injected

Retrieval is semantic rather than exact-match, which is the point: a pass-1 transcript
containing "mat plot lib" still retrieves the correct topic passage.

Two granularities (§6.4):
  * `lecture`   — all pass-1 transcripts of one recording are concatenated into a
                  single query, so errors average out. Primary condition: a deployed
                  system knows which lecture it is processing.
  * `utterance` — each utterance is its own query. Harder and more general; reported
                  as an ablation.

Context assembly is bounded by a fixed word budget and a fixed truncation policy, so
all conditions are comparable, and comes in two styles (§7.1):
  * `prose`    — the retrieved passages rendered as sentences. Primary, because Whisper
                 imitates the style of the text it is conditioned on: natural prose
                 using the target terms outperforms a comma-separated glossary, which is
                 liable to be transcribed verbatim rather than acted upon.
  * `glossary` — a comma-separated term list. The cheap, informative ablation row.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np

from common import ROOT, load_config, read_json, read_jsonl

# Framing sentence in Hindi: the injected context must look like the register of the
# audio (a Hindi matrix with English technical terms), not like English-only prose,
# and must never read as an instruction to the model (§7.1).
PROSE_FRAME = "यह {topic} पर एक spoken tutorial है।"
GENERIC_CONTEXT = (
    "यह एक technical spoken tutorial है। इसमें instructor कंप्यूटर पर काम करते हुए "
    "step by step समझाते हैं और Hindi के साथ English technical शब्दों का प्रयोग करते हैं। "
    "वे screen पर menu, toolbar और dialog box दिखाते हैं और keyboard से command "
    "type करके result देखते हैं।")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


class SyllabusRetriever:
    """Embedding search over the frozen syllabus chunk index."""

    def __init__(self, cfg=None):
        cfg = cfg or load_config()
        self.cfg = cfg
        idx = ROOT / cfg["syllabus"]["index_dir"]
        self.docs = read_jsonl(idx / "docs.jsonl")
        self.emb = np.load(idx / "emb.npy")
        self.terms_by_topic: dict[str, list[str]] = read_json(idx / "terms_by_topic.json")
        self.topics = sorted({d["topic"] for d in self.docs})
        p = idx / "rec2topic.json"
        self.rec2topic = {k: v for k, v in read_json(p).items()
                          if not k.startswith("_")} if p.exists() else {}
        self._model = None
        self._model_name = cfg["syllabus"]["embed_model"]

    # --- embedding search ----------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def topk(self, query: str, k: int = 3) -> list[tuple[dict, float]]:
        if not query.strip():
            return []
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        idx = np.argsort(-scores)[:k]
        return [(self.docs[i], float(scores[i])) for i in idx]

    def top_topic(self, query: str, k: int = 3) -> str | None:
        """Score-weighted majority topic over the top-k chunks."""
        hits = self.topk(query, k)
        if not hits:
            return None
        agg: dict[str, float] = {}
        for d, s in hits:
            agg[d["topic"]] = agg.get(d["topic"], 0.0) + s
        return max(agg, key=agg.get)

    # --- context assembly (§6.2 step 3) -------------------------------------
    def _topic_terms(self, topics: list[str], limit: int) -> list[str]:
        out, seen = [], set()
        for t in topics:
            for term in self.terms_by_topic.get(t, []):
                if len(term) > 3 and term not in seen:
                    seen.add(term)
                    out.append(term)
                    if len(out) >= limit:
                        return out
        return out

    def assemble_context(self, hits: list[tuple[dict, float]], style: str = "prose",
                         max_words: int = 120) -> str | None:
        """Bounded grounding context: topic label plus the terminology it contains.

        Truncation policy, fixed for every condition: take whole sentences (prose) or
        whole terms (glossary) in retrieval-rank order until the word budget is spent.
        """
        if not hits:
            return None
        topics = list(dict.fromkeys(d["topic"] for d, _ in hits))
        label = ", ".join(t.replace("_", " ") for t in topics)

        if style == "glossary":
            terms = self._topic_terms(topics, max_words)
            budget = max_words - len(label.split()) - 3
            kept: list[str] = []
            used = 0
            for t in terms:
                w = len(t.split())
                if used + w > budget:
                    break
                kept.append(t)
                used += w
            return f"विषय: {label}. Technical terms: " + ", ".join(kept)

        # prose: sentences from the retrieved passages, headed by a Hindi frame
        head = PROSE_FRAME.format(topic=label)
        used = len(head.split())
        kept = [head]
        for d, _ in hits:
            body = re.sub(r"^#+\s*", "", d["text"].replace("\n", " "))
            for sent in _SENT_SPLIT.split(body):
                sent = re.sub(r"\s+", " ", sent).strip(" .;:")
                if len(sent.split()) < 4:
                    continue
                w = len(sent.split())
                if used + w > max_words:
                    return " ".join(kept)
                kept.append(sent + ".")
                used += w
        return " ".join(kept)

    def context_for_topic(self, topic: str | None, style: str = "prose",
                          max_words: int = 120) -> str | None:
        """Context built from one named topic's own passages (C2, C3 and M-by-topic)."""
        if not topic:
            return None
        hits = [(d, 1.0) for d in self.docs if d["topic"] == topic]
        return self.assemble_context(hits, style, max_words)

    # --- hint terms for M2 / candidate terms for M3 -------------------------
    def hotword_terms(self, topics: list[str], n_terms: int) -> list[str]:
        return self._topic_terms(topics, n_terms)

    def candidate_terms(self, topics: list[str], limit: int = 400) -> list[str]:
        return self._topic_terms(topics, limit)

    def oracle_topic(self, row: dict) -> str | None:
        """Gold lecture topic from recording metadata (C3 upper bound only)."""
        rec = row.get("rec") or row["utt_id"].split("_")[1]
        return self.rec2topic.get(rec)


class RetrievalPlan:
    """Per-utterance retrieval decisions for one condition, computed once.

    Holds, for every utterance: the retrieved topics, the assembled context string, the
    hint-term list, and the candidate-term list for output-level correction. Building
    this up front keeps the decode loop free of retrieval work and makes the plan a
    dumpable artefact (runs/<tier>/<cond>/retrieval.json) that the retrieval-accuracy
    measurement in §6.3 reads directly.
    """

    def __init__(self, retriever: SyllabusRetriever, rows: list[dict],
                 pass1: dict[str, dict], granularity: str = "lecture", k: int = 3):
        self.r = retriever
        self.granularity = granularity
        self.k = k
        self.by_utt: dict[str, dict] = {}

        if granularity == "lecture":
            queries: dict[str, list[str]] = {}
            for row in rows:
                rec = row.get("rec") or row["utt_id"].split("_")[1]
                queries.setdefault(rec, []).append(
                    (pass1.get(row["utt_id"], {}) or {}).get("hyp", ""))
            per_rec = {}
            for rec, parts in queries.items():
                q = " ".join(p for p in parts if p)
                hits = retriever.topk(q, k)
                per_rec[rec] = hits
            for row in rows:
                rec = row.get("rec") or row["utt_id"].split("_")[1]
                self.by_utt[row["utt_id"]] = self._entry(per_rec.get(rec, []), rec)
        else:
            for row in rows:
                q = (pass1.get(row["utt_id"], {}) or {}).get("hyp", "")
                self.by_utt[row["utt_id"]] = self._entry(
                    retriever.topk(q, k), row.get("rec"))

    @staticmethod
    def _entry(hits, rec) -> dict:
        return {"rec": rec, "hits": hits,
                "topics": list(dict.fromkeys(d["topic"] for d, _ in hits)),
                "scores": [s for _, s in hits]}

    def topics(self, utt_id: str) -> list[str]:
        return self.by_utt.get(utt_id, {}).get("topics", [])

    def top1(self, utt_id: str) -> str | None:
        t = self.topics(utt_id)
        return t[0] if t else None

    def context(self, utt_id: str, style: str, max_words: int) -> str | None:
        e = self.by_utt.get(utt_id)
        if not e:
            return None
        return self.r.assemble_context(e["hits"], style, max_words)

    def hotwords(self, utt_id: str, n_terms: int) -> str | None:
        terms = self.r.hotword_terms(self.topics(utt_id), n_terms)
        return ", ".join(terms) if terms else None

    def candidates(self, utt_id: str, limit: int = 400) -> list[str]:
        return self.r.candidate_terms(self.topics(utt_id), limit)

    def dump(self) -> dict:
        return {"granularity": self.granularity, "top_k": self.k,
                "per_utterance": {u: {"topics": e["topics"], "scores": e["scores"]}
                                  for u, e in self.by_utt.items()}}


def random_topic_plan(retriever: SyllabusRetriever, rows: list[dict],
                      seed: int = 1337) -> dict[str, str]:
    """C2: one randomly chosen syllabus topic per *lecture* (not per utterance).

    Per-lecture assignment mirrors the structure of the retrieved condition, so C2
    differs from M1/M2 only in whether the syllabus document is the correct one.
    """
    rng = random.Random(seed)
    recs = sorted({row.get("rec") or row["utt_id"].split("_")[1] for row in rows})
    return {rec: rng.choice(retriever.topics) for rec in recs}
