"""Answer a question against one subject's lecture corpus, with citations.

Every claim is grounded in retrieved spans or notes, and each source carries the
originating lecture and timestamp so a student can verify any statement against
the recording (Section III-E).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.llm.client import complete
from app.models import Lecture, Subject
from app.notes.coverage import subject_coverage
from app.rag import store
from app.rag.router import Route, route as route_query

log = logging.getLogger(__name__)

# Output shape per query type. The retrieval is similar; what differs is what the
# student is asking to be done with it.
_STYLE = {
    "lookup": "Answer directly in one to three sentences. No preamble, no restating the question.",
    "explain": "Explain the concept as taught in these lectures, building from the "
               "lecturer's own framing. Use short paragraphs; add a worked example only "
               "if the lectures contained one.",
    "summary": "Give a structured summary with Markdown sub-headings, ordered as the "
               "material was taught.",
    "compare": "Compare the items point by point. A small Markdown table is appropriate "
               "when the dimensions of comparison are clean.",
    "quiz": "Produce practice questions of mixed difficulty. Give the questions first, "
            "then an 'Answers' section. Only ask about material the sources cover.",
    "outline": "List the topics as a nested Markdown outline in teaching order. No prose.",
    "smalltalk": "Reply briefly and plainly.",
}

_SYSTEM = """\
You are a study assistant over one student's own recorded lectures for a single \
subject. You answer from the supplied sources and nothing else.

The sources are excerpts from machine transcripts of Hindi-English code-switched \
lectures, plus notes generated from them. Transcripts contain recognition errors \
and mix Devanagari and Latin script — read through both. Answer in English unless \
the student writes to you in Hindi, in which case match their language while \
keeping technical terms in English.

Grounding rules, in order of importance:
- Use only what the sources say. Do not supply outside knowledge, even when you are \
confident it is correct and the lecture merely omitted it.
- If the sources do not answer the question, say so plainly and name what they do \
cover that is adjacent. Never fill a gap with a plausible-sounding answer.
- Cite with the bracketed source numbers, [1], [2], placed at the end of the \
sentence they support. Cite the source you actually used.
- Where a transcript is garbled but the intent is recoverable, use the correct \
technical term and do not comment on the transcription quality. Where it is not \
recoverable, treat it as absent.

Write for a student revising. Lead with the answer; supporting detail after.\
"""


@dataclass
class Answer:
    text: str
    query_type: str
    citations: list[dict[str, Any]] = field(default_factory=list)


def _resolve_lecture(db: Session, subject: Subject, hint: str) -> Lecture | None:
    if not hint:
        return None
    needle = hint.lower().strip()
    for lec in subject.lectures:
        if needle in lec.title.lower():
            return lec
    return None


def _gather(db: Session, subject: Subject, rt: Route) -> list[dict[str, Any]]:
    lecture = _resolve_lecture(db, subject, rt.lecture_hint)
    lecture_id = lecture.id if lecture else None

    primary = store.NOTES if rt.wants_notes else store.SPANS
    secondary = store.SPANS if rt.wants_notes else store.NOTES

    hits = store.search(primary, rt.search_query, subject.id, lecture_id=lecture_id)
    # Always mix in the other collection: notes are cleaner but lossy, transcripts
    # are noisier but complete, and most questions want a bit of each.
    hits += store.search(
        secondary, rt.search_query, subject.id, lecture_id=lecture_id, top_k=6
    )

    hits.sort(key=lambda h: h.get("score") or 0.0, reverse=True)
    return hits[:14]


def _render_sources(hits: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    blocks, citations = [], []
    for n, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        start = float(meta.get("start_s") or 0.0)
        stamp = f"{int(start) // 60:02d}:{int(start) % 60:02d}"
        label = "notes" if meta.get("kind") == "note" else "transcript"
        header = f"[{n}] {meta.get('lecture_title', 'Untitled')} — {label} @ {stamp}"
        blocks.append(f"{header}\n{hit['text']}")
        citations.append(
            {
                "n": n,
                "kind": meta.get("kind"),
                "lecture_id": meta.get("lecture_id"),
                "lecture_title": meta.get("lecture_title"),
                "start_s": start,
                "end_s": float(meta.get("end_s") or 0.0),
                "timestamp": stamp,
                "note_id": meta.get("note_id"),
            }
        )
    return "\n\n---\n\n".join(blocks), citations


def answer_question(
    db: Session,
    subject: Subject,
    question: str,
    history: list[dict] | None = None,
) -> Answer:
    rt = route_query(question, history)

    # Coverage is a database fact, not a retrieval problem — answering it from
    # embeddings would be guessing at something we can compute exactly.
    if rt.query_type == "coverage":
        cov = subject_coverage(db, subject)
        if not cov["total_units"]:
            return Answer(
                "No syllabus has been uploaded for this subject yet, so I can't tell "
                "you what's been covered. Upload the syllabus PDF and I'll track it "
                "against your lectures.",
                "coverage",
            )
        done = ", ".join(c["title"] for c in cov["covered"]) or "nothing yet"
        left = ", ".join(c["title"] for c in cov["outstanding"]) or "nothing — the syllabus is complete"
        text = (
            f"**{cov['covered_units']} of {cov['total_units']} syllabus units covered.**\n\n"
            f"**Covered:** {done}\n\n**Outstanding:** {left}"
        )
        return Answer(text, "coverage", [])

    if rt.query_type == "smalltalk":
        return Answer(
            f"I'm your assistant for {subject.name}. Ask me about anything from your "
            "recorded lectures — I'll answer from the transcripts and notes and point "
            "you at the timestamp.",
            "smalltalk",
        )

    hits = _gather(db, subject, rt)
    if not hits:
        return Answer(
            "I couldn't find anything about that in your lectures for this subject. "
            "Either it hasn't been covered yet, or the recording is still processing.",
            rt.query_type,
        )

    sources, citations = _render_sources(hits)
    convo = ""
    for msg in (history or [])[-6:]:
        convo += f"{msg['role']}: {msg['content']}\n"

    style = _STYLE.get(rt.query_type, _STYLE["lookup"])
    payload = (
        f"Subject: {subject.name}\n\n"
        f"Conversation so far:\n{convo or '(none)'}\n\n"
        f"Sources:\n\n{sources}\n\n---\n\n"
        f"Question: {question}\n\n"
        f"Response style for this question type ({rt.query_type}): {style}"
    )

    text = complete(_SYSTEM, payload, max_tokens=8_000)
    used = [c for c in citations if f"[{c['n']}]" in text]
    return Answer(text=text, query_type=rt.query_type, citations=used or citations[:3])
