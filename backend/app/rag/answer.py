"""Answer a question against one subject's lecture corpus, with citations.

Every claim is grounded in retrieved spans or notes, and each source carries the
originating lecture and timestamp so a student can verify any statement against
the recording (Section III-E).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.llm.client import LLMRateLimited, LLMUnavailable, complete
from app.models import Lecture, Subject
from app.notes.coverage import subject_coverage
from app.rag import store
from app.rag.router import Route, route as route_query

log = logging.getLogger(__name__)

# Output shape per query type. The retrieval is similar; what differs is what the
# student is asking to be done with it.
_STYLE = {
    "lookup": "Answer directly in one to three sentences. No preamble, no restating the question.",
    "explain": "Explain the concept using the uploaded sources and recordings. "
               "Use short paragraphs and plain language; add a worked example only "
               "if the sources contained one.",
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
You are a study assistant over one subject. You answer from the supplied sources \
and nothing else. The corpus is multimodal and stored locally: syllabus units, \
lecture voice transcripts, generated notes, PDF slides, images (OCR + captions), \
and text documents.

Sources are labelled by kind:
- syllabus: the planned curriculum. Not a recording of a class.
- transcript / notes: what was spoken in a lecture (voice), and notes from it.
- pdf / doc: uploaded readings and slides.
- image: a photographed slide, whiteboard, or diagram (text was OCR'd).

Answer in English unless the student writes to you in Hindi, in which case match \
their language while keeping technical terms in English.

Grounding rules, in order of importance:
- Use only what the sources say. Do not supply outside knowledge.
- Treat syllabus units as curriculum, not as spoken lecture content. If a topic is \
on the syllabus but no lecture, PDF, or image supports it, say it is on the \
syllabus and has not been taught in a recording yet.
- If the sources do not answer the question, say so plainly and name what they do \
cover that is adjacent. Never fill a gap with a plausible-sounding answer.
- Cite with the bracketed source numbers, [1], [2], placed at the end of the \
sentence they support. Cite the source you actually used.
- Where a transcript is garbled but the intent is recoverable, use the correct \
technical term and do not comment on the transcription quality.

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
    ordered = sorted(subject.lectures, key=lambda lec: lec.created_at)
    if needle.startswith("last "):
        return ordered[-1] if ordered else None
    number = re.search(r"(?:lecture|recording)\s+#?(\d+)", needle)
    if number:
        index = int(number.group(1)) - 1
        return ordered[index] if 0 <= index < len(ordered) else None
    for lec in subject.lectures:
        if needle in lec.title.lower():
            return lec
    return None


def _gather(db: Session, subject: Subject, rt: Route) -> list[dict[str, Any]]:
    lecture = _resolve_lecture(db, subject, rt.lecture_hint)
    if rt.lecture_hint and lecture is None:
        # An explicit recording scope must never silently become subject-wide.
        return []
    lecture_id = lecture.id if lecture else None

    primary = store.NOTES if rt.wants_notes else store.SPANS
    secondary = store.SPANS if rt.wants_notes else store.NOTES

    hits = store.search(primary, rt.search_query, subject.id, lecture_id=lecture_id)
    # Always mix in the other collection: notes are cleaner but lossy, transcripts
    # are noisier but complete, and most questions want a bit of each.
    hits += store.search(
        secondary, rt.search_query, subject.id, lecture_id=lecture_id, top_k=6
    )
    # A question explicitly about a recording must use that recording's evidence.
    if lecture_id is None:
        hits += store.search(store.UNITS, rt.search_query, subject.id, top_k=6)
        hits += store.search(store.MATERIALS, rt.search_query, subject.id, top_k=8)

    hits.sort(key=lambda h: h.get("score") or 0.0, reverse=True)
    return hits[:16]


def _render_sources(hits: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    blocks, citations = [], []
    for n, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        kind = meta.get("kind")
        if kind == "unit":
            title = meta.get("topic") or "Syllabus unit"
            header = f"[{n}] Syllabus — {title}"
            blocks.append(f"{header}\n{hit['text']}")
            citations.append(
                {
                    "n": n,
                    "kind": "unit",
                    "lecture_id": None,
                    "lecture_title": None,
                    "start_s": None,
                    "end_s": None,
                    "timestamp": None,
                    "note_id": None,
                    "unit_id": meta.get("unit_id"),
                    "unit_title": title,
                    "excerpt": hit["text"],
                }
            )
            continue

        if kind in ("pdf", "image", "doc"):
            page = re.search(r"\[page (\d+)\]", hit["text"])
            title = meta.get("topic") or "Material"
            header = f"[{n}] {kind.upper()} — {title}"
            blocks.append(f"{header}\n{hit['text']}")
            citations.append(
                {
                    "n": n,
                    "kind": kind,
                    "lecture_id": None,
                    "lecture_title": None,
                    "start_s": None,
                    "end_s": None,
                    "timestamp": None,
                    "note_id": None,
                    "material_id": meta.get("material_id"),
                    "material_title": title,
                    "excerpt": hit["text"],
                    "page": int(page.group(1)) if page else None,
                }
            )
            continue

        start = float(meta.get("start_s") or 0.0)
        stamp = f"{int(start) // 60:02d}:{int(start) % 60:02d}"
        label = "notes" if kind == "note" else "transcript"
        header = f"[{n}] {meta.get('lecture_title', 'Untitled')} — {label} @ {stamp}"
        blocks.append(f"{header}\n{hit['text']}")
        citations.append(
            {
                "n": n,
                "kind": kind,
                "lecture_id": meta.get("lecture_id"),
                "lecture_title": meta.get("lecture_title"),
                "start_s": start,
                "end_s": float(meta.get("end_s") or 0.0),
                "timestamp": stamp,
                "note_id": meta.get("note_id"),
                "excerpt": hit["text"],
            }
        )
    return "\n\n---\n\n".join(blocks), citations


def _extractive(hits: list[dict[str, Any]], citations: list[dict[str, Any]]) -> str:
    """Quoted local sources when Gemini is down. Never invents."""
    parts = [
        "Gemini is unavailable right now, so this is quoted from your local index "
        "rather than rewritten. Try the question again in a minute for a composed answer.\n"
    ]
    for cite, hit in zip(citations[:4], hits[:4]):
        snippet = " ".join((hit.get("text") or "").split())[:420]
        if not snippet:
            continue
        parts.append(f"**[{cite['n']}]** {snippet}")
    return "\n\n".join(parts)


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
            f"I'm your assistant for {subject.name}. Ask about the syllabus, lectures, "
            "PDFs, images, or notes — I'll answer from this subject's local corpus and cite "
            "where each claim came from.",
            "smalltalk",
        )

    hits = _gather(db, subject, rt)
    if not hits:
        return Answer(
            "I couldn't find anything about that in this subject's syllabus, lectures, "
            "or uploaded materials. Either it is outside the course, or a file is still processing.",
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

    try:
        text = complete(_SYSTEM, payload, max_tokens=8_000)
    except (LLMUnavailable, LLMRateLimited) as exc:
        log.warning("LLM unavailable, using extractive fallback: %s", exc)
        text = _extractive(hits, citations)
    used = [c for c in citations if f"[{c['n']}]" in text]
    # Remove unsupported citation markers rather than invent a source mapping.
    valid = {c["n"] for c in citations}
    text = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if int(m[1]) in valid else "", text)
    return Answer(text=text, query_type=rt.query_type, citations=used)
