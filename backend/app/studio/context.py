"""Collect the source material a Studio item is generated from.

Chat retrieves the top few passages for one question. A quiz or mind map over a
module needs the opposite: broad, ordered coverage of everything taught under
that scope. So scope is resolved against the database (notes are already
attributed to syllabus units, spans carry the units retrieval selected) and the
vector index is used only to pull in readings and slides, which have no unit
attribution of their own.

Everything returned is plain data; the caller closes the session before the
multi-second LLM call.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.config import settings
from app.ingest.materials import chunk_text
from app.models import Lecture, Material, Subject, Syllabus, SyllabusUnit
from app.rag import store
from app.rag.answer import _render_sources

SPAN_WINDOW = 4  # ~100 s of speech per transcript source block


class ScopeNotFound(LookupError):
    pass


@dataclass
class StudioContext:
    label: str
    sources: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    # True when nothing but syllabus units was found: the item can only restate
    # the planned curriculum, and the prompt says so.
    syllabus_only: bool = False


def _load_subject(db: Session, subject_id: str) -> Subject:
    subject = db.scalar(
        select(Subject)
        .where(Subject.id == subject_id)
        .options(
            selectinload(Subject.syllabus).selectinload(Syllabus.units),
            selectinload(Subject.lectures).selectinload(Lecture.notes),
            selectinload(Subject.lectures).selectinload(Lecture.spans),
            selectinload(Subject.materials),
        )
    )
    if subject is None:
        raise ScopeNotFound("subject not found")
    return subject


def _unit_hit(unit: SyllabusUnit, subject_id: str) -> dict[str, Any]:
    parts = [unit.title, unit.prose or ""]
    if unit.keywords:
        parts.append("Key terms: " + ", ".join(unit.keywords))
    return {
        "text": "\n\n".join(p for p in parts if p.strip()),
        "metadata": {"kind": "unit", "subject_id": subject_id, "unit_id": unit.id, "topic": unit.title},
    }


def _note_hits(lecture: Lecture, unit_id: str | None = None) -> list[dict[str, Any]]:
    out = []
    for note in sorted(lecture.notes, key=lambda n: n.order_index):
        if unit_id and note.unit_id != unit_id:
            continue
        out.append(
            {
                "text": f"{note.topic}\n\n{note.markdown}",
                "metadata": {
                    "kind": "note",
                    "lecture_id": lecture.id,
                    "lecture_title": lecture.title,
                    "note_id": note.id,
                    "start_s": float(note.start_s or 0.0),
                    "end_s": float(note.end_s or 0.0),
                },
            }
        )
    return out


def _span_hits(lecture: Lecture, unit_id: str | None = None) -> list[dict[str, Any]]:
    """Consecutive spans grouped into windows; optionally only spans decoded under `unit_id`."""
    spans = sorted(lecture.spans, key=lambda s: s.index)
    if unit_id:
        spans = [s for s in spans if (s.retrieved_unit_ids or [None])[0] == unit_id]
    out: list[dict[str, Any]] = []
    run: list = []

    def flush() -> None:
        if not run:
            return
        text = " ".join(s.text for s in run).strip()
        if text:
            out.append(
                {
                    "text": text,
                    "metadata": {
                        "kind": "span",
                        "lecture_id": lecture.id,
                        "lecture_title": lecture.title,
                        "start_s": float(run[0].start_s),
                        "end_s": float(run[-1].end_s),
                    },
                }
            )
        run.clear()

    for span in spans:
        if run and (span.index != run[-1].index + 1 or len(run) >= SPAN_WINDOW):
            flush()
        run.append(span)
    flush()
    return out


def _material_hits(material: Material) -> list[dict[str, Any]]:
    if material.status != "ready":
        return []
    return [
        {
            "text": chunk,
            "metadata": {"kind": material.kind, "material_id": material.id, "topic": material.title},
        }
        for chunk in chunk_text(material.text or "")
    ]


def _searched_materials(subject_id: str, query: str, top_k: int = 10) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    return store.search(store.MATERIALS, query, subject_id, top_k=top_k)


def _searched_lectures(subject_id: str, query: str) -> list[dict[str, Any]]:
    hits = store.search(store.NOTES, query, subject_id, top_k=10)
    hits += store.search(store.SPANS, query, subject_id, top_k=10)
    hits.sort(key=lambda h: h.get("score") or 0.0, reverse=True)
    return hits


def _dedupe(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for hit in hits:
        key = re.sub(r"\s+", " ", hit["text"]).strip()[:240]
        if key and key not in seen:
            seen.add(key)
            out.append(hit)
    return out


def _within_budget(hits: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    out, used = [], 0
    for hit in hits:
        size = len(hit["text"]) + 80
        if used + size > budget and out:
            continue  # a smaller later block may still fit
        out.append(hit)
        used += size
    return out


def _resolve_unit(subject: Subject, unit_id: str) -> SyllabusUnit:
    units = subject.syllabus.units if subject.syllabus else []
    unit = next((u for u in units if u.id == unit_id), None)
    if unit is None:
        raise ScopeNotFound("syllabus unit not found in this subject")
    return unit


def _resolve_lecture(subject: Subject, lecture_id: str) -> Lecture:
    lecture = next((lec for lec in subject.lectures if lec.id == lecture_id), None)
    if lecture is None:
        raise ScopeNotFound("lecture not found in this subject")
    return lecture


def scope_label(db: Session, subject_id: str, scope_type: str, scope_id: str | None) -> str:
    """Validate a scope and name it, without gathering anything."""
    subject = db.get(Subject, subject_id)
    if subject is None:
        raise ScopeNotFound("subject not found")
    if scope_type == "unit":
        unit = db.get(SyllabusUnit, scope_id) if scope_id else None
        if unit is None or unit.syllabus.subject_id != subject_id:
            raise ScopeNotFound("syllabus unit not found in this subject")
        return f"Unit {unit.order_index}: {unit.title}"
    if scope_type == "lecture":
        lecture = db.get(Lecture, scope_id) if scope_id else None
        if lecture is None or lecture.subject_id != subject_id:
            raise ScopeNotFound("lecture not found in this subject")
        return lecture.title
    return subject.name


def gather(
    db: Session,
    subject_id: str,
    scope_type: str,
    scope_id: str | None,
    focus: str | None = None,
    *,
    budget: int | None = None,
) -> StudioContext:
    budget = budget or settings.studio_context_chars
    subject = _load_subject(db, subject_id)
    focus = (focus or "").strip()
    lectures = sorted(subject.lectures, key=lambda lec: lec.created_at)

    units: list[dict[str, Any]] = []
    taught: list[dict[str, Any]] = []
    readings: list[dict[str, Any]] = []

    if scope_type == "unit":
        unit = _resolve_unit(subject, scope_id or "")
        label = f"Unit {unit.order_index}: {unit.title}"
        units = [_unit_hit(unit, subject.id)]
        for lec in lectures:
            taught += _note_hits(lec, unit.id)
        for lec in lectures:
            taught += _span_hits(lec, unit.id)
        query = " ".join([unit.title, *(unit.keywords or []), focus])
        readings = _searched_materials(subject.id, query)
    elif scope_type == "lecture":
        lecture = _resolve_lecture(subject, scope_id or "")
        label = lecture.title
        if lecture.summary:
            taught.append(
                {
                    "text": f"Lecture summary: {lecture.summary}",
                    "metadata": {
                        "kind": "note",
                        "lecture_id": lecture.id,
                        "lecture_title": lecture.title,
                        "start_s": 0.0,
                        "end_s": float(lecture.duration_s or 0.0),
                    },
                }
            )
        taught += _note_hits(lecture) + _span_hits(lecture)
    else:
        label = subject.name
        if subject.syllabus:
            units = [_unit_hit(u, subject.id) for u in subject.syllabus.units]
        if focus:
            # A steer narrows a subject-wide item to what retrieval ranks closest.
            taught = _searched_lectures(subject.id, focus)
            readings = _searched_materials(subject.id, focus, top_k=16)
        notes = [h for lec in lectures for h in _note_hits(lec)]
        spans = [h for lec in lectures for h in _span_hits(lec)]
        # Notes are the condensed record of every lecture; transcripts only
        # where no notes exist for that lecture.
        noted = {h["metadata"]["lecture_id"] for h in notes}
        taught += notes + [h for h in spans if h["metadata"]["lecture_id"] not in noted]
        readings += [h for m in subject.materials for h in _material_hits(m)]

    syllabus_only = bool(units) and not taught and not readings
    # Syllabus first (the frame), then lectures, then readings. Budget is
    # applied in that order so a long PDF cannot crowd out what was taught.
    hits = _within_budget(_dedupe(units + taught + readings), budget)
    if not hits:
        return StudioContext(label=label, sources="", citations=[], syllabus_only=False)
    sources, citations = _render_sources(hits)
    return StudioContext(label=label, sources=sources, citations=citations, syllabus_only=syllabus_only)
