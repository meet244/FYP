"""Write syllabus units, transcript spans, and notes into the local vector index.

Spans are indexed in overlapping windows rather than one-per-span: a ~25 s span is
short enough that a question often straddles a boundary, and the window carries
the surrounding sentence into the embedding without losing the timestamp anchor.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ingest.materials import chunk_text
from app.models import Lecture, Material, Note, Subject, Syllabus, SyllabusUnit, TranscriptSpan
from app.rag import store

WINDOW = 3  # spans per indexed chunk (~75 s)
STRIDE = 2  # overlap of one span


def index_lecture_spans(db: Session, lecture: Lecture) -> int:
    spans: list[TranscriptSpan] = sorted(lecture.spans, key=lambda s: s.index)
    if not spans:
        return 0

    ids, docs, metas = [], [], []
    for start in range(0, len(spans), STRIDE):
        window = spans[start : start + WINDOW]
        if not window:
            break
        text = " ".join(s.text for s in window).strip()
        if not text:
            continue
        ids.append(f"{lecture.id}:w{start}")
        docs.append(text)
        metas.append(
            {
                "kind": "span",
                "subject_id": lecture.subject_id,
                "lecture_id": lecture.id,
                "lecture_title": lecture.title,
                "start_s": float(window[0].start_s),
                "end_s": float(window[-1].end_s),
                "span_indices": ",".join(str(s.index) for s in window),
            }
        )
        if start + WINDOW >= len(spans):
            break

    store.upsert(store.SPANS, ids, docs, metas)
    return len(ids)


def index_lecture_notes(db: Session, lecture: Lecture) -> int:
    notes: list[Note] = sorted(lecture.notes, key=lambda n: n.order_index)
    ids, docs, metas = [], [], []
    for note in notes:
        body = f"{note.topic}\n\n{note.markdown}".strip()
        if not body:
            continue
        ids.append(f"note:{note.id}")
        docs.append(body)
        metas.append(
            {
                "kind": "note",
                "subject_id": lecture.subject_id,
                "lecture_id": lecture.id,
                "lecture_title": lecture.title,
                "note_id": note.id,
                "topic": note.topic,
                "unit_id": note.unit_id or "",
                "start_s": float(note.start_s or 0.0),
                "end_s": float(note.end_s or 0.0),
            }
        )
    store.upsert(store.NOTES, ids, docs, metas)
    return len(ids)


def index_syllabus_units(db: Session | None, subject: Subject) -> int:
    """Embed the subject's syllabus so chat can retrieve curriculum, not just lectures."""
    store.delete_subject_units(subject.id)

    units: list = []
    if db is not None:
        row = db.scalars(select(Syllabus).where(Syllabus.subject_id == subject.id)).first()
        if row is None:
            return 0
        units = list(
            db.scalars(
                select(SyllabusUnit)
                .where(SyllabusUnit.syllabus_id == row.id)
                .order_by(SyllabusUnit.order_index)
            )
        )
    else:
        syllabus = getattr(subject, "syllabus", None)
        if syllabus is None:
            return 0
        units = list(getattr(syllabus, "units", []) or [])

    if not units:
        return 0

    ids, docs, metas = [], [], []
    for unit in units:
        parts = [unit.title.strip(), (unit.prose or "").strip()]
        keywords = [k for k in (unit.keywords or []) if k]
        if keywords:
            parts.append(" ".join(keywords))
        text = "\n\n".join(p for p in parts if p)
        if not text:
            continue
        ids.append(f"unit:{unit.id}")
        docs.append(text)
        metas.append(
            {
                "kind": "unit",
                "subject_id": subject.id,
                "unit_id": unit.id,
                "unit_key": getattr(unit, "unit_key", "") or "",
                "topic": unit.title,
            }
        )

    store.upsert(store.UNITS, ids, docs, metas)
    return len(ids)


def index_material(material: Material) -> int:
    """Embed one PDF, image, or document as overlapping text chunks."""
    store.delete_material(material.id)
    body = (material.text or "").strip() or material.title
    chunks = chunk_text(body)
    if not chunks:
        return 0

    ids, docs, metas = [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(f"{material.id}:c{i}")
        docs.append(f"{material.title}\n\n{chunk}")
        metas.append(
            {
                "kind": material.kind,
                "subject_id": material.subject_id,
                "material_id": material.id,
                "topic": material.title,
                "chunk": i,
            }
        )
    store.upsert(store.MATERIALS, ids, docs, metas)
    return len(ids)


def reindex_lecture(db: Session | None, lecture: Lecture) -> dict[str, int]:
    store.delete_lecture(lecture.id)
    return {
        "spans": index_lecture_spans(db, lecture),
        "notes": index_lecture_notes(db, lecture),
        "units": index_syllabus_units(db, lecture.subject),
    }
