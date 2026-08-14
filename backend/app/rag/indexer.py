"""Write transcript spans and notes into the vector index.

Spans are indexed in overlapping windows rather than one-per-span: a ~25 s span is
short enough that a question often straddles a boundary, and the window carries
the surrounding sentence into the embedding without losing the timestamp anchor.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Lecture, Note, TranscriptSpan
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


def reindex_lecture(db: Session, lecture: Lecture) -> dict[str, int]:
    store.delete_lecture(lecture.id)
    return {
        "spans": index_lecture_spans(db, lecture),
        "notes": index_lecture_notes(db, lecture),
    }
