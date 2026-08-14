"""Syllabus coverage: which units have been delivered, which remain outstanding.

Free consequence of the method — every note already carries the syllabus unit
retrieval selected for its spans, so coverage is a group-by rather than a
separate classification pass.
"""
from __future__ import annotations

from collections import defaultdict

from sqlalchemy.orm import Session

from app.models import Lecture, Note, Subject, SyllabusUnit


def subject_coverage(db: Session, subject: Subject) -> dict:
    syllabus = subject.syllabus
    units: list[SyllabusUnit] = list(syllabus.units) if syllabus else []
    lectures: list[Lecture] = list(subject.lectures)

    hits: dict[str, list[dict]] = defaultdict(list)
    unattributed = 0
    for lec in lectures:
        for note in lec.notes:
            if note.unit_id:
                hits[note.unit_id].append(
                    {
                        "lecture_id": lec.id,
                        "lecture_title": lec.title,
                        "note_id": note.id,
                        "topic": note.topic,
                        "start_s": note.start_s,
                    }
                )
            else:
                unattributed += 1

    covered = [
        {
            "unit_id": u.id,
            "unit_key": u.unit_key,
            "title": u.title,
            "note_count": len(hits[u.id]),
            "notes": hits[u.id],
        }
        for u in units
        if hits[u.id]
    ]
    outstanding = [
        {"unit_id": u.id, "unit_key": u.unit_key, "title": u.title}
        for u in units
        if not hits[u.id]
    ]

    return {
        "subject_id": subject.id,
        "subject_name": subject.name,
        "total_units": len(units),
        "covered_units": len(covered),
        "coverage_fraction": round(len(covered) / len(units), 4) if units else None,
        "covered": covered,
        "outstanding": outstanding,
        "unattributed_notes": unattributed,
    }


def persist_notes(
    db: Session, lecture: Lecture, result, *, replace: bool = True
) -> list[Note]:
    """Write a SynthResult to the database, replacing any previous notes."""
    from app.models import LearningOutcome, Term

    if replace:
        for note in list(lecture.notes):
            db.delete(note)
        db.flush()

    created: list[Note] = []
    for sn in result.notes:
        note = Note(
            lecture_id=lecture.id,
            unit_id=sn.unit_id,
            order_index=sn.order_index,
            topic=sn.topic,
            markdown=sn.markdown,
            start_s=sn.start_s,
            end_s=sn.end_s,
        )
        db.add(note)
        db.flush()
        for t in sn.terms:
            db.add(Term(note_id=note.id, lecture_id=lecture.id,
                        term=t.term, definition=t.definition))
        for o in sn.outcomes:
            db.add(LearningOutcome(note_id=note.id, lecture_id=lecture.id,
                                   text=o.text, bloom_level=o.bloom_level))
        created.append(note)

    db.flush()
    return created
