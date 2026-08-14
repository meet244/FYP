from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.jobs import queue
from app.models import Subject, SyllabusUnit
from app.notes.coverage import subject_coverage
from app.schemas import (
    JobOut,
    SubjectCreate,
    SubjectOut,
    SyllabusOut,
    UnitOut,
    UnitUpdate,
)

router = APIRouter(prefix="/subjects", tags=["subjects"])


def _out(subject: Subject) -> SubjectOut:
    return SubjectOut(
        id=subject.id,
        name=subject.name,
        code=subject.code,
        description=subject.description,
        created_at=subject.created_at,
        has_syllabus=subject.syllabus is not None,
        lecture_count=len(subject.lectures),
    )


def _get(db: Session, subject_id: str) -> Subject:
    subject = db.get(Subject, subject_id)
    if subject is None:
        raise HTTPException(404, "subject not found")
    return subject


@router.post("", response_model=SubjectOut, status_code=201)
def create_subject(body: SubjectCreate, db: Session = Depends(get_db)) -> SubjectOut:
    subject = Subject(name=body.name, code=body.code, description=body.description)
    db.add(subject)
    db.commit()
    return _out(subject)


@router.get("", response_model=list[SubjectOut])
def list_subjects(db: Session = Depends(get_db)) -> list[SubjectOut]:
    subjects = db.scalars(select(Subject).order_by(Subject.created_at.desc())).all()
    return [_out(s) for s in subjects]


@router.get("/{subject_id}", response_model=SubjectOut)
def get_subject(subject_id: str, db: Session = Depends(get_db)) -> SubjectOut:
    return _out(_get(db, subject_id))


@router.delete("/{subject_id}", status_code=204, response_model=None)
def delete_subject(subject_id: str, db: Session = Depends(get_db)) -> None:
    from app.rag import store

    subject = _get(db, subject_id)
    for lec in subject.lectures:
        store.delete_lecture(lec.id)
    db.delete(subject)
    db.commit()


# --- syllabus ---
@router.post("/{subject_id}/syllabus", response_model=JobOut, status_code=202)
def upload_syllabus(
    subject_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)
) -> JobOut:
    """Upload the syllabus PDF.

    Parsing is a job, not a request: rewriting a syllabus into code-mixed
    narration is an LLM call and takes longer than a sensible HTTP timeout.
    """
    subject = _get(db, subject_id)
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "syllabus must be a PDF")

    dest = settings.uploads_dir / f"syllabus-{uuid.uuid4().hex}.pdf"
    dest.write_bytes(file.file.read())

    job = queue.enqueue(
        "ingest_syllabus",
        subject_id=subject.id,
        payload={"source_path": str(dest), "filename": file.filename},
    )
    return JobOut.model_validate(job)


@router.get("/{subject_id}/syllabus", response_model=SyllabusOut)
def get_syllabus(subject_id: str, db: Session = Depends(get_db)) -> SyllabusOut:
    subject = _get(db, subject_id)
    if subject.syllabus is None:
        raise HTTPException(404, "no syllabus uploaded for this subject")
    return SyllabusOut.model_validate(subject.syllabus)


@router.patch("/{subject_id}/syllabus/units/{unit_id}", response_model=UnitOut)
def update_unit(
    subject_id: str, unit_id: str, body: UnitUpdate, db: Session = Depends(get_db)
) -> UnitOut:
    """Hand-correct a unit.

    Worth exposing: the paper's own threat-to-validity section flags LLM-authored
    syllabi as a confound, and `prose` is the field SGCD actually conditions on.
    An instructor rewriting it in their own register is the highest-value edit in
    the system.
    """
    from app.asr import retrieve

    subject = _get(db, subject_id)
    unit = db.get(SyllabusUnit, unit_id)
    if unit is None or subject.syllabus is None or unit.syllabus_id != subject.syllabus.id:
        raise HTTPException(404, "unit not found on this subject")

    if body.title is not None:
        unit.title = body.title
    if body.prose is not None:
        unit.prose = body.prose
    if body.keywords is not None:
        unit.keywords = body.keywords
    db.commit()
    retrieve.invalidate(subject.syllabus.id)
    return UnitOut.model_validate(unit)


@router.get("/{subject_id}/coverage")
def get_coverage(subject_id: str, db: Session = Depends(get_db)) -> dict:
    return subject_coverage(db, _get(db, subject_id))
