from __future__ import annotations

import datetime as dt
import pathlib
import uuid

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.asr.catalog import ASROptions, resolve_options
from app.db import get_db
from app.jobs import queue
from app.models import Lecture, Subject, TranscriptionRun
from app.schemas import JobOut, LectureOut, NoteOut, RunOut, RunDetail, TranscriptOut
from app.ingest.uploads import save_upload

router = APIRouter(tags=["lectures"])

# What a phone recorder plausibly produces. ffmpeg handles the decoding.
AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav", ".aac", ".ogg", ".opus", ".flac", ".mp4", ".3gp", ".amr", ".webm"}


def _get_lecture(db: Session, lecture_id: str) -> Lecture:
    lecture = db.get(Lecture, lecture_id)
    if lecture is None:
        raise HTTPException(404, "lecture not found")
    return lecture


@router.post("/subjects/{subject_id}/lectures", response_model=JobOut, status_code=202)
def upload_lecture(
    subject_id: str,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    recorded_at: str | None = Form(None),
    model_id: str | None = Form(None),
    method: str | None = Form(None),
    language: str | None = Form(None),
    db: Session = Depends(get_db),
) -> JobOut:
    """Upload a classroom recording and start the pipeline.

    Returns a job immediately; the client polls `/jobs/{id}` while the selected
    model transcribes, generates notes and indexes the recording.
    """
    subject = db.get(Subject, subject_id)
    if subject is None:
        raise HTTPException(404, "subject not found")

    suffix = pathlib.Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(
            400, f"unsupported audio format {suffix!r}; expected one of {sorted(AUDIO_SUFFIXES)}"
        )

    when = None
    if recorded_at:
        try:
            when = dt.datetime.fromisoformat(recorded_at)
        except ValueError:
            raise HTTPException(400, "recorded_at must be ISO 8601")
    try:
        config = resolve_options(ASROptions(model_id=model_id, method=method, language=language))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    dest = settings.uploads_dir / f"lecture-{uuid.uuid4().hex}{suffix}"
    save_upload(file, dest)

    lecture = Lecture(
        subject_id=subject.id,
        title=title or pathlib.Path(file.filename or "Untitled lecture").stem,
        original_filename=file.filename,
        recorded_at=when,
        status="uploaded",
        source_path=str(dest),
        asr_config=config,
    )
    db.add(lecture)
    db.commit()

    job = queue.enqueue(
        "process_lecture",
        subject_id=subject.id,
        lecture_id=lecture.id,
        payload={"source_path": str(dest), "asr_config": config},
    )
    return JobOut.of(job)


@router.get("/subjects/{subject_id}/lectures", response_model=list[LectureOut])
def list_lectures(subject_id: str, db: Session = Depends(get_db)) -> list[LectureOut]:
    if db.get(Subject, subject_id) is None:
        raise HTTPException(404, "subject not found")
    rows = db.scalars(
        select(Lecture)
        .where(Lecture.subject_id == subject_id)
        .order_by(Lecture.created_at.desc())
    ).all()
    return [LectureOut.model_validate(r) for r in rows]


@router.get("/lectures/{lecture_id}", response_model=LectureOut)
def get_lecture(lecture_id: str, db: Session = Depends(get_db)) -> LectureOut:
    return LectureOut.model_validate(_get_lecture(db, lecture_id))


@router.get("/lectures/{lecture_id}/transcript", response_model=TranscriptOut)
def get_transcript(lecture_id: str, db: Session = Depends(get_db)) -> TranscriptOut:
    lecture = _get_lecture(db, lecture_id)
    spans = sorted(lecture.spans, key=lambda s: s.index)
    return TranscriptOut(
        lecture_id=lecture.id,
        title=lecture.title,
        status=lecture.status,
        duration_s=lecture.duration_s,
        text=" ".join(s.text for s in spans),
        spans=spans,
        asr_stats=lecture.asr_stats,
    )


@router.get("/lectures/{lecture_id}/notes", response_model=list[NoteOut])
def get_notes(lecture_id: str, db: Session = Depends(get_db)) -> list[NoteOut]:
    lecture = _get_lecture(db, lecture_id)
    return [
        NoteOut.model_validate(n)
        for n in sorted(lecture.notes, key=lambda n: n.order_index)
    ]


@router.get("/lectures/{lecture_id}/audio")
def get_audio(lecture_id: str, db: Session = Depends(get_db)) -> FileResponse:
    """Serve the normalised audio so the UI can seek to a citation's timestamp."""
    lecture = _get_lecture(db, lecture_id)
    if not lecture.audio_path or not pathlib.Path(lecture.audio_path).exists():
        raise HTTPException(404, "audio not available for this lecture")
    return FileResponse(lecture.audio_path, media_type="audio/wav", filename=f"{lecture.title}.wav")


@router.post("/lectures/{lecture_id}/reprocess", response_model=JobOut, status_code=202)
def reprocess(lecture_id: str, options: ASROptions | None = Body(None), db: Session = Depends(get_db)) -> JobOut:
    """Re-run the pipeline — e.g. after uploading or correcting the syllabus.

    Worth doing precisely because conditioning is what changes: the same audio
    decoded against a syllabus is not the same transcript.
    """
    lecture = _get_lecture(db, lecture_id)
    source = next((pathlib.Path(p) for p in (lecture.audio_path, lecture.source_path)
                   if p and pathlib.Path(p).is_file()), None)
    if source is None:
        raise HTTPException(409, "no stored audio to reprocess")

    try:
        config = resolve_options(options, lecture.asr_config)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with queue.operation_lock:
        if queue.active_for(db, lecture_id=lecture.id):
            raise HTTPException(409, "a job is already processing this lecture")
        job = queue.enqueue(
            "process_lecture", subject_id=lecture.subject_id, lecture_id=lecture.id,
            payload={"source_path": str(source), "asr_config": config},
        )
    return JobOut.of(job)


@router.post("/lectures/{lecture_id}/notes/regenerate", response_model=JobOut, status_code=202)
def regenerate_notes(lecture_id: str, db: Session = Depends(get_db)) -> JobOut:
    lecture = _get_lecture(db, lecture_id)
    if not lecture.spans:
        raise HTTPException(409, "lecture has no transcript yet")
    with queue.operation_lock:
        if queue.active_for(db, lecture_id=lecture_id):
            raise HTTPException(409, "a job is already processing this lecture")
        return JobOut.of(queue.enqueue("generate_notes", subject_id=lecture.subject_id,
                                       lecture_id=lecture_id))


@router.get("/lectures/{lecture_id}/runs", response_model=list[RunOut])
def list_runs(lecture_id: str, db: Session = Depends(get_db)):
    _get_lecture(db, lecture_id)
    return db.scalars(select(TranscriptionRun).where(TranscriptionRun.lecture_id == lecture_id)
                      .order_by(TranscriptionRun.created_at.desc())).all()


@router.get("/lectures/{lecture_id}/runs/{run_id}", response_model=RunDetail)
def get_run(lecture_id: str, run_id: str, db: Session = Depends(get_db)):
    row = db.get(TranscriptionRun, run_id)
    if row is None or row.lecture_id != lecture_id:
        raise HTTPException(404, "transcription run not found")
    return row


@router.delete("/lectures/{lecture_id}", status_code=204, response_model=None)
def delete_lecture(lecture_id: str, db: Session = Depends(get_db)) -> None:
    from app.rag import store

    lecture = _get_lecture(db, lecture_id)
    if queue.active_for(db, lecture_id=lecture.id):
        raise HTTPException(409, "cancel or finish active jobs before deleting this lecture")
    store.delete_lecture(lecture.id)
    if lecture.audio_path:
        pathlib.Path(lecture.audio_path).unlink(missing_ok=True)
    if lecture.source_path:
        pathlib.Path(lecture.source_path).unlink(missing_ok=True)
    db.delete(lecture)
    db.commit()
