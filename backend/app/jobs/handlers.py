"""Job handlers: the end-to-end pipelines.

process_lecture:  audio -> 16 kHz mono -> SGCD -> spans -> notes -> vector index
ingest_syllabus:  PDF -> code-mixed narrative units -> retrieval index bust
"""
from __future__ import annotations

import logging
import pathlib

from app.asr import retrieve, sgcd
from app.config import settings
from app.db import session_scope
from app.ingest import audio as audio_ingest
from app.ingest import syllabus as syllabus_ingest
from app.jobs.queue import progress, register, update
from app.models import Job, Lecture, Subject, Syllabus, SyllabusUnit, TranscriptSpan
from app.notes import coverage as coverage_mod
from app.notes import synthesize as synth
from app.rag import indexer
from app.views import SpanView, UnitView

log = logging.getLogger(__name__)


@register("process_lecture")
def process_lecture(job_id: str) -> None:
    with session_scope() as db:
        job = db.get(Job, job_id)
        lecture = db.get(Lecture, job.lecture_id)
        if lecture is None:
            raise ValueError(f"lecture {job.lecture_id} not found")
        subject = db.get(Subject, lecture.subject_id)
        source_path = pathlib.Path(job.payload["source_path"])
        lecture_id = lecture.id
        syllabus_id = subject.syllabus.id if subject.syllabus else None

    report = progress(job_id)

    # --- 1. normalise audio ---
    update(job_id, stage="audio", message="converting audio")
    wav_path = settings.audio_dir / f"{lecture_id}.wav"
    audio_ingest.to_wav16k_mono(source_path, wav_path)
    dur = audio_ingest.duration_s(wav_path)

    with session_scope() as db:
        lec = db.get(Lecture, lecture_id)
        lec.audio_path = str(wav_path)
        lec.duration_s = dur
        lec.status = "transcribing"
    report(0.05, f"audio ready ({dur / 60:.1f} min)")

    # --- 2. SGCD ---
    update(job_id, stage="transcribing")
    with session_scope() as db:
        units = []
        if syllabus_id:
            syl = db.get(Syllabus, syllabus_id)
            units = [UnitView.of(u) for u in syl.units] if syl else []

    result = sgcd.transcribe(
        str(wav_path),
        units=units,
        syllabus_id=syllabus_id,
        progress=lambda f, m: report(0.05 + 0.65 * f, m),
    )

    with session_scope() as db:
        lec = db.get(Lecture, lecture_id)
        for span in list(lec.spans):
            db.delete(span)
        db.flush()
        for s in result.spans:
            db.add(
                TranscriptSpan(
                    lecture_id=lecture_id,
                    index=s.index,
                    start_s=s.start_s,
                    end_s=s.end_s,
                    text=s.text,
                    baseline_text=s.baseline_text,
                    retrieved_unit_ids=s.retrieved_unit_ids,
                    prompt_tokens=s.prompt_tokens,
                    avg_logprob=s.avg_logprob,
                    compression_ratio=s.compression_ratio,
                    safeguard_fallback=s.safeguard_fallback,
                )
            )
        lec.asr_stats = result.stats
        lec.status = "transcribed"
    report(0.72, f"transcribed {result.stats['n_spans']} spans")

    # --- 3. notes ---
    # The LLM call runs with no session open: on SQLite a transaction held across
    # a multi-minute call blocks every other writer.
    update(job_id, stage="notes", message="generating notes")
    with session_scope() as db:
        lec = db.get(Lecture, lecture_id)
        lec.status = "summarising"
        lecture_title = lec.title
        span_views = [SpanView.of(s) for s in sorted(lec.spans, key=lambda s: s.index)]

    notes_error: str | None = None
    try:
        synth_result = synth.synthesize(lecture_title, span_views, units)
    except Exception as exc:  # noqa: BLE001
        # A missing API key or a bad LLM response must not cost the transcript —
        # that is the expensive artefact and it is already on disk. Stop at
        # `transcribed`; the client can retry notes with /reprocess.
        notes_error = f"{type(exc).__name__}: {exc}"
        log.warning("note synthesis failed for %s: %s", lecture_id, notes_error)
        with session_scope() as db:
            lec = db.get(Lecture, lecture_id)
            lec.status = "transcribed"
            lec.error = f"notes unavailable: {notes_error}"
    else:
        with session_scope() as db:
            lec = db.get(Lecture, lecture_id)
            coverage_mod.persist_notes(db, lec, synth_result)
            lec.status = "ready"
            lec.error = None
    report(0.9, "notes skipped" if notes_error else "notes written")

    # --- 4. index ---
    update(job_id, stage="indexing", message="indexing for search")
    with session_scope() as db:
        lec = db.get(Lecture, lecture_id)
        counts = indexer.reindex_lecture(db, lec)

    summary = f"indexed {counts['spans']} span windows, {counts['notes']} notes"
    if notes_error:
        summary += f" (notes unavailable: {notes_error})"
    report(1.0, summary)


@register("ingest_syllabus")
def ingest_syllabus(job_id: str) -> None:
    with session_scope() as db:
        job = db.get(Job, job_id)
        subject = db.get(Subject, job.subject_id)
        if subject is None:
            raise ValueError(f"subject {job.subject_id} not found")
        subject_id = subject.id
        subject_name = subject.name
        pdf_path = pathlib.Path(job.payload["source_path"])
        filename = job.payload.get("filename")

    update(job_id, stage="parsing", message="reading syllabus", progress=0.2)
    parsed = syllabus_ingest.parse_syllabus(pdf_path, subject_name)

    update(job_id, stage="writing", message="storing units", progress=0.7)
    with session_scope() as db:
        subject = db.get(Subject, subject_id)
        if subject.syllabus:
            old_id = subject.syllabus.id
            db.delete(subject.syllabus)
            db.flush()
            retrieve.invalidate(old_id)

        syl = Syllabus(
            subject_id=subject_id,
            source_filename=filename,
            source_path=str(pdf_path),
            provenance="real",
        )
        db.add(syl)
        db.flush()

        prefix = syllabus_ingest.slugify(subject_name)[:12]
        for i, u in enumerate(parsed["units"], start=1):
            db.add(
                SyllabusUnit(
                    syllabus_id=syl.id,
                    unit_key=f"{prefix}-u{i:02d}",
                    order_index=i,
                    title=u["title"],
                    prose=u["prose"],
                    keywords=u.get("keywords", []),
                )
            )
        n_units = len(parsed["units"])

    update(job_id, message=f"{n_units} units ingested", progress=1.0)
