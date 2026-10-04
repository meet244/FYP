"""Job handlers: the end-to-end pipelines.

process_lecture:  audio -> 16 kHz mono -> selected ASR -> spans -> notes -> vector index
ingest_syllabus:  PDF -> code-mixed narrative units -> retrieval index bust
ingest_material:  pdf/image/doc -> extracted text -> local Chroma chunks
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import asdict
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.asr import retrieve, sgcd
from app.config import settings
from app.db import session_scope
from app.ingest import audio as audio_ingest
from app.ingest import syllabus as syllabus_ingest
from app.jobs.queue import check_active, progress, register, update
from app.ingest import materials as materials_ingest
from app.models import Job, Lecture, Material, Subject, Syllabus, SyllabusUnit, TranscriptSpan, TranscriptionRun
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
        from app.asr.catalog import resolve_options
        config = job.payload.get("asr_config") or lecture.asr_config or resolve_options()
        checkpoint_id = job.payload.get("checkpoint_id") or job_id

    report = progress(job_id)

    # --- 1. normalise audio ---
    update(job_id, stage="audio", message="converting audio")
    wav_path = settings.audio_dir / f"{lecture_id}.wav"
    if not wav_path.exists():
        audio_ingest.to_wav16k_mono(source_path, wav_path)
    dur = audio_ingest.duration_s(wav_path)

    with session_scope() as db:
        lec = db.get(Lecture, lecture_id)
        lec.audio_path = str(wav_path)
        lec.duration_s = dur
        lec.status = "transcribing"
    report(0.05, f"audio ready ({dur / 60:.1f} min)")

    # --- 2. SGCD ---
    # First run downloads model weights with no span progress yet, which the
    # UI otherwise reads as a hang at 0%.
    update(
        job_id,
        stage="transcribing",
        message=f"loading {config['model_id']} (first run downloads weights)",
    )
    report(0.06, f"loading {config['model_id']} (first run downloads weights)")
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
        config=config,
        checkpoint_path=str(settings.data_dir / "jobs" / f"{checkpoint_id}.json"),
    )

    check_active(job_id)
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
        lec.asr_config = config
        lec.error = None
        # Old notes cannot remain attached to a different transcript.
        for note in list(lec.notes):
            db.delete(note)
        lec.summary = None
        lec.status = "transcribed"
        if db.query(TranscriptionRun).filter_by(job_id=job_id).first() is None:
            db.add(TranscriptionRun(lecture_id=lecture_id, job_id=job_id, config=config,
                                    stats=result.stats, spans=[asdict(s) for s in result.spans]))
    report(0.72, f"transcribed {result.stats['n_spans']} spans")

    _finish_lecture(job_id, lecture_id, report)


@register("generate_notes")
def generate_notes(job_id: str) -> None:
    with session_scope() as db:
        job = db.get(Job, job_id)
        lecture_id = job.lecture_id
    _finish_lecture(job_id, lecture_id, progress(job_id), require_notes=True)


def _finish_lecture(job_id, lecture_id, report, require_notes=False):
    # --- 3. notes ---
    # The LLM call runs with no session open: on SQLite a transaction held across
    # a multi-minute call blocks every other writer.
    update(job_id, stage="notes", message="generating notes")
    with session_scope() as db:
        lec = db.get(Lecture, lecture_id)
        lec.status = "summarising"
        lecture_title = lec.title
        span_views = [SpanView.of(s) for s in sorted(lec.spans, key=lambda s: s.index)]
        units = [UnitView.of(u) for u in lec.subject.syllabus.units] if lec.subject.syllabus else []

    notes_error: str | None = None
    try:
        synth_result = synth.synthesize(lecture_title, span_views, units)
    except Exception as exc:  # noqa: BLE001
        # A missing API key or a bad LLM response must not cost the transcript —
        # that is the expensive artefact and it is already on disk. Stop at
        # `transcribed`; the client can retry /notes/regenerate without ASR.
        notes_error = f"{type(exc).__name__}: {exc}"
        log.warning("note synthesis failed for %s: %s", lecture_id, notes_error)
        with session_scope() as db:
            lec = db.get(Lecture, lecture_id)
            lec.status = "transcribed"
            lec.error = f"notes unavailable: {notes_error}"
    else:
        check_active(job_id)
        with session_scope() as db:
            lec = db.get(Lecture, lecture_id)
            coverage_mod.persist_notes(db, lec, synth_result)
            lec.status = "ready"
            lec.error = None
            lec.summary = synth_result.summary
    report(0.9, "notes skipped" if notes_error else "notes written")

    # --- 4. index ---
    update(job_id, stage="indexing", message="indexing for search")
    check_active(job_id)
    with session_scope() as db:
        lec = db.scalar(select(Lecture).where(Lecture.id == lecture_id).options(
            selectinload(Lecture.spans), selectinload(Lecture.notes),
            selectinload(Lecture.subject).selectinload(Subject.syllabus).selectinload(Syllabus.units),
        ))
        db.expunge_all()
    counts = indexer.reindex_lecture(None, lec)
    check_active(job_id)

    summary = f"indexed {counts['spans']} span windows, {counts['notes']} notes"
    if counts.get("units"):
        summary += f", {counts['units']} syllabus units"
    if notes_error:
        summary += f" (notes unavailable: {notes_error})"
    report(1.0, summary)
    if notes_error and require_notes:
        raise RuntimeError(f"notes unavailable: {notes_error}")


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
    check_active(job_id)

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
        db.flush()

    check_active(job_id)
    with session_scope() as db:
        subject = db.scalar(select(Subject).where(Subject.id == subject_id).options(
            selectinload(Subject.syllabus).selectinload(Syllabus.units)))
        db.expunge_all()
    indexer.index_syllabus_units(None, subject)

    update(job_id, message=f"{n_units} units ingested", progress=1.0)


@register("ingest_material")
def ingest_material(job_id: str) -> None:
    with session_scope() as db:
        job = db.get(Job, job_id)
        material = db.get(Material, job.payload["material_id"])
        if material is None:
            raise ValueError("material not found")
        material_id = material.id
        kind = material.kind
        path = pathlib.Path(material.path)
        mime = material.mime or "application/octet-stream"
        material.status = "processing"

    report = progress(job_id)
    update(job_id, stage="extracting", message=f"reading {kind}")
    report(0.2, f"extracting {kind}")
    try:
        text = materials_ingest.extract(kind, path, mime)
        if not text.strip():
            raise ValueError("No extractable content. Scanned PDFs need OCR before upload.")
        check_active(job_id)
        update(job_id, stage="indexing", message="embedding chunks")
        report(0.7, "indexing")
        with session_scope() as db:
            mat = db.get(Material, material_id)
            mat.text = text
            db.expunge(mat)
        n = indexer.index_material(mat)
        check_active(job_id)
        with session_scope() as db:
            mat = db.get(Material, material_id)
            mat.text = text
            mat.n_chunks = n
            mat.status = "ready"
            mat.error = None
        report(1.0, f"indexed {n} chunks")
    except Exception as exc:
        with session_scope() as db:
            mat = db.get(Material, material_id)
            if mat is not None:
                mat.status = "failed"
                mat.error = f"{type(exc).__name__}: {exc}"
        raise
