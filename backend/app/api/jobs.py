from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Job, Lecture, Material, Subject
from app.jobs import queue
from app.schemas import JobOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return JobOut.of(job)


@router.get("", response_model=list[JobOut])
def list_jobs(
    subject_id: str | None = None,
    lecture_id: str | None = None,
    status: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[JobOut]:
    stmt = select(Job).order_by(Job.created_at.desc()).limit(min(limit, 200))
    if subject_id:
        stmt = stmt.where(Job.subject_id == subject_id)
    if lecture_id:
        stmt = stmt.where(Job.lecture_id == lecture_id)
    if status:
        stmt = stmt.where(Job.status == status)
    return [JobOut.of(j) for j in db.scalars(stmt).all()]


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, db: Session = Depends(get_db)):
    with queue.operation_lock:
        job = db.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job.status not in ("queued", "running"):
            raise HTTPException(409, "only queued or running jobs can be cancelled")
        job.status = "cancelling" if job.status == "running" else "cancelled"
        job.message = "cancelled; current inference call may finish before the worker stops"
        lecture = db.get(Lecture, job.lecture_id) if job.lecture_id else None
        if lecture:
            lecture.status = "ready" if lecture.notes else "transcribed" if lecture.spans else "uploaded"
        if job.kind == "ingest_material":
            mat = db.get(Material, (job.payload or {}).get("material_id"))
            if mat:
                mat.status, mat.error = "failed", "processing cancelled; retry the job"
        db.commit()
        return JobOut.of(job)


@router.post("/{job_id}/retry", response_model=JobOut, status_code=202)
def retry_job(job_id: str, db: Session = Depends(get_db)):
    with queue.operation_lock:
        job = db.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job.status not in ("failed", "cancelled"):
            raise HTTPException(409, "only failed or cancelled jobs can be retried")
        if job.subject_id and db.get(Subject, job.subject_id) is None:
            raise HTTPException(409, "subject was deleted")
        if job.lecture_id and db.get(Lecture, job.lecture_id) is None:
            raise HTTPException(409, "lecture was deleted")
        payload = dict(job.payload or {})
        if job.kind == "ingest_material" and db.get(Material, payload.get("material_id")) is None:
            raise HTTPException(409, "material was deleted")
        active = queue.active_for(db, lecture_id=job.lecture_id) if job.lecture_id else queue.active_for(
            db, subject_id=job.subject_id, material_id=payload.get("material_id"))
        if any(job.lecture_id or j.kind == job.kind for j in active):
            raise HTTPException(409, "a job already processes this source")
        if job.kind == "process_lecture":
            payload["checkpoint_id"] = payload.get("checkpoint_id") or job.id
        return JobOut.of(queue.enqueue(job.kind, subject_id=job.subject_id,
                                       lecture_id=job.lecture_id, payload=payload))
