from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Job
from app.schemas import JobOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return JobOut.model_validate(job)


@router.get("", response_model=list[JobOut])
def list_jobs(
    subject_id: str | None = None,
    lecture_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[JobOut]:
    stmt = select(Job).order_by(Job.created_at.desc()).limit(min(limit, 200))
    if subject_id:
        stmt = stmt.where(Job.subject_id == subject_id)
    if lecture_id:
        stmt = stmt.where(Job.lecture_id == lecture_id)
    if status:
        stmt = stmt.where(Job.status == status)
    return [JobOut.model_validate(j) for j in db.scalars(stmt).all()]
