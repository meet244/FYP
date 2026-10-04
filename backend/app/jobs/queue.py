"""In-process background job queue.

Transcription is minutes-long, so it cannot run inside a request. A worker thread
plus a `jobs` table is deliberately the whole mechanism — Celery and Redis would
add two services to a single-laptop deployment whose defining constraint is that
classroom audio never leaves the institution's machine.

Trade-off worth knowing: jobs live in this process. A restart mid-transcription
leaves the job `running` forever; `requeue_stale()` is called at startup to reset
those back to `queued`.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

from sqlalchemy import select, update as sql_update

from app.config import settings
from app.db import session_scope
from app.models import Job

log = logging.getLogger(__name__)

JobHandler = Callable[[str], None]

_registry: dict[str, JobHandler] = {}
_q: "queue.Queue[str]" = queue.Queue()
_workers: list[threading.Thread] = []
_started = False
_lock = threading.Lock()
_stop = threading.Event()
operation_lock = threading.RLock()


def active_for(db, *, lecture_id=None, subject_id=None, material_id=None):
    stmt = select(Job).where(Job.status.in_(("queued", "running", "cancelling")))
    if lecture_id:
        stmt = stmt.where(Job.lecture_id == lecture_id)
    if subject_id:
        stmt = stmt.where(Job.subject_id == subject_id)
    rows = db.scalars(stmt).all()
    if material_id:
        rows = [j for j in rows if (j.payload or {}).get("material_id") == material_id]
    return rows


class JobCancelled(RuntimeError):
    pass


def check_active(job_id: str) -> None:
    with session_scope() as db:
        job = db.get(Job, job_id)
        if job is None or job.status in ("cancelled", "cancelling"):
            raise JobCancelled("job was cancelled or its source was removed")


def register(kind: str) -> Callable[[JobHandler], JobHandler]:
    def deco(fn: JobHandler) -> JobHandler:
        _registry[kind] = fn
        return fn

    return deco


def enqueue(kind: str, **fields) -> Job:
    with session_scope() as db:
        job = Job(kind=kind, status="queued", **fields)
        db.add(job)
        db.flush()
        job_id = job.id
        db.expunge(job)
    _q.put(job_id)
    return job


def update(job_id: str, **fields) -> None:
    with session_scope() as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        if job.status in ("cancelled", "cancelling"):
            return
        for k, v in fields.items():
            setattr(job, k, v)


def progress(job_id: str) -> Callable[[float, str], None]:
    """Progress callback handed to long-running stages."""

    def cb(frac: float, message: str) -> None:
        check_active(job_id)
        update(job_id, progress=round(max(0.0, min(1.0, frac)), 4), message=message)

    return cb


def _cancelled(job):
    return job is None or job.status in ("cancelled", "cancelling")


def restore_cancelled(db, job) -> None:
    from app.models import Lecture, Material

    job.status = "cancelled"
    lec = db.get(Lecture, job.lecture_id) if job.lecture_id else None
    if lec:
        lec.status = "ready" if lec.notes else "transcribed" if lec.spans else "uploaded"
    if job.kind == "ingest_material":
        mat = db.get(Material, (job.payload or {}).get("material_id"))
        if mat:
            mat.status, mat.error = "failed", "processing cancelled; retry the job"


def _run_one(job_id: str) -> None:
    with session_scope() as db:
        job = db.get(Job, job_id)
        if job is None or job.status != "queued":
            return
        kind = job.kind
        # Atomic claim prevents duplicate queue delivery from running a job twice.
        claimed = db.execute(sql_update(Job).where(Job.id == job_id, Job.status == "queued")
                             .values(status="running", progress=0.0, error=None))
        if claimed.rowcount != 1:
            return

    handler = _registry.get(kind)
    if handler is None:
        update(job_id, status="failed", error=f"no handler registered for {kind!r}")
        return

    try:
        handler(job_id)
        check_active(job_id)
    except JobCancelled:
        log.info("job %s cancelled", job_id)
        with session_scope() as db:
            job = db.get(Job, job_id)
            if job:
                restore_cancelled(db, job)
    except Exception as exc:  # noqa: BLE001 — a failed job must not kill the worker
        log.exception("job %s (%s) failed", job_id, kind)
        from app.models import Lecture
        with operation_lock, session_scope() as db:
            job = db.get(Job, job_id)
            if _cancelled(job):
                if job:
                    restore_cancelled(db, job)
                return
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            lec = db.get(Lecture, job.lecture_id) if job.lecture_id else None
            if lec:
                lec.status = "transcribed" if lec.spans else "failed"
                lec.error = f"{type(exc).__name__}: {exc}"
    else:
        with operation_lock, session_scope() as db:
            job = db.get(Job, job_id)
            if job:
                if _cancelled(job):
                    restore_cancelled(db, job)
                else:
                    job.status, job.progress, job.stage = "succeeded", 1.0, "done"


def _loop() -> None:
    while not _stop.is_set():
        try:
            job_id = _q.get(timeout=0.5)
        except queue.Empty:
            # The jobs table is durable; also recover enqueue-before-notify crashes.
            with session_scope() as db:
                job_id = db.scalar(select(Job.id).where(Job.status == "queued")
                                   .order_by(Job.created_at).limit(1))
            if job_id:
                _run_one(job_id)
            continue
        try:
            _run_one(job_id)
        finally:
            _q.task_done()


def requeue_stale() -> int:
    """Reset jobs orphaned by a previous process and put them back on the queue."""
    with session_scope() as db:
        stopping = db.scalars(select(Job).where(Job.status == "cancelling")).all()
        for job in stopping:
            restore_cancelled(db, job)
        stale = db.scalars(select(Job).where(Job.status.in_(("queued", "running")))).all()
        ids = [j.id for j in stale]
        for job in stale:
            job.status = "queued"
            job.message = "requeued after restart"
    for job_id in ids:
        _q.put(job_id)
    return len(ids)


def start() -> None:
    global _started
    with _lock:
        if _started:
            return
        _stop.clear()
        n = requeue_stale()
        for i in range(max(1, settings.worker_threads)):
            t = threading.Thread(target=_loop, name=f"classscribe-worker-{i}", daemon=True)
            t.start()
            _workers.append(t)
        _started = True
    if n:
        log.info("requeued %d stale job(s)", n)


def stop() -> None:
    """Stop accepting jobs; permit the active handler to drain before DB shutdown."""
    global _started
    _stop.set()
    for worker in _workers:
        worker.join()
    _workers.clear()
    _started = False
