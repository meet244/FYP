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

from sqlalchemy import select

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
        for k, v in fields.items():
            setattr(job, k, v)


def progress(job_id: str) -> Callable[[float, str], None]:
    """Progress callback handed to long-running stages."""

    def cb(frac: float, message: str) -> None:
        update(job_id, progress=round(max(0.0, min(1.0, frac)), 4), message=message)

    return cb


def _run_one(job_id: str) -> None:
    with session_scope() as db:
        job = db.get(Job, job_id)
        if job is None or job.status not in ("queued", "running"):
            return
        kind = job.kind

    handler = _registry.get(kind)
    if handler is None:
        update(job_id, status="failed", error=f"no handler registered for {kind!r}")
        return

    update(job_id, status="running", progress=0.0, error=None)
    try:
        handler(job_id)
    except Exception as exc:  # noqa: BLE001 — a failed job must not kill the worker
        log.exception("job %s (%s) failed", job_id, kind)
        update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
    else:
        update(job_id, status="succeeded", progress=1.0, stage="done")


def _loop() -> None:
    while True:
        job_id = _q.get()
        try:
            _run_one(job_id)
        finally:
            _q.task_done()


def requeue_stale() -> int:
    """Reset jobs orphaned by a previous process and put them back on the queue."""
    with session_scope() as db:
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
        for i in range(max(1, settings.worker_threads)):
            t = threading.Thread(target=_loop, name=f"classscribe-worker-{i}", daemon=True)
            t.start()
            _workers.append(t)
        _started = True
    n = requeue_stale()
    if n:
        log.info("requeued %d stale job(s)", n)
