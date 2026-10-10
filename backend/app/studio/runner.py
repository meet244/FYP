"""Create Studio items and generate them in the background.

A dedicated thread pool rather than the job queue: the queue's single worker is
occupied for minutes by every transcription, and a student asking for a quiz
should not wait behind one. Status lives on the `studio_items` row, so the
frontend polls one table and a restart can requeue unfinished items.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import session_scope
from app.llm.client import LLMRateLimited, LLMUnavailable, complete_json_with_model
from app.models import StudioItem, Subject
from app.studio import context as ctx
from app.studio import generators as gen

log = logging.getLogger(__name__)

_pool: ThreadPoolExecutor | None = None


def _executor() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=settings.studio_workers, thread_name_prefix="classscribe-studio")
    return _pool


def create(
    db: Session,
    subject_id: str,
    kind: str,
    scope: dict[str, Any],
    options: dict[str, Any],
) -> StudioItem:
    """Validate, persist as `pending`, and schedule generation once committed."""
    if kind not in gen.SPECS:
        raise ValueError(f"unknown studio kind {kind!r}")
    scope_type = scope.get("type") or "subject"
    scope_id = scope.get("id") if scope_type != "subject" else None
    label = ctx.scope_label(db, subject_id, scope_type, scope_id)
    options = {k: v for k, v in options.items() if v not in (None, "")}
    item = StudioItem(
        subject_id=subject_id,
        kind=kind,
        title=gen.default_title(kind, options, label),
        scope={"type": scope_type, "id": scope_id, "label": label},
        options=options,
        status="pending",
    )
    db.add(item)
    db.flush()
    return item


def submit(item_id: str) -> None:
    _executor().submit(_safe_run, item_id)


def _safe_run(item_id: str) -> None:
    try:
        run(item_id)
    except Exception:  # noqa: BLE001 — never let a worker thread die silently
        log.exception("studio item %s crashed", item_id)


def _fail(item_id: str, message: str) -> None:
    with session_scope() as db:
        item = db.get(StudioItem, item_id)
        if item is not None:
            item.status, item.error = "failed", message


def run(item_id: str) -> None:
    with session_scope() as db:
        item = db.get(StudioItem, item_id)
        if item is None or item.status not in ("pending", "running"):
            return
        item.status, item.error = "running", None
        kind, options, scope = item.kind, dict(item.options or {}), dict(item.scope or {})
        subject_id = item.subject_id
        subject_name = db.get(Subject, subject_id).name

    try:
        with session_scope() as db:
            gathered = ctx.gather(
                db, subject_id, scope.get("type") or "subject", scope.get("id"), options.get("focus")
            )
    except ctx.ScopeNotFound as exc:
        _fail(item_id, f"{exc}. It may have been deleted.")
        return

    if not gathered.sources:
        _fail(
            item_id,
            "Nothing to generate from yet. Add a syllabus, a lecture recording or files "
            "for this scope, then try again.",
        )
        return

    spec = gen.SPECS[kind]
    prompt = gen.build_prompt(
        kind,
        options,
        subject_name=subject_name,
        scope_label=gathered.label,
        sources=gathered.sources,
        syllabus_only=gathered.syllabus_only,
    )
    try:
        raw, model = complete_json_with_model(gen.SYSTEM, prompt, spec.schema, max_tokens=spec.max_tokens)
        valid = {c["n"] for c in gathered.citations}
        content = spec.clean(raw if isinstance(raw, dict) else {}, valid)
    except LLMUnavailable:
        _fail(item_id, "No Gemini API key is configured. Set GEMINI_API_KEY in backend/.env and restart.")
        return
    except LLMRateLimited as exc:
        _fail(item_id, str(exc))
        return
    except Exception as exc:  # noqa: BLE001 — malformed output, network error
        log.warning("studio %s generation failed: %s", kind, exc)
        _fail(item_id, f"Generation failed: {exc}")
        return

    used = gen.referenced(content)
    with session_scope() as db:
        item = db.get(StudioItem, item_id)
        if item is None:  # deleted while generating
            return
        if content.get("title"):
            item.title = content["title"][:500]
        item.content = content
        item.citations = [c for c in gathered.citations if c["n"] in used]
        item.model = model
        item.status = "ready"
        item.error = None


def requeue_unfinished() -> int:
    """Items left pending/running by a previous process are generated again."""
    with session_scope() as db:
        ids = list(
            db.scalars(select(StudioItem.id).where(StudioItem.status.in_(("pending", "running"))))
        )
        for item_id in ids:
            db.get(StudioItem, item_id).status = "pending"
    for item_id in ids:
        submit(item_id)
    return len(ids)


def shutdown() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
