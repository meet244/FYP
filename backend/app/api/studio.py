from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import StudioItem, Subject
from app.schemas import StudioCreate, StudioItemOut
from app.studio import runner
from app.studio.context import ScopeNotFound

router = APIRouter(tags=["studio"])


def _get(db: Session, item_id: str) -> StudioItem:
    item = db.get(StudioItem, item_id)
    if item is None:
        raise HTTPException(404, "studio item not found")
    return item


@router.post("/subjects/{subject_id}/studio", response_model=StudioItemOut, status_code=202)
def create_item(subject_id: str, body: StudioCreate, db: Session = Depends(get_db)) -> StudioItemOut:
    """Queue a quiz, flashcards, mind map, report, slide deck or infographic."""
    if db.get(Subject, subject_id) is None:
        raise HTTPException(404, "subject not found")
    try:
        item = runner.create(
            db, subject_id, body.kind, body.scope.model_dump(), body.options.model_dump()
        )
    except ScopeNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    db.commit()
    runner.submit(item.id)
    return StudioItemOut.model_validate(item)


@router.get("/subjects/{subject_id}/studio", response_model=list[StudioItemOut])
def list_items(subject_id: str, db: Session = Depends(get_db)) -> list[StudioItemOut]:
    if db.get(Subject, subject_id) is None:
        raise HTTPException(404, "subject not found")
    rows = db.scalars(
        select(StudioItem)
        .where(StudioItem.subject_id == subject_id)
        .order_by(StudioItem.created_at.desc())
    ).all()
    return [StudioItemOut.model_validate(r) for r in rows]


@router.get("/studio/{item_id}", response_model=StudioItemOut)
def get_item(item_id: str, db: Session = Depends(get_db)) -> StudioItemOut:
    return StudioItemOut.model_validate(_get(db, item_id))


@router.post("/studio/{item_id}/retry", response_model=StudioItemOut, status_code=202)
def retry_item(item_id: str, db: Session = Depends(get_db)) -> StudioItemOut:
    item = _get(db, item_id)
    if item.status in ("pending", "running"):
        raise HTTPException(409, "this item is still generating")
    item.status, item.error, item.content, item.citations = "pending", None, None, None
    db.commit()
    runner.submit(item.id)
    return StudioItemOut.model_validate(item)


@router.delete("/studio/{item_id}", status_code=204, response_model=None)
def delete_item(item_id: str, db: Session = Depends(get_db)) -> None:
    db.delete(_get(db, item_id))
    db.commit()
