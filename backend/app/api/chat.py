from __future__ import annotations

import logging
import datetime as dt
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db import get_db
from app.llm.client import LLMRateLimited, LLMUnavailable
from app.models import ChatMessage, ChatSession, Subject, Syllabus, Lecture
from app.rag.answer import answer_question
from app.schemas import ChatMessageOut, ChatRequest, ChatResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

HISTORY_TURNS = 8


@router.post("/subjects/{subject_id}/chat", response_model=ChatResponse)
def chat(subject_id: str, body: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Ask a question against one subject's lecture corpus.

    Scoped per subject rather than globally: a student's Operating Systems
    question should not retrieve from their Networks lectures.
    """
    subject = db.scalar(select(Subject).where(Subject.id == subject_id).options(
        selectinload(Subject.syllabus).selectinload(Syllabus.units),
        selectinload(Subject.lectures).selectinload(Lecture.notes),
    ))
    if subject is None:
        raise HTTPException(404, "subject not found")

    if body.session_id:
        session = db.get(ChatSession, body.session_id)
        if session is None or session.subject_id != subject_id:
            raise HTTPException(404, "chat session not found for this subject")
    else:
        session = ChatSession(id=uuid.uuid4().hex, subject_id=subject_id, title=body.question[:120])

    history = [
        {"role": m.role, "content": m.content}
        for m in (session.messages[-HISTORY_TURNS:] if body.session_id else [])
    ]
    session_id = session.id
    # Eager-loaded subject is a detached read snapshot. Release SQLite before
    # retrieval/LLM calls so chat never holds the single writer for minutes.
    db.expunge_all()
    db.rollback()

    try:
        result = answer_question(None, subject, body.question, history)
    except (LLMUnavailable, LLMRateLimited) as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("chat failed")
        raise HTTPException(503, "Could not answer right now.") from exc

    if db.get(Subject, subject_id) is None:
        raise HTTPException(409, "subject was removed while answering")
    if body.session_id:
        session = db.get(ChatSession, session_id)
        if session is None:
            raise HTTPException(409, "chat session was removed while answering")
    else:
        session = ChatSession(id=session_id, subject_id=subject_id, title=body.question[:120])
        db.add(session)
        db.flush()
    session.updated_at = dt.datetime.now(dt.timezone.utc)
    db.add(ChatMessage(session_id=session_id, role="user", content=body.question))
    db.add(
        ChatMessage(
            session_id=session_id,
            role="assistant",
            content=result.text,
            query_type=result.query_type,
            citations=result.citations,
        )
    )
    db.commit()

    return ChatResponse(
        session_id=session_id,
        answer=result.text,
        query_type=result.query_type,
        citations=result.citations,
    )


@router.get("/subjects/{subject_id}/chat/sessions")
def list_sessions(subject_id: str, db: Session = Depends(get_db)) -> list[dict]:
    if db.get(Subject, subject_id) is None:
        raise HTTPException(404, "subject not found")
    rows = db.scalars(
        select(ChatSession)
        .where(ChatSession.subject_id == subject_id)
        .order_by(ChatSession.updated_at.desc())
    ).all()
    return [
        {
            "id": s.id,
            "title": s.title,
            "message_count": len(s.messages),
            "updated_at": s.updated_at,
        }
        for s in rows
    ]


@router.get("/chat/sessions/{session_id}", response_model=list[ChatMessageOut])
def get_session(session_id: str, db: Session = Depends(get_db)) -> list[ChatMessageOut]:
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(404, "chat session not found")
    return [ChatMessageOut.model_validate(m) for m in session.messages]


@router.delete("/chat/sessions/{session_id}", status_code=204, response_model=None)
def delete_session(session_id: str, db: Session = Depends(get_db)) -> None:
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(404, "chat session not found")
    db.delete(session)
    db.commit()
