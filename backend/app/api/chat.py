from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ChatMessage, ChatSession, Subject
from app.rag.answer import answer_question
from app.schemas import ChatMessageOut, ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

HISTORY_TURNS = 8


@router.post("/subjects/{subject_id}/chat", response_model=ChatResponse)
def chat(subject_id: str, body: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Ask a question against one subject's lecture corpus.

    Scoped per subject rather than globally: a student's Operating Systems
    question should not retrieve from their Networks lectures.
    """
    subject = db.get(Subject, subject_id)
    if subject is None:
        raise HTTPException(404, "subject not found")

    if body.session_id:
        session = db.get(ChatSession, body.session_id)
        if session is None or session.subject_id != subject_id:
            raise HTTPException(404, "chat session not found for this subject")
    else:
        session = ChatSession(subject_id=subject_id, title=body.question[:120])
        db.add(session)
        db.flush()

    history = [
        {"role": m.role, "content": m.content}
        for m in session.messages[-HISTORY_TURNS:]
    ]

    db.add(ChatMessage(session_id=session.id, role="user", content=body.question))
    db.flush()

    result = answer_question(db, subject, body.question, history)

    db.add(
        ChatMessage(
            session_id=session.id,
            role="assistant",
            content=result.text,
            query_type=result.query_type,
            citations=result.citations,
        )
    )
    db.commit()

    return ChatResponse(
        session_id=session.id,
        answer=result.text,
        query_type=result.query_type,
        citations=result.citations,
    )


@router.get("/subjects/{subject_id}/chat/sessions")
def list_sessions(subject_id: str, db: Session = Depends(get_db)) -> list[dict]:
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
