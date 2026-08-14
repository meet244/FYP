"""Pydantic request/response models — the contract the frontend codes against."""
from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- subjects ---
class SubjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    code: str | None = None
    description: str | None = None


class SubjectOut(ORM):
    id: str
    name: str
    code: str | None
    description: str | None
    created_at: dt.datetime
    has_syllabus: bool = False
    lecture_count: int = 0


# --- syllabus ---
class UnitOut(ORM):
    id: str
    unit_key: str
    order_index: int
    title: str
    prose: str
    keywords: list[str]


class SyllabusOut(ORM):
    id: str
    source_filename: str | None
    provenance: str
    created_at: dt.datetime
    units: list[UnitOut]


class UnitUpdate(BaseModel):
    title: str | None = None
    prose: str | None = None
    keywords: list[str] | None = None


# --- lectures ---
class SpanOut(ORM):
    index: int
    start_s: float
    end_s: float
    text: str
    retrieved_unit_ids: list[str]
    safeguard_fallback: bool


class LectureOut(ORM):
    id: str
    subject_id: str
    title: str
    status: str
    duration_s: float | None
    recorded_at: dt.datetime | None
    created_at: dt.datetime
    error: str | None = None
    asr_stats: dict[str, Any] | None = None


class TranscriptOut(BaseModel):
    lecture_id: str
    title: str
    status: str
    duration_s: float | None
    text: str
    spans: list[SpanOut]
    asr_stats: dict[str, Any] | None


# --- notes ---
class TermOut(ORM):
    term: str
    definition: str


class OutcomeOut(ORM):
    text: str
    bloom_level: str | None


class NoteOut(ORM):
    id: str
    topic: str
    markdown: str
    order_index: int
    start_s: float | None
    end_s: float | None
    unit_id: str | None
    terms: list[TermOut]
    outcomes: list[OutcomeOut]


# --- chat ---
class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None


class Citation(BaseModel):
    n: int
    kind: str | None = None
    lecture_id: str | None = None
    lecture_title: str | None = None
    start_s: float | None = None
    end_s: float | None = None
    timestamp: str | None = None
    note_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    query_type: str
    citations: list[Citation]


class ChatMessageOut(ORM):
    id: str
    role: str
    content: str
    query_type: str | None
    citations: list[dict[str, Any]] | None
    created_at: dt.datetime


# --- jobs ---
class JobOut(ORM):
    id: str
    kind: str
    status: str
    stage: str | None
    progress: float
    message: str | None
    error: str | None
    lecture_id: str | None
    subject_id: str | None
    created_at: dt.datetime
    updated_at: dt.datetime
