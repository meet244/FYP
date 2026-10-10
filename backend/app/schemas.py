"""Pydantic request/response models — the contract the frontend codes against."""
from __future__ import annotations

import datetime as dt
from typing import Any, Literal

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
    material_count: int = 0


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
    baseline_text: str | None = None
    prompt_tokens: int = 0
    avg_logprob: float | None = None
    compression_ratio: float | None = None


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
    asr_config: dict[str, Any] | None = None
    summary: str | None = None


class TranscriptOut(BaseModel):
    lecture_id: str
    title: str
    status: str
    duration_s: float | None
    text: str
    spans: list[SpanOut]
    asr_stats: dict[str, Any] | None


class RunOut(ORM):
    id: str
    lecture_id: str
    job_id: str
    config: dict[str, Any]
    stats: dict[str, Any]
    created_at: dt.datetime


class RunDetail(RunOut):
    spans: list[SpanOut]


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


# --- materials ---
class MaterialOut(ORM):
    id: str
    subject_id: str
    kind: str
    title: str
    original_filename: str | None
    mime: str | None
    status: str
    n_chunks: int
    error: str | None = None
    created_at: dt.datetime


class MaterialPreviewOut(MaterialOut):
    text: str | None = None


# --- chat ---
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
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
    unit_id: str | None = None
    unit_title: str | None = None
    material_id: str | None = None
    material_title: str | None = None
    excerpt: str | None = None
    page: int | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    query_type: str
    citations: list[Citation]
    studio_item_id: str | None = None


class ChatMessageOut(ORM):
    id: str
    role: str
    content: str
    query_type: str | None
    citations: list[dict[str, Any]] | None
    studio_item_id: str | None = None
    created_at: dt.datetime


# --- studio ---
StudioKind = Literal["quiz", "flashcards", "mindmap", "report", "slides", "infographic"]


class StudioScopeIn(BaseModel):
    type: Literal["subject", "unit", "lecture"] = "subject"
    id: str | None = None


class StudioOptions(BaseModel):
    # Questions / cards / slides. Ignored by kinds without a natural count.
    count: int | None = Field(default=None, ge=3, le=40)
    difficulty: Literal["easy", "medium", "hard", "mixed"] | None = None
    # Report flavour.
    format: Literal["study_guide", "briefing", "faq", "glossary"] | None = None
    # Free-text steer, e.g. "focus on backpropagation".
    focus: str | None = Field(default=None, max_length=500)


class StudioCreate(BaseModel):
    kind: StudioKind
    scope: StudioScopeIn = StudioScopeIn()
    options: StudioOptions = StudioOptions()


class StudioItemOut(ORM):
    id: str
    subject_id: str
    kind: str
    title: str
    scope: dict[str, Any]
    options: dict[str, Any]
    status: str
    content: dict[str, Any] | None = None
    citations: list[dict[str, Any]] | None = None
    model: str | None = None
    error: str | None = None
    created_at: dt.datetime
    updated_at: dt.datetime


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
    material_id: str | None = None
    asr_config: dict[str, Any] | None = None
    created_at: dt.datetime
    updated_at: dt.datetime

    @classmethod
    def of(cls, job: Any) -> "JobOut":
        payload = getattr(job, "payload", None) or {}
        out = cls.model_validate(job)
        if isinstance(payload, dict) and payload.get("material_id"):
            return out.model_copy(update={"material_id": payload["material_id"]})
        if isinstance(payload, dict) and payload.get("asr_config"):
            return out.model_copy(update={"asr_config": payload["asr_config"]})
        return out
