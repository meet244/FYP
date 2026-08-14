"""ORM models.

Naming follows the paper: a *subject* owns a *syllabus* of *units*; a *lecture*
is decoded into *spans*; spans are attached to the syllabus unit that retrieval
selected, which is what makes coverage tracking free (Section III-E).
"""
from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class Subject(Base, TimestampMixin):
    __tablename__ = "subjects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(256))
    code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    syllabus: Mapped["Syllabus | None"] = relationship(
        back_populates="subject", uselist=False, cascade="all, delete-orphan"
    )
    lectures: Mapped[list["Lecture"]] = relationship(
        back_populates="subject", cascade="all, delete-orphan"
    )


class Syllabus(Base, TimestampMixin):
    __tablename__ = "syllabi"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"))
    source_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # "real" (institutional PDF) | "generated-from-title" | "manual"
    provenance: Mapped[str] = mapped_column(String(32), default="real")

    subject: Mapped[Subject] = relationship(back_populates="syllabus")
    units: Mapped[list["SyllabusUnit"]] = relationship(
        back_populates="syllabus",
        cascade="all, delete-orphan",
        order_by="SyllabusUnit.order_index",
    )


class SyllabusUnit(Base, TimestampMixin):
    """A unit u_i = (t_i, d_i, K_i): title, code-mixed narration, terminology set."""

    __tablename__ = "syllabus_units"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    syllabus_id: Mapped[str] = mapped_column(ForeignKey("syllabi.id", ondelete="CASCADE"))
    unit_key: Mapped[str] = mapped_column(String(64))  # e.g. "os-u03"
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String(512))
    # d_i — fluent Hindi/English code-mixed narration in the dual-script convention.
    # This field, not `keywords`, is what SGCD actually feeds the decoder.
    prose: Mapped[str] = mapped_column(Text)
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)

    syllabus: Mapped[Syllabus] = relationship(back_populates="units")


class Lecture(Base, TimestampMixin):
    __tablename__ = "lectures"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(512))
    recorded_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    original_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    # uploaded -> transcribing -> transcribed -> summarising -> ready | failed
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    asr_stats: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    subject: Mapped[Subject] = relationship(back_populates="lectures")
    spans: Mapped[list["TranscriptSpan"]] = relationship(
        back_populates="lecture",
        cascade="all, delete-orphan",
        order_by="TranscriptSpan.index",
    )
    notes: Mapped[list["Note"]] = relationship(
        back_populates="lecture", cascade="all, delete-orphan"
    )


class TranscriptSpan(Base, TimestampMixin):
    """One ~25 s decode. Carries the full SGCD audit trail for the span."""

    __tablename__ = "transcript_spans"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    lecture_id: Mapped[str] = mapped_column(ForeignKey("lectures.id", ondelete="CASCADE"))
    index: Mapped[int] = mapped_column(Integer)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)

    text: Mapped[str] = mapped_column(Text)  # the accepted hypothesis
    baseline_text: Mapped[str | None] = mapped_column(Text, nullable=True)  # pass 1, unconditioned
    retrieved_unit_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    avg_logprob: Mapped[float | None] = mapped_column(Float, nullable=True)
    compression_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    # True when the safeguard reverted to the unconditioned hypothesis.
    safeguard_fallback: Mapped[bool] = mapped_column(Boolean, default=False)

    lecture: Mapped[Lecture] = relationship(back_populates="spans")


class Note(Base, TimestampMixin):
    __tablename__ = "notes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    lecture_id: Mapped[str] = mapped_column(ForeignKey("lectures.id", ondelete="CASCADE"))
    unit_id: Mapped[str | None] = mapped_column(
        ForeignKey("syllabus_units.id", ondelete="SET NULL"), nullable=True
    )
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    topic: Mapped[str] = mapped_column(String(512))
    markdown: Mapped[str] = mapped_column(Text)
    start_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_s: Mapped[float | None] = mapped_column(Float, nullable=True)

    lecture: Mapped[Lecture] = relationship(back_populates="notes")
    outcomes: Mapped[list["LearningOutcome"]] = relationship(
        back_populates="note", cascade="all, delete-orphan"
    )
    terms: Mapped[list["Term"]] = relationship(
        back_populates="note", cascade="all, delete-orphan"
    )


class LearningOutcome(Base, TimestampMixin):
    __tablename__ = "learning_outcomes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    note_id: Mapped[str] = mapped_column(ForeignKey("notes.id", ondelete="CASCADE"))
    lecture_id: Mapped[str] = mapped_column(ForeignKey("lectures.id", ondelete="CASCADE"))
    text: Mapped[str] = mapped_column(Text)
    bloom_level: Mapped[str | None] = mapped_column(String(32), nullable=True)

    note: Mapped[Note] = relationship(back_populates="outcomes")


class Term(Base, TimestampMixin):
    __tablename__ = "terms"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    note_id: Mapped[str] = mapped_column(ForeignKey("notes.id", ondelete="CASCADE"))
    lecture_id: Mapped[str] = mapped_column(ForeignKey("lectures.id", ondelete="CASCADE"))
    term: Mapped[str] = mapped_column(String(256))
    definition: Mapped[str] = mapped_column(Text)

    note: Mapped[Note] = relationship(back_populates="terms")


class Job(Base, TimestampMixin):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(64))  # "process_lecture" | "ingest_syllabus"
    subject_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    lecture_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class ChatSession(Base, TimestampMixin):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"))
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatMessage(Base, TimestampMixin):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    query_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)

    session: Mapped[ChatSession] = relationship(back_populates="messages")
