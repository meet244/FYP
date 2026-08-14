"""Detached, plain-data views of ORM rows.

Transcription and note synthesis take minutes. Holding a SQLAlchemy session open
across an LLM or Whisper call would keep a SQLite write transaction alive for the
duration and block every other request, so the worker reads what it needs into
these, closes the session, does the slow work, then reopens to write.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UnitView:
    """Duck-types SyllabusUnit for retrieve.SyllabusIndex and prompts.build_prompt."""

    id: str
    unit_key: str
    title: str
    prose: str
    keywords: list[str] = field(default_factory=list)

    @classmethod
    def of(cls, unit) -> "UnitView":
        return cls(
            id=unit.id,
            unit_key=unit.unit_key,
            title=unit.title,
            prose=unit.prose,
            keywords=list(unit.keywords or []),
        )


@dataclass
class SpanView:
    index: int
    start_s: float
    end_s: float
    text: str
    retrieved_unit_ids: list[str] = field(default_factory=list)

    @classmethod
    def of(cls, span) -> "SpanView":
        return cls(
            index=span.index,
            start_s=span.start_s,
            end_s=span.end_s,
            text=span.text,
            retrieved_unit_ids=list(span.retrieved_unit_ids or []),
        )
