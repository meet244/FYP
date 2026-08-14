"""Turn a corrected transcript into structured study material.

Section III-E: the transcript is segmented topically and converted into a topic
hierarchy, definitions of encountered terminology, and learning outcomes stated as
competency statements. Each note is attached to the syllabus unit it instantiates.

Unit attribution is retrieval-derived, not model-derived: SGCD already selected
units per span during decoding, so the majority vote over a note's spans is free
and grounded in what the decoder actually conditioned on. The model's own
suggestion is used only to break ties or fill gaps.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from app.llm.client import complete_json
from app.views import SpanView, UnitView

log = logging.getLogger(__name__)

_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Two or three sentences on the lecture as a whole."},
        "notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "first_span": {"type": "integer", "description": "Index of the first span this topic covers."},
                    "last_span": {"type": "integer", "description": "Index of the last span this topic covers."},
                    "markdown": {
                        "type": "string",
                        "description": "The study note in Markdown: sub-headings, bullet points, worked steps.",
                    },
                    "suggested_unit_key": {
                        "type": "string",
                        "description": "Syllabus unit key this topic instantiates, or empty string if none fits.",
                    },
                    "terms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "term": {"type": "string"},
                                "definition": {"type": "string"},
                            },
                            "required": ["term", "definition"],
                            "additionalProperties": False,
                        },
                    },
                    "outcomes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "A competency statement: 'Explain how ...', 'Compare ... with ...'.",
                                },
                                "bloom_level": {
                                    "type": "string",
                                    "enum": ["remember", "understand", "apply", "analyse", "evaluate", "create"],
                                },
                            },
                            "required": ["text", "bloom_level"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["topic", "first_span", "last_span", "markdown",
                             "suggested_unit_key", "terms", "outcomes"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "notes"],
    "additionalProperties": False,
}

_SYSTEM = """\
You write revision notes from a transcript of a Hindi-English code-switched \
university lecture.

About the transcript: it is machine-produced, so expect recognition errors, and it \
is code-mixed — Hindi in Devanagari carrying the explanation, English technical \
terms in Latin script. Read both. Where a Hindi word is clearly a garbled English \
technical term, use the correct English term in the notes.

Write the notes in ENGLISH. The student attended a code-mixed lecture but revises \
in English, and English keeps the technical vocabulary searchable.

Segment the transcript into 3-10 topics following the lecture's own structure. For \
each topic give the span index range it covers, so every note stays traceable to \
the recording.

Note bodies are for revision, not a transcript replay: state the actual content — \
definitions, mechanisms, steps, worked examples, the distinctions the lecturer drew \
— in Markdown, with sub-headings and bullets where they help. Do not describe the \
lecture ("the lecturer then explained..."); state what was taught.

`terms` holds technical terminology the lecture actually introduced, each with a \
one-sentence definition as used in this lecture.

`outcomes` are competency statements a student could be examined against, phrased \
as "Explain ...", "Derive ...", "Compare ...". One to three per topic. Only include \
outcomes the lecture genuinely supports.

If the syllabus unit list is supplied, set `suggested_unit_key` to the unit each \
topic instantiates, or the empty string when nothing fits. Never invent a key.

If a stretch of transcript is too garbled to interpret, leave it out rather than \
guessing at content.\
"""


@dataclass
class SynthTerm:
    term: str
    definition: str


@dataclass
class SynthOutcome:
    text: str
    bloom_level: str


@dataclass
class SynthNote:
    topic: str
    markdown: str
    order_index: int
    start_s: float | None
    end_s: float | None
    unit_id: str | None
    terms: list[SynthTerm] = field(default_factory=list)
    outcomes: list[SynthOutcome] = field(default_factory=list)


@dataclass
class SynthResult:
    summary: str
    notes: list[SynthNote]


def _render_transcript(spans: list[SpanView]) -> str:
    lines = []
    for s in spans:
        lines.append(f"[span {s.index} | {s.start_s:.0f}s-{s.end_s:.0f}s] {s.text}")
    return "\n".join(lines)


def _render_units(units: list[UnitView]) -> str:
    if not units:
        return "(no syllabus attached to this subject)"
    return "\n".join(f"- {u.unit_key}: {u.title}" for u in units)


def _unit_from_retrieval(
    spans: list[SpanView], first: int, last: int, valid_ids: set[str]
) -> str | None:
    """Majority vote over the units SGCD retrieved for the covered spans."""
    votes: Counter[str] = Counter()
    for s in spans:
        if first <= s.index <= last:
            for uid in s.retrieved_unit_ids or []:
                if uid in valid_ids:
                    votes[uid] += 1
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def synthesize(
    lecture_title: str, spans: list[SpanView], units: list[UnitView]
) -> SynthResult:
    spans = sorted(spans, key=lambda s: s.index)
    if not spans:
        raise ValueError("lecture has no transcript spans")

    by_key = {u.unit_key: u for u in units}
    valid_ids = {u.id for u in units}
    span_bounds = {s.index: (s.start_s, s.end_s) for s in spans}

    payload = (
        f"Lecture title: {lecture_title}\n\n"
        f"Syllabus units:\n{_render_units(units)}\n\n"
        f"Transcript:\n{_render_transcript(spans)}"
    )
    result = complete_json(_SYSTEM, payload, _SCHEMA, max_tokens=32_000)

    notes: list[SynthNote] = []
    for i, raw in enumerate(result.get("notes", [])):
        first = int(raw.get("first_span", 0))
        last = int(raw.get("last_span", first))
        if last < first:
            first, last = last, first

        unit_id = _unit_from_retrieval(spans, first, last, valid_ids)
        if unit_id is None:
            suggested = by_key.get((raw.get("suggested_unit_key") or "").strip())
            unit_id = suggested.id if suggested else None

        start_s = span_bounds.get(first, (None, None))[0]
        end_s = span_bounds.get(last, (None, None))[1]

        notes.append(
            SynthNote(
                topic=raw["topic"],
                markdown=raw["markdown"],
                order_index=i,
                start_s=start_s,
                end_s=end_s,
                unit_id=unit_id,
                terms=[SynthTerm(t["term"], t["definition"]) for t in raw.get("terms", [])],
                outcomes=[
                    SynthOutcome(o["text"], o.get("bloom_level", "understand"))
                    for o in raw.get("outcomes", [])
                ],
            )
        )

    return SynthResult(summary=result.get("summary", ""), notes=notes)
