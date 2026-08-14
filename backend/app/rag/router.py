"""Classify an incoming query so the right context and output shape are used.

"Based on the queries and the type of the queries, we generate relevant output" —
a definition lookup wants two sentences from one span; a revision request wants the
notes for a whole unit; a coverage question wants a database group-by and no
retrieval at all. Routing first keeps each of those from being answered as if it
were the others.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.llm.client import complete_json

log = logging.getLogger(__name__)

QUERY_TYPES = (
    "lookup",     # a fact, definition, or "what did she say about X"
    "explain",    # conceptual explanation, worked through
    "summary",    # summarise a lecture / topic / the subject so far
    "compare",    # contrast two things taught
    "quiz",       # generate practice questions or flashcards
    "outline",    # structure: what topics exist, in what order
    "coverage",   # syllabus progress — answered from the database, not retrieval
    "smalltalk",  # greetings, meta questions about the assistant
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "query_type": {"type": "string", "enum": list(QUERY_TYPES)},
        "search_query": {
            "type": "string",
            "description": "The query rewritten for semantic retrieval: self-contained, "
                           "pronouns resolved from the conversation, technical terms in English.",
        },
        "lecture_hint": {
            "type": "string",
            "description": "Lecture title or number the user referred to, or empty string.",
        },
        "wants_notes": {
            "type": "boolean",
            "description": "True when curated notes serve better than raw transcript.",
        },
    },
    "required": ["query_type", "search_query", "lecture_hint", "wants_notes"],
    "additionalProperties": False,
}

_SYSTEM = """\
You route questions in a study assistant built over a student's own lecture \
recordings. Classify the latest question and rewrite it for retrieval.

Types:
- lookup: a specific fact, definition, or "what did the lecturer say about X"
- explain: asks for a concept to be explained or worked through
- summary: summarise a lecture, a topic, or the course so far
- compare: contrast two or more things that were taught
- quiz: asks for practice questions, flashcards, or self-testing material
- outline: asks what topics exist and in what order
- coverage: asks about syllabus progress — what has been covered, what is left
- smalltalk: greetings or questions about the assistant itself

`search_query` must stand alone without the conversation: resolve "it", "that \
topic", "the last one" against the history, and put technical terms in English \
even when the question is asked in Hindi or transliterated Hindi, because the \
notes are written in English.

Set `wants_notes` true for summary, outline, quiz, and broad explain questions — \
curated notes serve those better. Set it false for lookup and for anything asking \
what was actually said, where raw transcript is the ground truth.\
"""


@dataclass
class Route:
    query_type: str
    search_query: str
    lecture_hint: str
    wants_notes: bool


def _fallback(question: str) -> Route:
    return Route(query_type="lookup", search_query=question, lecture_hint="", wants_notes=False)


def route(question: str, history: list[dict] | None = None) -> Route:
    convo = ""
    for msg in (history or [])[-6:]:
        convo += f"{msg['role']}: {msg['content']}\n"
    payload = f"Conversation so far:\n{convo or '(none)'}\n\nLatest question: {question}"

    try:
        raw = complete_json(_SYSTEM, payload, _SCHEMA, max_tokens=2_000, effort="low")
    except Exception as exc:
        # A routing failure should degrade to plain retrieval, not break the chat.
        log.warning("router failed, falling back to lookup: %s", exc)
        return _fallback(question)

    return Route(
        query_type=raw.get("query_type", "lookup"),
        search_query=raw.get("search_query") or question,
        lecture_hint=(raw.get("lecture_hint") or "").strip(),
        wants_notes=bool(raw.get("wants_notes")),
    )
