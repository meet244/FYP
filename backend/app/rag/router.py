"""Classify an incoming query so the right context and output shape are used.

Routing is heuristic: an LLM classifier doubled every chat into two Gemini
calls and burned the free-tier quota before the answer ran. Keyword routing is
good enough, and coverage/smalltalk never need a model at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

QUERY_TYPES = (
    "lookup",
    "explain",
    "summary",
    "compare",
    "quiz",
    "outline",
    "coverage",
    "smalltalk",
)


@dataclass
class Route:
    query_type: str
    search_query: str
    lecture_hint: str
    wants_notes: bool


_SMALLTALK = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|who are you|what can you do|help)\b",
    re.I,
)
_COVERAGE = re.compile(
    r"\b(coverage|been covered|what(?:'s| is) left|outstanding units|"
    r"syllabus progress|how much have we)\b",
    re.I,
)
_QUIZ = re.compile(r"\b(quiz|practice questions?|flashcards?|test me|mcqs?)\b", re.I)
_OUTLINE = re.compile(
    r"\b(outline|list the topics|what topics|what does the syllabus cover|"
    r"what is on the syllabus)\b",
    re.I,
)
_SUMMARY = re.compile(r"\b(summar(?:y|ise|ize)|recap|overview)\b", re.I)
_COMPARE = re.compile(r"\b(compare|contrast|difference|versus|\bvs\.?\b)\b", re.I)
_EXPLAIN = re.compile(r"\b(explain|how does|why does|walk me through|teach me)\b", re.I)
_LECTURE = re.compile(r"(?:lecture|recording)\s+#?(\d+)|last (?:lecture|recording)", re.I)


def _lecture_hint(question: str) -> str:
    match = _LECTURE.search(question)
    return match.group(0).strip() if match else ""


def route(question: str, history: list[dict] | None = None) -> Route:
    del history
    hint = _lecture_hint(question)
    q = question.strip()

    if _SMALLTALK.match(q) and len(q) < 60:
        return Route("smalltalk", q, "", False)
    if _COVERAGE.search(q):
        return Route("coverage", q, hint, False)
    if _QUIZ.search(q):
        return Route("quiz", q, hint, True)
    if _OUTLINE.search(q):
        return Route("outline", q, hint, True)
    if _SUMMARY.search(q):
        return Route("summary", q, hint, True)
    if _COMPARE.search(q):
        return Route("compare", q, hint, True)
    if _EXPLAIN.search(q):
        return Route("explain", q, hint, True)
    return Route("lookup", q, hint, False)
