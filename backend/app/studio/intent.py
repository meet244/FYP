"""Recognise a chat message that asks for a Studio item.

Heuristic, like `rag/router.py`, and for the same reason: an LLM classifier
would spend a Gemini request on every chat turn. A request needs a creation
verb (or "quiz me") plus an artefact noun, so "what is on the quiz?" stays a
normal question and "give me five practice questions" stays an inline answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_VERB = r"(?:make|create|generate|build|prepare|produce|draft|design|give\s+me|i\s+(?:want|need)|can\s+you\s+(?:make|create|generate)|turn\b.{0,40}\binto)"

_KINDS: list[tuple[str, str]] = [
    ("flashcards", r"flash\s*-?cards?"),
    ("mindmap", r"mind\s*-?\s*maps?|concept\s+maps?"),
    ("slides", r"slide\s*-?\s*decks?|slides|presentation|ppt"),
    ("infographic", r"info\s*-?graphics?|visual\s+summary|one-?pager"),
    ("quiz", r"quiz(?:zes)?|mcq\s+test|multiple[-\s]choice\s+(?:quiz|test)"),
    ("report", r"study\s+guide|briefing(?:\s+doc(?:ument)?)?|faq|glossary|report"),
]

_QUIZ_ME = re.compile(r"\bquiz\s+me\b", re.I)
_COUNT = re.compile(r"\b(\d{1,2})\s+(?:\w+\s+){0,2}(?:questions?|mcqs?|cards?|flash\s*cards?|slides?)\b", re.I)
_UNIT_NUM = re.compile(r"\b(?:unit|module|chapter)\s+#?(\d{1,2})\b", re.I)
_LECTURE = re.compile(r"\b(?:lecture|recording|class)\s+#?(\d{1,3})\b|\b(?:last|latest)\s+(?:lecture|recording|class)\b", re.I)


@dataclass
class StudioIntent:
    kind: str
    scope: dict[str, Any] = field(default_factory=lambda: {"type": "subject", "id": None})
    options: dict[str, Any] = field(default_factory=dict)


def _kind(text: str) -> str | None:
    if _QUIZ_ME.search(text):
        return "quiz"
    for kind, pattern in _KINDS:
        if re.search(rf"\b{_VERB}\b.{{0,60}}\b(?:{pattern})\b", text, re.I | re.S):
            return kind
    return None


def _report_format(text: str) -> str | None:
    t = text.lower()
    if "briefing" in t:
        return "briefing"
    if "faq" in t:
        return "faq"
    if "glossary" in t:
        return "glossary"
    if "study guide" in t:
        return "study_guide"
    return None


def _difficulty(text: str) -> str | None:
    match = re.search(r"\b(easy|medium|hard|difficult|tough|simple)\b", text, re.I)
    if not match:
        return None
    word = match.group(1).lower()
    return {"difficult": "hard", "tough": "hard", "simple": "easy"}.get(word, word)


def _scope(text: str, units: list[Any], lectures: list[Any]) -> dict[str, Any]:
    """Units: by number or by title words. Lectures: by number in upload order, or latest."""
    match = _UNIT_NUM.search(text)
    if match:
        n = int(match.group(1))
        unit = next((u for u in units if u.order_index == n), None)
        if unit is not None:
            return {"type": "unit", "id": unit.id}

    match = _LECTURE.search(text)
    if match and lectures:
        ordered = sorted(lectures, key=lambda lec: lec.created_at)
        if match.group(1):
            i = int(match.group(1)) - 1
            if 0 <= i < len(ordered):
                return {"type": "lecture", "id": ordered[i].id}
        else:
            return {"type": "lecture", "id": ordered[-1].id}

    lowered = text.lower()
    for unit in units:
        title = unit.title.lower().strip()
        if len(title) >= 4 and title in lowered:
            return {"type": "unit", "id": unit.id}
    return {"type": "subject", "id": None}


def detect(question: str, units: list[Any], lectures: list[Any]) -> StudioIntent | None:
    kind = _kind(question)
    if kind is None:
        return None
    options: dict[str, Any] = {}
    count = _COUNT.search(question)
    if count and kind in ("quiz", "flashcards", "slides"):
        options["count"] = max(3, min(40, int(count.group(1))))
    if kind == "quiz" and (level := _difficulty(question)):
        options["difficulty"] = level
    if kind == "report":
        options["format"] = _report_format(question) or "study_guide"
    if topic := _topic(question):
        options["focus"] = topic
    return StudioIntent(kind=kind, scope=_scope(question, units, lectures), options=options)


_FILLER = re.compile(
    r"\b(?:please|pls|can|could|you|would|make|create|generate|build|prepare|produce|draft|design|"
    r"give|me|i|want|need|turn|into|a|an|the|some|of|on|about|for|from|over|covering|with|and|"
    r"quiz|quizzes|mcqs?|test|multiple|choice|flash\s*-?cards?|cards?|mind\s*-?\s*maps?|concept|maps?|"
    r"slide\s*-?\s*decks?|slides?|presentation|ppt|info\s*-?graphics?|visual|summary|one-?pager|"
    r"study|guide|briefing|doc|document|faq|glossary|report|questions?|easy|medium|hard|difficult|"
    r"tough|simple|whole|entire|subject|course|this|everything|all|topics?|unit|module|chapter|"
    r"lecture|recording|class|last|latest|\d+)\b",
    re.I,
)


def _topic(question: str) -> str | None:
    """What is left once command words are removed, e.g. 'backpropagation'."""
    rest = _FILLER.sub(" ", question)
    rest = re.sub(r"[^\w\s\-+/']", " ", rest)
    rest = " ".join(rest.split())
    return rest[:500] if len(rest) >= 3 else None
