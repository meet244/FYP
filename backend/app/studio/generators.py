"""One schema and one instruction per Studio item kind.

Every kind is a single schema-constrained Gemini call over the same numbered
sources chat uses, so each question, card, slide and branch can point back at
the lecture moment or page it came from. Post-processing drops anything the
schema cannot enforce: out-of-range answer keys, citations to sources that
were never supplied, empty cards.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

KINDS = ("quiz", "flashcards", "mindmap", "report", "slides", "infographic")

LABELS = {
    "quiz": "Quiz",
    "flashcards": "Flashcards",
    "mindmap": "Mind map",
    "report": "Report",
    "slides": "Slide deck",
    "infographic": "Infographic",
}

REPORT_FORMATS = {
    "study_guide": (
        "Study guide",
        "A revision study guide: a short overview, then one section per major topic "
        "with the key ideas, definitions and any worked examples from the sources, "
        "then a 'Check yourself' list of 5 short-answer questions with answers.",
    ),
    "briefing": (
        "Briefing document",
        "A briefing document: an executive summary paragraph, then the main themes "
        "as sections with the most important facts, then 'Key takeaways' as bullets.",
    ),
    "faq": (
        "FAQ",
        "A frequently-asked-questions document: 8 to 12 questions a student revising "
        "this material would ask, each with a concise answer.",
    ),
    "glossary": (
        "Glossary",
        "An alphabetical glossary of every technical term the sources define or use, "
        "each with a one or two sentence definition as the sources use it.",
    ),
}

INFOGRAPHIC_ICONS = [
    "lightbulb", "gear", "chart", "book", "target", "layers",
    "network", "warning", "check", "clock", "cpu", "function",
]

SYSTEM = """\
You create study material for a university student from their own course \
sources: syllabus units, lecture voice transcripts (often Hindi-English \
code-mixed), notes generated from those lectures, and uploaded PDFs, slides, \
images and documents. Sources are numbered [1], [2], ...

Rules, in order of importance:
- Use only what the sources contain. Never add facts, formulas or examples from \
outside knowledge.
- Write in clear English. Keep technical terms exactly as the field uses them. \
Where a transcript is garbled but the intended technical term is recoverable, \
use the correct term without commenting on transcription quality.
- Syllabus units describe the planned curriculum, not what was taught. Prefer \
lecture and reading sources when they exist.
- Record which numbered sources support each element in its `sources` list. \
Only cite numbers that appear in the supplied sources.
- Write for revision: precise, concrete, no filler, no meta-commentary about \
the sources or about yourself.\
"""

_SOURCES = {"type": "array", "items": {"type": "integer"}, "description": "Supporting source numbers"}


@dataclass(frozen=True)
class Spec:
    schema: dict[str, Any]
    instruction: Callable[[dict[str, Any]], str]
    clean: Callable[[dict[str, Any], set[int]], dict[str, Any]]
    max_tokens: int = 16_000


def _refs(value: Any, valid: set[int]) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for v in value:
        if isinstance(v, int) and v in valid and v not in out:
            out.append(v)
    return out


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _strip_markers(text: str, valid: set[int]) -> str:
    return re.sub(r"\[(\d+)\]", lambda m: m.group(0) if int(m[1]) in valid else "", text)


# --- quiz -------------------------------------------------------------------

def _quiz_instruction(o: dict[str, Any]) -> str:
    n = o.get("count") or 10
    level = o.get("difficulty") or "mixed"
    spread = (
        "Mix easy recall, medium understanding and hard application questions."
        if level == "mixed"
        else f"Pitch every question at {level} difficulty."
    )
    return (
        f"Write a multiple-choice quiz of exactly {n} questions. {spread} "
        "Each question has 4 options with exactly one correct answer; distractors "
        "must be plausible misconceptions, not jokes. `answer_index` is the 0-based "
        "index of the correct option. `explanation` says why the answer is right in "
        "one or two sentences. `hint` nudges without giving the answer away. Cover "
        "the breadth of the sources rather than repeating one topic."
    )


QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "answer_index": {"type": "integer"},
                    "explanation": {"type": "string"},
                    "hint": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    "sources": _SOURCES,
                },
                "required": ["question", "options", "answer_index", "explanation", "sources"],
            },
        },
    },
    "required": ["title", "questions"],
}


def _quiz_clean(data: dict[str, Any], valid: set[int]) -> dict[str, Any]:
    questions = []
    for q in data.get("questions") or []:
        options = [_text(o) for o in q.get("options") or [] if _text(o)]
        answer = q.get("answer_index")
        if not _text(q.get("question")) or len(options) < 2:
            continue
        if not isinstance(answer, int) or not 0 <= answer < len(options):
            continue
        questions.append(
            {
                "question": _text(q["question"]),
                "options": options,
                "answer_index": answer,
                "explanation": _text(q.get("explanation")),
                "hint": _text(q.get("hint")) or None,
                "difficulty": q.get("difficulty") if q.get("difficulty") in ("easy", "medium", "hard") else None,
                "sources": _refs(q.get("sources"), valid),
            }
        )
    if not questions:
        raise ValueError("the model returned no usable quiz questions")
    return {"title": _text(data.get("title")), "questions": questions}


# --- flashcards ---------------------------------------------------------------

def _cards_instruction(o: dict[str, Any]) -> str:
    n = o.get("count") or 15
    return (
        f"Write exactly {n} flashcards. `front` is a term, a short question or a "
        "prompt (at most 15 words). `back` is the answer in at most 40 words, "
        "phrased so it can be checked from memory. Prefer definitions, mechanisms, "
        "cause and effect, and distinctions between easily confused ideas."
    )


CARDS_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "front": {"type": "string"},
                    "back": {"type": "string"},
                    "sources": _SOURCES,
                },
                "required": ["front", "back", "sources"],
            },
        },
    },
    "required": ["title", "cards"],
}


def _cards_clean(data: dict[str, Any], valid: set[int]) -> dict[str, Any]:
    cards = [
        {"front": _text(c.get("front")), "back": _text(c.get("back")), "sources": _refs(c.get("sources"), valid)}
        for c in data.get("cards") or []
        if _text(c.get("front")) and _text(c.get("back"))
    ]
    if not cards:
        raise ValueError("the model returned no usable flashcards")
    return {"title": _text(data.get("title")), "cards": cards}


# --- mind map -----------------------------------------------------------------

def _mindmap_instruction(o: dict[str, Any]) -> str:
    return (
        "Build a mind map of the material. The root is the overall topic. Give 4 to "
        "7 main branches in teaching order; each branch has 2 to 6 sub-topics, and a "
        "sub-topic may have up to 4 leaf details. Labels are short noun phrases (at "
        "most 6 words). `summary` on a branch is one sentence a student can read when "
        "they expand it."
    )


_LEAF = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
_SUB = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "summary": {"type": "string"},
        "children": {"type": "array", "items": _LEAF},
        "sources": _SOURCES,
    },
    "required": ["label"],
}
MINDMAP_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "root": {"type": "string"},
        "branches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "summary": {"type": "string"},
                    "children": {"type": "array", "items": _SUB},
                    "sources": _SOURCES,
                },
                "required": ["label", "children"],
            },
        },
    },
    "required": ["title", "root", "branches"],
}


def _mindmap_clean(data: dict[str, Any], valid: set[int]) -> dict[str, Any]:
    counter = iter(range(1, 10_000))

    def node(raw: dict[str, Any], depth: int) -> dict[str, Any] | None:
        label = _text(raw.get("label"))
        if not label:
            return None
        kids = [node(c, depth + 1) for c in raw.get("children") or []] if depth < 3 else []
        return {
            "id": f"n{next(counter)}",
            "label": label,
            "summary": _text(raw.get("summary")) or None,
            "sources": _refs(raw.get("sources"), valid),
            "children": [k for k in kids if k],
        }

    branches = [b for b in (node(b, 1) for b in data.get("branches") or []) if b]
    if not branches:
        raise ValueError("the model returned an empty mind map")
    root = {
        "id": "n0",
        "label": _text(data.get("root")) or _text(data.get("title")) or "Topic",
        "summary": None,
        "sources": [],
        "children": branches,
    }
    return {"title": _text(data.get("title")), "root": root}


# --- report -------------------------------------------------------------------

def _report_instruction(o: dict[str, Any]) -> str:
    _, shape = REPORT_FORMATS[o.get("format") or "study_guide"]
    return (
        f"Write {shape} Use Markdown with `##` section headings, short paragraphs, "
        "bullets and, where the sources compare things cleanly, a small table. Put "
        "bracketed source numbers such as [2] at the end of the sentences they "
        "support. Do not add a title heading; `title` holds it."
    )


REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "markdown": {"type": "string"},
    },
    "required": ["title", "markdown"],
}


def _report_clean(data: dict[str, Any], valid: set[int]) -> dict[str, Any]:
    body = _strip_markers(_text(data.get("markdown")), valid)
    if not body:
        raise ValueError("the model returned an empty report")
    return {"title": _text(data.get("title")), "markdown": body}


# --- slides -------------------------------------------------------------------

def _slides_instruction(o: dict[str, Any]) -> str:
    n = o.get("count") or 8
    return (
        f"Create a teaching slide deck of exactly {n} content slides that a student "
        "could present to classmates. Order them as a lesson: motivation, core "
        "ideas, how it works, examples from the sources, summary. Each slide has a "
        "short title and 3 to 5 bullets of at most 14 words each. `speaker_notes` "
        "is what the presenter says, 2 to 4 sentences. `subtitle` is a one-line "
        "framing for the title slide."
    )


SLIDES_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "subtitle": {"type": "string"},
        "slides": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                    "speaker_notes": {"type": "string"},
                    "sources": _SOURCES,
                },
                "required": ["title", "bullets", "sources"],
            },
        },
    },
    "required": ["title", "slides"],
}


def _slides_clean(data: dict[str, Any], valid: set[int]) -> dict[str, Any]:
    slides = []
    for s in data.get("slides") or []:
        bullets = [_text(b) for b in s.get("bullets") or [] if _text(b)]
        if not _text(s.get("title")) or not bullets:
            continue
        slides.append(
            {
                "title": _text(s["title"]),
                "bullets": bullets,
                "speaker_notes": _text(s.get("speaker_notes")) or None,
                "sources": _refs(s.get("sources"), valid),
            }
        )
    if not slides:
        raise ValueError("the model returned no usable slides")
    return {"title": _text(data.get("title")), "subtitle": _text(data.get("subtitle")) or None, "slides": slides}


# --- infographic --------------------------------------------------------------

def _infographic_instruction(o: dict[str, Any]) -> str:
    return (
        "Design the content of a one-page revision infographic. `headline` is a "
        "punchy title, `subtitle` one line. `stats` holds up to 4 figures that the "
        "sources state explicitly (a number or short value plus a label); leave it "
        "empty rather than invent a figure. `sections` holds 4 to 6 panels, each with "
        "a heading, an icon from the allowed list, and 2 to 4 points of at most 12 "
        "words. `process` is an optional ordered sequence (title plus 3 to 6 short "
        "steps) when the material describes a procedure or algorithm; otherwise "
        "leave its steps empty. `takeaway` is the single sentence to remember."
    )


INFOGRAPHIC_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "subtitle": {"type": "string"},
        "stats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"value": {"type": "string"}, "label": {"type": "string"}, "sources": _SOURCES},
                "required": ["value", "label"],
            },
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "icon": {"type": "string", "enum": INFOGRAPHIC_ICONS},
                    "points": {"type": "array", "items": {"type": "string"}},
                    "sources": _SOURCES,
                },
                "required": ["heading", "icon", "points"],
            },
        },
        "process": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "steps": {"type": "array", "items": {"type": "string"}},
            },
        },
        "takeaway": {"type": "string"},
    },
    "required": ["headline", "sections", "takeaway"],
}


def _infographic_clean(data: dict[str, Any], valid: set[int]) -> dict[str, Any]:
    sections = []
    for s in data.get("sections") or []:
        points = [_text(p) for p in s.get("points") or [] if _text(p)]
        if not _text(s.get("heading")) or not points:
            continue
        icon = s.get("icon") if s.get("icon") in INFOGRAPHIC_ICONS else "lightbulb"
        sections.append({"heading": _text(s["heading"]), "icon": icon, "points": points,
                         "sources": _refs(s.get("sources"), valid)})
    if not sections:
        raise ValueError("the model returned an empty infographic")
    stats = [
        {"value": _text(s.get("value")), "label": _text(s.get("label")), "sources": _refs(s.get("sources"), valid)}
        for s in (data.get("stats") or [])[:4]
        if _text(s.get("value")) and _text(s.get("label"))
    ]
    process = data.get("process") or {}
    steps = [_text(x) for x in process.get("steps") or [] if _text(x)]
    return {
        "title": _text(data.get("headline")),
        "subtitle": _text(data.get("subtitle")) or None,
        "stats": stats,
        "sections": sections,
        "process": {"title": _text(process.get("title")) or "How it works", "steps": steps} if len(steps) >= 2 else None,
        "takeaway": _text(data.get("takeaway")) or None,
    }


SPECS: dict[str, Spec] = {
    "quiz": Spec(QUIZ_SCHEMA, _quiz_instruction, _quiz_clean),
    "flashcards": Spec(CARDS_SCHEMA, _cards_instruction, _cards_clean),
    "mindmap": Spec(MINDMAP_SCHEMA, _mindmap_instruction, _mindmap_clean),
    "report": Spec(REPORT_SCHEMA, _report_instruction, _report_clean, max_tokens=24_000),
    "slides": Spec(SLIDES_SCHEMA, _slides_instruction, _slides_clean),
    "infographic": Spec(INFOGRAPHIC_SCHEMA, _infographic_instruction, _infographic_clean, max_tokens=8_000),
}


def default_title(kind: str, options: dict[str, Any], scope_label: str) -> str:
    label = LABELS[kind]
    if kind == "report":
        label = REPORT_FORMATS[options.get("format") or "study_guide"][0]
    return f"{label} · {scope_label}"


def build_prompt(
    kind: str,
    options: dict[str, Any],
    *,
    subject_name: str,
    scope_label: str,
    sources: str,
    syllabus_only: bool,
) -> str:
    focus = (options.get("focus") or "").strip()
    parts = [
        f"Subject: {subject_name}",
        f"Scope: {scope_label}",
    ]
    if focus:
        parts.append(f"Student's focus request: {focus}")
    if syllabus_only:
        parts.append(
            "Note: only syllabus units exist for this scope; nothing has been recorded "
            "or uploaded yet. Stay strictly within what the syllabus text states."
        )
    parts += [f"Sources:\n\n{sources}", "---", f"Task: {SPECS[kind].instruction(options)}"]
    return "\n\n".join(parts)


def referenced(content: dict[str, Any]) -> set[int]:
    """Every source number the cleaned content points at, structured or inline."""
    found: set[int] = set()

    def walk(value: Any, key: str | None = None) -> None:
        if key == "sources" and isinstance(value, list):
            found.update(v for v in value if isinstance(v, int))
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, k)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif isinstance(value, str):
            found.update(int(m) for m in re.findall(r"\[(\d+)\]", value))

    walk(content)
    return found
