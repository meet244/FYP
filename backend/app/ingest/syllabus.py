"""Turn an institutional syllabus PDF into SGCD-ready units.

This is the step that decides whether conditioning helps or hurts. A syllabus PDF
is a *terminology list* in prose clothing — bullet points, module tables, comma
separated topics. Fed to the decoder in that shape it reproduces the enumeration
failure mode (K-WER improves, U-WER degrades, aggregate WER +19.39). So ingestion
is not text extraction: it is a rewrite into fluent code-mixed instructional
narration matching the target dual-script convention.

The keyword set is retained per unit, but only as retrieval signal — it is never
rendered into the decoder prompt.
"""
from __future__ import annotations

import logging
import pathlib
import re

from pypdf import PdfReader

from app.llm.client import complete_json

log = logging.getLogger(__name__)

MAX_CHARS = 120_000  # a syllabus that exceeds this is a handbook, not a syllabus

_UNIT_SCHEMA = {
    "type": "object",
    "properties": {
        "course_title": {"type": "string"},
        "units": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short English title of the unit, as it appears in the syllabus.",
                    },
                    "prose": {
                        "type": "string",
                        "description": (
                            "20-35 words of flowing Hindi-English code-mixed instructional "
                            "narration describing this unit, Devanagari for Hindi function "
                            "words and Latin for English technical terms."
                        ),
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Technical terms in canonical Latin-script form.",
                    },
                },
                "required": ["title", "prose", "keywords"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["course_title", "units"],
    "additionalProperties": False,
}

_SYSTEM = """\
You convert university course syllabi into conditioning material for a \
Hindi-English code-switched speech recogniser. You are not summarising and you are \
not translating: you are writing what a lecturer would plausibly *say* while \
teaching each unit.

For every syllabus unit produce a `prose` field obeying all four rules:

1. REGISTER. Flowing instructional narration, the way an Indian lecturer opens a \
topic. Never a list. Never bullet points, never comma-separated terms, never \
"This unit covers X, Y, Z". Write full sentences a person would speak aloud.

2. DUAL SCRIPT. Hindi matrix in Devanagari; English technical terminology in \
Latin script, spelled canonically. Write "hum process scheduling समझेंगे", never \
"hum प्रोसेस शेड्यूलिंग समझेंगे" and never an all-English sentence. This convention \
is the single most important property of the output — the recogniser learns the \
target orthography from it.

3. TERMINAL WEIGHTING. Later words influence decoding more than earlier ones, so \
place the unit's most distinctive technical terms in the SECOND HALF of the \
sentence, and the most distinctive of all at or near the end.

4. LENGTH. 20-35 words. Code-mixed text costs roughly three tokens per word and \
the context channel is capped, so anything longer is truncated away.

Good example:
  "इस lecture में हम operating system के process scheduling के बारे में सीखेंगे। हम \
round robin, priority scheduling और context switch को समझेंगे।"

`keywords` holds the same unit's technical terms in canonical Latin spelling. It \
is used for retrieval only, never shown to the recogniser, so include \
abbreviations and variants freely.

Preserve the syllabus's own unit ordering. If the document has modules with \
sub-topics, emit one unit per module, not one per sub-topic. Aim for 6-14 units.\
"""


def extract_text(path: pathlib.Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # a single unreadable page shouldn't kill ingestion
            log.warning("page extraction failed in %s: %s", path.name, exc)
    text = "\n\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError(
            "no extractable text — the PDF is likely a scan; OCR it before uploading"
        )
    return text[:MAX_CHARS]


def parse_syllabus(path: pathlib.Path, subject_name: str | None = None) -> dict:
    """Return {"course_title": str, "units": [{title, prose, keywords}]}."""
    raw = extract_text(path)
    hint = f"The subject is called {subject_name!r}.\n\n" if subject_name else ""
    result = complete_json(
        system=_SYSTEM,
        user=f"{hint}Syllabus document:\n\n<syllabus>\n{raw}\n</syllabus>",
        schema=_UNIT_SCHEMA,
        max_tokens=16_000,
    )
    if not result.get("units"):
        raise ValueError("no units could be derived from this syllabus")
    return result


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "unit").lower()).strip("-")
    return s[:32] or "unit"
