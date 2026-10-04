"""Gemini client wrapper.

One place that knows how to talk to the model, so syllabus parsing, note
synthesis, and chat all share retry/JSON behaviour.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import settings

log = logging.getLogger(__name__)

# Free-tier RPD is per model. 3.6 Flash is 20/day; 3.1 Flash Lite is 500/day.
_FALLBACK_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
)


class LLMUnavailable(RuntimeError):
    """Raised when no API credential is configured."""


class LLMRateLimited(RuntimeError):
    """Raised when every candidate model is out of quota."""

    def __init__(self, message: str, retry_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_s = retry_s


def _api_key() -> str | None:
    for name in (
        settings.gemini_api_key,
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
        os.environ.get("CLASSSCRIBE_GEMINI_API_KEY"),
    ):
        if name and name.strip():
            return name.strip()
    return None


@functools.lru_cache(maxsize=1)
def get_client() -> genai.Client:
    key = _api_key()
    if not key:
        raise LLMUnavailable("no Gemini API key configured")
    return genai.Client(api_key=key)


def candidate_models(primary: str | None = None) -> list[str]:
    first = (primary or settings.llm_model).strip()
    out = [first]
    for name in _FALLBACK_MODELS:
        if name not in out:
            out.append(name)
    return out


def retry_after_s(exc: BaseException) -> float | None:
    match = re.search(r"retry in ([0-9.]+)\s*s", str(exc), re.I)
    return float(match.group(1)) if match else None


def is_rate_limit(exc: BaseException) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = str(exc)
    return code == 429 or "RESOURCE_EXHAUSTED" in text or text.startswith("429")


def _config(
    system: str,
    *,
    max_tokens: int,
    effort: str | None = None,
    schema: dict[str, Any] | None = None,
) -> types.GenerateContentConfig:
    del effort
    kwargs: dict[str, Any] = {
        "system_instruction": system,
        "max_output_tokens": max_tokens,
    }
    if schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_json_schema"] = schema
    return types.GenerateContentConfig(**kwargs)


def _reraise_auth(exc: BaseException) -> None:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = str(exc).lower()
    if code in (401, 403) or "api key" in text or "unauthoriz" in text:
        raise LLMUnavailable("no valid Gemini credential configured") from exc


def _generate(contents: Any, config: types.GenerateContentConfig) -> Any:
    last: BaseException | None = None
    for model in candidate_models():
        try:
            response = get_client().models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            if model != settings.llm_model:
                log.info("Gemini served by %s (primary %s unavailable)", model, settings.llm_model)
            return response
        except genai_errors.APIError as exc:
            _reraise_auth(exc)
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            if code in (404, 503) or is_rate_limit(exc):
                log.warning("Gemini model %s unavailable (%s); trying next", model, code or "error")
                last = exc
                continue
            raise
        except LLMUnavailable:
            raise
    retry = retry_after_s(last) if last else None
    raise LLMRateLimited(
        "Gemini is unavailable (quota or high demand). Try again in a minute.",
        retry,
    ) from last


def describe_image(data: bytes, mime: str) -> str:
    """OCR + caption a lecture slide or diagram. Local file bytes, remote LLM only."""
    response = _generate(
        [
            types.Part.from_bytes(data=data, mime_type=mime or "image/jpeg"),
            (
                "This is a classroom slide, whiteboard photo, or diagram. "
                "1) Transcribe every readable word exactly (OCR). "
                "2) Describe any diagram, chart, or figure in two short paragraphs. "
                "Keep technical terms in English. Do not invent content that is not visible."
            ),
        ],
        types.GenerateContentConfig(max_output_tokens=4_000),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("model returned an empty image description")
    return text


def complete(
    system: str,
    user: str,
    *,
    max_tokens: int = 16_000,
    effort: str | None = None,
) -> str:
    """Plain text completion."""
    response = _generate(user, _config(system, max_tokens=max_tokens, effort=effort))
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("model returned an empty response")
    return text


def complete_json(
    system: str,
    user: str,
    schema: dict[str, Any],
    *,
    max_tokens: int = 16_000,
    effort: str | None = None,
) -> Any:
    """Schema-constrained completion. Returns parsed JSON."""
    response = _generate(
        user,
        _config(system, max_tokens=max_tokens, effort=effort, schema=schema),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("model returned an empty JSON response")
    return json.loads(text)
