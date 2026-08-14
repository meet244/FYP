"""Anthropic client wrapper.

One place that knows how to talk to the model, so syllabus parsing, note
synthesis, query routing, and chat all share retry/streaming/JSON behaviour.
"""
from __future__ import annotations

import functools
import json
import logging
from typing import Any

import anthropic

from app.config import settings

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """Raised when no API credential is configured."""


@functools.lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    # A bare constructor also resolves ANTHROPIC_API_KEY / an `ant auth login`
    # profile, so an unset settings value is not by itself fatal.
    if settings.anthropic_api_key:
        return anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return anthropic.Anthropic()


def complete(
    system: str,
    user: str,
    *,
    max_tokens: int = 16_000,
    effort: str | None = None,
) -> str:
    """Plain text completion. Streams, so long note-synthesis calls don't time out."""
    client = get_client()
    try:
        with client.messages.stream(
            model=settings.llm_model,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": effort or settings.llm_effort},
            messages=[{"role": "user", "content": user}],
        ) as stream:
            message = stream.get_final_message()
    except anthropic.AuthenticationError as exc:  # pragma: no cover
        raise LLMUnavailable("no valid Anthropic credential configured") from exc

    if message.stop_reason == "refusal":
        detail = getattr(message.stop_details, "category", None)
        raise RuntimeError(f"model declined the request (category={detail})")

    return "".join(b.text for b in message.content if b.type == "text").strip()


def complete_json(
    system: str,
    user: str,
    schema: dict[str, Any],
    *,
    max_tokens: int = 16_000,
    effort: str | None = None,
) -> Any:
    """Schema-constrained completion. Returns parsed JSON.

    Uses `output_config.format`, which guarantees the response validates against
    `schema` — no prefill, no regex extraction, no retry-on-parse loop.
    """
    client = get_client()
    try:
        with client.messages.stream(
            model=settings.llm_model,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={
                "effort": effort or settings.llm_effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            messages=[{"role": "user", "content": user}],
        ) as stream:
            message = stream.get_final_message()
    except anthropic.AuthenticationError as exc:  # pragma: no cover
        raise LLMUnavailable("no valid Anthropic credential configured") from exc

    if message.stop_reason == "refusal":
        detail = getattr(message.stop_details, "category", None)
        raise RuntimeError(f"model declined the request (category={detail})")
    if message.stop_reason == "max_tokens":
        raise RuntimeError("response truncated — raise max_tokens or split the input")

    text = next(b.text for b in message.content if b.type == "text")
    return json.loads(text)
