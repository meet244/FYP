"""Research checkpoints and their actual runtime capabilities.

Availability means the runtime can load a model; weights are downloaded lazily.
It is deliberately separate from recognition quality and research validation.
"""
from __future__ import annotations

import importlib.util
import platform
from dataclasses import asdict, dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from app.config import settings


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    family: str
    repository: str
    portable_repository: str | None
    supports_context: bool
    languages: tuple[str, ...]
    default_language: str | None
    warning: str | None = None


MODELS = {
    m.id: m for m in (
        ModelSpec("whisper-turbo", "Whisper large-v3-turbo", "whisper",
                  "mlx-community/whisper-large-v3-turbo", "large-v3-turbo", True,
                  ("hi", "en"), "hi"),
        ModelSpec("whisper-large-v3", "Whisper large-v3", "whisper",
                  "mlx-community/whisper-large-v3-mlx", "large-v3", True,
                  ("hi", "en"), "hi",
                  "Whisper safeguard thresholds were fitted on turbo, not large-v3."),
        ModelSpec("qwen-0.6b", "Qwen3-ASR-0.6B", "qwen",
                  "Qwen/Qwen3-ASR-0.6B-hf", None, True, ("hi", "en"), "hi",
                  "Syllabus conditioning on Qwen is experimental; Whisper SGCD scores do not transfer."),
        ModelSpec("qwen-1.7b", "Qwen3-ASR-1.7B", "qwen",
                  "Qwen/Qwen3-ASR-1.7B-hf", None, True, ("hi", "en"), None,
                  "Forced Hindi produced Devanagari technical terms in the research. Auto language remains an unvalidated alternative."),
        ModelSpec("parakeet-rnnt-1.1b", "Parakeet RNNT 1.1B", "parakeet",
                  "mlx-community/parakeet-rnnt-1.1b", None, False, ("en",), "en",
                  "English only. This checkpoint cannot transcribe Hindi or use syllabus prompts."),
    )
}

METHODS = (
    {"id": "baseline", "name": "Baseline", "families": ["whisper", "qwen", "parakeet"], "experimental": False},
    {"id": "s5", "name": "Domain LM rescoring (S5)", "families": ["qwen"],
     "model_ids": ["qwen-0.6b"], "experimental": False,
     "warning": "Validated on 50 short SLR104 utterances. The frozen LM covers those tutorial domains; new subjects and long recordings need evaluation."},
    {"id": "sgcd", "name": "Syllabus narration (SGCD)", "families": ["whisper", "qwen"], "experimental": True},
)


def apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def runtime_for(spec: ModelSpec) -> tuple[str, str]:
    if spec.family == "whisper":
        return ("mlx", spec.repository) if apple_silicon() else ("faster-whisper", spec.portable_repository)
    return spec.family, spec.repository


def model_catalog() -> list[dict]:
    result = []
    for spec in MODELS.values():
        runtime, repo = runtime_for(spec)
        module = {"mlx": "mlx_whisper", "faster-whisper": "faster_whisper",
                  "qwen": "transformers", "parakeet": "parakeet_mlx"}[runtime]
        available = importlib.util.find_spec(module) is not None
        reason = None if available else f"Install the {runtime} runtime (see backend/README.md)."
        if runtime == "parakeet" and not apple_silicon():
            available, reason = False, "The research Parakeet runtime requires Apple silicon."
        result.append({**asdict(spec), "runtime": runtime, "runtime_repository": repo,
                       "available": available, "unavailable_reason": reason})
    return result


class ASROptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str | None = None
    method: str | None = None
    language: Literal["hi", "en", "auto"] | None = None

    @model_validator(mode="after")
    def validate_choice(self):
        if self.model_id is not None and self.model_id not in MODELS:
            raise ValueError(f"unknown model_id; choose one of {', '.join(MODELS)}")
        if self.method is not None and self.method not in {m["id"] for m in METHODS}:
            raise ValueError("unknown ASR method")
        if self.model_id:
            spec = MODELS[self.model_id]
            if self.method == "s5" and self.model_id != "qwen-0.6b":
                raise ValueError("S5's frozen domain LM is supported only with Qwen3-ASR-0.6B")
            if self.method and spec.family not in next(m["families"] for m in METHODS if m["id"] == self.method):
                raise ValueError(f"{self.method} is not supported by {spec.name}")
            if spec.family == "parakeet" and self.language == "hi":
                raise ValueError("Parakeet RNNT 1.1B supports English only")
        return self


def resolve_options(options: ASROptions | None = None, previous: dict | None = None) -> dict:
    """Freeze every effective option into the durable job payload."""
    requested = options.model_dump(exclude_none=True) if options else {}
    old = previous or {}
    model_id = requested.get("model_id", old.get("model_id"))
    if not model_id:
        model_id = next((s.id for s in MODELS.values() if settings.asr_model in
                         (s.repository, s.portable_repository)), "qwen-0.6b")
    spec = MODELS[model_id]
    runtime, repository = runtime_for(spec)
    if old and model_id == old.get("model_id"):
        runtime, repository = old.get("backend", runtime), old.get("model", repository)
    # Keep legacy environment overrides for default uploads.
    if not requested.get("model_id") and not old:
        runtime, repository = settings.asr_backend, settings.asr_model
    changing_model = "model_id" in requested and requested["model_id"] != old.get("model_id")
    default_method = settings.asr_method
    if not spec.supports_context or (default_method == "s5" and model_id != "qwen-0.6b"):
        default_method = "baseline"
    method = requested.get("method") or (None if changing_model else old.get("method")) or default_method
    language = requested.get("language")
    if language is None:
        language = (None if changing_model else old.get("language")) or spec.default_language or "auto"
        if not requested and not old:
            language = settings.asr_language or "auto"
    ASROptions(model_id=model_id, method=method, language=language)
    frozen = {
            "span_target_s": settings.span_target_s, "span_min_s": settings.span_min_s,
            "span_max_s": settings.span_max_s, "retrieval_k": settings.retrieval_k,
            "prompt_max_tokens": settings.prompt_max_tokens,
            "safeguard_d_logprob": settings.safeguard_d_logprob,
            "safeguard_max_cr": settings.safeguard_max_cr,
            "safeguard_len_ratio": settings.safeguard_len_ratio,
            "max_new_tokens": settings.asr_max_new_tokens,
            "repetition_penalty": settings.asr_repetition_penalty,
    }
    frozen.update(old)
    if method != old.get("method"):
        frozen.update(max_new_tokens=settings.asr_max_new_tokens,
                      repetition_penalty=1.0 if method == "s5" else settings.asr_repetition_penalty)
    if method == "s5":
        from app.asr.rescore import asset_info
        if old.get("method") != "s5" or not old.get("lm_sha256"):
            frozen.update(nbest_beams=5, lm_weight=0.2, length_bonus=0.5, **asset_info())
    frozen.update(model_id=model_id, method=method, language=language, backend=runtime, model=repository,
                  safeguard_enabled=(old.get("safeguard_enabled", settings.safeguard_enabled)
                                     if not changing_model else settings.safeguard_enabled)
                  and model_id == "whisper-turbo")
    return frozen
