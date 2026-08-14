"""Whisper backend adapter.

The paper's runs used `mlx-whisper` on Apple silicon (5.5-15.7x real time on a
consumer laptop). `faster-whisper` is the portable fallback; both expose the
`initial_prompt` slot, which is the only lever SGCD needs — no training, no access
to model internals.

Every backend returns the same DecodeResult so `sgcd.py` never branches on runtime.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from app.config import settings


@dataclass
class DecodeResult:
    text: str
    avg_logprob: float | None
    compression_ratio: float | None


# Frozen decode settings. temperature=0.0 is a scalar, which disables Whisper's
# temperature-fallback loop: deterministic, faster, and it keeps the safeguard's
# logprob comparison meaningful. condition_on_previous_text is off so the prompt
# is the *only* context — otherwise cross-span drift contaminates the experiment.
_DECODE = dict(
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,
    word_timestamps=False,
)


class WhisperBackend:
    def transcribe(self, audio: np.ndarray, prompt: str | None) -> DecodeResult:
        raise NotImplementedError


class MLXBackend(WhisperBackend):
    def __init__(self, model: str, language: str | None):
        import mlx_whisper  # noqa: F401  (fail fast if unavailable)

        self.model = model
        self.language = language

    def transcribe(self, audio: np.ndarray, prompt: str | None) -> DecodeResult:
        import mlx_whisper

        out = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model,
            initial_prompt=prompt,
            language=self.language,
            **_DECODE,
        )
        segs = out.get("segments") or []
        lps = [s["avg_logprob"] for s in segs if s.get("avg_logprob") is not None]
        crs = [s["compression_ratio"] for s in segs if s.get("compression_ratio") is not None]
        return DecodeResult(
            text=(out.get("text") or "").strip(),
            avg_logprob=float(np.mean(lps)) if lps else None,
            compression_ratio=float(max(crs)) if crs else None,
        )


class FasterWhisperBackend(WhisperBackend):
    def __init__(self, model: str, language: str | None):
        from faster_whisper import WhisperModel

        # faster-whisper wants a size or local path, not an mlx-community repo id.
        self.language = language
        self._model = WhisperModel(model, device="auto", compute_type="default")

    def transcribe(self, audio: np.ndarray, prompt: str | None) -> DecodeResult:
        segments, _info = self._model.transcribe(
            audio,
            initial_prompt=prompt,
            language=self.language,
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        texts, lps, crs = [], [], []
        for s in segments:  # generator — must be drained
            texts.append(s.text)
            if s.avg_logprob is not None:
                lps.append(s.avg_logprob)
            if s.compression_ratio is not None:
                crs.append(s.compression_ratio)
        return DecodeResult(
            text="".join(texts).strip(),
            avg_logprob=float(np.mean(lps)) if lps else None,
            compression_ratio=float(max(crs)) if crs else None,
        )


@functools.lru_cache(maxsize=2)
def get_backend(
    backend: str | None = None, model: str | None = None, language: str | None = None
) -> WhisperBackend:
    backend = (backend or settings.asr_backend).lower()
    model = model or settings.asr_model
    language = language if language is not None else settings.asr_language

    if backend in ("mlx", "mlx-whisper"):
        return MLXBackend(model, language)
    if backend in ("faster-whisper", "faster_whisper", "ctranslate2"):
        return FasterWhisperBackend(model, language)
    raise ValueError(f"unknown ASR backend: {backend!r}")
