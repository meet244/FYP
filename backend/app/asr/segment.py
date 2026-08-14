"""Partition lecture audio into spans bounded by the 30 s encoder receptive field.

This is not a detail. Section V-D: at the corpus's native 5.7 s utterances,
conditioning *regresses* (+5.11 WER) and needs the stability safeguard to break
even; at 26.2 s spans it gives -6.23 WER unaided. The paper frames this as a
deployment constraint rather than a limit on validity, precisely because a
classroom system controls its own segmentation. This module is where that control
is exercised, so the default target is 25 s and the ceiling is 28 s.

Boundaries are placed at the quietest point inside the admissible window so a cut
rarely lands mid-word. Energy-based rather than a neural VAD: one fewer model to
download, and the choice only affects where a boundary sits within a ~5 s slack
region, not whether conditioning works.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
_FRAME_S = 0.03


@dataclass(frozen=True)
class Span:
    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _frame_rms(audio: np.ndarray, frame_len: int) -> np.ndarray:
    n = len(audio) // frame_len
    if n == 0:
        return np.zeros(1, dtype=np.float32)
    trimmed = audio[: n * frame_len].reshape(n, frame_len)
    return np.sqrt((trimmed.astype(np.float32) ** 2).mean(axis=1) + 1e-12)


def plan_spans(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    target_s: float = 25.0,
    min_s: float = 8.0,
    max_s: float = 28.0,
) -> list[Span]:
    """Greedy left-to-right split, cutting at the quietest frame near `target_s`."""
    total_s = len(audio) / sample_rate
    if total_s <= max_s:
        return [Span(0, 0.0, total_s)]

    frame_len = max(1, int(_FRAME_S * sample_rate))
    rms = _frame_rms(audio, frame_len)
    frames_per_s = sample_rate / frame_len

    spans: list[Span] = []
    cursor = 0.0
    while cursor < total_s:
        remaining = total_s - cursor
        if remaining <= max_s:
            spans.append(Span(len(spans), cursor, total_s))
            break

        # Admissible boundary window around the target, clipped to [min, max].
        lo = cursor + max(min_s, target_s - 5.0)
        hi = cursor + min(max_s, target_s + 3.0)
        lo_f, hi_f = int(lo * frames_per_s), int(hi * frames_per_s)
        lo_f = max(0, min(lo_f, len(rms) - 1))
        hi_f = max(lo_f + 1, min(hi_f, len(rms)))

        window = rms[lo_f:hi_f]
        cut_f = lo_f + int(np.argmin(window)) if len(window) else hi_f
        cut_s = min(cut_f / frames_per_s, cursor + max_s)

        # Degenerate window (e.g. constant energy) — fall back to a hard cut.
        if cut_s - cursor < min_s:
            cut_s = cursor + target_s

        spans.append(Span(len(spans), cursor, cut_s))
        cursor = cut_s

    return spans


def slice_audio(audio: np.ndarray, span: Span, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    a = int(span.start_s * sample_rate)
    b = int(span.end_s * sample_rate)
    return audio[a:b]
