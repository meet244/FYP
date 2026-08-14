"""Syllabus-Grounded Contextual Decoding — the production pipeline.

    audio -> ~25 s spans
          -> pass 1, unconditioned                    (also the retrieval query)
          -> char n-gram TF-IDF over the syllabus     (k=3, ascending relevance)
          -> pass 2, conditioned on code-mixed prose  (<=200 tokens, left-truncated)
          -> stability safeguard                      (accept or revert to pass 1)

Cost is one supplementary decode per span: retrieval reuses the first-pass
hypothesis the baseline produces anyway. No training, no annotation, no auxiliary
acoustic model.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from app.asr import prompts, retrieve, segment
from app.asr.backends import DecodeResult, get_backend
from app.asr.normalize import script_mix
from app.config import settings

log = logging.getLogger(__name__)

ProgressCb = Callable[[float, str], None]


@dataclass
class SpanResult:
    index: int
    start_s: float
    end_s: float
    text: str
    baseline_text: str
    retrieved_unit_ids: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    safeguard_fallback: bool = False


@dataclass
class TranscriptionResult:
    spans: list[SpanResult]
    duration_s: float
    stats: dict


def load_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _should_revert(conditioned: DecodeResult, baseline: DecodeResult) -> bool:
    """Stability safeguard (Section III-D).

    Conditioning has two opposing effects: it repairs degenerate repetition in
    unconditioned output, and it also *induces* it, when generation continues the
    register of the supplied narration instead of terminating. Revert when any of
    the three signals fires.

    Thresholds were fitted on a development partition for whisper-large-v3-turbo
    and do not transfer across scales — applied unchanged to a smaller checkpoint
    they fired on 46% of utterances and made results worse. Refit per model.
    """
    if not settings.safeguard_enabled:
        return False

    if conditioned.avg_logprob is not None and baseline.avg_logprob is not None:
        if conditioned.avg_logprob < baseline.avg_logprob - settings.safeguard_d_logprob:
            return True
    if conditioned.compression_ratio is not None:
        if conditioned.compression_ratio > settings.safeguard_max_cr:
            return True
    base_words = len(baseline.text.split())
    if base_words and len(conditioned.text.split()) > settings.safeguard_len_ratio * base_words:
        return True
    return False


def transcribe(
    audio_path: str,
    units: Sequence = (),
    syllabus_id: str | None = None,
    progress: ProgressCb | None = None,
) -> TranscriptionResult:
    """Transcribe one lecture.

    `units` are the subject's SyllabusUnit rows. With none supplied the pipeline
    runs a single unconditioned pass — the paper's contentless register control
    (condition G) was only measured on short utterances, where it *regressed*
    aggregate WER, so it is not applied unvalidated at span length.
    """
    t0 = time.time()
    audio, sr = load_audio(audio_path)
    if sr != segment.SAMPLE_RATE:
        raise ValueError(f"expected {segment.SAMPLE_RATE} Hz mono, got {sr} Hz")

    duration_s = len(audio) / sr
    spans = segment.plan_spans(
        audio,
        sample_rate=sr,
        target_s=settings.span_target_s,
        min_s=settings.span_min_s,
        max_s=settings.span_max_s,
    )
    backend = get_backend()
    index = (
        retrieve.get_index(syllabus_id, units, k=settings.retrieval_k)
        if syllabus_id and units
        else None
    )

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    # ---- pass 1: unconditioned. Doubles as the retrieval query. ----
    baselines: list[DecodeResult] = []
    for i, span in enumerate(spans):
        baselines.append(backend.transcribe(segment.slice_audio(audio, span, sr), None))
        report(0.45 * (i + 1) / len(spans), f"first pass {i + 1}/{len(spans)}")

    # ---- pass 2: conditioned, per span ----
    results: list[SpanResult] = []
    n_fallback = 0
    for i, (span, base) in enumerate(zip(spans, baselines)):
        if index is None:
            results.append(
                SpanResult(
                    index=span.index,
                    start_s=span.start_s,
                    end_s=span.end_s,
                    text=base.text,
                    baseline_text=base.text,
                    avg_logprob=base.avg_logprob,
                    compression_ratio=base.compression_ratio,
                )
            )
            report(0.45 + 0.5 * (i + 1) / len(spans), f"span {i + 1}/{len(spans)}")
            continue

        picked = index.query(base.text)  # ascending relevance
        prompt = prompts.build_prompt(picked, settings.prompt_max_tokens)
        cond = backend.transcribe(segment.slice_audio(audio, span, sr), prompt)

        reverted = _should_revert(cond, base)
        n_fallback += int(reverted)
        results.append(
            SpanResult(
                index=span.index,
                start_s=span.start_s,
                end_s=span.end_s,
                text=base.text if reverted else cond.text,
                baseline_text=base.text,
                retrieved_unit_ids=[u.id for u in picked],
                prompt_tokens=prompts.n_tokens(prompt),
                avg_logprob=cond.avg_logprob,
                compression_ratio=cond.compression_ratio,
                safeguard_fallback=reverted,
            )
        )
        report(0.45 + 0.5 * (i + 1) / len(spans), f"span {i + 1}/{len(spans)}")

    elapsed = time.time() - t0
    full_text = " ".join(r.text for r in results)
    stats = {
        "n_spans": len(results),
        "mean_span_s": round(duration_s / max(1, len(results)), 2),
        "conditioned": index is not None,
        "safeguard_fallbacks": n_fallback,
        "safeguard_fallback_rate": round(n_fallback / max(1, len(results)), 4),
        "elapsed_s": round(elapsed, 1),
        "realtime_factor": round(duration_s / elapsed, 2) if elapsed else None,
        # Not a WER proxy — a descriptive check that the dual-script convention
        # survived. A collapse in `lat` is the Devanagari-transliteration failure.
        "script_mix": {k: round(v, 4) for k, v in script_mix(full_text).items()},
    }
    log.info("transcribed %s: %s", audio_path, stats)
    return TranscriptionResult(spans=results, duration_s=duration_s, stats=stats)
