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
import json
import hashlib
import pathlib
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import soundfile as sf

from app.asr import prompts, retrieve, segment
from app.asr.backends import DecodeResult, get_backend, inference_lock
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


def _should_revert(conditioned: DecodeResult, baseline: DecodeResult, config: dict | None = None) -> bool:
    """Stability safeguard (Section III-D).

    Conditioning has two opposing effects: it repairs degenerate repetition in
    unconditioned output, and it also *induces* it, when generation continues the
    register of the supplied narration instead of terminating. Revert when any of
    the three signals fires.

    Thresholds were fitted on a development partition for whisper-large-v3-turbo
    and do not transfer across scales — applied unchanged to a smaller checkpoint
    they fired on 46% of utterances and made results worse. Refit per model.
    """
    config = config or settings.model_dump()
    # Generic loop/empty detection is separate from Whisper's fitted thresholds.
    if not conditioned.text.strip() and baseline.text.strip():
        return True
    if _is_repetition_loop(conditioned.text) and not _is_repetition_loop(baseline.text):
        return True
    if not config["safeguard_enabled"]:
        return False

    if conditioned.avg_logprob is not None and baseline.avg_logprob is not None:
        if conditioned.avg_logprob < baseline.avg_logprob - config["safeguard_d_logprob"]:
            return True
    if conditioned.compression_ratio is not None:
        if conditioned.compression_ratio > config["safeguard_max_cr"]:
            return True
    base_words = len(baseline.text.split())
    if base_words and len(conditioned.text.split()) > config["safeguard_len_ratio"] * base_words:
        return True
    return False


def _is_repetition_loop(text: str) -> bool:
    """Catch the Whisper collapse where one or two tokens fill the span.

    temperature=0 disables Whisper's own repetition-repair loop, so this fires
    on unconditioned Hindi-forced decodes of English or noisy audio.
    """
    words = text.split()
    if len(words) < 12:
        return False
    return len(set(words)) / len(words) < 0.18


def transcribe(
    audio_path: str,
    units: Sequence = (),
    syllabus_id: str | None = None,
    progress: ProgressCb | None = None,
    config: dict | None = None,
    checkpoint_path: str | None = None,
) -> TranscriptionResult:
    # Keep the model resident for the entire recording; other workers may ingest
    # files while decoding, but cannot evict a model that is currently in use.
    with inference_lock:
        return _transcribe(audio_path, units, syllabus_id, progress, config, checkpoint_path)


def _transcribe(audio_path, units, syllabus_id, progress, config, checkpoint_path):
    """Transcribe one lecture.

    `units` are the subject's SyllabusUnit rows. With none supplied the pipeline
    runs a single unconditioned pass — the paper's contentless register control
    (condition G) was only measured on short utterances, where it *regressed*
    aggregate WER, so it is not applied unvalidated at span length.
    """
    t0 = time.time()
    from app.asr.catalog import resolve_options

    config = config or resolve_options()
    audio, sr = load_audio(audio_path)
    if sr != segment.SAMPLE_RATE:
        raise ValueError(f"expected {segment.SAMPLE_RATE} Hz mono, got {sr} Hz")

    duration_s = len(audio) / sr
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("audio must contain finite, nonempty samples")
    spans = segment.plan_spans(
        audio,
        sample_rate=sr,
        target_s=config["span_target_s"],
        min_s=config["span_min_s"],
        max_s=config["span_max_s"],
    )
    backend = get_backend(config["backend"], config["model"], config["language"], config=config)
    index = (
        retrieve.get_index(syllabus_id, units, k=config["retrieval_k"])
        if syllabus_id and units and config["method"] == "sgcd" and getattr(backend, "supports_context", True)
        else None
    )

    # A checkpoint is scoped to audio + configuration + exact syllabus content.
    # Requeues/retries reuse paid-for spans; editing a syllabus invalidates them.
    file = pathlib.Path(audio_path)
    fingerprint = hashlib.sha256(json.dumps({
        "audio": [str(file.resolve()), file.stat().st_size, file.stat().st_mtime_ns],
        "config": config, "units": [{"id": u.id, "title": u.title, "prose": u.prose,
                                     "keywords": u.keywords} for u in units],
    }, sort_keys=True).encode()).hexdigest()
    checkpoint = pathlib.Path(checkpoint_path) if checkpoint_path else None
    cache = {"fingerprint": fingerprint, "baselines": [], "results": []}
    if checkpoint and checkpoint.exists():
        try:
            previous = json.loads(checkpoint.read_text())
            if previous.get("fingerprint") == fingerprint:
                cache = previous
        except (ValueError, OSError):
            log.warning("ignoring unreadable ASR checkpoint %s", checkpoint)

    resumed_spans = len(cache["baselines"])

    def save():
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temp = checkpoint.with_suffix(".tmp")
            temp.write_text(json.dumps(cache, ensure_ascii=False))
            temp.replace(checkpoint)

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    # ---- pass 1: unconditioned. Doubles as the retrieval query. ----
    # Single-pass modes have no second decode. Reserve only a small fraction for
    # assembling results instead of presenting a nearly finished decode as 45%.
    first_weight = 0.45 if index is not None else 0.90
    remaining_weight = 0.95 - first_weight
    baselines: list[DecodeResult] = []
    for i, span in enumerate(spans):
        report(first_weight * i / len(spans), f"first pass {i + 1}/{len(spans)}")
        if i < len(cache["baselines"]):
            baselines.append(DecodeResult(**cache["baselines"][i]))
            continue
        clip = segment.slice_audio(audio, span, sr)
        hyp = (backend.transcribe_rescored(clip, config) if config["method"] == "s5"
               else backend.transcribe(clip, None))
        if config["method"] != "s5" and _is_repetition_loop(hyp.text):
            log.warning(
                "span %d looks like a repetition loop; retrying with language auto-detect",
                i,
            )
            retry = backend.transcribe(clip, None, language=None)
            if not _is_repetition_loop(retry.text):
                hyp = retry
        baselines.append(hyp)
        cache["baselines"].append(asdict(hyp))
        save()
        report(first_weight * (i + 1) / len(spans), f"first pass {i + 1}/{len(spans)}")

    # ---- pass 2: conditioned, per span ----
    results: list[SpanResult] = []
    n_fallback = 0
    for i, (span, base) in enumerate(zip(spans, baselines)):
        report(first_weight + remaining_weight * i / len(spans), f"span {i + 1}/{len(spans)}")
        if i < len(cache["results"]):
            cached = SpanResult(**cache["results"][i])
            results.append(cached)
            n_fallback += int(cached.safeguard_fallback)
            continue
        if index is None:
            results.append(
                SpanResult(
                    index=span.index,
                    start_s=span.start_s,
                    end_s=span.end_s,
                    text=base.text,
                    baseline_text=base.baseline_text or base.text,
                    avg_logprob=base.avg_logprob,
                    compression_ratio=base.compression_ratio,
                )
            )
            cache["results"].append(asdict(results[-1]))
            save()
            report(first_weight + remaining_weight * (i + 1) / len(spans), f"span {i + 1}/{len(spans)}")
            continue

        picked = index.query(base.text)  # ascending relevance
        prompt = prompts.build_prompt(picked, config["prompt_max_tokens"])
        cond = backend.transcribe(segment.slice_audio(audio, span, sr), prompt)

        reverted = _should_revert(cond, base, config)
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
        cache["results"].append(asdict(results[-1]))
        save()
        report(first_weight + remaining_weight * (i + 1) / len(spans), f"span {i + 1}/{len(spans)}")

    elapsed = time.time() - t0
    full_text = " ".join(r.text for r in results)
    stats = {
        "n_spans": len(results),
        "mean_span_s": round(duration_s / max(1, len(results)), 2),
        "conditioned": index is not None,
        "model_id": config["model_id"],
        "backend": config["backend"],
        "model": config["model"],
        "language": config["language"],
        "method": config["method"],
        "rescored": config["method"] == "s5",
        "lm_sha256": config.get("lm_sha256") if config["method"] == "s5" else None,
        "calibrated_safeguard": config["safeguard_enabled"],
        "safeguard_fallbacks": n_fallback,
        "safeguard_fallback_rate": round(n_fallback / max(1, len(results)), 4),
        "elapsed_s": round(elapsed, 1),
        "realtime_factor": round(duration_s / elapsed, 2) if elapsed else None,
        "rtf": round(elapsed / duration_s, 4) if duration_s else None,
        "resumed_spans": resumed_spans,
        # Not a WER proxy — a descriptive check that the dual-script convention
        # survived. A collapse in `lat` is the Devanagari-transliteration failure.
        "script_mix": {k: round(v, 4) for k, v in script_mix(full_text).items()},
    }
    log.info("transcribed %s: %s", audio_path, stats)
    return TranscriptionResult(spans=results, duration_s=duration_s, stats=stats)
