"""ASR backend and decode configuration (§4).

Whisper is run locally through faster-whisper / CTranslate2 because two of the three
grounding mechanisms need decoder-level access and per-token confidence values that a
hosted API does not expose (§4.1).

`DecodeConfig` carries everything that can change the model's output — model identity,
quantisation, decode parameters, injected context, hint terms — and hashes it into the
cache key (§4.4). Two conditions that differ in any of these fields therefore cannot
collide in the cache, and two that are genuinely identical share the decode.

One runtime constraint from §7.2 is enforced here rather than trusted: faster-whisper
silently ignores `hotwords` when `prefix` is set (get_prompt: `if hotwords and not
prefix`). Setting both is a configuration error and raises.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from common import stable_hash


@dataclass(frozen=True)
class ModelSpec:
    size: str = "large-v3-turbo"
    compute_type: str = "int8"
    device: str = "cpu"
    cpu_threads: int = 0
    num_workers: int = 1

    @property
    def name(self) -> str:
        return f"local-{self.size}-{self.compute_type}"


@dataclass
class DecodeConfig:
    """Everything that determines a decode. All fields enter the cache key."""
    language: Optional[str] = "hi"        # None => auto-detect
    beam_size: int = 5
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    vad_filter: bool = False
    word_timestamps: bool = True
    # --- grounding payloads -------------------------------------------------
    context: Optional[str] = None         # M1: initial_prompt (decoder text context)
    hotwords: Optional[str] = None        # M2: hint phrases (token-level biasing)
    prefix: Optional[str] = None          # unused; kept explicit for the §7.2 guard
    # Identifies the cut audio the decode read. Segment refinement changes the audio
    # without changing any decode parameter, so without this a refreeze of the corpus
    # would silently serve hypotheses decoded from the old cuts.
    audio_version: str = "raw"

    def __post_init__(self):
        if self.hotwords and self.prefix:
            raise ValueError(
                "§7.2: hotwords and prefix are mutually exclusive in faster-whisper — "
                "the prefix silently disables the hints. Never set both.")

    def key(self, model: ModelSpec) -> str:
        payload = {"model": asdict(model), "decode": asdict(self)}
        # cpu_threads / num_workers affect speed, not output; keep them out of the key
        # so a machine change does not invalidate a cache of identical hypotheses.
        payload["model"].pop("cpu_threads", None)
        payload["model"].pop("num_workers", None)
        return stable_hash(payload)

    def describe(self) -> dict:
        d = asdict(self)
        d["context_words"] = len(self.context.split()) if self.context else 0
        d["hotword_terms"] = len(self.hotwords.split(", ")) if self.hotwords else 0
        return d


def _word_records(segment) -> list[dict]:
    words = getattr(segment, "words", None) or []
    return [{"word": w.word, "start": w.start, "end": w.end,
             "prob": getattr(w, "probability", None)} for w in words]


class LocalWhisper:
    """faster-whisper wrapper returning text plus the confidence instrumentation."""

    def __init__(self, spec: ModelSpec = ModelSpec()):
        from faster_whisper import WhisperModel
        self.spec = spec
        self.name = spec.name
        self.model = WhisperModel(
            spec.size, device=spec.device, compute_type=spec.compute_type,
            cpu_threads=spec.cpu_threads, num_workers=spec.num_workers)

    def transcribe(self, audio_path: str, cfg: DecodeConfig) -> dict:
        segments, info = self.model.transcribe(
            audio_path,
            language=cfg.language,
            beam_size=cfg.beam_size,
            temperature=cfg.temperature,
            initial_prompt=cfg.context,
            hotwords=cfg.hotwords,
            prefix=cfg.prefix,
            condition_on_previous_text=cfg.condition_on_previous_text,
            vad_filter=cfg.vad_filter,
            word_timestamps=cfg.word_timestamps,
        )
        segs = []
        for s in segments:
            segs.append({
                "start": s.start, "end": s.end, "text": s.text,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
                "compression_ratio": s.compression_ratio,
                "words": _word_records(s),
            })
        return {
            "text": "".join(s["text"] for s in segs).strip(),
            "segments": segs,
            "language": getattr(info, "language", None),
            "language_prob": getattr(info, "language_probability", None),
        }


def get_backend(name: str, spec: ModelSpec | None = None):
    if name != "local":
        raise ValueError(
            f"backend {name!r} is not available: §4.1 requires local execution for "
            "decoder-level access and per-token confidences.")
    return LocalWhisper(spec or ModelSpec())


def model_spec_from_config(cfg, override_size: str | None = None) -> ModelSpec:
    m = cfg["model"]
    return ModelSpec(
        size=override_size or m["size"],
        compute_type=m["compute_type"],
        device=m["device"],
        cpu_threads=m.get("cpu_threads", 0),
        num_workers=m.get("num_workers", 1),
    )


def decode_config_from_config(cfg, **overrides) -> DecodeConfig:
    d = cfg["decode"]
    lang = d.get("language")
    base = dict(
        language=None if lang in (None, "auto", "none") else lang,
        beam_size=d["beam_size"],
        temperature=d["temperature"],
        condition_on_previous_text=d["condition_on_previous_text"],
        vad_filter=d["vad_filter"],
        word_timestamps=d.get("word_timestamps", True),
        audio_version=cfg.get_path("data.audio_version", "raw"),
    )
    base.update(overrides)
    return DecodeConfig(**base)
