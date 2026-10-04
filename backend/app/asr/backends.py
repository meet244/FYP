"""Whisper backend adapter.

The paper's runs used `mlx-whisper` on Apple silicon (5.5-15.7x real time on a
consumer laptop). `faster-whisper` is the portable fallback; both expose the
`initial_prompt` slot, which is the only lever SGCD needs — no training, no access
to model internals.

Every backend returns the same DecodeResult so `sgcd.py` never branches on runtime.
"""
from __future__ import annotations

import gc
import threading
import tempfile
import sys
import importlib
from dataclasses import dataclass

import numpy as np

from app.config import settings


@dataclass
class DecodeResult:
    text: str
    avg_logprob: float | None
    compression_ratio: float | None
    baseline_text: str | None = None


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
    supports_context = True

    def transcribe(
        self,
        audio: np.ndarray,
        prompt: str | None,
        *,
        language: str | None | object = ...,
    ) -> DecodeResult:
        raise NotImplementedError


class MLXBackend(WhisperBackend):
    def __init__(self, model: str, language: str | None):
        import mlx_whisper  # noqa: F401  (fail fast if unavailable)

        self.model = model
        self.language = language

    def transcribe(
        self,
        audio: np.ndarray,
        prompt: str | None,
        *,
        language: str | None | object = ...,
    ) -> DecodeResult:
        import mlx_whisper

        lang = self.language if language is ... else language
        out = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model,
            initial_prompt=prompt,
            language=lang,
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

    def transcribe(
        self,
        audio: np.ndarray,
        prompt: str | None,
        *,
        language: str | None | object = ...,
    ) -> DecodeResult:
        lang = self.language if language is ... else language
        segments, _info = self._model.transcribe(
            audio,
            initial_prompt=prompt,
            language=lang,
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


class QwenBackend(WhisperBackend):
    """Native Transformers Qwen3-ASR, using the public processor API.

    One bounded, deterministic generation per clip. Decoder limits and repetition
    penalty address the runaways recorded in the FYRP study. They change inference
    settings, so runtime output is not claimed to reproduce research WER.
    """

    def __init__(self, model: str, language: str | None, *, max_new_tokens=None, repetition_penalty=None):
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM

        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.language = language
        self.max_new_tokens = max_new_tokens or settings.asr_max_new_tokens
        self.repetition_penalty = repetition_penalty or settings.asr_repetition_penalty
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForMultimodalLM.from_pretrained(model, dtype=dtype).to(device).eval()

    def transcribe(self, audio, prompt, *, language=...):
        import torch

        lang = self.language if language is ... else language
        inputs = self.processor.apply_transcription_request(
            audio=audio, prompt=prompt, language=lang,
        ).to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, repetition_penalty=self.repetition_penalty,
                num_beams=1, num_return_sequences=1, use_cache=True,
                output_scores=False, return_dict_in_generate=False,
            )
        generated = output[:, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(generated, return_format="transcription_only")[0].strip()
        return DecodeResult(text, None, None)

    def transcribe_rescored(self, audio, config):
        """S5, without stacking syllabus prompting or shallow fusion.

        The retained comparison is the unrescored top beam, not S0 greedy.
        """
        import torch
        from app.asr.rescore import DomainLM, select_candidate

        signature = (config["lm_path"], config["lm_sha256"])
        if getattr(self, "_lm_signature", None) != signature:
            self._lm = DomainLM(self.processor.tokenizer, signature[0], signature[1])
            self._lm_signature = signature
        inputs = self.processor.apply_transcription_request(
            audio=audio, language=self.language, prompt=None,
        ).to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=config["max_new_tokens"], do_sample=False,
                repetition_penalty=config["repetition_penalty"], num_beams=config["nbest_beams"],
                num_return_sequences=config["nbest_beams"], length_penalty=1.0,
                output_scores=True, return_dict_in_generate=True,
            )
        scores = _beam_acoustic_scores(output, length_penalty=1.0)
        texts = self.processor.decode(output.sequences[:, inputs["input_ids"].shape[1]:],
                                      return_format="transcription_only")
        candidates = []
        for text, score in zip(texts, scores):
            candidates.append({"text": text.strip(), "am": float(score.item())})
        chosen = select_candidate(candidates, self._lm, config["lm_weight"], config["length_bonus"])
        return DecodeResult(chosen["text"], None, None, baseline_text=candidates[0]["text"])


def _beam_acoustic_scores(output, *, length_penalty: float):
    """Undo beam length normalization without gathering vocabulary-sized scores.

    Transformers' sequence scores are cumulative token log probabilities divided
    by the generated length raised to ``length_penalty``. Beam ancestry counts
    include EOS and exclude padding and the audio prompt. Do the small length
    reduction on CPU: the MPS transition-score gather can fail for short beams
    after a long recording. Limit ancestry to actual decoding steps so unused
    entries cannot inflate the length of an early-ending beam.
    """
    import torch

    ancestry = output.beam_indices.detach().cpu()[:, :len(output.scores)]
    lengths = (ancestry >= 0).sum(dim=1)
    scores = output.sequences_scores.detach().float().cpu()
    if (lengths == 0).any() or not torch.isfinite(scores).all():
        raise RuntimeError("S5 decoder returned invalid beam scores")
    return scores * lengths.to(scores.dtype).pow(length_penalty)


class ParakeetBackend(WhisperBackend):
    supports_context = False

    def __init__(self, model: str, language: str | None):
        from parakeet_mlx import from_pretrained

        self.model = from_pretrained(model)

    def transcribe(self, audio, prompt, *, language=...):
        import soundfile as sf

        if prompt:
            raise ValueError("Parakeet RNNT has no context-prompt channel")
        # The public MLX API accepts a filename. Always remove temporary audio.
        with tempfile.TemporaryDirectory(prefix="classscribe-parakeet-") as temp:
            path = f"{temp}/span.wav"
            sf.write(path, audio, 16000)
            result = self.model.transcribe(path)
        return DecodeResult((result.text or "").strip(), None, None)


# Only one checkpoint is resident, and an inference lease serializes ASR workers.
# Holding two Whisper/Qwen models on an 8 GB Mac causes swap or out-of-memory.
_cache_key = None
_cached_backend = None
_cache_lock = threading.RLock()
inference_lock = threading.RLock()


def clear_backend() -> None:
    global _cache_key, _cached_backend
    _cached_backend = None
    _cache_key = None
    # mlx-whisper owns a second cache internally; releasing our adapter is not enough.
    if "mlx_whisper" in sys.modules:
        holder = importlib.import_module("mlx_whisper.transcribe").ModelHolder
        holder.model, holder.model_path = None, None
    gc.collect()
    if "mlx.core" in sys.modules:
        sys.modules["mlx.core"].clear_cache()
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()


def get_backend(
    backend: str | None = None, model: str | None = None, language: str | None | object = ...,
    *, config: dict | None = None,
) -> WhisperBackend:
    global _cache_key, _cached_backend
    backend = (backend or settings.asr_backend).lower()
    model = model or settings.asr_model
    language = settings.asr_language if language is ... else language
    if language == "auto":
        language = None
    options = {k: config[k] for k in ("max_new_tokens", "repetition_penalty") if config and k in config} if backend == "qwen" else {}
    key = (backend, model, language, tuple(options.items()))
    with _cache_lock:
        if key == _cache_key:
            return _cached_backend
        clear_backend()
        cls = {"mlx": MLXBackend, "mlx-whisper": MLXBackend,
               "faster-whisper": FasterWhisperBackend, "faster_whisper": FasterWhisperBackend,
               "ctranslate2": FasterWhisperBackend, "qwen": QwenBackend,
               "parakeet": ParakeetBackend}.get(backend)
        if cls is None:
            raise ValueError(f"unknown ASR backend: {backend!r}")
        try:
            _cached_backend = cls(model, language, **options)
        except ImportError as exc:
            raise RuntimeError(f"ASR runtime {backend} is unavailable; install its requirements. {exc}") from exc
        _cache_key = key
        return _cached_backend
