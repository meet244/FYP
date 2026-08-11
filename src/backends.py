"""ASR backends. Both return {'text': str, 'segments': [...]}"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DecodeConfig:
    language: Optional[str] = "hi"  # code-switched HI-EN; 'hi' beats 'en' as a prior.
    temperature: float = 0.0        # None => auto-detect
    beam_size: int = 5
    prompt: Optional[str] = None    # syllabus context goes here
    extra: dict = field(default_factory=dict)

    def key(self) -> str:
        import hashlib
        import json
        blob = json.dumps(self.__dict__, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


class LocalWhisper:
    def __init__(self, model_size="large-v3", compute_type="int8", device="cpu",
                 cpu_threads=0):
        from faster_whisper import WhisperModel
        self.name = f"local-{model_size}-{compute_type}"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type,
                                  cpu_threads=cpu_threads)

    def transcribe(self, audio_path: str, cfg: DecodeConfig):
        segments, info = self.model.transcribe(
            audio_path,
            language=cfg.language,
            temperature=cfg.temperature,
            beam_size=cfg.beam_size,
            initial_prompt=cfg.prompt,
            condition_on_previous_text=False,   # utterances are independent
            vad_filter=False,                   # already sentence-segmented
            **cfg.extra,
        )
        segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        return {"text": "".join(s["text"] for s in segs).strip(),
                "segments": segs,
                "language": getattr(info, "language", None),
                "language_prob": getattr(info, "language_probability", None)}


class GroqWhisper:
    def __init__(self, model="whisper-large-v3"):
        from groq import Groq
        self.name = f"groq-{model}"
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def transcribe(self, audio_path: str, cfg: DecodeConfig):
        with open(audio_path, "rb") as fh:
            r = self.client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), fh.read()),
                model=self.model,
                language=cfg.language,
                temperature=cfg.temperature,
                prompt=cfg.prompt or "",
                response_format="verbose_json",
            )
        d = r if isinstance(r, dict) else r.model_dump()
        return {"text": d.get("text", "").strip(),
                "segments": d.get("segments", []),
                "language": d.get("language"),
                "language_prob": None}


def get_backend(name: str, **kw):
    return {"local": LocalWhisper, "groq": GroqWhisper}[name](**kw)
