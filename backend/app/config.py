"""Central configuration.

Defaults for every SGCD knob are the values frozen in the research
(`research/sgcd/PREREGISTRATION.md`). Override via environment or .env.
"""
from __future__ import annotations

import pathlib

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLASSSCRIBE_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    # --- storage ---
    data_dir: pathlib.Path = pathlib.Path("./data")
    db_url: str = "sqlite:///./data/classscribe.db"

    # --- ASR ---
    asr_backend: str = "qwen"  # qwen | mlx | faster-whisper | parakeet
    asr_model: str = "Qwen/Qwen3-ASR-0.6B-hf"
    asr_language: str | None = "hi"
    asr_method: str = "s5"
    asr_max_new_tokens: int = Field(default=512, ge=64, le=2048)
    asr_repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    max_upload_mb: int = Field(default=1024, ge=1)
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # --- SGCD, frozen from the DEV sweep ---
    # Span duration is load-bearing: at ~5.7 s conditioning regresses (+5.11 WER),
    # at ~26 s it helps (-6.23). 25 s sits inside the 30 s encoder receptive field.
    span_target_s: float = 25.0
    span_min_s: float = 8.0
    span_max_s: float = 28.0
    retrieval_k: int = 3
    prompt_max_tokens: int = 200

    safeguard_enabled: bool = True
    safeguard_d_logprob: float = 0.25
    safeguard_max_cr: float = 2.0
    safeguard_len_ratio: float = 1.5

    # --- embeddings / vector store ---
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    retrieval_top_k: int = 12

    # --- LLM ---
    # Accepts GEMINI_API_KEY, GOOGLE_API_KEY, or CLASSSCRIBE_GEMINI_API_KEY.
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "CLASSSCRIBE_GEMINI_API_KEY",
        ),
    )
    llm_model: str = "gemini-3.1-flash-lite"
    llm_effort: str = "high"

    # --- jobs ---
    worker_threads: int = Field(default=1, ge=1, le=4)

    @model_validator(mode="after")
    def validate_spans(self):
        if not 0 < self.span_min_s <= self.span_target_s <= self.span_max_s <= 30:
            raise ValueError("span durations must satisfy 0 < min <= target <= max <= 30")
        if not 1 <= self.retrieval_k <= 20 or not 1 <= self.prompt_max_tokens <= 200:
            raise ValueError("retrieval_k must be 1–20 and prompt_max_tokens 1–200")
        return self

    @property
    def uploads_dir(self) -> pathlib.Path:
        return self.data_dir / "uploads"

    @property
    def audio_dir(self) -> pathlib.Path:
        return self.data_dir / "audio"

    @property
    def chroma_dir(self) -> pathlib.Path:
        return self.data_dir / "chroma"

    @property
    def materials_dir(self) -> pathlib.Path:
        return self.data_dir / "materials"

    def ensure_dirs(self) -> None:
        for p in (
            self.data_dir,
            self.uploads_dir,
            self.audio_dir,
            self.chroma_dir,
            self.materials_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
