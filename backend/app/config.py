"""Central configuration.

Defaults for every SGCD knob are the values frozen in the research
(`research/sgcd/PREREGISTRATION.md`). Override via environment or .env.
"""
from __future__ import annotations

import pathlib

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLASSSCRIBE_", env_file=".env", extra="ignore"
    )

    # --- storage ---
    data_dir: pathlib.Path = pathlib.Path("./data")
    db_url: str = "sqlite:///./data/classscribe.db"

    # --- ASR ---
    asr_backend: str = "mlx"  # "mlx" | "faster-whisper"
    asr_model: str = "mlx-community/whisper-large-v3-turbo"
    asr_language: str | None = "hi"

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
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-5"
    llm_effort: str = "high"

    # --- jobs ---
    worker_threads: int = 1

    @property
    def uploads_dir(self) -> pathlib.Path:
        return self.data_dir / "uploads"

    @property
    def audio_dir(self) -> pathlib.Path:
        return self.data_dir / "audio"

    @property
    def chroma_dir(self) -> pathlib.Path:
        return self.data_dir / "chroma"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.uploads_dir, self.audio_dir, self.chroma_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
