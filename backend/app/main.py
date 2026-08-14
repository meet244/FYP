"""ClassScribe backend.

Syllabus-Grounded Contextual Decoding for code-switched classroom lectures,
served as a REST API for a NotebookLM-style chat frontend.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, jobs, lectures, subjects
from app.config import settings
from app.db import init_db
from app.ingest.audio import ffmpeg_available
from app.jobs import handlers  # noqa: F401 — registers job handlers
from app.jobs import queue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("classscribe")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    queue.start()
    if not ffmpeg_available():
        log.warning("ffmpeg not found on PATH — lecture uploads will fail until it is installed")
    log.info(
        "ready | asr=%s/%s | span=%.0fs k=%d | safeguard=%s | llm=%s",
        settings.asr_backend,
        settings.asr_model,
        settings.span_target_s,
        settings.retrieval_k,
        settings.safeguard_enabled,
        settings.llm_model,
    )
    yield


app = FastAPI(
    title="ClassScribe",
    version="0.1.0",
    description=(
        "Records classroom lectures, transcribes them with Syllabus-Grounded "
        "Contextual Decoding, and answers questions over the resulting corpus."
    ),
    lifespan=lifespan,
)

# Wide open by default — this is a single-machine deployment whose whole point is
# that audio stays local. Narrow this before exposing the service on a network.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(subjects.router)
app.include_router(lectures.router)
app.include_router(chat.router)
app.include_router(jobs.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {
        "status": "ok",
        "ffmpeg": ffmpeg_available(),
        "asr": {
            "backend": settings.asr_backend,
            "model": settings.asr_model,
            "language": settings.asr_language,
        },
        "sgcd": {
            "span_target_s": settings.span_target_s,
            "retrieval_k": settings.retrieval_k,
            "prompt_max_tokens": settings.prompt_max_tokens,
            "safeguard_enabled": settings.safeguard_enabled,
        },
        "llm_model": settings.llm_model,
    }
