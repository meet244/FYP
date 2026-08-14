"""Normalise phone recordings to what the encoder expects: 16 kHz mono PCM.

Phone recorders emit m4a/aac/opus at 44.1 or 48 kHz stereo. Everything downstream
(segmentation, both decode passes) assumes 16 kHz mono float, so the conversion
happens once, here, at upload time rather than per span.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import soundfile as sf

SAMPLE_RATE = 16_000


class AudioError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def to_wav16k_mono(src: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
    """Transcode `src` to 16 kHz mono WAV at `dst`."""
    if not ffmpeg_available():
        raise AudioError("ffmpeg not found on PATH — required to decode phone recordings")

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-i", str(src),
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-vn",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise AudioError("ffmpeg failed: " + " | ".join(tail))
    return dst


def duration_s(path: pathlib.Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)
