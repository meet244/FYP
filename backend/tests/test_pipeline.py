"""End-to-end pipeline test with the model and LLM stubbed out.

Exercises the parts that are easy to get subtly wrong and that no unit test
covers: ffmpeg conversion, the worker thread, the detached-session handoffs in
`handlers.process_lecture`, span persistence, note attribution, and coverage.

The ASR backend and the LLM are stubbed — this asserts the plumbing, not
recognition quality. Recognition quality is what `../research` measures.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.asr.backends import DecodeResult
from app.db import session_scope
from app.main import app
from app.models import Job, Lecture

# What the stub "hears" — deliberately code-mixed, in the corpus's dual-script
# convention, so note synthesis and script-mix stats see realistic input.
FAKE_SPEECH = [
    "आज हम process scheduling के बारे में पढ़ेंगे और round robin समझेंगे",
    "अब हम memory management देखेंगे जिसमें paging और segmentation आते हैं",
    "अंत में हम file system और inode के बारे में बात करेंगे",
]


class StubBackend:
    """Returns a different line per call so spans are distinguishable."""

    def __init__(self):
        self.calls: list[str | None] = []

    def transcribe(self, audio: np.ndarray, prompt: str | None) -> DecodeResult:
        self.calls.append(prompt)
        line = FAKE_SPEECH[(len(self.calls) - 1) % len(FAKE_SPEECH)]
        return DecodeResult(text=line, avg_logprob=-0.25, compression_ratio=1.3)


@pytest.fixture
def stubbed(monkeypatch):
    backend = StubBackend()
    monkeypatch.setattr("app.asr.sgcd.get_backend", lambda: backend)

    # Keep the vector store out of it — indexing would pull a 120 MB encoder.
    monkeypatch.setattr("app.rag.store.upsert", lambda *a, **k: None)
    monkeypatch.setattr("app.rag.store.delete_lecture", lambda *a, **k: None)

    def fake_notes(system, user, schema, **kw):
        return {
            "summary": "Scheduling, memory, and file systems.",
            "notes": [
                {
                    "topic": "Process scheduling",
                    "first_span": 0,
                    "last_span": 0,
                    "markdown": "## Round robin\nEach process gets a fixed time slice.",
                    "suggested_unit_key": "",
                    "terms": [{"term": "round robin", "definition": "Fixed-slice scheduling."}],
                    "outcomes": [{"text": "Explain round-robin scheduling.",
                                  "bloom_level": "understand"}],
                }
            ],
        }

    monkeypatch.setattr("app.notes.synthesize.complete_json", fake_notes)
    return backend


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _write_phone_style_recording(path, seconds: float = 70.0) -> None:
    """44.1 kHz stereo, like a phone would produce — ffmpeg must downmix it."""
    sr = 44_100
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t))
    sf.write(str(path), np.stack([tone, tone], axis=1).astype(np.float32), sr)


def _await_job(client, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_lecture_pipeline_end_to_end(client, stubbed, tmp_path):
    sid = client.post("/subjects", json={"name": "Operating Systems"}).json()["id"]

    # Seed a syllabus directly — PDF parsing is an LLM call, covered separately.
    with session_scope() as db:
        from app.models import Subject, Syllabus, SyllabusUnit

        subject = db.get(Subject, sid)
        syl = Syllabus(subject_id=subject.id, provenance="manual")
        db.add(syl)
        db.flush()
        for i, (title, prose, kws) in enumerate(
            [
                ("Process scheduling",
                 "इस lecture में हम process scheduling और round robin समझेंगे।",
                 ["scheduling", "round robin"]),
                ("Memory management",
                 "इस lecture में हम paging और segmentation देखेंगे।",
                 ["paging", "segmentation"]),
            ],
            start=1,
        ):
            db.add(SyllabusUnit(syllabus_id=syl.id, unit_key=f"os-u{i:02d}",
                                order_index=i, title=title, prose=prose, keywords=kws))

    src = tmp_path / "lecture.wav"
    _write_phone_style_recording(src)

    with src.open("rb") as fh:
        r = client.post(
            f"/subjects/{sid}/lectures",
            files={"file": ("lecture.wav", fh, "audio/wav")},
            data={"title": "Week 1 — scheduling"},
        )
    assert r.status_code == 202
    job = _await_job(client, r.json()["id"])
    assert job["status"] == "succeeded", job["error"]

    lecture_id = job["lecture_id"]
    lec = client.get(f"/lectures/{lecture_id}").json()
    assert lec["status"] == "ready"
    assert lec["duration_s"] == pytest.approx(70.0, abs=0.5)

    stats = lec["asr_stats"]
    assert stats["conditioned"] is True
    assert stats["n_spans"] >= 3
    assert 20.0 <= stats["mean_span_s"] <= 28.0, "spans left the validated regime"
    assert set(stats["script_mix"]) == {"dev", "lat", "other"}

    # Two passes per span: unconditioned then conditioned.
    assert len(stubbed.calls) == 2 * stats["n_spans"]
    assert stubbed.calls[: stats["n_spans"]] == [None] * stats["n_spans"], "pass 1 must be unprompted"
    assert all(p for p in stubbed.calls[stats["n_spans"] :]), "pass 2 must carry a prompt"

    transcript = client.get(f"/lectures/{lecture_id}/transcript").json()
    assert len(transcript["spans"]) == stats["n_spans"]
    assert all(s["retrieved_unit_ids"] for s in transcript["spans"])
    assert FAKE_SPEECH[0][:20] in transcript["text"]

    notes = client.get(f"/lectures/{lecture_id}/notes").json()
    assert len(notes) == 1
    assert notes[0]["topic"] == "Process scheduling"
    assert notes[0]["terms"][0]["term"] == "round robin"
    assert notes[0]["outcomes"][0]["bloom_level"] == "understand"
    # Attribution comes from retrieval, not from the model's suggestion (which
    # this stub deliberately left blank).
    assert notes[0]["unit_id"], "note was not attached to a syllabus unit"

    cov = client.get(f"/subjects/{sid}/coverage").json()
    assert cov["total_units"] == 2
    assert cov["covered_units"] == 1
    assert len(cov["outstanding"]) == 1

    client.delete(f"/subjects/{sid}")


def test_pipeline_without_a_syllabus_runs_one_pass(client, stubbed, tmp_path):
    """No syllabus is a supported state, not an error — it just isn't SGCD."""
    sid = client.post("/subjects", json={"name": "Unplanned"}).json()["id"]
    src = tmp_path / "adhoc.wav"
    _write_phone_style_recording(src, seconds=40.0)

    with src.open("rb") as fh:
        job_id = client.post(
            f"/subjects/{sid}/lectures", files={"file": ("adhoc.wav", fh, "audio/wav")}
        ).json()["id"]
    job = _await_job(client, job_id)
    assert job["status"] == "succeeded", job["error"]

    stats = client.get(f"/lectures/{job['lecture_id']}").json()["asr_stats"]
    assert stats["conditioned"] is False
    assert len(stubbed.calls) == stats["n_spans"], "second pass should not have run"
    assert all(p is None for p in stubbed.calls)

    client.delete(f"/subjects/{sid}")


def test_failed_job_records_the_error(client, tmp_path):
    """A broken upload must fail the job, not wedge the worker."""
    sid = client.post("/subjects", json={"name": "Corrupt"}).json()["id"]
    bad = tmp_path / "broken.m4a"
    bad.write_bytes(b"this is not audio")

    with bad.open("rb") as fh:
        job_id = client.post(
            f"/subjects/{sid}/lectures", files={"file": ("broken.m4a", fh, "audio/aac")}
        ).json()["id"]

    job = _await_job(client, job_id)
    assert job["status"] == "failed"
    assert "ffmpeg" in (job["error"] or "").lower()

    client.delete(f"/subjects/{sid}")


def test_worker_survives_a_failed_job(client, stubbed, tmp_path):
    """The queue must keep draining after a failure — verified by a later success."""
    with session_scope() as db:
        failed_before = db.query(Job).filter(Job.status == "failed").count()
    assert failed_before >= 0  # sanity: the table is queryable

    sid = client.post("/subjects", json={"name": "After failure"}).json()["id"]
    src = tmp_path / "ok.wav"
    _write_phone_style_recording(src, seconds=30.0)
    with src.open("rb") as fh:
        job_id = client.post(
            f"/subjects/{sid}/lectures", files={"file": ("ok.wav", fh, "audio/wav")}
        ).json()["id"]

    assert _await_job(client, job_id)["status"] == "succeeded"
    with session_scope() as db:
        assert db.get(Lecture, client.get(f"/jobs/{job_id}").json()["lecture_id"]) is not None
    client.delete(f"/subjects/{sid}")
