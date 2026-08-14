"""Smoke tests — no audio, no API key, no model download required.

Covers the deterministic parts of SGCD (segmentation, retrieval, prompt
construction, the safeguard) plus the API surface. The parts that need a real
recording or a real credential are exercised by scripts/demo.py instead.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.asr import prompts, retrieve, segment
from app.asr.backends import DecodeResult
from app.asr.normalize import normalize, script_mix
from app.asr.sgcd import _should_revert
from app.main import app
from app.views import UnitView


# --- segmentation ------------------------------------------------------------
def test_spans_land_in_the_lecture_length_regime():
    """Section V-D: conditioning needs ~25 s spans, not utterance-length ones."""
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.05, 16_000 * 300).astype(np.float32)  # 5 minutes

    spans = segment.plan_spans(audio, target_s=25.0, min_s=8.0, max_s=28.0)

    assert len(spans) >= 10
    assert all(s.duration_s <= 28.0 + 1e-6 for s in spans), "span exceeds encoder window"
    # Every span but the final remainder must clear the minimum.
    assert all(s.duration_s >= 8.0 for s in spans[:-1])
    # Contiguous, no gaps or overlaps.
    for a, b in zip(spans, spans[1:]):
        assert a.end_s == pytest.approx(b.start_s)
    assert spans[-1].end_s == pytest.approx(len(audio) / 16_000)


def test_short_audio_is_a_single_span():
    audio = np.zeros(16_000 * 10, dtype=np.float32)
    assert len(segment.plan_spans(audio)) == 1


def test_boundary_prefers_a_silent_region():
    sr = 16_000
    audio = np.ones(sr * 60, dtype=np.float32) * 0.3
    quiet_at = 24  # seconds — inside the admissible window around a 25 s target
    audio[quiet_at * sr : (quiet_at * sr) + sr // 2] = 0.0

    spans = segment.plan_spans(audio, target_s=25.0)

    assert spans[0].end_s == pytest.approx(quiet_at, abs=0.6)


# --- retrieval ---------------------------------------------------------------
UNITS = [
    UnitView("u1", "os-u01", "Process scheduling",
             "इस lecture में हम process scheduling सीखेंगे। हम round robin और priority scheduling देखेंगे।",
             ["process", "scheduling", "round robin", "priority"]),
    UnitView("u2", "os-u02", "Memory management",
             "इस tutorial में हम memory management समझेंगे। हम paging, segmentation और virtual memory देखेंगे।",
             ["memory", "paging", "segmentation", "virtual memory"]),
    UnitView("u3", "os-u03", "File systems",
             "इस lecture में हम file system के बारे में सीखेंगे। हम inode, directory और file permissions समझेंगे।",
             ["file system", "inode", "directory", "file permissions"]),
]


def test_retrieval_ranks_the_matching_unit_last():
    """Terminal weighting: the most relevant unit must render at the END."""
    index = retrieve.get_index("syl-test", UNITS, k=3)
    picked = index.query("आज हम paging और virtual memory के बारे में बात करेंगे")

    assert picked[-1].unit_key == "os-u02", "best match must be last, not first"
    assert len(picked) == 3


def test_retrieval_survives_a_noisy_first_pass():
    """Char n-grams, not word tokens: the query is a degraded hypothesis."""
    index = retrieve.get_index("syl-test", UNITS, k=1)
    picked = index.query("hum फाइल सिस्टम aur inod ke bare mein")  # misspelt, mixed script

    assert picked[0].unit_key == "os-u03"


def test_empty_query_does_not_crash():
    index = retrieve.get_index("syl-test", UNITS, k=2)
    assert len(index.query("")) == 2


def test_index_cache_busts_when_units_change():
    a = retrieve.get_index("syl-bust", UNITS, k=2)
    assert retrieve.get_index("syl-bust", UNITS, k=2) is a
    retrieve.invalidate("syl-bust")
    assert retrieve.get_index("syl-bust", UNITS, k=2) is not a


# --- prompt construction -----------------------------------------------------
def test_prompt_is_narration_not_a_term_list():
    """Enumeration is the published failure mode; it must not be constructible here."""
    prompt = prompts.build_prompt(UNITS[:2])

    assert "," not in prompt.split("।")[0] or "सीखेंगे" in prompt
    assert prompt.count("।") >= 2, "expected flowing sentences"
    for unit in UNITS[:2]:
        assert unit.prose.strip() in prompt


def test_prompt_preserves_order_so_the_best_match_lands_last():
    prompt = prompts.build_prompt([UNITS[0], UNITS[1]])
    assert prompt.index(UNITS[0].prose[:20]) < prompt.index(UNITS[1].prose[:20])


def test_prompt_truncates_from_the_left():
    long_units = UNITS * 12
    prompt = prompts.build_prompt(long_units, max_tokens=60)

    assert prompts.n_tokens(prompt) <= 60
    # The terminal region — the high-influence tokens — must survive.
    assert prompt.rstrip().endswith(long_units[-1].prose.strip()[-12:])


def test_no_units_falls_back_to_the_register_prompt():
    assert prompts.build_prompt([]) == prompts.GENERIC_PROMPT


# --- stability safeguard -----------------------------------------------------
def _dr(text: str, lp: float | None = -0.3, cr: float | None = 1.4) -> DecodeResult:
    return DecodeResult(text=text, avg_logprob=lp, compression_ratio=cr)


def test_safeguard_accepts_a_stable_conditioned_hypothesis():
    assert not _should_revert(_dr("process scheduling की बात करते हैं"),
                              _dr("प्रोसेस शेड्यूलिंग की बात करते हैं"))


def test_safeguard_reverts_on_logprob_collapse():
    assert _should_revert(_dr("garbage", lp=-2.0), _dr("baseline", lp=-0.3))


def test_safeguard_reverts_on_degenerate_repetition():
    assert _should_revert(_dr("loop loop loop", cr=4.0), _dr("baseline"))


def test_safeguard_reverts_on_runaway_length():
    assert _should_revert(_dr(" ".join(["word"] * 40)), _dr(" ".join(["word"] * 10)))


def test_safeguard_tolerates_missing_metrics():
    assert not _should_revert(_dr("ok", lp=None, cr=None), _dr("base", lp=None, cr=None))


# --- normalisation -----------------------------------------------------------
def test_normalisation_matches_the_research_implementation():
    assert normalize("File Permissions समझें।") == "file permissions समझें"
    assert normalize("१२३ pages") == "123 pages"


def test_script_mix_detects_the_dual_script_convention():
    mix = script_mix("हम file permissions समझेंगे")
    assert mix["lat"] == pytest.approx(2 / 4)
    assert mix["dev"] == pytest.approx(2 / 4)


# --- API ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_the_frozen_sgcd_config(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["sgcd"]["span_target_s"] >= 20.0, "span target below the validated regime"
    assert body["sgcd"]["retrieval_k"] == 3
    assert body["sgcd"]["prompt_max_tokens"] == 200


def test_subject_lifecycle(client):
    created = client.post("/subjects", json={"name": "Operating Systems", "code": "ITC501"})
    assert created.status_code == 201
    sid = created.json()["id"]

    assert client.get(f"/subjects/{sid}").json()["name"] == "Operating Systems"
    assert any(s["id"] == sid for s in client.get("/subjects").json())

    # No syllabus and no lectures yet.
    assert client.get(f"/subjects/{sid}/syllabus").status_code == 404
    assert client.get(f"/subjects/{sid}/lectures").json() == []

    cov = client.get(f"/subjects/{sid}/coverage").json()
    assert cov["total_units"] == 0 and cov["covered_units"] == 0

    assert client.delete(f"/subjects/{sid}").status_code == 204
    assert client.get(f"/subjects/{sid}").status_code == 404


def test_unknown_ids_are_404_not_500(client):
    assert client.get("/subjects/nope").status_code == 404
    assert client.get("/lectures/nope").status_code == 404
    assert client.get("/jobs/nope").status_code == 404


def test_rejects_non_audio_upload(client):
    sid = client.post("/subjects", json={"name": "Networks"}).json()["id"]
    r = client.post(
        f"/subjects/{sid}/lectures",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )
    assert r.status_code == 400
    client.delete(f"/subjects/{sid}")


def test_rejects_non_pdf_syllabus(client):
    sid = client.post("/subjects", json={"name": "DBMS"}).json()["id"]
    r = client.post(
        f"/subjects/{sid}/syllabus",
        files={"file": ("syllabus.docx", b"x", "application/msword")},
    )
    assert r.status_code == 400
    client.delete(f"/subjects/{sid}")
