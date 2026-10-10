"""Studio: quiz, flashcards, mind map, report, slides, infographic.

Gemini is stubbed; these tests cover scope resolution against a real database,
schema clean-up of model output, chat intent detection, and the API lifecycle
(create -> background generation -> ready, retry, delete).
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.db import session_scope
from app.main import app
from app.models import Lecture, Note, Subject, Syllabus, SyllabusUnit, TranscriptSpan
from app.studio import context as ctx
from app.studio import generators as gen
from app.studio.intent import detect


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.rag.store.search", lambda *a, **k: [])
    monkeypatch.setattr("app.rag.store.delete_subject", lambda *a, **k: None)
    with TestClient(app) as c:
        yield c


def _seed(name: str = "Neural Networks") -> dict[str, str]:
    with session_scope() as db:
        subject = Subject(name=name)
        db.add(subject)
        db.flush()
        syl = Syllabus(subject_id=subject.id, provenance="manual")
        db.add(syl)
        db.flush()
        u1 = SyllabusUnit(syllabus_id=syl.id, unit_key="nn-u01", order_index=1, title="Perceptrons",
                          prose="Single-layer perceptron and its learning rule.", keywords=["perceptron"])
        u2 = SyllabusUnit(syllabus_id=syl.id, unit_key="nn-u02", order_index=2, title="Backpropagation",
                          prose="Gradient descent through layers.", keywords=["gradient"])
        db.add_all([u1, u2])
        db.flush()
        lec = Lecture(subject_id=subject.id, title="L01 Perceptrons", status="ready", summary="Perceptrons and XOR.")
        db.add(lec)
        db.flush()
        db.add_all([
            TranscriptSpan(lecture_id=lec.id, index=0, start_s=0, end_s=25,
                           text="perceptron weighted sum करता है", retrieved_unit_ids=[u1.id]),
            TranscriptSpan(lecture_id=lec.id, index=1, start_s=25, end_s=50,
                           text="XOR linearly separable नहीं है", retrieved_unit_ids=[u1.id]),
            TranscriptSpan(lecture_id=lec.id, index=2, start_s=50, end_s=75,
                           text="next time backprop देखेंगे", retrieved_unit_ids=[u2.id]),
            Note(lecture_id=lec.id, unit_id=u1.id, order_index=0, topic="Perceptron rule",
                 markdown="w <- w + lr * error * x", start_s=0, end_s=50),
        ])
        return {"subject": subject.id, "u1": u1.id, "u2": u2.id, "lecture": lec.id}


# --- context -------------------------------------------------------------------

def test_unit_scope_uses_attributed_notes_and_spans_only(client):
    ids = _seed()
    with session_scope() as db:
        got = ctx.gather(db, ids["subject"], "unit", ids["u1"])
    assert got.label == "Unit 1: Perceptrons"
    assert "learning rule" in got.sources
    assert "w <- w + lr" in got.sources
    assert "XOR linearly separable" in got.sources
    assert "backprop देखेंगे" not in got.sources  # decoded under unit 2
    kinds = [c["kind"] for c in got.citations]
    assert kinds[0] == "unit" and "note" in kinds and "span" in kinds
    assert not got.syllabus_only


def test_lecture_scope_excludes_syllabus(client):
    ids = _seed()
    with session_scope() as db:
        got = ctx.gather(db, ids["subject"], "lecture", ids["lecture"])
    assert "Perceptrons and XOR" in got.sources
    assert "backprop" in got.sources
    assert all(c["kind"] != "unit" for c in got.citations)


def test_unit_without_lectures_is_marked_syllabus_only(client):
    ids = _seed()
    with session_scope() as db:
        for span in db.query(TranscriptSpan).all():
            if span.retrieved_unit_ids == [ids["u2"]]:
                db.delete(span)
        db.flush()
        db.expire_all()
        got = ctx.gather(db, ids["subject"], "unit", ids["u2"])
    assert got.syllabus_only


def test_unknown_scope_raises(client):
    ids = _seed()
    with session_scope() as db, pytest.raises(ctx.ScopeNotFound):
        ctx.gather(db, ids["subject"], "unit", "missing")


# --- generators ----------------------------------------------------------------

def test_quiz_clean_drops_broken_questions_and_foreign_citations():
    raw = {
        "title": "Perceptron quiz",
        "questions": [
            {"question": "Q1", "options": ["a", "b", "c", "d"], "answer_index": 2,
             "explanation": "because", "sources": [1, 99]},
            {"question": "Q2", "options": ["a", "b"], "answer_index": 5, "explanation": "", "sources": []},
            {"question": "", "options": ["a", "b"], "answer_index": 0, "explanation": "", "sources": []},
        ],
    }
    out = gen.SPECS["quiz"].clean(raw, {1, 2})
    assert len(out["questions"]) == 1
    assert out["questions"][0]["sources"] == [1]
    assert gen.referenced(out) == {1}


def test_mindmap_clean_assigns_ids_and_caps_depth():
    raw = {"title": "NN", "root": "Neural nets", "branches": [
        {"label": "Perceptron", "children": [{"label": "Rule", "children": [{"label": "lr"}]}]},
        {"label": "", "children": []},
    ]}
    out = gen.SPECS["mindmap"].clean(raw, set())
    root = out["root"]
    assert root["label"] == "Neural nets" and len(root["children"]) == 1
    leaf = root["children"][0]["children"][0]["children"][0]
    assert leaf["label"] == "lr" and leaf["children"] == []
    ids = {root["id"], root["children"][0]["id"], leaf["id"]}
    assert len(ids) == 3


def test_report_strips_markers_for_unsupplied_sources():
    out = gen.SPECS["report"].clean({"title": "T", "markdown": "Fact [1]. Made up [7]."}, {1})
    assert out["markdown"] == "Fact [1]. Made up ."
    assert gen.referenced(out) == {1}


def test_infographic_drops_process_with_one_step():
    raw = {"headline": "H", "sections": [{"heading": "S", "icon": "nope", "points": ["p"]}],
           "process": {"title": "x", "steps": ["only"]}, "takeaway": "t"}
    out = gen.SPECS["infographic"].clean(raw, set())
    assert out["process"] is None
    assert out["sections"][0]["icon"] == "lightbulb"


def test_every_kind_has_a_prompt():
    for kind in gen.KINDS:
        prompt = gen.build_prompt(kind, {"focus": "xor"}, subject_name="NN", scope_label="All",
                                  sources="[1] x", syllabus_only=True)
        assert "Task:" in prompt and "xor" in prompt and "only syllabus units" in prompt


# --- intent --------------------------------------------------------------------

_UNITS = [SimpleNamespace(id="u1", order_index=1, title="Perceptrons"),
          SimpleNamespace(id="u2", order_index=2, title="Backpropagation")]
_LECTURES = [SimpleNamespace(id="l1", created_at=1), SimpleNamespace(id="l2", created_at=2)]


@pytest.mark.parametrize(
    "text,kind,scope",
    [
        ("Make a quiz on unit 2", "quiz", {"type": "unit", "id": "u2"}),
        ("quiz me on backpropagation", "quiz", {"type": "unit", "id": "u2"}),
        ("create 20 flashcards from lecture 1", "flashcards", {"type": "lecture", "id": "l1"}),
        ("generate a mind map of the last lecture", "mindmap", {"type": "lecture", "id": "l2"}),
        ("can you make a slide deck about perceptrons", "slides", {"type": "unit", "id": "u1"}),
        ("give me an infographic", "infographic", {"type": "subject", "id": None}),
        ("prepare a study guide for module 1", "report", {"type": "unit", "id": "u1"}),
    ],
)
def test_detect_kind_and_scope(text, kind, scope):
    got = detect(text, _UNITS, _LECTURES)
    assert got is not None
    assert got.kind == kind
    assert got.scope == scope


@pytest.mark.parametrize(
    "text",
    ["Give me five practice questions", "What is on the quiz?", "Explain the slides", "what is a perceptron"],
)
def test_ordinary_questions_are_not_studio_requests(text):
    assert detect(text, _UNITS, _LECTURES) is None


def test_detect_options():
    got = detect("make 12 hard quiz questions on gradient clipping", _UNITS, _LECTURES)
    assert got.options["count"] == 12
    assert got.options["difficulty"] == "hard"
    assert got.options["focus"] == "gradient clipping"
    assert detect("make a briefing doc", _UNITS, _LECTURES).options["format"] == "briefing"


# --- API -----------------------------------------------------------------------

def _await_item(client, item_id: str, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/studio/{item_id}").json()
        if body["status"] in ("ready", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("studio item did not finish")


def _fake_quiz(system, user, schema, **kw):
    assert "[1] Syllabus" in user
    return {"title": "Perceptron check", "questions": [
        {"question": "What does a perceptron compute?", "options": ["sum", "sort", "hash", "copy"],
         "answer_index": 0, "explanation": "A weighted sum.", "sources": [1, 2]},
    ]}, "gemini-3.5-flash"


def test_api_lifecycle(client, monkeypatch):
    ids = _seed()
    monkeypatch.setattr("app.studio.runner.complete_json_with_model", _fake_quiz)
    sid = ids["subject"]

    r = client.post(f"/subjects/{sid}/studio", json={"kind": "quiz", "scope": {"type": "unit", "id": ids["u1"]},
                                                    "options": {"count": 5}})
    assert r.status_code == 202, r.text
    item = _await_item(client, r.json()["id"])
    assert item["status"] == "ready", item["error"]
    assert item["title"] == "Perceptron check"
    assert item["model"] == "gemini-3.5-flash"
    assert item["scope"]["label"] == "Unit 1: Perceptrons"
    assert [c["n"] for c in item["citations"]] == [1, 2]

    assert [i["id"] for i in client.get(f"/subjects/{sid}/studio").json()] == [item["id"]]

    retried = client.post(f"/studio/{item['id']}/retry")
    assert retried.status_code == 202
    assert _await_item(client, item["id"])["status"] == "ready"

    assert client.delete(f"/studio/{item['id']}").status_code == 204
    assert client.get(f"/studio/{item['id']}").status_code == 404


def test_api_rejects_foreign_scope(client):
    a, b = _seed("A"), _seed("B")
    r = client.post(f"/subjects/{a['subject']}/studio", json={"kind": "quiz", "scope": {"type": "unit", "id": b["u1"]}})
    assert r.status_code == 404


def test_missing_key_fails_item_with_actionable_message(client, monkeypatch):
    from app.llm.client import LLMUnavailable

    def no_key(*a, **k):
        raise LLMUnavailable("no key")

    monkeypatch.setattr("app.studio.runner.complete_json_with_model", no_key)
    ids = _seed()
    r = client.post(f"/subjects/{ids['subject']}/studio", json={"kind": "flashcards"})
    item = _await_item(client, r.json()["id"])
    assert item["status"] == "failed"
    assert "GEMINI_API_KEY" in item["error"]


def test_chat_request_creates_item_and_card(client, monkeypatch):
    ids = _seed()
    monkeypatch.setattr("app.studio.runner.complete_json_with_model", _fake_quiz)
    r = client.post(f"/subjects/{ids['subject']}/chat", json={"question": "Make a quiz on unit 1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query_type"] == "studio" and body["studio_item_id"]
    assert "Unit 1: Perceptrons" in body["answer"]
    assert _await_item(client, body["studio_item_id"])["status"] == "ready"

    saved = client.get(f"/chat/sessions/{body['session_id']}").json()
    assert saved[-1]["studio_item_id"] == body["studio_item_id"]
