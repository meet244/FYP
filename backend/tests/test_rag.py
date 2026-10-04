"""Subject-scoped RAG: syllabus units sit in the same local Chroma store as
lecture spans and notes, so the chatbot can answer from the curriculum even
before a lecture is uploaded — and never from a remote vector DB.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.rag import indexer, store
from app.rag.answer import Answer, _gather, _render_sources, answer_question
from app.llm.client import candidate_models, is_rate_limit, retry_after_s
from app.rag.router import Route, route
from app.schemas import JobOut


def test_index_syllabus_units_upserts_into_the_local_units_collection(monkeypatch):
    captured: dict = {}
    deleted: list[str] = []

    monkeypatch.setattr(
        "app.rag.indexer.store.upsert",
        lambda name, ids, docs, metas: captured.update(
            name=name, ids=ids, docs=docs, metas=metas
        ),
    )
    monkeypatch.setattr(
        "app.rag.indexer.store.delete_subject_units", deleted.append
    )

    unit = SimpleNamespace(
        id="u1",
        unit_key="ml-u01",
        order_index=1,
        title="Perceptron",
        prose="A perceptron takes weighted inputs and produces an output.",
        keywords=["perceptron", "weights"],
    )
    subject = SimpleNamespace(
        id="s1",
        syllabus=SimpleNamespace(id="syl1", units=[unit]),
    )

    n = indexer.index_syllabus_units(None, subject)

    assert n == 1
    assert deleted == ["s1"]
    assert captured["name"] == store.UNITS
    assert captured["ids"] == ["unit:u1"]
    assert "Perceptron" in captured["docs"][0]
    assert "weighted inputs" in captured["docs"][0]
    meta = captured["metas"][0]
    assert meta["kind"] == "unit"
    assert meta["subject_id"] == "s1"
    assert meta["unit_id"] == "u1"
    assert meta["topic"] == "Perceptron"
    assert "lecture_id" not in meta


def test_index_syllabus_units_clears_vectors_when_there_is_no_syllabus(monkeypatch):
    deleted: list[str] = []
    monkeypatch.setattr("app.rag.indexer.store.delete_subject_units", deleted.append)
    monkeypatch.setattr(
        "app.rag.indexer.store.upsert",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not upsert")),
    )

    n = indexer.index_syllabus_units(None, SimpleNamespace(id="s1", syllabus=None))
    assert n == 0
    assert deleted == ["s1"]


def test_gather_always_searches_syllabus_units(monkeypatch):
    searched: list[str] = []

    def fake_search(name, query, subject_id, **kw):
        searched.append(name)
        assert "lecture_id" not in kw or name != store.UNITS
        return []

    monkeypatch.setattr("app.rag.answer.store.search", fake_search)

    subject = SimpleNamespace(id="s1", lectures=[])
    rt = Route(
        query_type="explain",
        search_query="perceptron",
        lecture_hint="",
        wants_notes=False,
    )
    _gather(None, subject, rt)

    assert store.UNITS in searched
    assert store.SPANS in searched
    assert store.NOTES in searched
    assert store.MATERIALS in searched


def test_syllabus_hits_render_without_a_lecture_timestamp():
    hits = [
        {
            "id": "unit:u1",
            "text": "Perceptron is a linear classifier.",
            "metadata": {
                "kind": "unit",
                "subject_id": "s1",
                "unit_id": "u1",
                "unit_key": "ml-u01",
                "topic": "Perceptron",
            },
            "score": 0.91,
        }
    ]
    sources, citations = _render_sources(hits)
    assert "Syllabus" in sources
    assert "Perceptron" in sources
    assert citations[0]["kind"] == "unit"
    assert citations[0]["unit_id"] == "u1"
    assert citations[0]["lecture_id"] is None
    assert citations[0]["timestamp"] is None


def test_answer_from_syllabus_when_no_lectures_exist(monkeypatch):
    monkeypatch.setattr(
        "app.rag.answer.route_query",
        lambda q, h=None: Route(
            query_type="explain",
            search_query="perceptron",
            lecture_hint="",
            wants_notes=False,
        ),
    )

    def fake_search(name, query, subject_id, **kw):
        if name == store.UNITS:
            return [
                {
                    "id": "unit:u1",
                    "text": "A perceptron takes weighted inputs and produces an output.",
                    "metadata": {
                        "kind": "unit",
                        "subject_id": "s1",
                        "unit_id": "u1",
                        "topic": "Perceptron",
                        "unit_key": "ml-u01",
                    },
                    "score": 0.88,
                }
            ]
        return []

    monkeypatch.setattr("app.rag.answer.store.search", fake_search)
    monkeypatch.setattr(
        "app.rag.answer.complete",
        lambda *a, **k: "A perceptron is a neuron with weighted inputs. [1]",
    )

    subject = SimpleNamespace(id="s1", name="Deep Learning", lectures=[], syllabus=object())
    result = answer_question(None, subject, "What is a perceptron?")

    assert isinstance(result, Answer)
    assert "perceptron" in result.text.lower()
    assert result.citations
    assert result.citations[0]["kind"] == "unit"


def test_chunk_text_overlaps_and_keeps_short_strings():
    from app.ingest.materials import chunk_text

    assert chunk_text("") == []
    assert chunk_text("short") == ["short"]
    parts = chunk_text("alpha " * 400, size=80, overlap=20)
    assert len(parts) > 3
    assert parts[0][:5] == "alpha"
    # Adjacent windows share tokens — that is the point of overlap.
    assert any(w in parts[1] for w in parts[0].split()[-4:])


def test_index_material_upserts_pdf_chunks(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "app.rag.indexer.store.upsert",
        lambda name, ids, docs, metas: captured.update(
            name=name, ids=ids, docs=docs, metas=metas
        ),
    )
    monkeypatch.setattr("app.rag.indexer.store.delete_material", lambda *_: None)

    material = SimpleNamespace(
        id="m1",
        subject_id="s1",
        kind="pdf",
        title="Lecture slides",
        text="Perceptron slides. A perceptron takes weighted inputs and produces an output. " * 20,
    )
    n = indexer.index_material(material)
    assert n >= 1
    assert captured["name"] == store.MATERIALS
    assert captured["metas"][0]["kind"] == "pdf"
    assert captured["metas"][0]["material_id"] == "m1"
    assert "Lecture slides" in captured["docs"][0]


def test_pdf_hits_render_as_material_citations():
    hits = [
        {
            "id": "m1:c0",
            "text": "Backprop updates weights using the chain rule.",
            "metadata": {
                "kind": "pdf",
                "subject_id": "s1",
                "material_id": "m1",
                "topic": "Week 2 slides",
                "chunk": 0,
            },
            "score": 0.8,
        }
    ]
    sources, citations = _render_sources(hits)
    assert "PDF" in sources
    assert citations[0]["kind"] == "pdf"
    assert citations[0]["material_id"] == "m1"
    assert citations[0]["lecture_id"] is None


def test_route_is_heuristic_no_llm():
    assert route("What syllabus units have been covered?").query_type == "coverage"
    assert route("Give me five practice questions").query_type == "quiz"
    assert route("hello").query_type == "smalltalk"
    assert route("What does the syllabus cover?").query_type == "outline"
    assert route("Explain the last lecture from the recording").query_type == "explain"
    lookup = route("What is a perceptron?")
    assert lookup.query_type == "lookup"
    assert lookup.search_query == "What is a perceptron?"


def test_gemini_quota_helpers():
    assert retry_after_s(RuntimeError("Please retry in 25.8s.")) == pytest.approx(25.8)
    assert is_rate_limit(RuntimeError("429 RESOURCE_EXHAUSTED. {'error': {}}"))
    models = candidate_models("gemini-3.6-flash")
    assert models[0] == "gemini-3.6-flash"
    assert "gemini-3.1-flash-lite" in models


def test_job_out_reads_material_id_from_payload():
    job = SimpleNamespace(
        id="j1",
        kind="ingest_material",
        status="running",
        stage="extracting",
        progress=0.2,
        message="reading pdf",
        error=None,
        lecture_id=None,
        subject_id="s1",
        payload={"material_id": "m9"},
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        updated_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    out = JobOut.of(job)
    assert out.material_id == "m9"


def test_extractive_fallback_when_gemini_is_down(monkeypatch):
    from app.llm.client import LLMRateLimited

    monkeypatch.setattr(
        "app.rag.answer.route_query",
        lambda q, h=None: Route("lookup", "perceptron", "", False),
    )

    def fake_search(name, query, subject_id, **kw):
        if name == store.UNITS:
            return [
                {
                    "id": "unit:u1",
                    "text": "A perceptron takes weighted inputs and produces an output.",
                    "metadata": {
                        "kind": "unit",
                        "subject_id": "s1",
                        "unit_id": "u1",
                        "topic": "Perceptron",
                    },
                    "score": 0.9,
                }
            ]
        return []

    monkeypatch.setattr("app.rag.answer.store.search", fake_search)
    monkeypatch.setattr(
        "app.rag.answer.complete",
        lambda *a, **k: (_ for _ in ()).throw(LLMRateLimited("quota")),
    )

    subject = SimpleNamespace(id="s1", name="DL", lectures=[], syllabus=object())
    result = answer_question(None, subject, "What is a perceptron?")
    assert "perceptron" in result.text.lower()
    assert result.citations
    assert "[1]" in result.text


