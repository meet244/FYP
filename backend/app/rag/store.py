"""Chroma-backed vector index over transcript spans and generated notes.

Multilingual embeddings, not English ones: the corpus is Devanagari matrix with
Latin technical terms in the same sentence, and a monolingual encoder collapses
the Hindi half. Everything is keyed by subject so the chat interface can scope a
query to one course.
"""
from __future__ import annotations

import functools
import logging
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from app.config import settings

log = logging.getLogger(__name__)

SPANS = "spans"
NOTES = "notes"


@functools.lru_cache(maxsize=1)
def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


@functools.lru_cache(maxsize=1)
def _embedder():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embed_model
    )


@functools.lru_cache(maxsize=4)
def collection(name: str) -> chromadb.Collection:
    return _client().get_or_create_collection(
        name=name,
        embedding_function=_embedder(),
        metadata={"hnsw:space": "cosine"},
    )


def upsert(name: str, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
    if not ids:
        return
    collection(name).upsert(ids=ids, documents=documents, metadatas=metadatas)


def delete_lecture(lecture_id: str) -> None:
    for name in (SPANS, NOTES):
        try:
            collection(name).delete(where={"lecture_id": lecture_id})
        except Exception as exc:  # collection may not exist yet
            log.debug("delete on %s skipped: %s", name, exc)


def search(
    name: str,
    query: str,
    subject_id: str,
    *,
    top_k: int | None = None,
    lecture_id: str | None = None,
) -> list[dict[str, Any]]:
    top_k = top_k or settings.retrieval_top_k
    where: dict[str, Any] = {"subject_id": subject_id}
    if lecture_id:
        where = {"$and": [{"subject_id": subject_id}, {"lecture_id": lecture_id}]}

    try:
        res = collection(name).query(query_texts=[query], n_results=top_k, where=where)
    except Exception as exc:
        log.warning("vector search on %s failed: %s", name, exc)
        return []

    hits: list[dict[str, Any]] = []
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for i, doc, meta, dist in zip(ids, docs, metas, dists):
        hits.append(
            {
                "id": i,
                "text": doc,
                "metadata": meta or {},
                "score": 1.0 - float(dist) if dist is not None else None,
            }
        )
    return hits
