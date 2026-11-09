"""Chunks, embeddings and the vector store.

Embeddings come from a Hugging Face sentence-transformers model (BAAI/bge-small-en-v1.5, 384 dims).
Two stores implement the same two methods:
  ChromaStore    embedded Chroma, the default (no server needed)
  PgVectorStore  Postgres + pgvector, used when DATABASE_URL is set (CI runs it as a service container)
Both take an optional `dates` filter, which is what the agent uses once it knows which meeting a
question is about: FOMC statements are near-identical boilerplate, so similarity alone can't tell
two meetings apart.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# bge models want this prefix on queries (not on passages)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Chunk:
    id: str
    doc_id: str
    date: str
    title: str
    url: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, higher is closer


def chunk(docs: list[dict]) -> list[Chunk]:
    """One chunk per paragraph, each prefixed with its statement's title so the date travels with it."""
    out = []
    for d in docs:
        for i, para in enumerate(p for p in d["text"].split("\n\n") if len(p.split()) >= 8):
            out.append(Chunk(f"{d['id']}#{i}", d["id"], d["date"], d["title"], d["url"], f"{d['title']}. {para}"))
    return out


def own_weights(model):
    """Copy every weight into freshly allocated memory.

    Transformers memory-maps safetensors files; when a tensor starts at an unaligned offset in the file
    (seen: data_ptr() % 64 == 2), Apple's Accelerate BLAS returns NaN from a plain matmul on it. The
    cross-encoder scored everything NaN until this; cloning costs ~100 MB once and fixes it everywhere.
    """
    import torch

    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.clone()
    return model


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    # CPU by default: matches CI, and some macOS versions crash in torch's MPS backend
    m = SentenceTransformer(EMBED_MODEL, device=os.environ.get("DEVICE", "cpu"))
    own_weights(m)
    return m


def embed(texts: list[str], query: bool = False) -> list[list[float]]:
    texts = [QUERY_PREFIX + t for t in texts] if query else texts
    return _model().encode(texts, normalize_embeddings=True, batch_size=32).tolist()


@lru_cache(maxsize=1)
def _cross():
    from sentence_transformers import CrossEncoder

    m = CrossEncoder(RERANK_MODEL, device=os.environ.get("DEVICE", "cpu"))
    own_weights(m.model)
    return m


def rerank(query: str, hits: list[Hit], top: int = int(os.environ.get("RERANK_TOP", 2))) -> list[Hit]:
    """Re-score hits with a cross-encoder (reads query and passage together, unlike the bi-encoder),
    keep the best `top`. Scores become the cross-encoder's sigmoid probability."""
    if not hits:
        return hits
    import math

    raw = _cross().predict([(query, h.chunk.text) for h in hits])
    scored = sorted(zip(hits, raw), key=lambda x: -x[1])[:top]
    return [Hit(h.chunk, 1 / (1 + math.exp(-float(r)))) for h, r in scored]


class Store(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...
    def search(self, query: str, k: int = 5, dates: list[str] | None = None) -> list[Hit]: ...
    def count(self) -> int: ...


class ChromaStore:
    def __init__(self, path: str | None = None, name: str = "fomc"):
        import chromadb

        client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self.col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk]) -> None:
        self.col.upsert(
            ids=[c.id for c in chunks],
            embeddings=embed([c.text for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "date": c.date, "title": c.title, "url": c.url} for c in chunks],
        )

    def search(self, query: str, k: int = 5, dates: list[str] | None = None) -> list[Hit]:
        where = None if not dates else ({"date": dates[0]} if len(dates) == 1 else {"date": {"$in": dates}})
        r = self.col.query(query_embeddings=embed([query], query=True), n_results=k, where=where)
        hits = []
        for cid, text, meta, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]):
            hits.append(Hit(Chunk(cid, meta["doc_id"], meta["date"], meta["title"], meta["url"], text), 1 - dist))
        return hits

    def count(self) -> int:
        return self.col.count()


class PgVectorStore:
    """Postgres + pgvector: one table, an HNSW cosine index, and a plain SQL date filter."""

    def __init__(self, url: str, dim: int = 384):
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(url, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(
            f"""CREATE TABLE IF NOT EXISTS chunks (
                  id text PRIMARY KEY, doc_id text NOT NULL, date date NOT NULL, title text NOT NULL,
                  url text NOT NULL, text text NOT NULL, embedding vector({dim}) NOT NULL)"""
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)")

    def add(self, chunks: list[Chunk]) -> None:
        import numpy as np

        vecs = embed([c.text for c in chunks])
        with self.conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding""",
                [(c.id, c.doc_id, c.date, c.title, c.url, c.text, np.array(v)) for c, v in zip(chunks, vecs)],
            )

    def search(self, query: str, k: int = 5, dates: list[str] | None = None) -> list[Hit]:
        import numpy as np

        q = np.array(embed([query], query=True)[0])
        sql = "SELECT id, doc_id, date::text, title, url, text, 1 - (embedding <=> %s) FROM chunks"
        args: list = [q]
        if dates:
            sql += " WHERE date = ANY(%s::date[])"
            args.append(dates)
        sql += " ORDER BY embedding <=> %s LIMIT %s"
        args += [q, k]
        rows = self.conn.execute(sql, args).fetchall()
        return [Hit(Chunk(*r[:6]), float(r[6])) for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


def open_store(path: str | None = None) -> Store:
    url = os.environ.get("DATABASE_URL")
    return PgVectorStore(url) if url else ChromaStore(path)


def build(store: Store, docs: list[dict]) -> int:
    cs = chunk(docs)
    store.add(cs)
    return len(cs)
