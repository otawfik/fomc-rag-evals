"""HTTP API.

    uvicorn fomc.api:app --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Who dissented in December 2024?"}'

The index is built at startup from data/statements.jsonl (seconds; ~240 chunks). With DATABASE_URL set it
goes to pgvector instead of the in-process Chroma store.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import Agent
from .ingest import load
from .store import build, open_store

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    docs = load()
    store = open_store()
    state["chunks"] = build(store, docs)
    state["agent"] = Agent(store, [d["date"] for d in docs])
    state["agent"].ask("warm-up")
    yield


app = FastAPI(title="FOMC RAG", lifespan=lifespan)


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    naive: bool = False


@app.get("/health")
def health():
    a = state.get("agent")
    return {"ok": bool(a), "chunks": state.get("chunks"), "answerer": a.reader.name if a else None}


@app.post("/ask")
def ask(body: Ask):
    a = state["agent"]
    return a.naive(body.question) if body.naive else a.ask(body.question)
