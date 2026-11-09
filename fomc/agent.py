"""The question-answering agent, as a LangGraph state machine.

    resolve ─► retrieve ─► grade ─┬─► rerank ─► answer ─► verify ─► END
                  ▲                │
                  └──── widen ◄────┘  (nothing relevant: drop to the month, then to no filter; at most twice)

resolve   routes on intent (range / decision / dissent / other) and finds which meeting(s) the question means: a full date ("December 18, 2024", "2024-12-18"),
          a month ("March 2022") or a year, matched against the meetings actually in the store
retrieve  vector search, filtered to those meetings when there are any
grade     keeps hits above a similarity floor; if none survive, widen the filter and retry
rerank    a cross-encoder re-reads question + passage together and keeps the best two paragraphs
answer    the configured answerer (local Hugging Face reader, or Bedrock)
verify    the answer must come from a retrieved statement, else it's flagged as unsupported

`naive()` is the baseline the evals compare against: one unfiltered vector search, then the same answerer.
"""
from __future__ import annotations

import re
import time
from datetime import date, datetime
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .llm import Answer, answerer
from .store import Hit, Store, rerank

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
MON = "|".join(list(MONTHS) + [m[:3] for m in MONTHS] + ["sept"])
FLOOR = 0.45  # cosine similarity below which a chunk isn't about the question
K = 5


def _month(s: str) -> int:
    s = s.lower().rstrip(".")
    return MONTHS.get(s) or next(v for k, v in MONTHS.items() if k.startswith(s[:3]))


def resolve_dates(question: str, meetings: list[str]) -> tuple[list[str], str]:
    """(meeting dates the question points at, how specific: 'day' | 'month' | 'year' | 'none')"""
    q = question.lower()
    found: set[str] = set()
    for y, m, d in re.findall(r"\b(20\d\d)-(\d\d)-(\d\d)\b", q):
        found.add(f"{y}-{m}-{d}")
    for mon, d, y in re.findall(rf"\b({MON})\.? (\d{{1,2}}),? (20\d\d)\b", q):
        found.add(date(int(y), _month(mon), int(d)).isoformat())
    for d, mon, y in re.findall(rf"\b(\d{{1,2}}) ({MON})\.? (20\d\d)\b", q):
        found.add(date(int(y), _month(mon), int(d)).isoformat())
    hit = [m for m in meetings if m in found]
    if hit:
        return hit, "day"
    months = {(int(y), _month(mon)) for mon, y in re.findall(rf"\b({MON})\.? (20\d\d)\b", q)}
    hit = [m for m in meetings if (int(m[:4]), int(m[5:7])) in months]
    if hit:
        return hit, "month"
    years = set(re.findall(r"\b(20\d\d)\b", q))
    hit = [m for m in meetings if m[:4] in years]
    return (hit, "year") if hit else ([], "none")


# intent -> the canonical question the reranker and reader get. A small extractive reader is literal:
# "did they raise, cut or hold?" pulls it toward any sentence with "range" or "action" in it (the
# "full range of tools" boilerplate, the voting roll). Routing on intent and asking it the way the
# statements phrase the answer is what lifts decision accuracy (README, "What moved the numbers").
INTENTS = [
    ("dissent", r"\b(dissent|voted against|vote against|against the (policy )?action|who opposed)", "Who voted against the action?"),
    ("decision", r"\b(raise|rais|cut|hold|held|hike|lower|decid|decision|policy action|what did the (fed|fomc|committee) do)",
     "What did the Committee decide to do with the target range for the federal funds rate?"),
    ("range", r"\b(target range|range for the federal funds|federal funds (rate )?target|where did the fed put)",
     "What is the target range for the federal funds rate?"),
]


def intent(question: str) -> tuple[str, str]:
    q = question.lower()
    for name, rx, canonical in INTENTS:
        if re.search(rx, q):
            return name, canonical
    return "other", question


class State(TypedDict, total=False):
    question: str
    intent: str
    ask: str
    dates: list[str]
    scope: str
    hits: list[Hit]
    tries: int
    answer: Answer
    supported: bool
    trace: list[str]


class Agent:
    def __init__(self, store: Store, meetings: list[str], reader=None, *, dates=True, rerank=True, intents=True, vote_rule=True):
        self.store, self.meetings, self.reader = store, sorted(meetings), reader or answerer()
        # switches for the ablation in evals/ablation.py; all on in the real agent
        self.use = {"dates": dates, "rerank": rerank, "intents": intents, "vote_rule": vote_rule}
        g = StateGraph(State)
        g.add_node("resolve", self.resolve)
        g.add_node("retrieve", self.retrieve)
        g.add_node("grade", self.grade)
        g.add_node("widen", self.widen)
        g.add_node("rerank", self.rerank)
        g.add_node("answer", self.answer)
        g.add_node("verify", self.verify)
        g.set_entry_point("resolve")
        g.add_edge("resolve", "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", lambda s: "rerank" if s["hits"] or s["tries"] >= 2 else "widen")
        g.add_edge("rerank", "answer")
        g.add_edge("widen", "retrieve")
        g.add_edge("answer", "verify")
        g.add_edge("verify", END)
        self.graph = g.compile()

    def resolve(self, s: State) -> State:
        dates, scope = resolve_dates(s["question"], self.meetings) if self.use["dates"] else ([], "none")
        kind, ask = intent(s["question"]) if self.use["intents"] else ("other", s["question"])
        return {"dates": dates, "scope": scope, "intent": kind, "ask": ask, "tries": 0,
                "trace": [f"resolve: {kind}, {scope} {dates[:3]}{'…' if len(dates) > 3 else ''}"]}

    def retrieve(self, s: State) -> State:
        hits = self.store.search(s["question"], K, s["dates"] or None)
        return {"hits": hits, "trace": s["trace"] + [f"retrieve: {len(hits)} hits"]}

    def grade(self, s: State) -> State:
        keep = [h for h in s["hits"] if h.score >= FLOOR]
        return {"hits": keep, "tries": s["tries"] + 1, "trace": s["trace"] + [f"grade: kept {len(keep)}/{len(s['hits'])}"]}

    def widen(self, s: State) -> State:
        if s["scope"] == "day" and s["dates"]:
            y, m = s["dates"][0][:4], s["dates"][0][5:7]
            dates = [d for d in self.meetings if d[:7] == f"{y}-{m}"]
            return {"dates": dates, "scope": "month", "trace": s["trace"] + ["widen: to the month"]}
        return {"dates": [], "scope": "none", "trace": s["trace"] + ["widen: no filter"]}

    def rerank(self, s: State) -> State:
        hits = s["hits"]
        # the voting paragraph says what dissenters *preferred* ("preferred to lower the target range ...
        # to 3-1/2 to 3-3/4 percent"): evidence for who dissented, a trap for what was decided
        if self.use["vote_rule"] and s["intent"] in ("range", "decision"):
            hits = [h for h in hits if not re.search(r"Voting (for|against)", h.chunk.text)] or hits
        hits = rerank(s["ask"], hits) if self.use["rerank"] else hits
        return {"hits": hits, "trace": s["trace"] + [f"rerank: top {len(hits)}"]}

    def answer(self, s: State) -> State:
        # the extractive reader gets the canonical question; an LLM gets the user's own words
        q = s["question"] if getattr(self.reader, "name", "") == "bedrock" else s["ask"]
        return {"answer": self.reader.answer(q, s["hits"])}

    def verify(self, s: State) -> State:
        ok = s["answer"].source is not None and any(h.chunk.doc_id == s["answer"].source for h in s["hits"])
        return {"supported": ok, "trace": s["trace"] + [f"verify: {'cited ' + s['answer'].source if ok else 'unsupported'}"]}

    def ask(self, question: str) -> dict:
        t = time.perf_counter()
        s = self.graph.invoke({"question": question})
        return result(question, s["answer"], s["hits"], time.perf_counter() - t, s["supported"], s["trace"])

    def naive(self, question: str) -> dict:
        """Baseline: one unfiltered vector search, same answerer, no date handling."""
        t = time.perf_counter()
        hits = self.store.search(question, K)
        a = self.reader.answer(question, hits)
        return result(question, a, hits, time.perf_counter() - t, a.source is not None, ["naive"])


def result(question: str, a: Answer, hits: list[Hit], secs: float, supported: bool, trace: list[str]) -> dict:
    return {
        "question": question,
        "answer": a.text,
        "source": a.source,
        "confidence": round(a.confidence, 3),
        "supported": supported,
        "retrieved": [h.chunk.doc_id for h in hits],
        "sources": [{"doc_id": h.chunk.doc_id, "title": h.chunk.title, "url": h.chunk.url, "score": round(h.score, 3)} for h in hits],
        "latency_ms": round(secs * 1000, 1),
        "trace": trace,
    }
