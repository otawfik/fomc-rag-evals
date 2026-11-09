"""What each step buys: the golden set and the held-out set through cumulative versions of the agent.

    python -m evals.ablation      # writes evals/ABLATION.md
"""
import json
from pathlib import Path

from fomc.agent import Agent
from fomc.ingest import load
from fomc.store import build, open_store

from .golden import OUT as GOLDEN, correct
from .heldout import items as heldout_items

STEPS = [
    ("vector search only (naive RAG)", None),
    ("+ resolve the meeting date, filter retrieval", dict(dates=True, rerank=False, intents=False, vote_rule=False)),
    ("+ cross-encoder rerank", dict(dates=True, rerank=True, intents=False, vote_rule=False)),
    ("+ intent routing, canonical question", dict(dates=True, rerank=True, intents=True, vote_rule=False)),
    ("+ keep the voting roll out of range/decision answers", dict(dates=True, rerank=True, intents=True, vote_rule=True)),
]


def acc(fn, items):
    return sum(correct(it, fn(it["question"])["answer"]) for it in items) / len(items)


def main():
    docs = load()
    store = open_store()
    build(store, docs)
    golden = [json.loads(x) for x in GOLDEN.read_text().splitlines() if x.strip()]
    held = heldout_items()
    rows = ["| pipeline | golden accuracy | held-out accuracy |", "|---|---|---|"]
    for name, flags in STEPS:
        a = Agent(store, [d["date"] for d in docs], **(flags or {}))
        fn = a.naive if flags is None else a.ask
        rows.append(f"| {name} | {acc(fn, golden):.1%} | {acc(fn, held):.1%} |")
        print(rows[-1], flush=True)
    Path(__file__).with_name("ABLATION.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
