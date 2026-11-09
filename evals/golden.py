"""The golden test set, built from the statements' own facts, and the scorers that check answers against it.

    python -m evals.golden        # writes evals/golden.jsonl

For every statement: the target range it set and what the Committee decided (raise / lower / hold);
for every statement with a dissent, who voted against. Three phrasings per kind, picked by meeting,
and three ways of naming the meeting ("December 18, 2024", "2024-12-18", "December 2024" when it's the
only meeting that month), so the set tests date handling and not one template.

The answers are facts parsed out of the documents (fomc/ingest.py), not written by hand, so the set
grows by itself when new statements are ingested; tests/test_ingest.py pins the parser on real text.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from fomc.ingest import load, parse_range

OUT = Path(__file__).resolve().parent / "golden.jsonl"

RANGE_Q = [
    "What target range for the federal funds rate did the FOMC set {when}?",
    "{When_cap}, what was the federal funds rate target range after the FOMC's decision?",
    "Where did the Fed put the federal funds target range {when}?",
]
DECISION_Q = [
    "What did the FOMC decide to do with the federal funds rate target range {when}?",
    "{When_cap}, did the Fed raise, cut or hold the federal funds rate?",
    "What policy action did the Committee take on the federal funds rate {when}?",
]
DISSENT_Q = [
    "Who voted against the FOMC's policy action {when}?",
    "{When_cap}, which members dissented from the FOMC decision?",
    "Who dissented at the FOMC meeting {when}?",
]


def when(d: date, k: int, alone_in_month: bool) -> str:
    style = k % 3 if alone_in_month else k % 2
    if style == 0:
        return f"at its {d:%B} {d.day}, {d.year} meeting"
    if style == 1:
        return f"at the {d.isoformat()} meeting"
    return f"in {d:%B} {d.year}"


def build() -> list[dict]:
    docs = load()
    months = {}
    for x in docs:
        months[x["date"][:7]] = months.get(x["date"][:7], 0) + 1
    out = []
    for i, x in enumerate(docs):
        d = date.fromisoformat(x["date"])
        f = x["facts"]
        alone = months[x["date"][:7]] == 1
        for kind, templates, gold in (
            ("range", RANGE_Q, f["range"]),
            ("decision", DECISION_Q, f["decision"]),
            ("dissent", DISSENT_Q, f["dissents"]),
        ):
            if kind == "dissent" and not gold:
                continue
            k = i + len(out)
            w = when(d, k, alone)
            q = templates[k % 3].format(when=w, When_cap=w[0].upper() + w[1:])
            out.append({"id": f"{kind}-{x['date']}", "kind": kind, "question": q, "doc_id": x["id"], "gold": gold})
    return out


# ---------------------------------------------------------------- scoring

DEC_WORDS = {
    "raise": r"\b(rais|increas|hik|tighten)",
    "lower": r"\b(lower|cut|reduc|decreas|eas)",
    "hold": r"\b(maintain|keep|kept|hold|held|unchanged|leave|left)",
}
_DEC = r"(\d+(?:\.\d+)?)\s*%?\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)\s*(?:%|percent)"


def answer_range(text: str) -> tuple[float, float] | None:
    """A range in the Fed's fractions ('4-1/4 to 4-1/2 percent') or decimals ('4.25% to 4.5%')."""
    r = parse_range(text.replace("‑", "-"))
    if r:
        return r
    m = re.search(_DEC, text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def answer_decision(text: str) -> str | None:
    found = [k for k, rx in DEC_WORDS.items() if re.search(rx, text.lower())]
    return found[0] if len(found) == 1 else None


def surname(name: str) -> str:
    return name.split()[-1].lower()


def correct(item: dict, answer: str) -> bool:
    if item["kind"] == "range":
        return answer_range(answer) == tuple(item["gold"])
    if item["kind"] == "decision":
        return answer_decision(answer) == item["gold"]
    a = answer.lower()
    return all(surname(n) in a for n in item["gold"])


if __name__ == "__main__":
    items = build()
    OUT.write_text("".join(json.dumps(x) + "\n" for x in items))
    kinds = {}
    for x in items:
        kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
    print(len(items), "questions", kinds, "->", OUT)
