"""Held-out questions: phrasings the golden templates never use, written by hand to check that intent
routing and date handling generalise instead of matching the templates. Gold answers still come from
the parsed statements (fomc/ingest.py), never typed in."""
from fomc.ingest import load

# (question, meeting, kind)
QUESTIONS = [
    ("How much did the Fed cut rates by in December 2024, and to what range?", "2024-12-18", "range"),
    ("After the Fed's first hike in March 2022, what was the new fed funds range?", "2022-03-16", "range"),
    ("What was the policy rate band set on 2023-07-26?", "2023-07-26", "range"),
    ("fed funds target after the sept 17 2025 meeting?", "2025-09-17", "range"),
    ("In June 2023 did the Committee pause or keep hiking?", "2023-06-14", "decision"),
    ("Did rates go up, down or stay put at the November 2022 FOMC?", "2022-11-02", "decision"),
    ("Was there a rate cut at the 2024-09-18 FOMC meeting?", "2024-09-18", "decision"),
    ("What happened to interest rates at the Fed's January 2025 meeting?", "2025-01-29", "decision"),
    ("Which FOMC members opposed the decision in September 2024?", "2024-09-18", "dissent"),
    ("Name the dissenters at the July 30, 2025 meeting.", "2025-07-30", "dissent"),
    ("Who were the no votes at the October 2026 FOMC meeting?", "2026-10-28", "dissent"),
    ("Who broke ranks with the majority in March 2025?", "2025-03-19", "dissent"),
]


def items() -> list[dict]:
    facts = {d["date"]: d for d in load()}
    out = []
    for q, day, kind in QUESTIONS:
        f = facts[day]["facts"]
        gold = {"range": f["range"], "decision": f["decision"], "dissent": f["dissents"]}[kind]
        out.append({"id": f"heldout-{kind}-{day}", "kind": kind, "question": q, "doc_id": f"fomc-{day}", "gold": gold})
    return out
