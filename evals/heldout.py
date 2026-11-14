"""Held-out questions: phrasings the golden templates never use, written by hand to check that intent
routing and date handling generalise instead of matching the templates. Gold answers still come from
the parsed statements (fomc/ingest.py), never typed in.

Two lists. QUESTIONS is the original 12 and is frozen, so its score stays comparable across versions.
EXTRA was added with semantic routing and was written and committed before the router existed (see the
git history), so nothing in the router could have been fitted to it. Neither list is tuned on: tuning
happens on evals/dev.py only."""
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

EXTRA = [
    ("Where did the federal funds rate stand once the May 2023 FOMC meeting wrapped up?", "2023-05-03", "range"),
    ("Give me the fed funds corridor announced on Dec. 14, 2022.", "2022-12-14", "range"),
    ("How high did the Fed take its benchmark rate in July 2023?", "2023-07-26", "range"),
    ("What's the upper and lower bound of the policy rate as of the 2025-12-10 statement?", "2025-12-10", "range"),
    ("Coming out of the November 2024 meeting, borrowing costs were pegged at what level?", "2024-11-07", "range"),
    ("the fed's benchmark rate in jan 2022, what was it?", "2022-01-26", "range"),
    ("What interest rate did policymakers settle on at the 29 October 2025 meeting?", "2025-10-29", "range"),
    ("Following the September 2026 FOMC, what band is the overnight rate supposed to trade in?", "2026-09-16", "range"),
    ("Did the Fed tighten policy in May 2022?", "2022-05-04", "decision"),
    ("Was the Fed's last meeting of 2023 a pause?", "2023-12-13", "decision"),
    ("Did borrowing costs come down at the Fed's October 2025 meeting?", "2025-10-29", "decision"),
    ("Were rates left alone on March 20, 2024?", "2024-03-20", "decision"),
    ("Is it true the FOMC stood pat in June 2025?", "2025-06-18", "decision"),
    ("Which way did the Fed move rates at its 2022-09-21 meeting?", "2022-09-21", "decision"),
    ("Did the central bank ease in December 2025?", "2025-12-10", "decision"),
    ("What was the outcome of the FOMC's September 2026 rate vote?", "2026-09-16", "decision"),
    ("Any change to the policy rate at the 1 February 2023 meeting?", "2023-02-01", "decision"),
    ("Was the June 2022 rate move unanimous? If not, who disagreed?", "2022-06-15", "dissent"),
    ("Which officials objected to the outcome of the December 2025 meeting?", "2025-12-10", "dissent"),
    ("Who was the lone holdout at the FOMC's December 18, 2024 meeting?", "2024-12-18", "dissent"),
    ("Did anyone on the Committee push back on the January 2026 hold?", "2026-01-28", "dissent"),
    ("List the policymakers who didn't go along with the majority on 2026-07-29.", "2026-07-29", "dissent"),
    ("In March 2022, one member wanted a bigger move. Who?", "2022-03-16", "dissent"),
    ("Whose votes went the other way at the April 2026 FOMC?", "2026-04-29", "dissent"),
    ("Which governors disagreed with the Committee's call at the October 2026 meeting?", "2026-10-28", "dissent"),
]


def _items(questions, prefix: str) -> list[dict]:
    facts = {d["date"]: d for d in load()}
    out = []
    for q, day, kind in questions:
        f = facts[day]["facts"]
        gold = {"range": f["range"], "decision": f["decision"], "dissent": f["dissents"]}[kind]
        out.append({"id": f"{prefix}-{kind}-{day}", "kind": kind, "question": q, "doc_id": f"fomc-{day}", "gold": gold})
    return out


def items() -> list[dict]:
    """The original 12."""
    return _items(QUESTIONS, "heldout")


def expanded() -> list[dict]:
    """The original 12 plus EXTRA."""
    return _items(QUESTIONS, "heldout") + _items(EXTRA, "heldout2")
