"""Dev set: hand-written paraphrases used to tune the semantic intent router (fomc/router.py).

This is the only set the router's exemplars, threshold and margin were tuned on. The held-out
questions (evals/heldout.py) are scored once at the end and never looked at while tuning. Gold answers
come from the parsed statements, as everywhere else.

OTHER holds questions that are none of range / decision / dissent. They have no parsed gold answer;
they only check that the router leaves them alone.

    python -m evals.dev      # routing accuracy and answer accuracy, keyword router vs semantic
"""
import sys

from fomc.ingest import load

# (question, meeting, kind)
QUESTIONS = [
    ("What was the fed funds rate after the June 2022 meeting?", "2022-06-15", "range"),
    ("What range were rates in following the FOMC's 2023-03-22 decision?", "2023-03-22", "range"),
    ("Tell me the interest rate the Fed announced in November 2022.", "2022-11-02", "range"),
    ("How low did the Fed bring rates in September 2025?", "2025-09-17", "range"),
    ("What number did the Fed land on for rates at the Jan 31, 2024 meeting?", "2024-01-31", "range"),
    ("What's the Fed's rate target as of March 2026?", "2026-03-18", "range"),
    ("At what level did officials set short-term interest rates in July 2022?", "2022-07-27", "range"),
    ("fed rate after dec 2021 meeting", "2021-12-15", "range"),
    ("What were the top and bottom of the funds rate range after the 7 May 2025 meeting?", "2025-05-07", "range"),
    ("Where were interest rates when the September 2023 meeting ended?", "2023-09-20", "range"),
    ("Did the Fed move rates in July 2024?", "2024-07-31", "decision"),
    ("Was there a hike at the December 2022 meeting?", "2022-12-14", "decision"),
    ("Did the FOMC loosen policy on November 7, 2024?", "2024-11-07", "decision"),
    ("Did the Fed do nothing in May 2025?", "2025-05-07", "decision"),
    ("Rates: higher, lower, or flat in March 2023?", "2023-03-22", "decision"),
    ("Did policymakers leave rates unchanged at the 2026-06-17 meeting?", "2026-06-17", "decision"),
    ("Was there an increase in rates in February 2023?", "2023-02-01", "decision"),
    ("Did the Fed reduce rates in September 2024?", "2024-09-18", "decision"),
    ("What direction did rates go at the Fed's September 2026 meeting?", "2026-09-16", "decision"),
    ("Was April 2021 a no-move meeting?", "2021-04-28", "decision"),
    ("Was anyone opposed to the March 2022 hike?", "2022-03-16", "dissent"),
    ("Who wasn't on board with the Committee in June 2022?", "2022-06-15", "dissent"),
    ("Which member cast the only contrary vote at the September 2024 meeting?", "2024-09-18", "dissent"),
    ("Was the December 18, 2024 decision backed by every voter? Who didn't back it?", "2024-12-18", "dissent"),
    ("Who refused to support the Fed's decision at the 2025-09-17 meeting?", "2025-09-17", "dissent"),
    ("Which officials split from the rest of the Committee in January 2026?", "2026-01-28", "dissent"),
    ("Who did not support the July 2026 policy action?", "2026-07-29", "dissent"),
    ("Name the members who wanted something different in July 2025.", "2025-07-30", "dissent"),
    ("Were all members in agreement at the October 2025 meeting, and if not who wasn't?", "2025-10-29", "dissent"),
    ("Who stood apart from the consensus at the March 2026 FOMC?", "2026-03-18", "dissent"),
]

OTHER = [
    "What is the Committee's longer-run inflation goal?",
    "How did the statement describe the labor market in June 2024?",
    "What did the Fed say about its balance sheet in May 2022?",
    "How does the FOMC describe economic activity in March 2025?",
    "What risks to the outlook did the Committee mention in March 2023?",
    "Did the statement mention the war in Ukraine in March 2022?",
    "What did the Committee say about inflation in December 2021?",
    "Is the Fed still reducing its holdings of Treasury securities as of June 2024?",
]


def items() -> list[dict]:
    facts = {d["date"]: d for d in load()}
    out = []
    for q, day, kind in QUESTIONS:
        f = facts[day]["facts"]
        gold = {"range": f["range"], "decision": f["decision"], "dissent": f["dissents"]}[kind]
        out.append({"id": f"dev-{kind}-{day}", "kind": kind, "question": q, "doc_id": f"fomc-{day}", "gold": gold})
    return out


def main() -> int:
    from fomc import router
    from fomc.agent import Agent, intent, route
    from fomc.store import build, open_store

    from .golden import correct

    dev = items()
    labelled = [(it["question"], it["kind"]) for it in dev] + [(q, "other") for q in OTHER]
    verbose = "-v" in sys.argv
    for name, fn in (("keyword", lambda q: intent(q)[0]), ("semantic + fallback", lambda q: route(q)[0])):
        wrong = [(q, k, fn(q)) for q, k in labelled if fn(q) != k]
        print(f"routing, {name}: {len(labelled) - len(wrong)}/{len(labelled)} right")
        for q, k, got in wrong if verbose else []:
            print(f"   want {k:8} got {got:8} {q}")
    if verbose:
        for q, k in labelled:
            got, s, m = router.route(q)
            print(f"   {k:8} -> {str(got):8} score {s:.3f} margin {m:.3f}  {q}")

    docs = load()
    store = open_store()
    build(store, docs)
    for name, flag in (("keyword", False), ("semantic + fallback", True)):
        a = Agent(store, [d["date"] for d in docs], semantic=flag)
        rows = [(it, a.ask(it["question"])) for it in dev]
        ok = [correct(it, r["answer"]) for it, r in rows]
        print(f"answers, {name}: {sum(ok)}/{len(dev)} = {sum(ok) / len(dev):.1%}")
        for (it, r), good in zip(rows, ok) if verbose else []:
            if not good:
                print(f"   {it['kind']:8} {r['trace'][0]} | {it['question']} -> {r['answer'][:110]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
