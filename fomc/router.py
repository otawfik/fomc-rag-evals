"""Semantic intent routing: which kind of question is this (range / decision / dissent / other)?

The keyword router in fomc/agent.py only knows the words it was given ("raise", "dissent", "target
range"). This one embeds the question with the same bge-small model the retriever uses and compares
it with a few example questions per intent. The intent whose examples sit closest wins, provided it
is close enough (FLOOR) and clearly ahead of the runner-up (MARGIN); otherwise the caller falls back
to the keyword router.

The meeting date is stripped first: "December 2024" says which statement, not which kind of question,
and it otherwise dominates the similarity between two short questions.

The examples, FLOOR and MARGIN were tuned on evals/dev.py only. The held-out questions were not
looked at while tuning.
"""
from __future__ import annotations

import re
from functools import lru_cache

EXAMPLES = {
    "range": [
        "What is the target range for the federal funds rate?",
        "What level was the federal funds rate set at?",
        "What was the interest rate after the meeting?",
        "Where did the Fed set its policy rate?",
        "What percent is the fed funds rate?",
        "How low is the rate now?",
    ],
    "decision": [
        "Did the Fed raise, lower or hold interest rates?",
        "Did the Committee change the federal funds rate or leave it where it was?",
        "Was there a rate increase or a rate cut?",
        "Did rates go up or down?",
        "Did the Fed keep rates steady?",
        "Did the Fed loosen policy or not?",
    ],
    "dissent": [
        "Who voted against the policy action?",
        "Which members dissented?",
        "Did every member vote in favor, or did someone vote against?",
        "Did any member not support the decision?",
        "Who did not agree with the rest of the Committee?",
        "Which voters preferred a different policy?",
    ],
    "other": [
        "What did the statement say about inflation?",
        "How did the Committee describe the labor market and economic activity?",
        "What are the risks to the economic outlook?",
        "What is the Committee doing with its holdings of Treasury securities?",
        "What is the Fed's longer-run goal?",
        "What did the statement mention about recent events?",
    ],
}
# Swept on the dev set (floor 0.6-0.75, margin 0-0.08): answers barely move below floor 0.7 / margin 0.05
# and drop above them, so these are mid-plateau values, not a sharp optimum.
FLOOR = 0.6    # best example must be at least this similar
MARGIN = 0.03  # and this far ahead of the best example of an intent that reads a different paragraph
SAME_SENTENCE = {"range", "decision"}

_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|"
          r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_DATE = re.compile(rf"\b20\d\d-\d\d-\d\d\b|\b{_MONTH}\b\.?|\b\d{{1,2}}(?:st|nd|rd|th)?\b,?|\b20\d\d\b", re.I)


def strip_dates(question: str) -> str:
    """The question without its date words: 'Did the Fed cut in Dec. 2024?' -> 'Did the Fed cut in ?'"""
    return re.sub(r"\s+", " ", _DATE.sub(" ", question)).strip()


@lru_cache(maxsize=1)
def _examples():
    import numpy as np

    from .store import embed

    names = [k for k, v in EXAMPLES.items() for _ in v]
    return names, np.array(embed([q for v in EXAMPLES.values() for q in v]))


def scores(question: str) -> dict[str, float]:
    """Each intent's best cosine similarity to the question."""
    import numpy as np

    from .store import embed

    names, vecs = _examples()
    sims = vecs @ np.array(embed([strip_dates(question)])[0])
    out = {k: -1.0 for k in EXAMPLES}
    for n, s in zip(names, sims):
        out[n] = max(out[n], float(s))
    return out


def route(question: str) -> tuple[str | None, float, float]:
    """(intent or None when not confident, best score, margin over the runner-up)"""
    ranked = sorted(scores(question).items(), key=lambda x: -x[1])
    best, s = ranked[0]
    # range and decision are answered by the same sentence ("decided to lower the target range ... to
    # 4-1/4 to 4-1/2 percent"), so a close call between those two costs nothing: the margin that matters
    # is over the nearest intent that would send the question to a different paragraph
    rivals = [v for k, v in ranked[1:] if not {best, k} <= SAME_SENTENCE]
    return (best if s >= FLOOR and s - rivals[0] >= MARGIN else None), s, s - rivals[0]
