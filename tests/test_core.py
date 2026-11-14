"""Fast unit tests: parsing, date resolution, intent routing, scoring. No models are loaded here."""
import pytest

from evals.golden import answer_decision, answer_range, correct
from fomc import router
from fomc.agent import intent, resolve_dates, route
from fomc.ingest import decision, dissents, parse_range, rate
from fomc.llm import sentence

MEETINGS = ["2022-03-16", "2022-05-04", "2024-12-18", "2025-07-30", "2026-07-29"]


@pytest.mark.parametrize("s,v", [("4-1/4", 4.25), ("1/4", 0.25), ("5", 5.0), ("3-3/4", 3.75), ("0", 0.0)])
def test_rate(s, v):
    assert rate(s) == v


def test_parse_range_takes_the_decision_not_the_dissent():
    text = ("the Committee decided to lower the target range for the federal funds rate by 1/4 percentage point "
            "to 4-1/4 to 4-1/2 percent. ... Voting against the action was Beth M. Hammack, who preferred to "
            "maintain the target range for the federal funds rate at 4-1/2 to 4-3/4 percent.")
    assert parse_range(text) == (4.25, 4.5)
    assert decision(text) == "lower"


def test_decision_words():
    assert decision("the Committee decided to keep the target range for the federal funds rate at 0 to 1/4 percent") == "hold"
    assert decision("the Committee decided to raise the target range for the federal funds rate to 1-1/2 to 1-3/4 percent") == "raise"


def test_dissents_keep_middle_initials_and_lists():
    one = "Voting against the action was Esther L. George, who preferred at this meeting to raise the target range."
    many = ("Voting against the monetary policy action were Beth M. Hammack, Neel Kashkari, and Lorie K. Logan, "
            "who preferred to raise the target range.")
    two = "Voting against the action were Michelle W. Bowman and Christopher J. Waller, who preferred to lower."
    assert dissents(one) == ["Esther L. George"]
    assert dissents(many) == ["Beth M. Hammack", "Neel Kashkari", "Lorie K. Logan"]
    assert dissents(two) == ["Michelle W. Bowman", "Christopher J. Waller"]
    assert dissents("Voting for the monetary policy action were Jerome H. Powell, Chair.") == []


@pytest.mark.parametrize("q,dates,scope", [
    ("What did they do on December 18, 2024?", ["2024-12-18"], "day"),
    ("what about 2025-07-30", ["2025-07-30"], "day"),
    ("At its 18 December 2024 meeting?", ["2024-12-18"], "day"),
    ("in March 2022", ["2022-03-16"], "month"),
    ("fed funds after the sept 2025 meeting", ["2025-07-30"], "year"),  # no 2025-09 meeting here: falls back to the year
    ("anything in 2022?", ["2022-03-16", "2022-05-04"], "year"),
    ("what is the inflation goal?", [], "none"),
])
def test_resolve_dates(q, dates, scope):
    assert resolve_dates(q, MEETINGS) == (dates, scope)


@pytest.mark.parametrize("q,kind", [
    ("Who voted against the action in July 2026?", "dissent"),
    ("Did the Fed raise, cut or hold in May 2022?", "decision"),
    ("What target range did the FOMC set?", "range"),
    ("What is the Committee's inflation objective?", "other"),
])
def test_intent(q, kind):
    assert intent(q)[0] == kind


@pytest.mark.parametrize("q,bare", [
    ("Did the Fed cut in Dec. 2024?", "Did the Fed cut in ?"),
    ("What was the range on 2023-07-26?", "What was the range on ?"),
    ("Who dissented at the July 30, 2025 meeting?", "Who dissented at the meeting?"),
    ("fed funds after the sept 17 2025 meeting", "fed funds after the meeting"),
    ("What may the Committee do?", "What the Committee do?"),  # "may" goes too: harmless, it's only the router's view
])
def test_strip_dates(q, bare):
    assert router.strip_dates(q) == bare


def _fake_scores(monkeypatch, **s):
    monkeypatch.setattr(router, "scores", lambda q: {"range": 0.0, "decision": 0.0, "dissent": 0.0, "other": 0.0, **s})


def test_route_trusts_a_confident_semantic_match(monkeypatch):
    _fake_scores(monkeypatch, dissent=0.8, decision=0.6)
    kind, ask, how = route("Did anyone push back on the hold?")  # keyword router alone says "decision"
    assert (kind, ask, how) == ("dissent", "Who voted against the action?", "semantic")


def test_route_falls_back_to_keywords_when_unsure(monkeypatch):
    _fake_scores(monkeypatch, dissent=0.80, other=0.79)  # too close to call
    assert route("Who dissented?")[::2] == ("dissent", "keyword")
    _fake_scores(monkeypatch, dissent=0.5)  # clear winner, but not similar enough to anything
    assert route("What target range did the FOMC set?")[::2] == ("range", "keyword")


def test_route_ignores_a_close_call_between_range_and_decision(monkeypatch):
    _fake_scores(monkeypatch, range=0.80, decision=0.79, other=0.6)  # same sentence answers both
    assert route("Where did rates end up?")[::2] == ("range", "semantic")


def test_route_other_keeps_the_users_question(monkeypatch):
    _fake_scores(monkeypatch, other=0.9)
    assert route("What did the statement say about inflation?") == ("other", "What did the statement say about inflation?", "semantic")


def test_route_can_be_switched_off():
    assert route("Did the Fed raise, cut or hold?", semantic=False)[::2] == ("decision", "keyword")


def test_sentence_keeps_initials_together():
    ctx = "Inflation remains elevated. Voting against the action was Esther L. George, who preferred to raise. Next."
    assert sentence(ctx, "Esther L. George") == "Voting against the action was Esther L. George, who preferred to raise."


def test_scorers():
    assert answer_range("to 4-1/4 to 4-1/2 percent") == (4.25, 4.5)
    assert answer_range("between 4.25% to 4.5%") == (4.25, 4.5)
    assert answer_decision("the Committee decided to maintain the target range") == "hold"
    assert answer_decision("decided to raise the range; one member preferred to maintain it") is None  # ambiguous
    item = {"kind": "dissent", "gold": ["Michelle W. Bowman", "Christopher J. Waller"]}
    assert correct(item, "Bowman and Waller voted against")
    assert not correct(item, "Bowman voted against")
