"""Fetch every FOMC policy statement listed on the Fed's calendar page and parse it.

    python -m fomc.ingest            # writes data/statements.jsonl

FOMC statements are works of the US federal government (public domain), so the parsed text is
committed to the repo and the app and evals run offline. Each record holds the statement text
plus the facts the golden test set is built from: the target range, the decision (raise / lower /
hold) and who voted against.
"""
from __future__ import annotations

import html
import json
import re
import time
from datetime import date, datetime
from pathlib import Path

import requests

BASE = "https://www.federalreserve.gov"
CALENDAR = f"{BASE}/monetarypolicy/fomccalendars.htm"
UA = {"User-Agent": "fomc-rag-evals (https://github.com/otawfik/fomc-rag-evals)"}
OUT = Path(__file__).resolve().parent.parent / "data" / "statements.jsonl"

_FRACTION = {"": 0.0, "1/8": 0.125, "1/4": 0.25, "3/8": 0.375, "1/2": 0.5, "5/8": 0.625, "3/4": 0.75, "7/8": 0.875}
# "4-1/4 to 4-1/2 percent", "0 to 1/4 percent", "5 to 5-1/4 percent"
RANGE = re.compile(r"(\d+(?:-\d/\d)?|\d/\d)\s+to\s+(\d+(?:-\d/\d)?|\d/\d)\s+percent")


def rate(s: str) -> float:
    """'4-1/4' -> 4.25, '1/4' -> 0.25, '5' -> 5.0"""
    if "/" in s and "-" not in s:
        return _FRACTION[s]
    whole, _, frac = s.partition("-")
    return int(whole) + _FRACTION[frac]


def parse_range(text: str) -> tuple[float, float] | None:
    m = RANGE.search(text)
    return (rate(m.group(1)), rate(m.group(2))) if m else None


def decision(text: str) -> str | None:
    """raise / lower / hold, from the sentence that sets the target range."""
    m = re.search(r"(?:Committee )?decided to (raise|lower|maintain|keep|reduce|increase)[^.]*target range", text)
    if not m:
        return None
    return {"raise": "raise", "increase": "raise", "lower": "lower", "reduce": "lower", "maintain": "hold", "keep": "hold"}[m.group(1)]


def dissents(text: str) -> list[str]:
    """Names in 'Voting against the action was/were ...' (empty when the vote was unanimous)."""
    m = re.search(r"Voting against (?:the |this )?(?:monetary policy )?action (?:was|were) (.*)", text)
    if not m:
        return []
    rest = m.group(1).split(" who ")[0]
    # end of sentence: a period not straight after a lone capital (keeps "Esther L. George" whole)
    rest = re.split(r"(?<![A-Z])\.(?:\s|$)", rest)[0].rstrip(",; ")
    names = re.split(r",\s*(?:and\s+)?|\s+and\s+", rest)
    return [re.sub(r"\s+", " ", n).strip() for n in names if n.strip()]


def statement_text(page: str) -> tuple[str, str] | None:
    """(release date line, body) from a press-release page."""
    i = page.find('id="article"')
    if i < 0:
        return None
    seg = page[i : page.find("For media inquiries", i)]
    # the Fed's pages use non-breaking hyphens in fractions ("1\u20111/2") and nbsp; normalise both
    seg = html.unescape(seg).replace("\u2011", "-").replace("\u2010", "-").replace("\xa0", " ")
    # paragraphs end at block tags; inline tags (<a>, <span>) are removed without breaking the sentence
    seg = re.sub(r"(?i)</?(p|div|br|li|h\d|tr|table|ul|ol)\b[^>]*>", "\n", seg)
    lines = [x.strip() for x in re.sub(r"<[^>]+>", "", seg).split("\n")]
    lines = [re.sub(r"\s+", " ", x) for x in lines if x and x not in ('id="article">', "Share")]
    if len(lines) < 4:
        return None
    skip = ("For release at", "For immediate release")
    # older pages have no "For media inquiries" line: stop at the first thing that isn't the statement
    # (implementation-note and related-release links, the update stamp, the site footer)
    stop = ("Implementation Note", "Last Update", "Board of Governors of the Federal Reserve System", "Related Information",
            "Plans for Reducing", "Principles for Reducing")
    body = []
    for x in lines[2:]:
        if x.startswith(stop) or re.fullmatch(r"[A-Z][a-z]+ \d{1,2}, \d{4}", x):
            break
        # the Board's street address sits inside the article on some 2021 pages
        if not x.startswith(skip) and "Constitution Avenue" not in x and "Washington, DC" not in x:
            body.append(x)
    return lines[0], "\n\n".join(body)


def fetch() -> list[dict]:
    cal = requests.get(CALENDAR, headers=UA, timeout=60).text
    paths = sorted(set(re.findall(r"/newsevents/pressreleases/monetary(\d{8})a\.htm", cal)))
    out = []
    for d in paths:
        url = f"{BASE}/newsevents/pressreleases/monetary{d}a.htm"
        r = requests.get(url, headers=UA, timeout=60)
        r.encoding = "utf-8"  # the server sends no charset; without this "1\u20111/2" arrives as mojibake
        got = statement_text(r.text)
        time.sleep(0.5)
        if not got:
            continue
        title_date, body = got
        rng = parse_range(body)
        # only policy statements set a target range (the calendar also links other releases)
        if not rng or "Federal Open Market Committee" not in body and "Committee" not in body:
            continue
        day = datetime.strptime(d, "%Y%m%d").date()
        out.append({
            "id": f"fomc-{day.isoformat()}",
            "date": day.isoformat(),
            "title": f"FOMC statement, {day:%B} {day.day}, {day.year}",
            "url": url,
            "text": body,
            "facts": {"range": list(rng), "decision": decision(body), "dissents": dissents(body)},
        })
    return out


def load(path: Path = OUT) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


if __name__ == "__main__":
    docs = fetch()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("".join(json.dumps(d) + "\n" for d in docs))
    print(f"{len(docs)} statements -> {OUT} ({docs[0]['date']} to {docs[-1]['date']})")
    for d in docs:
        print(d["date"], d["facts"])
