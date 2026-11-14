"""Run the golden set through the agent and the naive baseline, score both, and gate on thresholds.

    python -m evals.run                 # agent + naive, writes evals/report.json and evals/REPORT.md
    python -m evals.run --check         # also exit 1 if any metric falls below evals/thresholds.json
    JUDGE=bedrock python -m evals.run   # add an LLM-as-judge pass (needs AWS credentials)

Metrics, per pipeline:
  hit@1 / hit@5 / MRR   is the right statement the top hit / in the top five, and how high
  accuracy              answer matches the parsed fact (range exact, decision class, every dissenter named)
  cited_right           the answer's source is the gold statement
  p50 / p95 latency     per question, after a warm-up (model loading isn't counted)
CI runs `--check` on every push: a change that makes answers worse fails the build.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

from fomc.agent import Agent
from fomc.ingest import load
from fomc.store import build, open_store

from .golden import OUT as GOLDEN, correct

HERE = Path(__file__).resolve().parent


def score(rows: list[dict], items: list[dict]) -> dict:
    n = len(items)
    rr, h1, h5, acc, cited = 0.0, 0, 0, 0, 0
    by_kind: dict[str, list[int]] = {}
    for r, it in zip(rows, items):
        ret = r["retrieved"]
        if ret and ret[0] == it["doc_id"]:
            h1 += 1
        if it["doc_id"] in ret:
            h5 += 1
            rr += 1 / (ret.index(it["doc_id"]) + 1)
        ok = correct(it, r["answer"])
        r["correct"] = ok
        acc += ok
        cited += r["source"] == it["doc_id"]
        by_kind.setdefault(it["kind"], []).append(ok)
    lat = sorted(r["latency_ms"] for r in rows)
    return {
        "n": n,
        "hit@1": round(h1 / n, 3),
        "hit@5": round(h5 / n, 3),
        "mrr": round(rr / n, 3),
        "accuracy": round(acc / n, 3),
        **{f"accuracy_{k}": round(sum(v) / len(v), 3) for k, v in sorted(by_kind.items())},
        "cited_right": round(cited / n, 3),
        "p50_ms": round(statistics.median(lat), 1),
        "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 1),
    }


def judge(rows: list[dict], items: list[dict]) -> float:
    """LLM-as-judge on Bedrock: does the answer state the gold fact? Fraction judged correct."""
    import boto3

    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    model = os.environ["BEDROCK_JUDGE_MODEL_ID"]
    yes = 0
    for r, it in zip(rows, items):
        prompt = (
            "You grade answers about Federal Reserve FOMC statements. Reply with exactly YES or NO.\n"
            f"Question: {it['question']}\nReference fact: {json.dumps(it['gold'])}\nAnswer: {r['answer']}\n"
            "Does the answer state the reference fact (ranges in percent; decision raise/lower/hold; dissenters by name)?"
        )
        out = client.converse(modelId=model, messages=[{"role": "user", "content": [{"text": prompt}]}],
                              inferenceConfig={"maxTokens": 3, "temperature": 0})
        r["judge"] = out["output"]["message"]["content"][0]["text"].strip().upper().startswith("YES")
        yes += r["judge"]
    return round(yes / len(rows), 3)


def markdown(report: dict) -> str:
    keys = [k for k in report["agent"] if k != "n"]
    lines = [f"Golden set: {report['agent']['n']} questions over {report['docs']} FOMC statements "
             f"({report['chunks']} chunks), answerer `{report['answerer']}`, store `{report['store']}`.", "",
             "| metric | naive RAG | agent |", "|---|---|---|"]
    for k in keys:
        lines.append(f"| {k} | {report['naive'].get(k, '')} | {report['agent'][k]} |")
    for key, what in (("heldout", "Held-out phrasings, the original set"),
                      ("heldout_expanded", "Held-out phrasings, expanded set (includes the original)"),
                      ("dev", "Dev set (the router was tuned on this one, so it flatters)")):
        if report.get(key):
            h = report[key]
            kinds = " / ".join(f"{k[9:]} {v}" for k, v in h.items() if k.startswith("accuracy_"))
            lines += ["", f"{what}, {h['n']} hand-written questions: accuracy {h['accuracy']} ({kinds}), "
                      f"hit@1 {h['hit@1']}, cited_right {h['cited_right']}."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    docs = load()
    items = [json.loads(x) for x in GOLDEN.read_text().splitlines() if x.strip()]
    if args.limit:
        items = items[: args.limit]
    store = open_store()
    chunks = build(store, docs)
    agent = Agent(store, [d["date"] for d in docs])
    agent.ask("warm-up: what is the Committee's inflation objective?")

    report = {"docs": len(docs), "chunks": chunks, "answerer": agent.reader.name,
              "store": type(store).__name__, "naive": {}, "agent": {}}
    rows = {}
    for name, fn in (("naive", agent.naive), ("agent", agent.ask)):
        rows[name] = [fn(it["question"]) for it in items]
        report[name] = score(rows[name], items)
    # held-out phrasings: reported separately, so tuning on the golden templates can't hide overfitting
    from .dev import items as dev_items
    from .heldout import expanded, items as heldout_items

    extra_rows = []
    for key, its in (("heldout", heldout_items()), ("heldout_expanded", expanded()), ("dev", dev_items())):
        got = [agent.ask(it["question"]) for it in its]
        report[key] = score(got, its)
        extra_rows += [{**r, "id": it["id"], "set": key, "gold": it["gold"]} for r, it in zip(got, its) if key != "heldout"]
    if os.environ.get("JUDGE") == "bedrock":
        report["agent"]["judge_correct"] = judge(rows["agent"], items)

    (HERE / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (HERE / "REPORT.md").write_text(markdown(report))
    (HERE / "answers.jsonl").write_text("".join(json.dumps({**r, "id": it["id"], "gold": it["gold"]}) + "\n" for r, it in zip(rows["agent"], items)) + "".join(json.dumps(r) + "\n" for r in extra_rows))
    print(markdown(report))

    if args.check:
        th = json.loads((HERE / "thresholds.json").read_text())
        # "heldout_accuracy" reads report["heldout"]["accuracy"]; everything else reads the agent's golden scores
        got = {**report["agent"], **{f"{name}_{k}": v for name in ("heldout", "heldout_expanded", "dev") for k, v in report[name].items()}}
        bad = [f"{k}: {got[k]} < {v}" for k, v in th.items() if not k.endswith("_ms") and got[k] < v]
        bad += [f"{k}: {got[k]} > {v}" for k, v in th.items() if k.endswith("_ms") and got[k] > v]
        if bad:
            print("QUALITY GATE FAILED\n  " + "\n  ".join(bad), file=sys.stderr)
            return 1
        print("quality gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
