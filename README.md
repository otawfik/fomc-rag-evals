# fomc-rag-evals

Question answering over every Federal Reserve FOMC policy statement since 2021, with an evaluation
suite that runs on every push and **fails the build if answer quality drops**.

Ask *"Who dissented in December 2024?"* or *"What target range did the FOMC set at its September 17,
2025 meeting?"* and get the sentence that answers it, cited to the statement on federalreserve.gov.

![ci](https://github.com/otawfik/fomc-rag-evals/actions/workflows/ci.yml/badge.svg)

**Stack:** Hugging Face Transformers + sentence-transformers (embeddings, cross-encoder reranker,
extractive reader) · Chroma and Postgres/pgvector · LangGraph · Amazon Bedrock (optional answerer and
LLM-as-judge) · FastAPI · Docker · GitHub Actions

## Results

108 golden questions over 47 statements (242 chunks), plus 37 hand-written held-out questions (the
original 12 and 25 added later). From `evals/report.json`; CI recomputes these on every push, on
Chroma and on pgvector.

| | naive RAG | agent |
|---|---|---|
| right statement retrieved first (hit@1) | 28.7% | **100%** |
| right statement in the top 5 (hit@5) | 54.6% | **100%** |
| answer correct | 30.6% | **100%** |
| ↳ target range / decision / dissenters | 23.4% / 34.0% / 42.9% | 100% / 100% / 100% |
| answer cites the right statement | 24.1% | **100%** |
| latency p50 / p95 (CPU, M-series laptop) | 182 / 226 ms | 141 / 241 ms |
| **held-out phrasings, original 12**, answer correct | 25.0% | **91.7%** (11 of 12) |
| **held-out phrasings, all 37**, answer correct | 13.5% | **81.1%** (30 of 37) |

The held-out numbers are the honest ones. The golden questions come from templates; the held-out ones
are phrased differently ("did rates go up, down or stay put", "who were the no votes") and are never
tuned on. On the 37, the agent retrieves and cites the right statement 36 times; the other is a date
the resolver can't read ("the last meeting of 2023").

**Semantic routing moved held-out from 58.3% to 91.7% on the original 12.** Before it, the keyword
intent router didn't recognise most paraphrases. Now the question is embedded and compared with
example questions per intent, with the keyword router as the fallback (`fomc/router.py`). Golden
stayed at 100%. How that number was kept honest:

- The router was tuned on a separate dev set (`evals/dev.py`, 30 paraphrases, 46.7% → 96.7%). That
  number flatters, since the examples and thresholds were chosen with it in view.
- The 25 new held-out questions were written and committed before the router existed (the git history
  shows the order), and the held-out sets were scored once, after tuning was finished.
- A caveat: the dev and held-out questions were written by the same author at the same time, so
  they share habits of phrasing. Questions from real users would be a harder test than either.
- 37 questions is a small sample: one question is 2.7 points, and the 95% interval around 81% runs
  from roughly 66% to 91%.

The latency row was re-measured on a slower day for the laptop (naive RAG, which this change doesn't
touch, went from 112 to 182 ms p50). Measured back to back, the semantic router adds about 12 ms per
question: one extra embedding of the question.

### What moved the numbers

Each row adds one step to the one above (`python -m evals.ablation`):

| pipeline | golden (108) | held-out, original 12 | held-out, all 37 | dev (30, tuned on) |
|---|---|---|---|---|
| vector search only (naive RAG) | 30.6% | 25.0% | 13.5% | 10.0% |
| + resolve the meeting date, filter retrieval | 74.1% | 41.7% | 29.7% | 33.3% |
| + cross-encoder rerank | 84.3% | 50.0% | 43.2% | 53.3% |
| + keyword intent routing, canonical question for the reader | 88.9% | 50.0% | 48.6% | 53.3% |
| + keep the voting roll out of range/decision answers | **100%** | 58.3% | 48.6% | 46.7% |
| + semantic intent routing, keywords as the fallback | **100%** | **91.7%** | **81.1%** | 96.7% |

The voting-roll rule is what takes the golden set to 100%, but it only fires when the intent is known,
so on paraphrases the keyword router mislabels its effect is mixed (the original 12 gain one question,
dev loses two). Semantic routing is the step that makes the earlier ones apply to questions outside the
templates.

**Why naive RAG fails here:** FOMC statements are near-identical boilerplate from meeting to meeting.
Embedding similarity can't tell *December 2024* from *January 2024*, so plain vector search retrieves a
plausible statement from the wrong meeting 71% of the time and answers confidently from it.

**The voting-roll trap:** at meetings with a dissent, the voting paragraph says the dissenter
"preferred to lower the target range ... to 3-1/2 to 3-3/4 percent". It matches range and decision
questions well, and it's the wrong answer. It's evidence for *who dissented*, never for *what was
decided*.

## How it works

```
question ─► resolve ─► retrieve ─► grade ─┬─► rerank ─► answer ─► verify ─► answer + citation
             │            ▲                │
             │            └──── widen ◄────┘  nothing relevant: widen to the month, then drop the filter
             └ intent (range / decision / dissent / other): by meaning, keywords as the fallback; and the meeting date(s)
```

A LangGraph state machine (`fomc/agent.py`):

- **resolve** parses the meeting out of the question ("December 18, 2024", "2024-12-18", "March
  2022", a year) against the meetings actually indexed, and routes on intent. The router
  (`fomc/router.py`) strips the date words, embeds what's left with the retriever's bge-small model and
  compares it with six example questions per intent. The closest intent wins if it is similar enough
  and clearly ahead of the runner-up; otherwise the keyword router decides. A close call between
  *range* and *decision* is accepted, because the same sentence answers both.
- **retrieve** is vector search with bge-small-en-v1.5 embeddings, filtered to that meeting: a
  metadata `where` in Chroma, a `WHERE date = ANY(...)` in pgvector.
- **grade** drops chunks below a similarity floor. If none survive, **widen** relaxes the filter and
  retries, at most twice.
- **rerank** re-scores the survivors with a cross-encoder (ms-marco-MiniLM-L-6-v2) against a canonical
  form of the question, and keeps the best two paragraphs.
- **answer** runs a Hugging Face extractive reader (DistilBERT fine-tuned on SQuAD) over them and
  returns the whole sentence containing the best span. With `LLM=bedrock` it calls Bedrock instead.
- **verify** flags any answer whose source isn't among the retrieved statements.

## Evals

- **Golden set** (`evals/golden.py`): for every statement, the target range, the decision (raise /
  lower / hold) and every dissenter. The answers are **parsed from the documents**, not typed by hand,
  so the set grows with each new meeting. Three phrasings per question type and three ways of naming
  the meeting.
- **Held-out set** (`evals/heldout.py`): 37 hand-written questions in phrasings the templates don't
  use: the original 12, frozen so the number stays comparable, and 25 written before the semantic
  router existed. Scored and gated separately, and reported both ways.
- **Dev set** (`evals/dev.py`): 30 more paraphrases, plus 8 questions that are none of the three
  kinds. The only set the router is tuned on. `python -m evals.dev -v` shows every routing decision
  with its score and margin.
- **Scoring:** ranges must match exactly, whether written as Fed fractions ("4-1/4 to 4-1/2 percent")
  or decimals ("4.25% to 4.5%"); decisions are classified from the answer; every dissenter must be
  named. Plus hit@1/hit@5/MRR, citation correctness and p50/p95 latency.
- **LLM-as-judge:** `JUDGE=bedrock python -m evals.run` adds a Bedrock pass that grades each answer
  against the reference fact. It's optional because it needs AWS credentials; the deterministic
  scorers above are what gate CI.
- **The gate:** `evals/thresholds.json`. CI runs `python -m evals.run --check` on Chroma and on a real
  Postgres + pgvector service container. Any metric below its floor fails the build, and the score
  table is posted to the run's summary page.

The evals also caught three bugs in their own test data before any model work. The parser had cut
"Esther L. George" at her middle initial. It had missed dissents phrased "voting against the
*monetary policy* action", so July 2026's three dissenters were missing from the gold answers. And it
had split sentences at inline HTML tags.

## Run it

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python -m fomc.ingest          # refresh data/statements.jsonl from federalreserve.gov (optional: it's committed)
python -m evals.golden         # rebuild the golden set from the statements
python -m evals.run --check    # score naive vs agent, enforce the thresholds
python -m evals.ablation       # the "what moved the numbers" table
python -m evals.dev -v         # the router on the dev set: keyword vs semantic, every decision
pytest -q && pytest -q -m slow # unit tests; end-to-end API test

uvicorn fomc.api:app --port 8000
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Who dissented at the FOMC meeting in December 2024?"}'
```

- **pgvector:** `DATABASE_URL=postgresql://user:pass@host/db` switches the store, e.g.
  `docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16`.
- **Bedrock:** `LLM=bedrock BEDROCK_MODEL_ID=<a model your account has enabled> AWS_REGION=us-east-1`,
  with AWS credentials in the environment.
- **Docker:** `docker build -t fomc-rag . && docker run -p 8000:8000 fomc-rag`. The three models are
  baked into the image, so a container starts without downloading them.
- **Deploying on AWS:** the image runs as-is on App Runner or ECS Fargate (port 8000, `/health` for the
  health check), with RDS Postgres + pgvector via `DATABASE_URL`. Hosting isn't included here because
  it costs money.

## Engineering notes

- **NaN from a cross-encoder.** The reranker scored every passage NaN. The weights loaded cleanly with
  no missing keys. The cause: Transformers memory-maps safetensors, one tensor started at an unaligned
  address (`data_ptr() % 64 == 2`), and Apple's Accelerate BLAS returns NaN from a plain matmul on it.
  The fix is `own_weights()`, which copies the weights into aligned memory once. An earlier version of
  the eval numbers was inflated by the NaN ordering; everything above was measured after the fix.
- Transformers 5 removed the question-answering pipeline, so `fomc/llm.py:extract` does the span
  search directly: start × end probabilities over the context tokens, forward spans of at most 40
  tokens, mapped back to characters.
- The Fed's server sends no charset, and statements write fractions with non-breaking hyphens
  ("1‑1/2"). Decoded as Latin-1, those became mojibake and five meetings silently failed to parse.

## Limitations

- **Seven of the 37 held-out questions still fail**, for four different reasons:
  - *The router isn't confident and the keyword fallback is wrong (4).* "fed funds corridor", "what
    band is the overnight rate supposed to trade in" and "whose votes went the other way" fall back to
    keywords and land on "other". "Who was the lone holdout" falls back and the keyword router sees
    "hold" and calls it a decision. Six examples per intent don't cover market slang.
  - *The router is confident and wrong (1).* "Did the Committee pause or keep hiking?" is routed to
    dissent. The same happens to one dev question ("Was April 2021 a no-move meeting?"). Short
    questions about a hold have little for the embedding to go on.
  - *Routed correctly, wrong paragraph retrieved (1).* "Did the central bank ease?" is routed to
    decision, but retrieval inside the statement still searches with the user's words, and the
    decision paragraph isn't in the top five. Searching with the canonical question as well is the
    obvious next step; it wasn't done here because it would have been a fix chosen by looking at a
    held-out failure.
  - *Date not understood (1).* "The Fed's last meeting of 2023" resolves to the whole year. The date
    resolver reads dates, months and years, not ordinals.
- **Routing only has four intents.** A question that mixes two ("was it unanimous, and what was the
  range?") gets one of them.
- **The held-out set is small and has one author** (see Results). 81% on 37 questions is an estimate
  with a wide interval, not a guarantee.
- The extractive reader can only quote, so answers are sentences from the statement, not syntheses.
  The Bedrock answerer writes free-form answers, but it isn't part of the CI gate.
- The corpus is policy statements only, not minutes, press-conference transcripts or projections.

Data: FOMC statements are works of the US federal government, in the public domain.
