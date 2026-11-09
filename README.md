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

108 golden questions over 47 statements (242 chunks), plus 12 hand-written held-out questions. From
`evals/report.json`; CI recomputes these on every push, on Chroma and on pgvector.

| | naive RAG | agent |
|---|---|---|
| right statement retrieved first (hit@1) | 28.7% | **100%** |
| right statement in the top 5 (hit@5) | 54.6% | **100%** |
| answer correct | 30.6% | **100%** |
| ↳ target range / decision / dissenters | 23.4% / 34.0% / 42.9% | 100% / 100% / 100% |
| answer cites the right statement | 24.1% | **100%** |
| latency p50 / p95 (CPU, M-series laptop) | 112 / 128 ms | 78 / 112 ms |
| **held-out phrasings**, answer correct | 25.0% | **58.3%** (retrieval and citations still 100%) |

The held-out number is the honest one. The golden questions come from templates; the 12 held-out ones
are phrased differently ("did rates go up, down or stay put", "who were the no votes") and are never
tuned on. The agent still finds and cites the right statement every time; where it misses, the keyword
intent router didn't recognise the phrasing (see Limitations).

### What moved the numbers

Each row adds one step to the one above (`python -m evals.ablation`):

| pipeline | golden | held-out |
|---|---|---|
| vector search only (naive RAG) | 30.6% | 25.0% |
| + resolve the meeting date, filter retrieval | 74.1% | 41.7% |
| + cross-encoder rerank | 84.3% | 50.0% |
| + intent routing, canonical question for the reader | 88.9% | 50.0% |
| + keep the voting roll out of range/decision answers | **100%** | **58.3%** |

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
             └ intent (range / decision / dissent / other) and the meeting date(s)
```

A LangGraph state machine (`fomc/agent.py`):

- **resolve** parses the meeting out of the question ("December 18, 2024", "2024-12-18", "March
  2022", a year) against the meetings actually indexed, and routes on intent.
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
- **Held-out set** (`evals/heldout.py`): 12 hand-written questions in phrasings the templates don't
  use. Scored and gated separately.
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

- **Intent routing is keyword-based.** Paraphrases it doesn't know fall through to "other" (held-out
  58%). The next step is semantic routing: embed the question and compare it with intent
  descriptions.
- The extractive reader can only quote, so answers are sentences from the statement, not syntheses.
  The Bedrock answerer writes free-form answers, but it isn't part of the CI gate.
- The corpus is policy statements only, not minutes, press-conference transcripts or projections.

Data: FOMC statements are works of the US federal government, in the public domain.
