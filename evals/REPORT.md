Golden set: 108 questions over 47 FOMC statements (242 chunks), answerer `extractive`, store `ChromaStore`.

| metric | naive RAG | agent |
|---|---|---|
| hit@1 | 0.287 | 1.0 |
| hit@5 | 0.546 | 1.0 |
| mrr | 0.376 | 1.0 |
| accuracy | 0.306 | 1.0 |
| accuracy_decision | 0.34 | 1.0 |
| accuracy_dissent | 0.429 | 1.0 |
| accuracy_range | 0.234 | 1.0 |
| cited_right | 0.241 | 1.0 |
| p50_ms | 169.4 | 131.1 |
| p95_ms | 414.6 | 198.8 |

Held-out phrasings, the original set, 12 hand-written questions: accuracy 0.917 (decision 0.75 / dissent 1.0 / range 1.0), hit@1 1.0, cited_right 1.0.

Held-out phrasings, expanded set (includes the original), 37 hand-written questions: accuracy 0.811 (decision 0.769 / dissent 0.833 / range 0.833), hit@1 0.973, cited_right 0.973.

Dev set (the router was tuned on this one, so it flatters), 30 hand-written questions: accuracy 0.967 (decision 0.9 / dissent 1.0 / range 1.0), hit@1 1.0, cited_right 1.0.
