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
| p50_ms | 110.0 | 74.2 |
| p95_ms | 138.8 | 123.8 |

Held-out phrasings (12 hand-written questions the templates never use): accuracy 0.583, hit@1 1.0, cited_right 1.0.
