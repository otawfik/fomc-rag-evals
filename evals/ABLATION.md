| pipeline | golden (108) | held-out, original 12 | held-out, expanded 37 | dev (30, tuned on) |
|---|---|---|---|---|
| vector search only (naive RAG) | 30.6% | 25.0% | 13.5% | 10.0% |
| + resolve the meeting date, filter retrieval | 74.1% | 41.7% | 29.7% | 33.3% |
| + cross-encoder rerank | 84.3% | 50.0% | 43.2% | 53.3% |
| + intent routing, canonical question | 88.9% | 50.0% | 48.6% | 53.3% |
| + keep the voting roll out of range/decision answers | 100.0% | 58.3% | 48.6% | 46.7% |
| + semantic intent routing, keywords as the fallback | 100.0% | 91.7% | 81.1% | 96.7% |
