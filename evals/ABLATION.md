| pipeline | golden accuracy | held-out accuracy |
|---|---|---|
| vector search only (naive RAG) | 30.6% | 25.0% |
| + resolve the meeting date, filter retrieval | 74.1% | 41.7% |
| + cross-encoder rerank | 84.3% | 50.0% |
| + intent routing, canonical question | 88.9% | 50.0% |
| + keep the voting roll out of range/decision answers | 100.0% | 58.3% |
