# Not Building — and why

The brief grades what I chose *not* to build. This is that list, kept honest as the build goes.

## Locked out of scope from the start (D11)

| Not built | Why |
|---|---|
| **Fine-tuning** (of the classifier or the drafter) | The evaluation is worth more than the model. Fine-tuning would consume most of the budget, needs far more than 220 labels to beat a well-prompted model on 11 classes, and produces an artifact I cannot explain line-by-line in a live interview. Prompting + retrieval is the defensible baseline to beat first. |
| **Agentic tool use** (the agent calling Spotify APIs, issuing refunds, looking up accounts) | The dataset contains no account state, so any tool would be a mock. A mocked tool call cannot be evaluated, and an unevaluated capability is worse than a missing one for this brief. Escalation *is* the tool call here. |
| **Multilingual support** | SpotifyCares answers in many languages; I filter to English. Handling more would need per-language golden labels I cannot produce or verify, and unverifiable coverage is a liability. Non-English volume is measured and reported in the data funnel so the cost of this choice is visible. |
| **A web UI** | Zero evaluation signal. A CLI (`agent --demo`) demonstrates the same behaviour and keeps the reproduction path under 15 minutes. |
| **A vector database** (FAISS/Chroma/pgvector) | The corpus is O(10k) threads. A numpy matrix and an in-memory BM25 index answer in milliseconds, add no infrastructure to the repro path, and I can explain every line of the retrieval code under questioning. A vector DB would be resume-driven development. |
| **Multi-turn conversational agent** | Scope is single customer message + preceding thread context → one decision. Multi-turn would require simulating the customer, which is unevaluable against this dataset. |

## Deliberately deferred (would be next, ranked in REPORT.md §7)

*(populated as the build proceeds — see `reports/REPORT.md` "What I'd do with one more week")*
