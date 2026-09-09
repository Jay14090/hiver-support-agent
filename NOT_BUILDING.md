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

## Intents I deliberately collapsed (Phase 2)

Clustering proposed finer distinctions than a support router can use. Collapsed because
**a support agent would route them identically** — the test I applied to every merge:

| Induced clusters | Collapsed into | Why |
|---|---|---|
| `content_availability_issue` + `album_availability_request` + `music_feature_requests` (1,127 msgs, ~50% of the sample) | `content_missing_or_metadata` | "why isn't X on Spotify", "where's the new album", "so many songs aren't on here" get the same answer. Keeping three labels would have inflated macro-F1 by adding easy, near-duplicate classes. |
| `service_availability_request` ("launch in India") + `app_update_request` ("make an Apple Watch app") + `playlist_management_requests` | `feature_request_or_complaint` | All are "something that does not exist yet". Splitting geography from software from playlist tooling adds label noise with no routing consequence. |
| `service_access_issue` (556 msgs) | `other` | Genuinely incoherent — "you ok spotify?", "I have an issue I can't find an answer to". Forcing it into a topic would have manufactured a class the classifier could not learn and the labels could not defend. |

I also **did not split** any cluster, because none of the nine mixed routing outcomes in
a way that survived reading the members.

**Not adding a `service_outage` intent**, despite outage chatter being visibly present:
it is time-correlated rather than customer-specific, and a router should handle it with
an incident banner, not a per-message intent.

## Deferred, ranked (from REPORT.md §7)

1. **A second human annotator** — every number is capped by label quality, and I have
   machine–machine κ where human–human κ is needed.
2. **Retrieval evaluation (recall@k)** — retrieval feeds both classification and drafting
   and is the largest completely untested link in the chain.
3. **Judge validation against human scores** — the CLI exists; 70 replies would convert
   every reply-quality claim from provisional to evidence.
4. **Per-intent routing thresholds** — one global τ is wrong when `billing_charge_dispute`
   and `praise_or_chitchat` have opposite cost profiles.
5. **Fixing failure mode 1** (profanity read as distress) — 54% of needless escalations.
6. **Re-weighting metrics to the production intent mix** — would turn the 73.6% deflection
   rate from a benchmark artefact into a deployable estimate.
