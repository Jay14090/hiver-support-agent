# Decision Log

Non-obvious decisions, in the format: **Decision → alternatives considered → why this one
→ what would change my mind.** Ordered roughly by how much each one affects the results.

---

### D1. Time-based split, not random

**Decision.** Sort threads by root-tweet timestamp. Oldest 70% → retrieval corpus and
few-shot pool. Next 15% → dev (threshold tuning). Newest 15% → test (golden set drawn
from here only).

**Alternatives.** Random 70/15/15; grouped split by customer id; k-fold CV.

**Why.** A random split puts near-duplicate *future* threads into the retrieval index.
Support data is enormously repetitive — the same "my downloads vanished after the update"
appears hundreds of times — so a random split lets the agent retrieve what is effectively
the answer key and inflates reply quality invisibly. Time-based also mirrors deployment:
you always answer today's ticket using yesterday's history. `assert_no_leakage()` runs on
every decision and fails loudly if a retrieved id is ever in dev or test.

**What would change my mind.** If the goal were a static benchmark rather than a
deployment estimate, k-fold would give tighter CIs on n=220. I would still never use a
random split with a retrieval component.

---

### D2. The resolution-bearing filter — and why it is the biggest bias in the project

**Decision.** Keep only threads containing at least one brand reply with *actionable
content* (troubleshooting steps, a support link, a policy explanation, or a confirmed
action). Pure handoffs — "we've sent you a DM", "sorry to hear that!" — are dropped.
A transparent keyword/heuristic scorer, never an LLM call.

**Alternatives.** Keep every thread with any brand reply; LLM-score every reply;
length threshold only.

**Why.** You cannot ground a drafted reply in a thread whose only brand turn is "please
DM us". Roughly 64% of SpotifyCares threads are exactly that. A heuristic (not an LLM)
because it runs over 28k threads for free, is deterministic so the funnel reproduces, and
I have to defend every one of its decisions live.

**Measured, not asserted.** Validated on a random sample of 100 replies against an
independent adjudicator: **precision 0.806, recall 0.853, F1 0.829, Cohen's κ 0.736**,
heuristic positive rate 0.36 vs adjudicator 0.34.

**⚠ This filter is the single largest source of survivorship bias.** Everything is
evaluated on the subset of problems the brand chose to answer publicly *and*
substantively. Billing and account work moves to DM and is largely invisible.

**What would change my mind.** Access to DM transcripts would remove the need for the
filter entirely and would change the intent distribution substantially.

---

### D3. Two blind machine label passes + human adjudication, never "hand-labelled"

**Decision.** Pass A (`gpt-4o-mini`, intent-first framing) and Pass B (`gpt-4.1-mini`,
escalation-first framing, shuffled intent order, shuffled item order, no access to A).
All disagreements plus a random 20% audit of agreements go to a human queue. Only rows a
human opened are marked `human_adjudicated`; everything else carries `provisional: true`
and `run_eval.py` stamps PROVISIONAL on every number while any remain.

**Alternatives.** Label 220 by hand from scratch (~6 hours); single LLM pass; three-way
majority vote.

**Why.** A single LLM pass gives no reliability estimate at all. Two *differently framed*
passes give a real one — and it turned out to matter enormously (see D4). The 20% audit
of agreements exists so that human rubber-stamping is detectable rather than silent.

**What would change my mind.** A second human annotator would let me report a genuine
inter-annotator κ, which is strictly better evidence than machine–machine κ. That is the
single highest-value thing another week would buy.

---

### D4. Reporting that my escalation ground truth is unstable

**Decision.** Report prominently that Pass A escalates **15.0%** of the golden set and
Pass B escalates **22.3%** of the *same* items (intent κ 0.655, escalate κ 0.703).

**Alternatives.** Pick the "better" pass and report only that; average them; quietly
report κ without the rate gap.

**Why.** The two prompts differ only in framing. A 7-point swing from framing alone means
the escalation boundary is genuinely ambiguous — not that one prompt is broken. Since
router "accuracy" is measured *against* that boundary, this caps how much any routing
number can mean. Hiding it would make the results look better and be less true.

**What would change my mind.** Nothing about reporting it. A real support org's
historical routing decisions would replace my invented policy with an observable one.

---

### D5. Asymmetric cost model (10:1), reported with a sensitivity curve

**Decision.** cost(false-auto on a should-escalate) = 10 × cost(false-escalate). Primary
routing metric is **expected cost per 100 messages**, not accuracy or F1.

**Alternatives.** Accuracy; F1 on escalate; a symmetric cost.

**Why.** A router is a decision system. Accuracy treats "annoyed a customer by escalating
needlessly" and "auto-replied to a fraud report" as equal errors, which is absurd. The
asymmetry is what makes always-escalate a genuinely strong baseline rather than a joke.

**The 10:1 ratio is asserted, not measured** — I have no data on the true cost of either
error. So `cost_sensitivity()` reports the verdict at ratios 1, 2, 5, 10, 20 and 50, and
the report shows whether the ranking of systems is stable across them.

**What would change my mind.** Real handling-cost and churn data would replace the guess.

---

### D6. Hybrid router: deterministic rules → model → confidence gates

**Decision.** Three layers in that fixed order.

**Alternatives.** Pure LLM router; pure rules; a trained classifier on routing labels.

**Why.** Abuse, legal/privacy demands, refund requests and account compromise are
safety-critical: they must not depend on a model's mood, and I must be able to point at
the exact line that fired. A model that is right 97% of the time is not good enough where
being wrong is a headline. But rules are bad at judgement calls ("have they already tried
the fix?", "is this frustration or genuine distress?"), so the model handles those.
Confidence gates come last as a humility layer: not knowing is itself a reason to
escalate. Every decision carries an enum reason code **and** a human sentence.

**What would change my mind.** If the rule layer produced many false escalations in the
error analysis, I would narrow the patterns rather than remove the layer.

---

### D7. `must_include` / `must_not_include` instead of reference replies

**Decision.** Each golden example carries 2–3 things any acceptable reply must do and 1–3
things it must not do, checked mechanically.

**Alternatives.** One gold reference reply + BLEU/ROUGE/embedding similarity; LLM judge only.

**Why.** There is no single correct reply to an open-ended support message, so similarity
to one arbitrary reference mostly measures paraphrase distance. Must/must-not encodes
what actually matters ("acknowledge the double charge", "don't promise an amount"), costs
nothing, and is deterministic. The check is a crude token-overlap floor, not a semantic
judgement — the LLM judge covers semantics separately, and the crudeness is documented
rather than dressed up.

**What would change my mind.** With more annotation budget, 3–5 reference replies per
example plus a learned metric would be better.

---

### D8. Judge from a different model, four axes scored in separate calls

**Decision.** Judge = `gpt-4.1-mini`, generator = `gpt-4o-mini`. Each of groundedness /
actionability / brand-fit / safety is scored in its own API call, plus a separate binary
"would a support agent send this unedited?".

**Alternatives.** One call returning four scores (4× cheaper); same model as generator;
human-only scoring.

**Why.** One call for four scores produces a halo — a fluent reply scores well on
groundedness because it *read* well. Four calls measure four things instead of one thing
four times. Different model because self-preference is a known LLM-judge pathology.

**Honest limitation.** Only `OPENAI_API_KEY` exists on this machine, so judge and
generator share a provider — a weaker separation than a different family. That is
precisely why the self-preference probe is **mandatory** here and its measured delta is
reported rather than assumed away.

**What would change my mind.** An Anthropic or Google key would give real family
separation and I would switch immediately.

---

### D9. Committed record/replay LLM cache

**Decision.** Every response is written to `cache/llm/<sha256>.json` and committed.
`make demo` runs with `LLM_OFFLINE=1`, where a cache miss is a hard error, not a fallback
to a live call.

**Alternatives.** Ship results.json only; require the grader to supply a key; mock the LLM.

**Why.** The brief requires headline results reproducible in under 15 minutes, and a
grader will not have my key. Making a miss a *hard error* is the important half: it means
`make demo` cannot silently succeed while secretly needing network access, so the cache is
provably complete for that path.

**Cost.** ~24MB of JSON in the repo. Worth it.

**What would change my mind.** If the cache pushed the repo over ~100MB I would prune it
to only the calls `make demo` replays.

---

### D10. SpotifyCares as the brand, with a pre-registered fallback rule

**Decision.** SpotifyCares, with an automatic fallback to AppleSupport then Delta if the
brand yielded <2,000 usable threads or <15% resolution-bearing replies.

**Alternatives.** An airline (highest volume); a telco; AppleSupport.

**Why.** Airlines and telcos overwhelmingly reply "please DM us" — grounded reply drafting
is impossible without observable public resolutions. Spotify's intents are also crisply
separable (playback / billing / account / plan) rather than one undifferentiated blob.

**Note on process.** The fallback rule **fired** on my first run (12.2% resolution rate)
and I did *not* immediately switch brands — I first checked whether my own filter was
wrong. It was: every Spotify support link is a `t.co` shortlink and my link patterns only
matched `<URL>` and `spotify.com/support`, so the strongest positive signal never fired.
Fixing that bug took the rate to 36.4% and the check passed honestly. The full progression
(12.2% → 58.5% over-permissive → 36.4% final) is in ASSUMPTIONS.md [A-07].

**What would change my mind.** Nothing now; the D3 check passes on real numbers.

---

### D11. Not fine-tuning anything

**Decision.** Prompting + retrieval only.

**Alternatives.** Fine-tune a small classifier on the machine-labelled corpus; LoRA a
small generator on real brand replies.

**Why.** The brief says the proof is worth more than the system, and evaluation is where
the budget belongs. Fine-tuning on ~7k *machine* labels would teach a model to imitate
`gpt-4o-mini`, capping it at the labeller's accuracy while looking impressive. It also
produces an artifact I cannot fully explain line-by-line under live questioning.

**What would change my mind.** A few thousand human-labelled examples would make a
fine-tuned classifier both cheaper and better than prompting at inference time.

---

### D12. No vector database

**Decision.** BM25 (`rank_bm25`) + a numpy matrix of MiniLM embeddings, fused with
Reciprocal Rank Fusion.

**Alternatives.** FAISS, Chroma, pgvector, LanceDB.

**Why.** The corpus is ~7,000 threads. A numpy matmul answers in milliseconds. A vector DB
would add infrastructure to the reproduction path, more dependencies to the 15-minute
budget, and nothing to the results. RRF specifically because it needs no score
normalisation between two incomparable scoring scales.

**What would change my mind.** ~1M+ threads, or a latency requirement under load.

---

### D13. Hybrid retrieval rather than dense-only

**Decision.** BM25 + dense, RRF-fused, then reranked by an LLM-free heuristic (fusion
score, then resolution quality, then recency).

**Alternatives.** Dense only; BM25 only; a cross-encoder reranker.

**Why.** Support text is full of exact tokens dense search blurs (`Duo`, `Hulu`, `403`,
`Autoplay`) and full of paraphrase BM25 misses ("won't play" / "no sound" / "stuck
buffering"). Recency is in the rerank because product eras matter: a 2017 fix for an old
app version is worse than a newer equivalent answer.

**What would change my mind.** A cross-encoder would likely improve top-3 quality, but
retrieval quality is currently *unmeasured* (see the one-more-week list) — I would measure
recall@k before adding a reranker I could not justify.

---

### D14. Reporting a taxonomy that clustering only half-supported

**Decision.** Ship 11 intents, and mark in `reports/figures/taxonomy.png` which ones have
**no cluster support**.

**Alternatives.** Ship only the ~5 intents clustering found; claim all 11 were "induced
from the data"; force a larger k until every intent appeared.

**Why.** HDBSCAN found **0 clusters (100% noise)** and KMeans silhouette was ~0.04 at
every k from 6 to 20 — there is very little natural structure in short support tweets, and
the topic mass is wildly unbalanced (content-missing alone is ~50%). Clustering found the
big blobs. The remaining intents are real (confirmed by keyword probes and reading) but
their provenance is *manual*, and saying "induced from the data" without that caveat would
overstate the statistical support.

**What would change my mind.** Stronger embeddings (e5/bge) or clustering the *brand
replies* instead of customer messages might find real structure — in the one-week list.

---

### D15. Fixing my own filters rather than tuning until the numbers looked good

**Decision.** Every filter that produced a suspicious number was *diagnosed by reading
samples*, then either fixed as a bug or measured and left alone.

**Alternatives.** Move the threshold until the number cleared the gate; drop the
inconvenient filter entirely; switch brands to escape a low resolution rate.

**Examples.** (a) The resolution scorer's t.co blindness — a bug, fixed. (b) "Can you DM
us your account's email" scoring as a resolution — a genuine gap in the handoff patterns,
fixed. (c) The English filter passing an entire cluster of romanised Indonesian — the
script check was necessary but not sufficient, so an English function-word gate was added,
then measured (it produced 3 false positives, which were fixed). (d) A "hard case" quota
that flagged 24% of the sample because plain negation appears in nearly every support
complaint — the definition was too loose, and was tightened to land at exactly 15%.

**Why this is a decision and not just debugging.** The tempting alternative each time was
to move a threshold until the number looked acceptable. The rule I applied instead: change
the code only where reading the data shows the code is *wrong*, and where the number is
merely inconvenient, measure it and report it.

**What would change my mind.** Nothing. This is the part I would most want to be asked
about.
