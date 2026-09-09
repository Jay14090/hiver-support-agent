# Walkthrough — a 10-minute tour, and the questions I'd least like to be asked

Written for a live session where you can ask me to change any of this on the spot.

---

## Part 1 — the 10-minute code tour

**Start here, in this order.** Every module is under ~250 lines and has no inheritance,
no metaprogramming, and no framework.

| # | File | What to look at | Why it matters |
|---|---|---|---|
| 1 | `src/support_agent/config.py` | The whole file | Every constant in the system. When you ask "where does that threshold come from?", the answer is always one file. |
| 2 | `src/support_agent/threads.py` | `_longest_brand_chain`, `_norm_id`, `split_children` | Thread reconstruction, with the five named traps (T1–T5). Each has a test. |
| 3 | `src/support_agent/resolution.py` | `score_reply` | The filter that decides what data exists. Transparent scoring, no LLM. |
| 4 | `src/support_agent/retrieval.py` | `search`, `assert_no_leakage` | BM25 + dense, RRF-fused. The leakage assertion runs on every decision. |
| 5 | `src/support_agent/route.py` | `DETERMINISTIC_RULES`, `route` | Three layers: rules → model → confidence gates, in that fixed order. |
| 6 | `src/support_agent/agent.py` | `SupportAgent.handle` | 40 lines. Thin orchestrator, all the interesting logic lives in the heads. |
| 7 | `eval/metrics.py` | `expected_cost_per_100`, `bootstrap_ci`, `is_meaningful_gap` | Why routing is scored as a decision, and why no gap smaller than the CI is ever claimed. |
| 8 | `eval/run_eval.py` | `provisional_status` | The mechanism that makes it impossible to report machine labels as adjudicated. |

**One-line mental model:** retrieve 3 similar *resolved* threads from the past → classify
the intent → draft a reply grounded in those threads and cite them → route through rules,
then the model, then confidence gates.

**To add a new intent** (a thing you might well ask me to do live):
1. add it to `INTENTS` in `config.py`,
2. add a `## \`intent_name\`` section to `golden/CODEBOOK.md` with a definition, 3 real
   positives, 2 near-misses and a tie-break rule against its most confusable neighbour,
3. `python scripts/check_codebook.py` (the gate fails if any intent lacks an entry),
4. re-run the label passes and the eval.
Nothing else needs to change — the codebook is the single source of truth, parsed by
`codebook.py` and injected into every prompt.

---

## Part 2 — the fifteen questions

### 1. Why a time-based split?
Because retrieval makes a random split cheat. Support data is enormously repetitive; a
random split puts near-duplicate *future* threads into the index, so the agent retrieves
something close to the answer key and reply quality inflates invisibly. Time-based also
matches deployment: you answer today's ticket with yesterday's history. `assert_no_leakage()`
fails loudly on every decision if a retrieved id is ever in dev or test.

### 2. Why should I trust your judge?
Partly you shouldn't, and I measured how much. I report Spearman ρ, Krippendorff's α,
exact-match and within-±1 against human scores on the same rubric, plus three bias probes
(position-swap flip rate, length correlation, self-preference delta). Judge and generator
share a provider — only `OPENAI_API_KEY` exists on this machine — so the separation is
weaker than a different model family, which is exactly why the self-preference probe is
mandatory here rather than optional. If ρ < 0.6 the report says the judge-based deltas are
weak evidence, in those words.

### 3. What is the weakest part of this system?
**The escalation ground truth.** Pass A escalates 15.0% of the golden set; Pass B, differing
only in prompt framing, escalates 22.3% of the *same* items. A 7-point swing from framing
alone means the boundary is genuinely ambiguous — and since router "accuracy" is measured
against that boundary, every routing number inherits that ambiguity. The escalation policy
is also *mine*, not Spotify's, because their real routing is unobservable from public
tweets.

### 4. Where does it break at 10× volume?
Not in retrieval — BM25 plus a 7k×384 numpy matrix is microseconds, and it would still be
fine at 70k. It breaks on **cost and latency**: three sequential LLM calls per message
(classify, draft, route). At 10× that is the whole bill. The fix is cheap and I did not
build it: batch the classify+route calls into one, cache by near-duplicate message hash
(support traffic is extremely repetitive — a big fraction of inbound would hit cache), and
only call the drafter for messages that route to `auto`. Escalated messages don't need a
draft at all, and that alone removes ~15% of drafting calls.

### 5. Why did the copy-paste baseline beat you on groundedness?
If it did — and it plausibly does — it's because B1 copies a **real human reply verbatim**
from a similar past thread. That reply is guaranteed to contain no hallucinated policy and
to be in perfect brand voice, because a Spotify agent wrote it. The LLM has to *generate*
and can drift. What B1 cannot do is be *relevant* when no near-duplicate exists: it copies
the nearest reply regardless of whether it answers this question. So I'd expect B1 to win
on groundedness and safety and lose on actionability and relevance. That trade is exactly
why both are reported per-axis rather than as one number.

### 6. Your macro-F1 went up 0.03 after the Phase 8 fix. Is that real?
Only if the CIs don't overlap, and at n=220 the 95% CI half-width on macro-F1 is roughly
±0.05–0.07. So a 0.03 gap is **not a result** and I don't claim it. `is_meaningful_gap()`
is called for every pairwise comparison and `run_eval.py` prints "CIs OVERLAP — this gap is
NOT a result and must not be claimed" when they do.

### 7. Show me where the survivorship bias is.
`src/support_agent/resolution.py`. It drops every thread whose only brand reply is a
handoff ("we've sent you a DM") or an empty acknowledgement — about 64% of SpotifyCares
threads. So the whole system is evaluated on the subset of problems the brand chose to
answer publicly *and* substantively. Billing and account work moves to DM and is largely
invisible: `billing_charge_dispute` is 0.8% of the labelled corpus. I validated the filter
itself (P=0.806, R=0.853, κ=0.736 against an independent adjudicator on 100 random replies)
so at least the bias is *characterised* rather than unknown.

### 8. Your golden labels are machine-generated. Isn't that circular?
Partly, and the repo never calls them hand labels. Two independent passes with different
models and different framings give a reliability estimate (intent κ 0.655, escalate κ 0.703);
everything they disagree on plus a random 20% audit of agreements goes to a human queue.
Rows a human hasn't opened carry `provisional: true`, and `run_eval.py` stamps PROVISIONAL
on every number while any remain. The circularity that *does* survive: the classifier and
the labeller are the same model family, so the classifier is being graded partly on
matching its own biases. A second human annotator is the fix.

### 9. Why is `always_escalate` in your results table?
Because it's a genuinely strong policy under a 10:1 cost asymmetry — zero missed
escalations by construction — and because if my agent can't beat it on expected cost, that
is the finding. It costs 85.0 per 100 messages on the golden set. Notably **B1's keyword
router costs 106.4**, i.e. worse than escalating everything, despite deflecting 93.6% of
traffic. That is the single clearest illustration of why accuracy is the wrong routing
metric.

### 10. Why is the taxonomy 11 intents and not 8, or 20?
Clustering gave weak evidence either way: HDBSCAN found 0 clusters (100% noise) and KMeans
silhouette was ~0.04 at every k from 6 to 20. So k=9 was a judgement call inside the
brief's target band, not an argmax — silhouette argmax was k=7. The clusters established
topic *mass* (content-missing is ~50%) and produced 5 intents; the other 6 were confirmed
by keyword probes and reading, and `reports/figures/taxonomy.png` marks them in red as
having **no cluster support**. I'd rather show that honestly than claim 11 data-induced
intents.

### 11. Your `must_include` check is just token overlap. Isn't that crude?
Yes, deliberately, and it's documented as a floor rather than a semantic judgement. It
catches "the reply never mentioned the double charge at all"; it does not catch subtle
misstatement. The LLM judge covers semantics on four separate axes. The alternative — one
gold reference reply plus BLEU/embedding similarity — would mostly measure paraphrase
distance, which is worse and looks more rigorous.

### 12. Why three router layers instead of just asking the model?
Because the cases where being wrong is a headline — abuse, legal demands, refund requests,
account compromise — must not depend on a model's mood, and I need to point at the exact
line that fired. But rules are bad at judgement ("have they already tried the fix?"), so
the model handles those. Confidence gates come last because *not knowing* is itself a
reason to involve a human. Every decision returns an enum reason code **and** a human
sentence.

### 13. You found bugs in your own filters. How do I know you didn't just tune until the
numbers looked good?
Fair question, and it's why `ASSUMPTIONS.md` records the full progression of every filter
with numbers, including the embarrassing middle states. The resolution filter went 12.2% →
58.5% (over-permissive) → 36.4%. The rule I applied: **change the code only where reading
the data shows the code is wrong; where the number is merely inconvenient, measure it and
report it.** The t.co fix was a real bug (every Spotify link is a shortlink, and my link
patterns couldn't match one). The D3 brand-viability check *fired* on my first run and I
fixed my filter rather than switching brands to escape it — then it passed honestly.

### 14. What would you do first with another week?
In expected-value order: (1) a second human annotator on 100 items, to get a real
inter-annotator κ instead of machine–machine κ — every other number is capped by label
quality; (2) **measure retrieval**, recall@k against manually-marked relevant threads —
retrieval is currently the largest completely untested link in the chain; (3) per-intent
routing thresholds instead of one global τ; (4) a shadow-mode A/B harness.

### 15. What's the one thing you'd want me to take away?
That I treated the evaluation as the deliverable. The most useful number I produced isn't
the agent's macro-F1 — it's that two competent labellers, differing only in prompt framing,
disagree about escalation by 7 percentage points. That number tells you what this system
can and cannot be trusted to do, and it's the kind of thing that only shows up if you
build the measurement before you fall in love with the model.
