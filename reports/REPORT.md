# SpotifyCares Support Agent — Report

**Dataset:** `thoughtvector/customer-support-on-twitter` (2,811,774 tweets) · **Brand:**
`SpotifyCares` · **Golden set:** 220 examples · **Total LLM spend:** $2.46

> ⚠ **STATUS: PROVISIONAL.** 220/220 golden rows are machine-labelled and awaiting human
> adjudication (96-item queue prepared). Judge-vs-human agreement is **not yet measured**.
> Every number below is measured against machine labels. `run_eval.py` stamps this
> automatically and cannot be silenced without doing the human pass.

---

## 1. What is misleading about my headline number?

Written first, deliberately, so the results section is honest by construction. My headline
is **macro-F1 0.789 [0.726, 0.837] and expected cost 31.4 per 100 messages**. Here is why
you should discount it.

1. **The labels are machine-made, and the two machines disagreed a lot.** Pass A and
   Pass B agree on intent at **κ = 0.655** and raw 69.6%. Nearly a third of the golden set
   (76/220) had a disagreement. My 0.789 is agreement with *one* labelling pass, and a
   second competent pass would have scored it differently.

2. **My escalation ground truth is unstable, and this is the worst problem here.** Pass A
   escalates **15.0%** of the golden set; Pass B, differing *only in prompt framing*,
   escalates **22.3%** of the same items. A 7-point swing from framing alone means the
   boundary is genuinely ambiguous. Since routing "accuracy" is measured against that
   boundary, **every routing number inherits that ambiguity**.

3. **n=220 makes the CI wider than most differences I could claim.** The 95% CI half-width
   on macro-F1 is ±0.056. Any gap below ~0.11 between two systems is not a result. My
   Phase 8 fix improved cost by 12.3 per 100 and I *still* report it as not clearing the
   noise floor (paired CI **[−30.0, +0.92]**, includes zero).

4. **The escalation policy is my invention, not Spotify's.** Their real routing is
   unobservable from public tweets. The router is measured on agreement with a policy I
   wrote in `config.py`, not on correctness.

5. **Survivorship bias: I only see problems answered publicly and substantively.** The
   resolution-bearing filter drops 64% of threads (28,220 → 10,284). Billing and account
   work moves to DM, so `billing_charge_dispute` is 0.8% of the corpus. The whole system
   is evaluated on the easy, public half of support.

6. **The golden set is not production traffic.** Stratification with a floor of 12 per
   intent over-represents rare classes: `feature_request_or_complaint` is 49.7% of the
   real test split but 26% of the golden set. **Macro-F1 flatters rare classes**, and the
   **73.6% deflection rate does not transfer** to real inbound volume.

7. **The 10:1 cost ratio is asserted, not measured — and the ranking depends on it.** At a
   **1:1** ratio B1 actually *beats* the agent (12.3 vs 15.0 per 100). The agent only wins
   from roughly 2:1 upward. If the true cost of a needless escalation is closer to that of
   a missed one, my headline conclusion inverts.

8. **The judge is unvalidated against a human.** I have measured its *biases* — position
   flip rate 5.0%, length correlation |ρ| < 0.11, **self-preference +0.225 on a 1–5
   scale** — but Spearman ρ against human scores is **not yet computed**. Until it is,
   every reply-quality number is one model grading another, and the +0.225 self-preference
   means the judge mildly favours its own family, which includes the generator.

9. **Time-based splitting limits leakage but does not eliminate it.** Brand vocabulary,
   product eras and recurring incidents persist across the boundary. The agent can still
   benefit from having seen "clear your cache" phrasing a thousand times.

10. **One of my own metrics was broken and I nearly reported it.** The deterministic
    `must_include` check scored the agent at **7.4%** — that is not performance, it is a
    measurement failure (abstract requirements vs concrete replies share no vocabulary).
    The semantic re-check gives **53.8%**. A 7× error in my own instrument, found only by
    reading examples.

11. **B1 is trained on machine labels**, so it learns to imitate `gpt-4o-mini`. Its ceiling
    is the labeller's accuracy, which makes it a slightly *unfair* baseline — it is handicapped
    in a way a human-labelled classifier would not be.

12. **The hard-case quota did not do what I intended.** Hard-flagged examples scored
    *higher* (84.4% vs 79.3%) and produced **zero** missed escalations, because my hard
    flags select for escalation keywords, which make routing *easier*.

---

## 2. Problem framing, and what I chose not to build

"Good" for SpotifyCares specifically: **short, fast, safe, and deflecting volume without
burning trust.** Their public support is high-volume, low-complexity, and heavily
repetitive; the expensive failure is not a clumsy reply but auto-answering something that
needed a human.

So the system is scored as a **decision system**, not a classifier: expected cost per 100
messages under an asymmetric penalty, with missed escalations reported in **raw counts**.

**Not built** (full reasoning in `NOT_BUILDING.md`): fine-tuning (would teach a model to
imitate my labeller and cost the evaluation budget), agentic tool use (no account state
exists in this dataset, so every tool would be a mock and unevaluable), multilingual
support (I cannot verify labels I cannot read), a vector DB (7k threads; numpy is faster
than the infrastructure), and a UI (zero evaluation signal).

---

## 3. Data

![funnel](figures/data_funnel.png)

2,811,774 tweets → 43,265 by SpotifyCares → **28,220 threads** → 10,284 resolution-bearing
(**36.4%**) → 9,702 clean → **6,791 / 1,455 / 1,456** corpus/dev/test, split by time.

**Reconstruction traps hit** (one test each, `tests/test_threads.py`): `response_tweet_id`
is sometimes a comma-separated list; pandas silently turns ids into floats (`123` →
`123.0`) unless dtype is forced; cycles and self-references exist; threads branch, so the
branch containing the brand's reply must be selected.

**The resolution-bearing filter is the most consequential data decision**, so I validated
it rather than asserting it: on 100 randomly sampled replies judged by an independent
model, **precision 0.806, recall 0.853, F1 0.829, κ 0.736**, with positive rates matching
(0.36 heuristic vs 0.34 adjudicator). It went through three measured versions — 12.2% →
58.5% → 36.4% — the first because every Spotify link is a `t.co` shortlink my patterns
could not match, the second because "can you DM us your account's email" is a handoff that
matched no handoff pattern.

---

## 4. Approach

```
message ─► hybrid retrieval (BM25 + MiniLM, RRF-fused, top-3 by recency/quality rerank)
             │  └─ assert_no_leakage(): fails loudly if any hit is in dev/test
             ├─► classify  → {intent, confidence, rationale}   [codebook + tie-break rules]
             ├─► draft     → {reply ≤280 chars, grounded_in[ids]}   [hard safety constraints]
             └─► route     → rules → intent-corroboration → model → confidence gates
                              └─ always returns an enum code AND a human sentence
```

**Retrieval is hybrid** because support text contains both exact tokens dense search blurs
(`Duo`, `Hulu`, `Autoplay`) and paraphrase BM25 misses ("won't play" / "no sound"). RRF
needs no score normalisation between the two.

**Routing is hybrid by design.** Safety-critical cases (abuse, legal, refunds, compromise)
fire on deterministic rules so I can point at the line that fired; judgement calls go to
the model; confidence gates come last. **Threshold tuning on dev produced a negative
result about my own design**: raising τ strictly *increased* cost (at τ=0.55 it escalated
31 extra messages and caught **zero** additional misses), so the confidence gate is
**disabled**. σ=0.30 is exactly cost-neutral and is kept only as anti-hallucination
insurance.

**Taxonomy: 11 intents, weakly induced.** HDBSCAN found **0 clusters (100% noise)**;
KMeans silhouette was ~0.04 at every k from 6 to 20. Clustering isolated 5 intents; the
other 6 rest on keyword probes and reading, and are marked red in `figures/taxonomy.png`.

---

## 5. Results

![table](figures/baseline_table.png)

| system | intent macro-F1 (95% CI) | cost/100 | missed escalations | deflection |
|---|---|---|---|---|
| B0 always-escalate | 0.038 | 85.0 | **0** of 220 | 0.0% |
| B0 always-auto | 0.038 | 150.0 | 33 of 220 | 100.0% |
| B1 TF-IDF + copy-paste | 0.606 [0.531, 0.671] | 106.4 | 23 of 220 | 93.6% |
| **agent** | **0.789 [0.726, 0.837]** | **31.4 [15, 52]** | **4** of 220 | 73.6% |

**Where a baseline beats the agent, stated plainly:**

- **B0 always-escalate has zero missed escalations. The agent has four.** On the single
  safety metric that matters most, the dumbest possible policy wins, and it always will.
  The agent's claim is that it removes 73.6% of the human workload for those 4 errors.
- **At a 1:1 cost ratio, B1 beats the agent** (12.3 vs 15.0 per 100). My headline depends
  on the asymmetry being real.
- **B1's keyword router costs 106.4 — worse than escalating everything (85.0)** — despite
  deflecting 93.6%. This is the cleanest demonstration in the project that accuracy is the
  wrong routing metric.

**Reply quality** (LLM judge, 4 axes in separate calls, 220 replies each):

| axis | B1 (verbatim human copy) | agent (generated) |
|---|---|---|
| groundedness | 3.03 [2.82, 3.22] | **4.07 [3.92, 4.22]** |
| actionability | 2.00 [1.86, 2.16] | **2.90 [2.73, 3.06]** |
| brand fit | 2.67 [2.54, 2.79] | **3.97 [3.88, 4.05]** |
| safety | 4.58 [4.45, 4.68] | **4.92 [4.88, 4.96]** |
| would send unedited | 0.9% | **10.0%** |
| semantic requirement compliance | 30.0% | **53.8%** |

I expected the copy-paste baseline to win on groundedness — it emits *real human text*
with zero hallucination risk. It lost (3.03 vs 4.07), and the reason is the interesting
part: **a verbatim human reply transplanted from a similar-but-different thread makes
claims that are not supported for *this* customer's problem.** Groundedness is a property
of the match, not of the authorship. B1 cites a source for 100% of its replies and still
scores a point lower.

**The most sobering number is 10%.** Only 10% of the agent's replies would be sent
unedited by the judge acting as a support agent. This is a drafting aid, not an
autoresponder.

**Calibration** (![calibration](figures/calibration.png)): ECE **0.079**. Well-calibrated
enough to report, not well-calibrated enough to gate on — which is exactly what the
threshold sweep independently found.

**Pipeline reliability:** 0 parse failures across 220 classify, 220 draft and 216 route
calls; 0 replies over the 280-character cap; 21.4% of replies ungrounded (cite nothing).

---

## 6. Failure analysis

Full version with verbatim examples in `FAILURE_ANALYSIS.md`. I read all 66 agent errors.

1. **Frustration read as distress** — 14 of 26 needless escalations. The model escalates on
   profanity ("I don't pay $10 a month for this shit") rather than on genuine distress.
   All from the model layer, none from the rules.
2. **`feature_request` vs `content_missing`** (8) — mostly **label disputes, not model
   errors**; the agent's confidence is 0.80 and I think it is arguably right.
3. **`praise_or_chitchat` → `other`** (8) — agent confidence 0.3–0.4; it knows it is
   guessing, and the confusion has **no routing consequence** either way.
4. **`app_bug` vs `playback`** (5) — the label requires inferring a *cause* the customer
   never states.
5. **Missed escalations** (4 post-fix) — in every case **the intent classifier was already
   right** and the router auto-handled anyway.

**Phase 8 targeted fix**, aimed at mode 5: escalate when a high-stakes *intent* is
corroborated by a money amount or a security noun, regardless of phrasing.

| | before | after |
|---|---|---|
| cost / 100 | 43.6 | **31.4** |
| missed escalations | 7 | **4** |
| needless escalations | 26 | 29 |

**Paired bootstrap on the per-example cost difference: −12.3 [−30.0, +0.92].** The interval
**includes zero**, so *this fix does not clear the noise floor at n=220* and I do not claim
it as a result — even though it helps in 94.7% of resamples. It changed 6 decisions:
**3 correct, 3 wrong**. It nets out favourably only because a missed escalation costs 10×.

---

## 7. What I'd do with one more week

Ranked by expected value per hour.

1. **A second human annotator on 100 items (≈3h).** Every number here is capped by label
   quality, and I currently have machine–machine κ where I need human–human κ. This is
   worth more than any modelling change.
2. **Measure retrieval (≈4h).** Recall@k against manually-marked relevant threads.
   Retrieval feeds classification *and* drafting and is currently the **largest completely
   untested link in the chain** — I have no idea whether top-3 contains the right thread.
3. **Validate the judge against human scores (≈2h).** The CLI exists
   (`eval/human_judge_cli.py`); 70 replies would give Spearman ρ and Krippendorff's α and
   would turn every reply-quality claim from provisional into evidence.
4. **Per-intent routing thresholds (≈3h).** One global τ is clearly wrong when
   `billing_charge_dispute` and `praise_or_chitchat` have opposite cost profiles.
5. **Fix mode 1 (≈2h).** Distinguish profanity-about-product from abuse-at-a-person; it is
   54% of needless escalations and the cheapest remaining win.
6. **Re-weight metrics to production intent mix (≈1h).** Would convert the 73.6%
   deflection rate from a benchmark artefact into a deployable estimate.

---

*Every number in this report traces to `reports/results.json`, `reports/p8_delta.json`,
`reports/judge_validation.json`, `golden/ab_agreement.json` or
`reports/resolution_filter_validation.json`, all committed.*
