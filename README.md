# SpotifyCares Support Agent

An AI support agent for `@SpotifyCares` built from 2.8M real customer-support tweets, and
— more importantly — **the evaluation that says how much to trust it**.

> ⚠ **Results are currently PROVISIONAL.** The golden set is machine-labelled with a
> 96-item human adjudication queue prepared but not yet worked. `run_eval.py` stamps this
> on every number automatically. See [What's outstanding](#whats-outstanding).

## Reproduce the headline results in under 15 minutes

```bash
git clone <repo> && cd hiver-support-agent
make setup     # venv + deps, ~3 min
make demo      # replays the committed LLM cache. NO API KEY NEEDED. ~4 min
               # prints the headline table and regenerates every figure
```

`make demo` runs with `LLM_OFFLINE=1`, where a cache miss is a **hard error** rather than
a silent fallback to a live call — so it cannot quietly succeed by hitting the network.

*No `make` (e.g. Windows)?* `python run.py demo` runs the identical commands.
*Live run:* `make full` (needs `OPENAI_API_KEY`, ~$2.50, rebuilds the cache from scratch).

## What to read first

If you have five minutes: **[`reports/REPORT.md` §1 "What is misleading about my headline
number?"](reports/REPORT.md)** — twelve reasons to discount my own results, written before
the results section so it could not be retrofitted.

Then: [`reports/DECISION_LOG.md`](reports/DECISION_LOG.md) (15 non-obvious decisions) and
[`reports/WALKTHROUGH.md`](reports/WALKTHROUGH.md) (code tour + the 15 questions I'd least
like to be asked).

## Headline results

220-example golden set, drawn from a held-out time-based test split. Bootstrap 95% CIs.

| system | intent macro-F1 (95% CI) | cost/100 msgs | missed escalations | deflection |
|---|---|---|---|---|
| B0 always-escalate | 0.038 | 85.0 | **0** of 220 | 0.0% |
| B0 always-auto | 0.038 | 150.0 | 33 of 220 | 100.0% |
| B1 TF-IDF + BM25 copy-paste | 0.606 [0.531, 0.671] | 106.4 | 23 of 220 | 93.6% |
| **agent** | **0.789 [0.726, 0.837]** | **31.4 [15, 52]** | **4** of 220 | 73.6% |

`cost/100` is the primary routing metric: a missed escalation costs **10×** a needless one
(a router is a decision system, not a classifier). Three findings worth more than the
table itself:

- **B1's keyword router (106.4) costs more than escalating everything (85.0)** despite
  deflecting 93.6% — accuracy is the wrong routing metric.
- **At a 1:1 cost ratio, B1 beats the agent.** The headline depends on the asymmetry.
- **Only 10% of the agent's replies would be sent unedited.** This is a drafting aid.

## Architecture

```
customer message
      │
      ├─► hybrid retrieval    BM25 + MiniLM cosine, RRF-fused, top-3 reranked
      │      │                (corpus split ONLY — assert_no_leakage() on every call)
      │      └─► returns FULL past threads: customer turn + brand's resolution
      │
      ├─► classify   {intent, confidence, rationale}      codebook + tie-break rules
      ├─► draft      {reply ≤280 chars, grounded_in[ids]} hard safety constraints
      └─► route      deterministic rules → intent-corroboration → model → confidence gates
                     always returns an enum reason code AND a human sentence
```

Eleven intents induced from the data then hand-edited; see
[`golden/CODEBOOK.md`](golden/CODEBOOK.md).

## Repository map

| path | what |
|---|---|
| `src/support_agent/` | the system — every module <250 lines, no framework |
| `eval/` | metrics, LLM judge, judge validation, `run_eval.py`, figures |
| `baselines/` | B0 trivial, B1 classical |
| `golden/` | 220-example set, codebook, sampling note, labelling CLI |
| `reports/` | REPORT, DECISION_LOG, FAILURE_ANALYSIS, WALKTHROUGH, results.json, figures |
| `cache/llm/` | committed record/replay cache — this is what makes `make demo` work |
| `ASSUMPTIONS.md` | every decision I made instead of asking, with what would change my mind |

## Commands

```bash
make test      # 72 tests: thread traps, PII scrubbing, metrics, determinism, leakage
make eval      # evaluate all systems -> reports/results.json
make gate      # the submission checklist, as an executable
python -m src.support_agent.agent --demo        # 10 dev messages, full decisions
python golden/label_cli.py                      # the human adjudication pass
python eval/human_judge_cli.py                  # human judge scoring
```

## What's outstanding

Two human-in-the-loop steps are built and prepared but not yet done. Both are blocking a
non-provisional result:

1. **Golden-set adjudication** — `python golden/label_cli.py` (96 queued items, ~40 min),
   then `python scripts/build_golden.py`.
2. **Judge validation against a human** — `python eval/human_judge_cli.py` (70 replies),
   then `python -m eval.judge_validation --report`.

## Citations & Borrowings

**Data.** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
(Thought Vector, Kaggle, CC BY-NC-SA 4.0). No Kaggle credentials were available on this
machine, so `scripts/get_data.sh` tries the Kaggle CLI first and falls back to the
byte-identical HuggingFace mirror [`SunidhiSriram/twcs`](https://huggingface.co/datasets/SunidhiSriram/twcs).

**Libraries.** `pandas`, `numpy`, `scikit-learn` (TF-IDF, LogisticRegression, KMeans,
HDBSCAN, silhouette), [`rank_bm25`](https://github.com/dorianbrown/rank_bm25) (BM25Okapi),
[`sentence-transformers`](https://www.sbert.net/) with
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
`matplotlib`, `openai`, `pytest`, `pyarrow`.

**Models.** OpenAI `gpt-4o-mini` (generation, labelling) and `gpt-4.1-mini` (judge,
second labelling pass). Only `OPENAI_API_KEY` was available, so judge and generator share
a provider — a weaker separation than a different model family, which is why the
self-preference probe is mandatory here and its result (+0.225 on a 1–5 scale) is reported
as a limitation.

**Methods and patterns borrowed, not invented by me.**
- *Reciprocal Rank Fusion* for combining lexical and dense retrieval — Cormack, Clarke &
  Buettcher (2009), standard constant k=60.
- *LLM-as-a-judge, and its known pathologies* (position bias, verbosity bias,
  self-preference) — Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
  (2023). The three bias probes in `eval/judge_validation.py` follow that paper's framing.
- *Expected Calibration Error / reliability diagrams* — Guo et al., "On Calibration of
  Modern Neural Networks" (2017).
- *Krippendorff's α* (interval) and *Cohen's κ* — standard inter-rater formulations,
  implemented directly in `eval/metrics.py` rather than pulled from a library so the
  arithmetic is inspectable.
- *Percentile bootstrap* for confidence intervals — Efron & Tibshirani.
- The *record/replay cache* pattern is the standard HTTP-fixture idea (VCR/`betamax`),
  reimplemented in ~40 lines keyed on a SHA256 of model+prompt+params.

**AI assistance — disclosed in full.** This project was built with **Claude Code
(Claude Opus 5)** working autonomously from a detailed brief I wrote. It was used for:
scaffolding and writing effectively all of the code; designing prompts; drafting the
report, codebook and decision log; and running the analysis. I directed the work, set the
constraints and locked the key decisions (brand, split strategy, cost model, judge
separation, what not to build) in advance. Specific things the AI found that I had not
anticipated are recorded honestly in `ASSUMPTIONS.md` — including three real bugs in my
own filters (`t.co` shortlinks invisible to the resolution scorer, romanised Indonesian
passing an English filter, and a stratification/labelling mismatch that broke the
per-intent floor). Commits are co-authored accordingly.

**Not borrowed:** the intent taxonomy, the escalation policy, the resolution-bearing
scorer, the cost model, the `must_include`/`must_not_include` grading design, and every
number in the report.

## Licence / data note

The dataset is CC BY-NC-SA 4.0 (non-commercial). All committed text is PII-scrubbed
(`scripts/scan_secrets.py` verifies the working tree and the full git history); tweet ids
are retained so every example in the report is traceable.
