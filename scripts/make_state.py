"""One-shot generator for STATE.json. Kept in the repo so the task ledger is auditable."""
import json

T = [
    # phase, id, title, gate
    (0, "P0.1", "Create directory tree + .gitignore + requirements.txt + .env.example", "test -d src/support_agent"),
    (0, "P0.2", "Write STATE.json / PROGRESS.md / ASSUMPTIONS.md / BLOCKERS.md / NOT_BUILDING.md", "test -f STATE.json"),
    (0, "P0.3", "Write src/support_agent/config.py (all constants in one place)", "python -c \"import src.support_agent.config\""),
    (0, "P0.4", "Write src/support_agent/llm.py: cache + retry + cost meter + --offline", "pytest tests/test_llm_cache.py -q"),
    (0, "P0.5", "Write Makefile with all targets", "test -f Makefile"),
    (0, "P0.6", "Commit P0 scaffold", "git log --oneline | grep -q P0"),

    (1, "P1.1", "scripts/get_data.sh: kaggle -> HF mirror -> local search -> synthetic", "test -f data/raw/twcs.csv"),
    (1, "P1.2", "ingest.py: load csv with forced str dtypes, verify schema + row count", "python -m src.support_agent.ingest --verify-schema"),
    (1, "P1.3", "threads.py: parent->child graph, comma-list ids, cycle guard, depth cap, branch pick", "pytest tests/test_threads.py -q"),
    (1, "P1.4", "Brand slice for SpotifyCares + D3 fallback check (>=2000 threads, >=15% resolution rate)", "python -m src.support_agent.ingest --brand-stats"),
    (1, "P1.5", "Resolution-bearing scorer (transparent heuristic, no LLM) + rate reported", "pytest tests/test_resolution.py -q"),
    (1, "P1.6", "clean.py: mention/signature strip, PII scrub, emoji keep, punctuation normalise", "pytest tests/test_clean.py -q"),
    (1, "P1.7", "Language filter, min-token filter, near-duplicate dedupe", "pytest tests/test_clean.py -q"),
    (1, "P1.8", "Time-based 70/15/15 split on root created_at -> corpus/dev/test.jsonl", "test -s data/processed/test.jsonl"),
    (1, "P1.9", "Commit processed subsample (<40MB) + data funnel figure", "test -f reports/figures/data_funnel.png"),
    (1, "P1.10", "Gate: ingest --verify prints funnel; thread+clean tests pass", "python -m src.support_agent.ingest --verify"),

    (2, "P2.1", "Embed 3000 sampled corpus customer messages (MiniLM)", "test -f data/interim/taxonomy_emb.npy"),
    (2, "P2.2", "Cluster (HDBSCAN else KMeans sweep k=6..20 by silhouette) + save sweep plot", "test -f reports/figures/cluster_sweep.png"),
    (2, "P2.3", "LLM-name each cluster from 20 centroid-near + 10 random members", "test -f data/interim/cluster_names.json"),
    (2, "P2.4", "Hand-edit names; merge/split by routing outcome; land 8-12 intents + other", "python scripts/check_intents.py"),
    (2, "P2.5", "Write golden/CODEBOOK.md: def + 3 pos + 2 near-miss + tie-break per intent", "python scripts/check_codebook.py"),
    (2, "P2.6", "taxonomy.png cluster->intent map; record collapsed intents in NOT_BUILDING.md", "test -f reports/figures/taxonomy.png"),

    (3, "P3.1", "Stratified sample 220 from TEST split w/ floors + 15% hard-case quota; fixed seed", "test -f golden/sample_220.jsonl"),
    (3, "P3.2", "Write golden/SAMPLING_NOTE.md (method + honesty statement)", "test -f golden/SAMPLING_NOTE.md"),
    (3, "P3.3", "Pass A prelabels (codebook in context)", "test -f golden/prelabel_A.jsonl"),
    (3, "P3.4", "Pass B prelabels (different model + framing, shuffled, blind to A)", "test -f golden/prelabel_B.jsonl"),
    (3, "P3.5", "Cohen's kappa A-vs-B on intent and on escalate; write to SAMPLING_NOTE.md", "python scripts/ab_agreement.py"),
    (3, "P3.6", "Build golden/v1.jsonl (adjudicated where A==B, flagged where not) + JSON schema validation", "python scripts/validate_golden.py"),
    (3, "P3.7", "Build golden/label_cli.py: resumable, saves per item, shows both labels + retrieved", "python golden/label_cli.py --selftest"),
    (3, "P3.8", "Build human review queue (all disagreements + random 20% of agreements)", "test -f golden/review_queue.jsonl"),

    (4, "P4.1", "baselines/b0_trivial.py: majority intent, canned reply, always-auto AND always-escalate", "python -m baselines.b0_trivial --eval"),
    (4, "P4.2", "baselines/b1_simple.py: TFIDF+LogReg / BM25 top-1 reply copy / keyword router", "python -m baselines.b1_simple --eval"),
    (4, "P4.3", "baseline_table.png + markdown table", "test -f reports/figures/baseline_table.png"),

    (5, "P5.1", "retrieval.py: BM25 + dense, RRF fuse, top-8 -> heuristic rerank top-3, corpus-only assertion", "pytest tests/test_retrieval.py -q"),
    (5, "P5.2", "classify.py: codebook prompt + dynamic few-shot, strict JSON, parse-failure counter", "python -m src.support_agent.classify --smoke"),
    (5, "P5.3", "draft.py: grounded reply, hard constraints, 280 char cap, grounding citations", "python -m src.support_agent.draft --smoke"),
    (5, "P5.4", "route.py: deterministic rules -> LLM judgement -> confidence gates (tau, sigma)", "python -m src.support_agent.route --smoke"),
    (5, "P5.5", "Tune tau/sigma on DEV only; save threshold sweep figure", "test -f reports/figures/threshold_sweep.png"),
    (5, "P5.6", "agent.py orchestrator: handle(message, context) -> Decision", "python -m src.support_agent.agent --demo"),

    (6, "P6.1", "eval/metrics.py: macro-F1, per-class PRF, confusion, ECE, bootstrap CI, cost metric", "pytest tests/test_metrics.py -q"),
    (6, "P6.2", "Routing metrics: P/R/F1, expected cost per 100 @10:1, deflection curve, auto-but-should-escalate raw count", "pytest tests/test_metrics.py -q"),
    (6, "P6.3", "eval/judge.py: 4 axes scored in separate calls + binary send-unedited + must_include checks", "python -m eval.judge --smoke"),
    (6, "P6.4", "Human judge scoring CLI for 60-80 replies", "test -f eval/human_judge_cli.py"),
    (6, "P6.5", "judge_validation.py: Spearman, Krippendorff alpha, exact + within-1 rates", "python -m eval.judge_validation --report"),
    (6, "P6.6", "Bias probes: position (40 swapped pairs), length (rho), self-preference (delta)", "python -m eval.judge_validation --probes"),
    (6, "P6.7", "Calibration: reliability diagram + ECE figure", "test -f reports/figures/calibration.png"),
    (6, "P6.8", "eval/run_eval.py: one command, all 3 systems, writes reports/results.json", "python -m eval.run_eval --system agent --split golden"),
    (6, "P6.9", "Determinism test: same command twice -> identical numbers", "pytest tests/test_determinism.py -q"),
    (6, "P6.10", "eval/figures.py: all report figures regenerated from results.json", "python -m eval.figures"),

    (7, "P7.1", "Dump all errors to reports/errors.jsonl and cluster them", "test -f reports/errors.jsonl"),
    (7, "P7.2", "Read >=40 errors manually; name top 5 failure modes", "test -f reports/FAILURE_ANALYSIS.md"),
    (7, "P7.3", "Per mode: count, 2 verbatim PII-scrubbed examples, mechanism hypothesis, cheap confirming test", "python scripts/check_failure_analysis.py"),

    (8, "P8.1", "Pick top 1-2 modes; implement targeted fix only", "git log --oneline | grep -q P8"),
    (8, "P8.2", "Re-run full eval; before/after table with CIs", "test -f reports/iteration_delta.md"),
    (8, "P8.3", "State plainly whether delta clears the noise floor", "grep -qiE 'noise floor|statistically' reports/iteration_delta.md"),

    (9, "P9.1", "REPORT.md section 1: What is misleading about my headline number (>=8 items) FIRST", "python scripts/check_report.py --misleading"),
    (9, "P9.2", "REPORT.md sections 2-7", "test -f reports/REPORT.md"),
    (9, "P9.3", "DECISION_LOG.md: 10-15 entries, decision/alternatives/why/what-would-change-my-mind", "python scripts/check_report.py --decisions"),
    (9, "P9.4", "WALKTHROUGH.md: code tour + 15 hostile questions with answers", "test -f reports/WALKTHROUGH.md"),
    (9, "P9.5", "README.md: <15min repro, headline table, architecture, Citations & Borrowings, AI disclosure", "test -f README.md"),
    (9, "P9.6", "Verify make demo from a clean clone in a temp dir, no API key, timed", "test -f reports/repro_timing.txt"),

    (10, "P10.1", "Run full submission checklist", "python scripts/submission_gate.py"),
    (10, "P10.2", "Secret/PII scan of full git history", "python scripts/scan_secrets.py"),
    (10, "P10.3", "Print SUBMISSION GATE block and STOP", "true"),
]

state = {
    "project": "hiver-support-agent",
    "brand": "SpotifyCares",
    "budget_usd_cap": 25.0,
    "current_phase": 0,
    "tasks": [
        {"id": i, "phase": p, "title": t, "status": "todo", "attempts": 0, "gate": g, "artifacts": []}
        for (p, i, t, g) in T
    ],
}
with open("STATE.json", "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)
print("STATE.json written: %d tasks across phases 0-10" % len(T))
