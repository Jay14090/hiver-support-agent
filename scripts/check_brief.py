"""Check the repo against the ASSIGNMENT BRIEF's literal wording, not my own checklist.

`submission_gate.py` checks the plan I set myself. This checks what the brief actually
asked for, quoting it, so that any gap is visible as a gap rather than hidden behind a
checklist I wrote.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.support_agent import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
G, R, Y, B, X = "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m"
if os.name == "nt":
    os.system("")

rows = []


def req(quote: str, status: str, evidence: str):
    """status: MET | PARTIAL | NOT_MET"""
    rows.append((quote, status, evidence))


def jl(p: Path) -> int:
    return sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0


def main() -> int:
    REP, GOLD = config.REPORTS, config.GOLDEN
    report = (REP / "REPORT.md").read_text(encoding="utf-8") if (REP / "REPORT.md").exists() else ""
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

    # ---- 1. subsample
    funnel = json.loads((config.DATA_INTERIM / "funnel.json").read_text(encoding="utf-8"))
    req("We will not run your code on the full dataset - a subsample is expected", "MET",
        f"one brand slice of {funnel['total_tweets']:,} tweets -> {funnel['threads_clean']:,} "
        f"threads; data/processed/*.jsonl committed ({sum(f.stat().st_size for f in config.DATA_PROCESSED.glob('*.jsonl'))/1024**2:.1f}MB)")

    # ---- 2. repo link public / access granted
    try:
        vis = subprocess.run(["gh", "repo", "view", "Jay14090/hiver-support-agent",
                              "--json", "visibility,url", "-q", ".visibility+\" \"+.url"],
                             cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        vis = "unknown"
    is_pub = vis.lower().startswith("public")
    req("Include the repo link (public, or private with access granted to us)",
        "MET" if is_pub else "NOT_MET",
        f"{vis}" + ("" if is_pub else "  <-- MUST be made public, or access granted"))

    # ---- 3. runnable pipeline, README reproduces headline in <15 min
    timing = (REP / "repro_timing.txt").read_text(encoding="utf-8") if (REP / "repro_timing.txt").exists() else ""
    m = re.search(r"TOTAL\s*:\s*(\d+(?:\.\d+)?)\s*s", timing)
    secs = float(m.group(1)) if m else None
    req("README must let us reproduce your headline results in under 15 minutes",
        "MET" if (secs and secs < 900 and "PASS" in timing) else "NOT_MET",
        f"verified from a clean clone with NO API key: {secs:.0f}s ({secs/60:.1f} min)" if secs else "not timed")

    # ---- 4. golden set: 150-250 HAND-LABELLED examples you built yourself
    gold = [json.loads(l) for l in (GOLD / "v1.jsonl").open(encoding="utf-8")] if (GOLD / "v1.jsonl").exists() else []
    n_human = sum(1 for g in gold if g.get("labeler") == "human_adjudicated")
    size_ok = 150 <= len(gold) <= 250
    req("Golden evaluation set - 150-250 hand-labelled examples you built yourself",
        "MET" if (size_ok and n_human == len(gold)) else "PARTIAL",
        f"{len(gold)} examples, size OK={size_ok}; but human_adjudicated = {n_human}/{len(gold)}. "
        f"Labels are 3 blind model passes + 2-of-3 majority. NOT hand-labelled.")

    req("...with a short note on how you sampled and labelled them", "MET" if (GOLD / "SAMPLING_NOTE.md").exists() else "NOT_MET",
        f"golden/SAMPLING_NOTE.md ({len((GOLD / 'SAMPLING_NOTE.md').read_text(encoding='utf-8').split())} words), "
        f"states the machine-labelling honestly")

    # ---- 5. eval harness: automated metrics + judge rubric + judge-vs-human evidence
    res = json.loads((REP / "results.json").read_text(encoding="utf-8")) if (REP / "results.json").exists() else {}
    req("Evaluation harness - automated metrics", "MET" if res.get("systems") else "NOT_MET",
        f"one command produces every number; {len(res.get('systems', {}))} systems, CIs on all headline metrics")
    req("...+ an LLM-as-judge rubric for reply quality", "MET" if (REP / "judge_scores.jsonl").exists() else "NOT_MET",
        f"4 axes scored in separate calls + binary; {jl(REP / 'judge_scores.jsonl')} replies scored")

    jv = json.loads((REP / "judge_validation.json").read_text(encoding="utf-8")) if (REP / "judge_validation.json").exists() else {}
    ha = jv.get("human_agreement", {})
    indep = ha.get("independent_human_scoring")
    req("...including evidence of how well your judge agrees with a human",
        "MET" if indep is True else ("PARTIAL" if ha.get("overall_mean_rho") is not None else "NOT_MET"),
        f"rho={ha.get('overall_mean_rho')} over n={ha.get('n_human_scored')}, plus 3 bias probes. "
        f"BUT independent_human_scoring={indep}: scores were AI-generated then human-ENDORSED.")

    # ---- 6. report sections
    def has(pat):
        return bool(re.search(pat, report, re.I))
    words = len(report.split())
    req("Report (max 6 pages)", "MET" if words <= 3600 else "NOT_MET", f"~{words} words (~{words/600:.1f} pages)")
    req("  Problem framing: what 'good' means for this brand", "MET" if has(r"##\s*2\..*framing") else "NOT_MET",
        "REPORT.md section 2")
    req("  ...and what you chose not to build", "MET" if (ROOT / "NOT_BUILDING.md").exists() and has(r"not built") else "NOT_MET",
        f"NOT_BUILDING.md + REPORT.md section 2")
    n_sys = len([k for k in res.get("systems", {})])
    req("  Results vs at least two baselines (a trivial one and a simple one)",
        "MET" if n_sys >= 3 else "NOT_MET",
        f"B0 trivial (2 degenerate routing policies) + B1 simple (TF-IDF/BM25/keywords) + agent = {n_sys} systems")
    fa = (REP / "FAILURE_ANALYSIS.md").read_text(encoding="utf-8") if (REP / "FAILURE_ANALYSIS.md").exists() else ""
    n_modes = len(re.findall(r"^##\s+Mode\s+\d", fa, re.M))
    n_ids = len(re.findall(r"id=\d+", fa))
    req("  Failure analysis: top 5 failure modes with real examples and hypotheses",
        "MET" if (n_modes == 5 and n_ids >= 10) else "NOT_MET",
        f"{n_modes} modes, {n_ids} real PII-scrubbed examples cited by tweet id, hypothesis + cheap test each")
    mm = re.search(r"##\s*1\..*?misleading.*?\n(.*?)(?=\n## )", report, re.S | re.I)
    n_items = len(re.findall(r"^\s*\d+\.\s+\*\*", mm.group(1), re.M)) if mm else 0
    req("  'What is misleading about my headline number?' - mandatory section",
        "MET" if n_items >= 1 and mm and report.index(mm.group(0)) < report.index("## 2.") else "NOT_MET",
        f"section 1, FIRST in the report, {n_items} numbered items")
    req("  What you'd do next with one more week", "MET" if has(r"one more week") else "NOT_MET",
        "REPORT.md section 7, ranked by expected value per hour")
    dl = (REP / "DECISION_LOG.md").read_text(encoding="utf-8") if (REP / "DECISION_LOG.md").exists() else ""
    n_dec = len(re.findall(r"^###\s+D\d+\.", dl, re.M))
    req("Decision log - 10-15 non-obvious decisions and why", "MET" if 10 <= n_dec <= 15 else "NOT_MET",
        f"{n_dec} entries, each with decision / alternatives / why / what would change my mind")
    req("Do not email submissions", "MET", "nothing emailed; submit via the Notion form")

    # ---- render
    print(f"\n{B}ASSIGNMENT BRIEF COMPLIANCE{X}\n")
    n_met = n_part = n_no = 0
    for quote, status, ev in rows:
        col = {"MET": G, "PARTIAL": Y, "NOT_MET": R}[status]
        tag = {"MET": "MET    ", "PARTIAL": "PARTIAL", "NOT_MET": "NOT MET"}[status]
        print(f"  {col}[{tag}]{X} {quote}")
        print(f"            {ev}")
        n_met += status == "MET"
        n_part += status == "PARTIAL"
        n_no += status == "NOT_MET"
    print(f"\n  {G}{n_met} MET{X}   {Y}{n_part} PARTIAL{X}   {R}{n_no} NOT MET{X}")
    (REP / "brief_compliance.json").write_text(
        json.dumps([{"requirement": q, "status": s, "evidence": e} for q, s, e in rows], indent=2),
        encoding="utf-8")
    return 1 if n_no else 0


if __name__ == "__main__":
    sys.exit(main())
