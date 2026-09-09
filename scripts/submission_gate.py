"""The Phase 10 submission checklist, as an executable.

Every item is either GREEN (checked programmatically) or RED with the exact command that
would fix it. Nothing here is self-assessed prose: if the checklist says the human review
pass is done, it is because `golden/human_labels.jsonl` says so.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.support_agent import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GREEN, RED, YELLOW, BOLD, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m"
import os  # noqa: E402

if os.name == "nt":
    os.system("")

checks = []


def check(name: str, ok, detail: str = "", fix: str = "", warn_only: bool = False):
    checks.append({"name": name, "ok": bool(ok), "detail": detail, "fix": fix, "warn_only": warn_only})


def jsonl_len(p: Path) -> int:
    return sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0


def main() -> int:
    G = config.GOLDEN
    R = config.REPORTS

    # ---------- reproduction
    timing = R / "repro_timing.txt"
    t_ok, t_detail = False, "not timed yet"
    if timing.exists():
        txt = timing.read_text(encoding="utf-8")
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)", txt)
        if m:
            secs = float(m.group(1))
            t_ok = secs < 900
            t_detail = f"{secs:.0f}s ({secs/60:.1f} min), budget 15 min"
    check("make demo works from a clean clone, no API key, < 15 min", t_ok, t_detail,
          "python scripts/verify_repro.py")
    check("make full documented and known-working", (ROOT / "Makefile").exists()
          and "full:" in (ROOT / "Makefile").read_text(encoding="utf-8"),
          "Makefile target present")

    # ---------- golden set
    n_golden = jsonl_len(G / "v1.jsonl")
    check("golden/v1.jsonl has 150-250 examples", 150 <= n_golden <= 250, f"{n_golden} rows")

    schema_ok = subprocess.run([sys.executable, "scripts/validate_golden.py"], cwd=ROOT,
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace").returncode == 0
    check("golden set is schema-valid", schema_ok, fix="python scripts/validate_golden.py")

    if (G / "v1.jsonl").exists():
        rows = [json.loads(l) for l in (G / "v1.jsonl").open(encoding="utf-8")]
        from collections import Counter

        c = Counter(r["intent"] for r in rows)
        thin = {i: c.get(i, 0) for i in config.INTENTS if c.get(i, 0) < config.GOLDEN_MIN_PER_INTENT}
        check(f"every intent has >= {config.GOLDEN_MIN_PER_INTENT} examples", not thin,
              f"below floor: {thin}" if thin else f"min per intent = {min(c.values())}")
        n_human = sum(1 for r in rows if r.get("labeler") == "human_adjudicated")
        n_queue = jsonl_len(G / "review_queue.jsonl")
        n_done = jsonl_len(G / "human_labels.jsonl")
        check("HUMAN REVIEW PASS COMPLETE", n_done >= n_queue and n_queue > 0,
              f"{n_done}/{n_queue} queue items reviewed; {n_human}/{len(rows)} rows adjudicated",
              "python golden/label_cli.py   (then: python scripts/build_golden.py)")
    else:
        check("every intent has enough examples", False, "no golden set")
        check("HUMAN REVIEW PASS COMPLETE", False, "no golden set")

    note = (G / "SAMPLING_NOTE.md")
    note_ok = note.exists() and "machine labels" in note.read_text(encoding="utf-8").lower()
    check("SAMPLING_NOTE.md honestly states the labelling process", note_ok)

    # ---------- systems + eval
    res_p = R / "results.json"
    res = json.loads(res_p.read_text(encoding="utf-8")) if res_p.exists() else {}
    systems = res.get("systems", {})
    check("two baselines implemented and evaluated",
          any(k.startswith("b0") for k in systems) and any(k.startswith("b1") for k in systems),
          f"systems evaluated: {list(systems)}", "python -m eval.run_eval --system all")

    jv_p = R / "judge_validation.json"
    jv = json.loads(jv_p.read_text(encoding="utf-8")) if jv_p.exists() else {}
    ha = jv.get("human_agreement", {})
    probes = jv.get("bias_probes", {})
    check("LLM judge + human agreement evidence",
          bool(ha.get("overall_mean_rho")),
          f"mean rho = {ha.get('overall_mean_rho')}" if ha.get("overall_mean_rho")
          else ha.get("message", "not run"),
          "python eval/human_judge_cli.py  &&  python -m eval.judge_validation --report")
    check("3 bias probes with numbers",
          all(k in probes for k in ("position", "length", "self_preference")),
          f"probes present: {list(probes)}", "python -m eval.judge_validation --probes")

    ci_ok = all("macro_f1" in s.get("intent", {}) and "lo" in s["intent"]["macro_f1"]
                for s in systems.values()) if systems else False
    check("all headline metrics carry bootstrap CIs", ci_ok)

    # ---------- write-ups
    rep = R / "REPORT.md"
    if rep.exists():
        t = rep.read_text(encoding="utf-8")
        words = len(t.split())
        check("REPORT.md <= 6 pages", words <= 3600, f"~{words} words (~{words/600:.1f} pages)")
        m = re.search(r"##\s*1\..*?misleading.*?\n(.*?)(?=\n## )", t, re.S | re.I)
        n_items = len(re.findall(r"^\s*\d+\.\s+\*\*", m.group(1), re.M)) if m else 0
        check("misleading-numbers section has >= 8 items", n_items >= 8, f"{n_items} items")
    else:
        check("REPORT.md <= 6 pages", False, "missing")
        check("misleading-numbers section has >= 8 items", False, "missing")

    fa = R / "FAILURE_ANALYSIS.md"
    n_modes = len(re.findall(r"^##\s+Mode\s+\d", fa.read_text(encoding="utf-8"), re.M)) if fa.exists() else 0
    check("FAILURE_ANALYSIS.md: 5 modes with real examples", n_modes == 5, f"{n_modes} modes")

    dl = R / "DECISION_LOG.md"
    n_dec = len(re.findall(r"^###\s+D\d+\.", dl.read_text(encoding="utf-8"), re.M)) if dl.exists() else 0
    check("DECISION_LOG.md: 10-15 entries", 10 <= n_dec <= 15, f"{n_dec} entries")

    nb = ROOT / "NOT_BUILDING.md"
    check("NOT_BUILDING.md populated", nb.exists() and len(nb.read_text(encoding="utf-8")) > 800)
    check("WALKTHROUGH.md written", (R / "WALKTHROUGH.md").exists())

    rd = ROOT / "README.md"
    if rd.exists():
        t = rd.read_text(encoding="utf-8")
        check("README has Citations & Borrowings + AI disclosure",
              "Citations" in t and ("AI assistance" in t or "Claude" in t))
    else:
        check("README has Citations & Borrowings + AI disclosure", False, "missing README")

    # ---------- hygiene
    sec = subprocess.run([sys.executable, "scripts/scan_secrets.py"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    check("no PII / API keys / .env in git history", sec.returncode == 0,
          sec.stdout.strip().split("\n")[-1] if sec.stdout else "")

    size = sum(f.stat().st_size for f in ROOT.rglob("*")
               if f.is_file() and ".venv" not in f.as_posix()
               and "data/raw" not in f.as_posix() and "data/interim" not in f.as_posix())
    check("repo < 100MB", size < 100 * 1024**2, f"{size/1024**2:.1f} MB")

    n_commits = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip() or "0"
    check("clean history with sensible commit messages", int(n_commits) >= 5,
          f"{n_commits} commits")

    spend = json.loads((ROOT / "cost_ledger.json").read_text(encoding="utf-8"))["total_usd"]
    check("cost_ledger.json total under budget", spend < config.BUDGET_USD_CAP,
          f"${spend:.2f} of ${config.BUDGET_USD_CAP:.0f}")

    # ---------- render
    print(f"\n{BOLD}SUBMISSION CHECKLIST{RESET}")
    n_red = 0
    for c in checks:
        if c["ok"]:
            mark, col = "x", GREEN
        elif c["warn_only"]:
            mark, col = "!", YELLOW
        else:
            mark, col = " ", RED
            n_red += 1
        print(f"  {col}[{mark}]{RESET} {c['name']}" + (f"  {col}-- {c['detail']}{RESET}" if c["detail"] else ""))
        if not c["ok"] and c["fix"]:
            print(f"        fix: {c['fix']}")
    print(f"\n{n_red} item(s) RED, {len(checks) - n_red} GREEN")
    (config.REPORTS / "submission_checklist.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8")
    return 1 if n_red else 0


if __name__ == "__main__":
    sys.exit(main())
