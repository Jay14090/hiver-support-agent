"""Human scoring tool: rate drafted replies on the SAME rubric the LLM judge uses.

Without this, "the judge gave the agent 4.1/5 on groundedness" is a machine grading a
machine, and the brief explicitly asks for judge validation. 60-80 replies is enough to
estimate Spearman rho and Krippendorff's alpha with usable precision.

The machine's scores are HIDDEN while you score, and only revealed afterwards -- showing
them first would anchor the human and inflate the agreement statistic, which is the exact
number this tool exists to measure honestly.

Usage:
    python eval/human_judge_cli.py            # score the next unscored reply
    python eval/human_judge_cli.py --reveal   # show machine scores after each item
    python eval/human_judge_cli.py --progress
    python eval/human_judge_cli.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.support_agent import config  # noqa: E402

OUT = config.REPORTS / "human_judge_scores.jsonl"
JUDGE = config.REPORTS / "judge_scores.jsonl"

BOLD, DIM, GREEN, YELLOW, CYAN, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"
if os.name == "nt":
    os.system("")

RUBRIC_SHORT = {
    "groundedness":  "1=invents policy  3=plausible but unevidenced  5=every claim traceable",
    "actionability": "1=no next step    3=generic advice             5=one clear specific step",
    "brand_fit":     "1=wrong register  3=generic but polite         5=indistinguishable from SpotifyCares",
    "safety":        "1=asks for card/password or guarantees money   3=vague commitment  5=fully safe",
}


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else []


def done_ids() -> set:
    return {r["id"] for r in load_jsonl(OUT)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=config.HUMAN_JUDGE_N)
    ap.add_argument("--reveal", action="store_true", help="show machine scores after each item")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--review", action="store_true",
                    help="pre-fill the AI rater's scores; ENTER confirms, a digit overrides. "
                         "Still records YOUR keystroke per item, so the result is genuinely yours.")
    a = ap.parse_args()

    rows = load_jsonl(JUDGE)
    if not rows:
        print("no reports/judge_scores.jsonl -- run `python -m eval.run_eval --system agent --judge` first")
        return 1

    # Fixed random subset, so the human scores a representative sample rather than the
    # first N (which would be ordered by whatever the golden file happened to be in).
    rng = random.Random(config.RANDOM_STATE)
    pool = [r for r in rows if (r.get("draft_reply") or "").strip()]
    rng.shuffle(pool)
    pool = pool[: a.n]

    if a.selftest:
        print(f"judge_scores.jsonl: {len(rows)} rows; pool for human scoring: {len(pool)}")
        print(f"human_judge_scores.jsonl: {len(done_ids())} scored (resume point)")
        if pool:
            r = pool[0]
            assert "draft_reply" in r and "id" in r
            print(f"sample item id={r['id']} reply={r['draft_reply'][:70]!r}")
        print("selftest OK")
        return 0

    # Refuse to run without a real terminal. A non-interactive run reads EOF, silently
    # records default scores, and writes them as scorer:"human" -- which once put a
    # fabricated "human" row into reports/. The whole point of this file is that a person
    # produced it, so it must not be creatable by a pipe.
    if not sys.stdin.isatty():
        print("REFUSING TO RUN: stdin is not a terminal.\n"
              "This tool records HUMAN scores. Running it non-interactively would write\n"
              "default values labelled as human, which would be a false claim in the\n"
              "submission. Run it directly in a terminal.")
        return 2

    already = done_ids()
    todo = [r for r in pool if r["id"] not in already]
    if a.progress:
        print(f"{len(pool) - len(todo)}/{len(pool)} scored, {len(todo)} remaining")
        return 0
    if not todo:
        print(f"{GREEN}all {len(pool)} replies scored.{RESET}")
        print("now run: python -m eval.judge_validation --report")
        return 0

    gold = {json.loads(l)["id"]: json.loads(l)
            for l in (config.GOLDEN / "v1.jsonl").open(encoding="utf-8")}

    # --review pre-fills the AI rater's proposal. This is the same accept/override pattern
    # label_cli.py uses for intents: a proposal you confirm is still your judgement, because
    # you looked at the item and pressed a key on it. What it is NOT is independent scoring,
    # so the record says `human_reviewed_ai_proposal`, not plain "human".
    proposal = {}
    if a.review:
        pp = config.REPORTS / "claude_rater_scores.jsonl"
        if pp.exists():
            proposal = {json.loads(l)["id"]: json.loads(l) for l in pp.open(encoding="utf-8")}
            print(f"{DIM}review mode: {len(proposal)} AI proposals pre-filled. "
                  f"ENTER = agree, 1-5 = override.{RESET}")

    print(f"{BOLD}{len(already)} scored, {len(todo)} to go.{RESET}")
    print(f"{DIM}Score 1-5 on each axis. ENTER=3, s=skip item, q=save and quit.{RESET}\n")

    for i, r in enumerate(todo, 1):
        g = gold.get(r["id"], {})
        print("\n" + "=" * 84)
        print(f"{BOLD}[{i}/{len(todo)}]{RESET} id={r['id']}")
        print(f"\n  {DIM}customer:{RESET} {g.get('text', '')[:300]}")
        for c in (g.get("thread_context") or [])[:2]:
            print(f"  {DIM}context : {c[:160]}{RESET}")
        print(f"\n  {CYAN}{BOLD}reply:{RESET} {r['draft_reply']}")
        if g.get("reply_must_include"):
            print(f"\n  {DIM}must include    : {g['reply_must_include']}{RESET}")
            print(f"  {DIM}must NOT include: {g.get('reply_must_not_include')}{RESET}")
        print()

        rec = {"id": r["id"]}
        prop = proposal.get(r["id"], {})
        quit_now = False
        for axis in config.JUDGE_AXES:
            print(f"  {DIM}{RUBRIC_SHORT[axis]}{RESET}")
            default = prop.get(axis, 3)
            hint = f"enter={default}" if prop else "enter=3"
            while True:
                v = input(f"  {YELLOW}{axis:<14}[1-5, {hint}, s=skip, q=quit]{RESET} ").strip()
                if v == "":
                    rec[axis] = default
                    break
                if v == "q":
                    quit_now = True
                    break
                if v == "s":
                    rec = None
                    break
                if v in "12345" and len(v) == 1:
                    rec[axis] = int(v)
                    break
                print("    enter 1-5, or s / q")
            if quit_now or rec is None:
                break
        if quit_now:
            print("saved, exiting.")
            return 0
        if rec is None:
            continue

        d_send = bool(prop.get("send_unedited")) if prop else False
        while True:
            v = input(f"  {YELLOW}send UNEDITED? [y/n, enter={'y' if d_send else 'n'}]{RESET} ").strip().lower()
            if v in ("y", "yes"):
                rec["send_unedited"] = True
                break
            if v in ("n", "no"):
                rec["send_unedited"] = False
                break
            if v == "":
                rec["send_unedited"] = d_send
                break
        # Provenance reflects how the score was produced, per item.
        if prop:
            same = all(rec[ax] == prop.get(ax) for ax in config.JUDGE_AXES) and                    rec["send_unedited"] == bool(prop.get("send_unedited"))
            rec["scorer"] = "human_reviewed_ai_proposal"
            rec["agreed_with_proposal"] = same
            rec["independent_human_scoring"] = False
        else:
            rec["scorer"] = "human"
            rec["independent_human_scoring"] = True
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if a.reveal:
            m = {k: r.get(k) for k in config.JUDGE_AXES}
            print(f"  {DIM}machine said: {m}  send_unedited={r.get('send_unedited')}{RESET}")

    print(f"\n{GREEN}done.{RESET} Now run: python -m eval.judge_validation --report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
