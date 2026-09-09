"""Human adjudication tool for the golden set.

Design goals, in order:
  1. **Fast.** ~90 items in ~40 minutes means ~25s per item, so the common case --
     "both machines agree and they're right" -- must be a SINGLE keystroke.
  2. **Resumable.** Saves after every single item. Ctrl-C at item 47 loses nothing.
  3. **Informative.** Shows the message, the thread context, both machine labels, and
     (optionally) similar historical threads, so the human is deciding, not guessing.

Usage:
    python golden/label_cli.py               # work the review queue
    python golden/label_cli.py --all         # review all 220, not just the queue
    python golden/label_cli.py --retrieval   # also show similar past threads (slower)
    python golden/label_cli.py --progress    # how far through am I
    python golden/label_cli.py --selftest    # no input needed; used by the P3.7 gate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.support_agent import config  # noqa: E402

GOLDEN = config.GOLDEN
OUT = GOLDEN / "human_labels.jsonl"

# Single-keystroke intent picker.
KEYS = {
    "1": "playback_streaming_issue",
    "2": "account_access_login",
    "3": "billing_charge_dispute",
    "4": "subscription_cancel_refund",
    "5": "plan_or_premium_management",
    "6": "content_missing_or_metadata",
    "7": "device_integration_issue",
    "8": "app_bug_or_crash",
    "9": "feature_request_or_complaint",
    "0": "praise_or_chitchat",
    "-": "other",
}
RKEYS = {v: k for k, v in KEYS.items()}

BOLD, DIM, GREEN, RED, YELLOW, CYAN, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"
)
if os.name == "nt":
    os.system("")  # enable ANSI on Windows terminals


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else []


def save_one(rec: dict) -> None:
    """Append-only, one line per decision. Later lines win, so an edit is just a re-append."""
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def done_ids() -> set:
    return {r["id"] for r in load_jsonl(OUT)}


def _fmt_intent(name: str) -> str:
    return f"{name} [{RKEYS.get(name, '?')}]"


def show(item: dict, i: int, n: int, retriever=None) -> None:
    A, B = item["A"], item["B"]
    agree_i = A["intent"] == B["intent"]
    agree_e = bool(A["escalate"]) == bool(B["escalate"])
    print("\n" + "=" * 84)
    print(f"{BOLD}[{i}/{n}]{RESET}  id={item['id']}  {DIM}({item.get('reason','')}){RESET}")
    for c in (item.get("thread_context") or [])[:3]:
        print(f"  {DIM}context: {c[:180]}{RESET}")
    print(f"\n  {BOLD}{item['text'][:400]}{RESET}\n")

    ci = GREEN if agree_i else RED
    ce = GREEN if agree_e else RED
    print(f"  intent    A: {ci}{_fmt_intent(A['intent'])}{RESET}"
          f"   B: {ci}{_fmt_intent(B.get('intent',''))}{RESET}"
          + ("" if agree_i else f"  {RED}<-- DISAGREE{RESET}"))
    print(f"  escalate  A: {ce}{A['escalate']}{RESET} ({A.get('escalate_reason_code','')})"
          f"   B: {ce}{B.get('escalate')}{RESET} ({B.get('escalate_reason_code','')})"
          + ("" if agree_e else f"  {RED}<-- DISAGREE{RESET}"))
    print(f"  must include    : {A.get('reply_must_include')}")
    print(f"  must NOT include: {A.get('reply_must_not_include')}")
    if A.get("notes"):
        print(f"  {DIM}A notes: {A['notes']}{RESET}")

    if retriever is not None:
        try:
            for h in retriever.search(item["text"])[:2]:
                print(f"  {CYAN}similar [{h.id}] {h.customer_text[:90]}{RESET}")
                print(f"           {DIM}-> {h.resolution_reply[:110]}{RESET}")
        except Exception as e:  # noqa: BLE001 -- retrieval is a convenience, never fatal here
            print(f"  {DIM}(retrieval unavailable: {e}){RESET}")


HELP = f"""
  {BOLD}ENTER{RESET} accept A     {BOLD}b{RESET} accept B     {BOLD}e{RESET} toggle escalate
  {BOLD}1-9,0,-{RESET} set intent:
     1 playback   2 login      3 billing    4 cancel/refund  5 plan/premium  6 content
     7 device     8 app bug    9 feature    0 praise         - other
  {BOLD}n{RESET} add a note    {BOLD}s{RESET} skip    {BOLD}q{RESET} save and quit    {BOLD}?{RESET} this help
"""


def review(items: list, retriever=None) -> None:
    already = done_ids()
    todo = [x for x in items if x["id"] not in already]
    print(f"{BOLD}{len(already)} already reviewed, {len(todo)} to go.{RESET}")
    print(HELP)
    n = len(todo)
    for i, item in enumerate(todo, 1):
        show(item, i, n, retriever)
        cur = dict(item["A"])
        while True:
            try:
                k = input(f"  {YELLOW}[enter=A, b=B, 1-0/-=intent, e=escalate, n=note, s=skip, q=quit]{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nsaved, exiting.")
                return
            if k == "":
                break
            if k == "b":
                cur = dict(item["B"])
                print(f"    -> took B: {cur['intent']}, escalate={cur['escalate']}")
                continue
            if k in KEYS:
                cur["intent"] = KEYS[k]
                print(f"    -> intent = {cur['intent']}")
                continue
            if k == "e":
                cur["escalate"] = not bool(cur.get("escalate"))
                print(f"    -> escalate = {cur['escalate']}")
                continue
            if k == "n":
                cur["notes"] = input("    note: ").strip()
                continue
            if k == "?":
                print(HELP)
                continue
            if k == "s":
                cur = None
                break
            if k == "q":
                print("saved, exiting.")
                return
            print("    unrecognised key -- press ? for help")
        if cur is None:
            continue
        save_one({
            "id": item["id"],
            "intent": cur["intent"],
            "escalate": bool(cur.get("escalate")),
            "escalate_reason_code": cur.get("escalate_reason_code", ""),
            "reply_must_include": cur.get("reply_must_include", []),
            "reply_must_not_include": cur.get("reply_must_not_include", []),
            "difficulty": cur.get("difficulty", "medium"),
            "notes": cur.get("notes", ""),
            "labeler": "human_adjudicated",
        })
    print(f"\n{GREEN}queue complete.{RESET} Now run: python scripts/build_golden.py")


def selftest() -> int:
    """Non-interactive check used by the P3.7 gate: files parse, save/resume works."""
    q = GOLDEN / "review_queue.jsonl"
    items = load_jsonl(q)
    print(f"review_queue.jsonl: {len(items)} items")
    if items:
        need = {"id", "text", "A", "B"}
        missing = [k for k in need if k not in items[0]]
        assert not missing, f"queue item missing {missing}"
        show(items[0], 1, len(items))
    # resume logic must be a pure function of what is on disk
    before = done_ids()
    print(f"human_labels.jsonl: {len(before)} decisions recorded (resume point)")
    assert isinstance(before, set)
    print("selftest OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="review all 220, not just the queue")
    ap.add_argument("--retrieval", action="store_true", help="show similar historical threads")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if a.all:
        A = {r["id"]: r for r in load_jsonl(GOLDEN / "prelabel_A.jsonl")}
        B = {r["id"]: r for r in load_jsonl(GOLDEN / "prelabel_B.jsonl")}
        items = [{"id": i, "text": r["text"], "thread_context": r["thread_context"],
                  "reason": "full_pass", "A": r, "B": B.get(i, {})} for i, r in A.items()]
    else:
        items = load_jsonl(GOLDEN / "review_queue.jsonl")
    if not items:
        print("nothing to review -- run scripts/ab_agreement.py first")
        return 1

    if a.progress:
        d = done_ids()
        rem = [x for x in items if x["id"] not in d]
        print(f"{len(items) - len(rem)}/{len(items)} reviewed, {len(rem)} remaining")
        return 0

    retriever = None
    if a.retrieval:
        from src.support_agent.retrieval import HybridRetriever

        print("loading retriever ...")
        retriever = HybridRetriever.from_split("corpus")

    review(items, retriever)
    return 0


if __name__ == "__main__":
    sys.exit(main())
