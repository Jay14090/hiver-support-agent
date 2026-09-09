"""Draw the 220-example golden sample from the TEST split.

Design, all of it consequential and all of it recorded in golden/SAMPLING_NOTE.md:

  - **Test split only.** The retrieval corpus is the oldest 70%; the golden set comes
    from the newest 15%. Nothing the agent can retrieve is ever something it is graded on.
  - **Stratified, not random.** Proportional allocation across intents with a floor of
    `GOLDEN_MIN_PER_INTENT`, so per-class F1 is not computed on 2 examples. Where an
    intent genuinely does not have enough examples in the test split (billing, for one),
    it takes everything available and the shortfall is REPORTED rather than padded.
  - **A deliberate ~15% hard-case quota.** Negation, sarcasm, multi-intent, very short
    messages, and messages carrying escalation triggers. Support systems fail on exactly
    these, and a golden set of easy cases would measure nothing interesting.

Consequence, stated here and in the report: **the golden set is NOT a random sample of
production traffic.** Macro-F1 over it flatters rare classes, and the deflection rate
measured on it does not transfer to real inbound volume.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from src.support_agent import config  # noqa: E402

GOLDEN = config.GOLDEN
TARGET = config.GOLDEN_TARGET_N
FLOOR = config.GOLDEN_MIN_PER_INTENT

# ------------------------------------------------------------------ hard-case detectors
NEGATION = re.compile(r"\b(not|n't|never|no longer|nothing|neither|nor|without|un\w+ed)\b", re.I)
SARCASM = re.compile(r"(\bthanks a lot\b|\bgreat job\b|\bwonderful\b|\blove how\b|\bnice one\b|"
                     r"\bbrilliant\b|🙄|😒|/s\b|\bapparently\b|\bsupposedly\b)", re.I)
MULTI_INTENT = re.compile(r"\b(also|and also|another thing|plus|as well as|second(ly)?|"
                          r"on top of that|besides)\b", re.I)
ESCALATION_TRIGGER = re.compile(
    r"\b(refund|charged twice|double charge|unauthorized|unauthorised|hack(ed)?|cancel|"
    r"lawyer|legal|gdpr|data protection|fraud|scam|stolen|dispute|chargeback)\b", re.I)


def hard_flags(text: str, context) -> list:
    f = []
    if NEGATION.search(text):
        f.append("negation")
    if SARCASM.search(text):
        f.append("sarcasm")
    if MULTI_INTENT.search(text):
        f.append("multi_intent")
    if len(text.split()) <= 8:
        f.append("very_short")
    if ESCALATION_TRIGGER.search(text):
        f.append("escalation_trigger")
    if context:
        f.append("has_context")
    return f


def is_hard_case(flags: list) -> bool:
    """What actually counts as a hard case.

    Plain negation does NOT, on its own. Almost every support complaint contains a
    negation ("my music *won't* play"), so counting it alone flagged a quarter of the
    sample as hard and made the quota meaningless. Negation is only interesting when it
    combines with something else -- sarcasm, a second intent, or near-absent context.
    """
    strong = {"sarcasm", "multi_intent", "very_short", "escalation_trigger", "labeler_says_hard"}
    hits = set(flags)
    if hits & strong:
        return True
    return "negation" in hits and "has_context" in hits


def main() -> None:
    rng = random.Random(config.RANDOM_STATE)
    # Stratify on the FULL Pass-A run, i.e. the exact procedure that produces the final
    # golden labels. Using a cheaper, different intent pass here (as this script first
    # did) let the sample satisfy a 12-per-intent floor that the real labels then broke.
    src = GOLDEN / "prelabel_A_full.jsonl"
    if not src.exists():
        raise SystemExit("run `python scripts/prelabel.py --stage A-full` first")
    rows = [json.loads(l) for l in src.open(encoding="utf-8")]
    for r in rows:
        r["hard_flags"] = hard_flags(r["text"], r.get("thread_context"))
        # The labeller's own difficulty judgement counts as a hard-case signal too: it
        # catches the sarcasm and context-dependence that regexes cannot see.
        if r.get("difficulty") == "hard":
            r["hard_flags"].append("labeler_says_hard")
        r["is_hard"] = is_hard_case(r["hard_flags"])

    by_intent = defaultdict(list)
    for r in rows:
        by_intent[r["intent"]].append(r)
    for v in by_intent.values():
        rng.shuffle(v)

    pool_counts = {k: len(v) for k, v in by_intent.items()}
    print(f"test split: {len(rows)} messages across {len(by_intent)} intents")

    # ---- allocation: floor first, then proportional over what remains
    alloc, shortfalls = {}, {}
    for intent in config.INTENTS:
        avail = pool_counts.get(intent, 0)
        take = min(FLOOR, avail)
        alloc[intent] = take
        if avail < FLOOR:
            shortfalls[intent] = {"available": avail, "floor": FLOOR}
    remaining = TARGET - sum(alloc.values())
    if remaining > 0:
        leftover = {i: pool_counts.get(i, 0) - alloc[i] for i in config.INTENTS}
        total_left = sum(max(0, v) for v in leftover.values())
        if total_left > 0:
            for intent in config.INTENTS:
                extra = int(round(remaining * max(0, leftover[intent]) / total_left))
                alloc[intent] = min(pool_counts.get(intent, 0), alloc[intent] + extra)
    # trim/pad to land exactly on TARGET
    while sum(alloc.values()) > TARGET:
        biggest = max(alloc, key=lambda k: alloc[k] - FLOOR)
        alloc[biggest] -= 1
    while sum(alloc.values()) < TARGET:
        cands = [i for i in config.INTENTS if alloc[i] < pool_counts.get(i, 0)]
        if not cands:
            break
        biggest = max(cands, key=lambda k: pool_counts.get(k, 0) - alloc[k])
        alloc[biggest] += 1

    # ---- pick, honouring the hard-case quota inside each intent
    want_hard = int(round(TARGET * config.HARD_CASE_QUOTA))
    picked, hard_taken = [], 0
    for intent in config.INTENTS:
        n = alloc[intent]
        if n <= 0:
            continue
        pool = by_intent.get(intent, [])
        hard = [r for r in pool if r["is_hard"]]
        easy = [r for r in pool if not r["is_hard"]]
        n_hard = min(len(hard), max(1, int(round(n * config.HARD_CASE_QUOTA))))
        take = hard[:n_hard] + easy[: n - n_hard]
        if len(take) < n:  # intent is hard-case heavy; backfill from whatever is left
            take += [r for r in hard[n_hard:] if r not in take][: n - len(take)]
        picked.extend(take[:n])
        hard_taken += sum(1 for r in take[:n] if r["is_hard"])

    rng.shuffle(picked)
    out = GOLDEN / "sample_220.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps({
                "id": r["id"], "text": r["text"], "thread_context": r["thread_context"],
                "stratify_intent": r["intent"], "escalate": r.get("escalate"), "hard_flags": r["hard_flags"],
                "is_hard": r["is_hard"],
            }, ensure_ascii=False) + "\n")

    meta = {
        "n": len(picked),
        "target": TARGET,
        "random_state": config.RANDOM_STATE,
        "source_split": "test (newest 15% by root tweet timestamp)",
        "floor_per_intent": FLOOR,
        "allocation": {k: v for k, v in alloc.items() if v},
        "pool_counts": pool_counts,
        "shortfalls": shortfalls,
        "hard_cases": hard_taken,
        "hard_case_rate": round(hard_taken / max(1, len(picked)), 3),
        "hard_flag_counts": dict(Counter(f for r in picked for f in r["hard_flags"])),
    }
    (GOLDEN / "sampling_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {out}")
    if shortfalls:
        print(f"\n!! intents BELOW the {FLOOR}-example floor: {list(shortfalls)}")
        print("   These take everything available. Per-class metrics for them are reported")
        print("   with raw support counts and explicitly refused as evidence.")


if __name__ == "__main__":
    main()
