"""Top up under-floor intents in the golden sample, using the same 3-pass procedure.

Why this is legitimate and not cherry-picking: the per-intent floor is a **pre-registered
design requirement** of the stratified sample (`config.GOLDEN_MIN_PER_INTENT`). The sample
was drawn to satisfy it under Pass A labels; 3-pass majority re-adjudication then moved
rows between intents and `billing_charge_dispute` fell to 8. Restoring the floor is
completing the stratification, not selecting for a result -- the candidates are drawn at
RANDOM from the remaining test-split pool, and they are labelled by the identical blind
A/B/C procedure before being accepted.

What would make it cherry-picking, and is therefore forbidden here: choosing candidates by
whether the *agent* gets them right. The agent is never consulted, and this script runs
before any re-evaluation.
"""
from __future__ import annotations

import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
from scripts.adjudicate_pass_c import MODEL_C, PASS_C_SYSTEM, _fmt  # noqa: E402
from scripts.prelabel import PASS_B_SYSTEM  # noqa: E402
from scripts.sample_golden import hard_flags, is_hard_case  # noqa: E402
from src.support_agent import config, llm  # noqa: E402
from src.support_agent.codebook import load_codebook_text  # noqa: E402

GOLDEN = config.GOLDEN
_lock = threading.Lock()


def main() -> int:
    golden = [json.loads(l) for l in (GOLDEN / "v1.jsonl").open(encoding="utf-8")]
    counts = Counter(g["intent"] for g in golden)
    under = {i: counts.get(i, 0) for i in config.INTENTS
             if counts.get(i, 0) < config.GOLDEN_MIN_PER_INTENT}
    if not under:
        print("all intents meet the floor; nothing to do")
        return 0
    print(f"under floor: {under}")

    used = {g["id"] for g in golden}
    a_full = [json.loads(l) for l in (GOLDEN / "prelabel_A_full.jsonl").open(encoding="utf-8")]
    sample_rows = {r["id"]: r for r in
                   (json.loads(l) for l in (GOLDEN / "sample_220.jsonl").open(encoding="utf-8"))}

    import random

    rng = random.Random(config.RANDOM_STATE + 99)
    added_A, added_B, added_C, added_sample = [], [], [], []

    for intent, have in under.items():
        need = config.GOLDEN_MIN_PER_INTENT - have
        pool = [r for r in a_full if r["intent"] == intent and r["id"] not in used]
        rng.shuffle(pool)
        print(f"\n[{intent}] need {need} more; {len(pool)} unused candidates in the test split")
        if not pool:
            print("  POOL EXHAUSTED -- the test split genuinely has no more of this intent.")
            continue

        cb = load_codebook_text()
        sys_b = PASS_B_SYSTEM % cb
        sys_c = PASS_C_SYSTEM % cb

        def label(row, system, model, tag):
            parsed, _ = llm.complete_json(_fmt(row), system=system, model=model,
                                          max_tokens=320, tag=tag)
            return parsed or {}

        accepted = 0
        for row in pool:
            if accepted >= need:
                break
            b = label(row, sys_b, config.LABELER_B_MODEL, "prelabel_B")
            c = label(row, sys_c, MODEL_C, "prelabel_C")
            votes = [row["intent"], b.get("intent"), c.get("intent")]
            top, n = Counter(votes).most_common(1)[0]
            status = "accept" if (n >= 2 and top == intent) else "reject"
            print(f"  {status}: id={row['id']} votes={votes}")
            if status == "reject":
                continue
            accepted += 1
            added_A.append(row)
            added_B.append({"id": row["id"], "text": row["text"],
                            "thread_context": row["thread_context"],
                            "intent": b.get("intent") if b.get("intent") in config.INTENTS else "other",
                            "escalate": bool(b.get("escalate", False)),
                            "escalate_reason_code": b.get("escalate_reason_code", ""),
                            "reply_must_include": (b.get("reply_must_include") or [])[:3],
                            "reply_must_not_include": (b.get("reply_must_not_include") or [])[:3],
                            "difficulty": b.get("difficulty", "medium"),
                            "notes": str(b.get("notes", ""))[:160],
                            "labeler": f"machine_pass_B:{config.LABELER_B_MODEL}",
                            "parse_failed": not b})
            added_C.append({"id": row["id"], "text": row["text"],
                            "thread_context": row["thread_context"],
                            "intent": c.get("intent") if c.get("intent") in config.INTENTS else "other",
                            "escalate": bool(c.get("escalate", False)),
                            "escalate_reason_code": c.get("escalate_reason_code", ""),
                            "difficulty": c.get("difficulty", "medium"),
                            "notes": str(c.get("notes", ""))[:160],
                            "labeler": f"machine_pass_C:{MODEL_C}",
                            "parse_failed": not c})
            hf = hard_flags(row["text"], row.get("thread_context"))
            if row.get("difficulty") == "hard":
                hf.append("labeler_says_hard")
            added_sample.append({"id": row["id"], "text": row["text"],
                                 "thread_context": row.get("thread_context") or [],
                                 "stratify_intent": row["intent"],
                                 "escalate": row.get("escalate"),
                                 "hard_flags": hf, "is_hard": is_hard_case(hf)})
        print(f"  accepted {accepted}/{need}")

    llm.flush_ledger()
    if not added_A:
        print("\nnothing added")
        return 0

    for name, rows in (("prelabel_A.jsonl", added_A), ("prelabel_B.jsonl", added_B),
                       ("prelabel_C.jsonl", added_C), ("sample_220.jsonl", added_sample)):
        p = GOLDEN / name
        with p.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"appended {len(rows)} rows to {p.name}")

    print("\nnow re-run: python scripts/build_golden.py && python scripts/validate_golden.py")
    print(f"spend ${llm.spend_so_far():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
