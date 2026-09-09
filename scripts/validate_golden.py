"""P3.6 gate: schema-validate golden/v1.jsonl.

Deliberately strict. A golden set with a silently malformed row produces metrics that
look fine and mean nothing, and that failure is invisible in every chart downstream.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

sys.path.insert(0, ".")
from src.support_agent import config  # noqa: E402

SCHEMA = {
    "id": str,
    "text": str,
    "thread_context": list,
    "intent": str,
    "escalate": bool,
    "escalate_reason_code": str,
    "reply_must_include": list,
    "reply_must_not_include": list,
    "difficulty": str,
    "labeler": str,
    "provisional": bool,
}
VALID_DIFFICULTY = {"easy", "medium", "hard"}
VALID_REASONS = set(config.ESCALATION_REASON_CODES) | set(config.AUTO_REASON_CODES) | {""}
VALID_LABELERS = {"human_adjudicated", "machine_agreed", "model_majority_3pass", "machine_unresolved"}


def main() -> int:
    p = config.GOLDEN / "v1.jsonl"
    if not p.exists():
        print(f"FAIL: {p} does not exist")
        return 1
    rows = [json.loads(l) for l in p.open(encoding="utf-8")]
    errors = []

    if not (150 <= len(rows) <= 250):
        errors.append(f"row count {len(rows)} outside the required 150-250 band")

    seen = set()
    for i, r in enumerate(rows):
        where = f"row {i} (id={r.get('id')})"
        for k, t in SCHEMA.items():
            if k not in r:
                errors.append(f"{where}: missing field `{k}`")
            elif not isinstance(r[k], t):
                errors.append(f"{where}: `{k}` should be {t.__name__}, got {type(r[k]).__name__}")
        if r.get("id") in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(r.get("id"))
        if r.get("intent") not in config.INTENTS:
            errors.append(f"{where}: intent {r.get('intent')!r} not in config.INTENTS")
        if r.get("difficulty") not in VALID_DIFFICULTY:
            errors.append(f"{where}: difficulty {r.get('difficulty')!r} invalid")
        if r.get("escalate_reason_code") not in VALID_REASONS:
            errors.append(f"{where}: reason code {r.get('escalate_reason_code')!r} invalid")
        if r.get("labeler") not in VALID_LABELERS:
            errors.append(f"{where}: labeler {r.get('labeler')!r} invalid")
        if not str(r.get("text", "")).strip():
            errors.append(f"{where}: empty text")
        # An escalation must carry an escalation reason, never an auto reason.
        if r.get("escalate") and r.get("escalate_reason_code") in set(config.AUTO_REASON_CODES):
            errors.append(f"{where}: escalate=True but reason is an auto code")

    counts = Counter(r["intent"] for r in rows)
    thin = {i: counts.get(i, 0) for i in config.INTENTS if counts.get(i, 0) < config.GOLDEN_MIN_PER_INTENT}

    print(f"golden/v1.jsonl: {len(rows)} rows")
    print("\nper-intent counts:")
    for i in config.INTENTS:
        n = counts.get(i, 0)
        flag = "  << below floor" if n < config.GOLDEN_MIN_PER_INTENT else ""
        print(f"  {i:<32} {n:>4}{flag}")
    lab = Counter(r["labeler"] for r in rows)
    print("\nlabel provenance:", dict(lab))
    prov = sum(1 for r in rows if r.get("provisional"))
    print(f"provisional rows: {prov}/{len(rows)}")

    if thin:
        # A warning, not a failure: an intent that is genuinely rare in public data
        # cannot be conjured, and padding it would be worse than reporting it.
        print(f"\nWARNING: intents below the {config.GOLDEN_MIN_PER_INTENT}-example floor: {thin}")
        print("  per-class metrics for these are reported with raw support and not used as evidence")

    if errors:
        print(f"\nFAIL: {len(errors)} schema error(s)")
        for e in errors[:25]:
            print("  -", e)
        return 1
    print("\nOK: schema valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
