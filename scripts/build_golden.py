"""Assemble golden/v1.jsonl from the two machine passes plus any human adjudications.

Provenance is tracked PER EXAMPLE in the `labeler` field, and it is the whole point:

  human_adjudicated   -- a human opened it in label_cli.py and confirmed or overrode
  machine_agreed      -- A and B agreed and no human has looked at it yet
  machine_unresolved  -- A and B disagreed and no human has broken the tie (A is used
                         as a provisional value and the row is flagged)

`provisional` is True for anything not human-adjudicated. Downstream, `run_eval.py`
stamps PROVISIONAL on every number while any provisional rows remain, so it is
impossible to quietly report machine-labelled results as if they were adjudicated.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

sys.path.insert(0, ".")
from src.support_agent import config  # noqa: E402

GOLDEN = config.GOLDEN

REQUIRED = ["id", "text", "thread_context", "intent", "escalate", "escalate_reason_code",
            "reply_must_include", "reply_must_not_include", "difficulty", "labeler"]
VALID_DIFFICULTY = {"easy", "medium", "hard"}
VALID_REASONS = set(config.ESCALATION_REASON_CODES) | set(config.AUTO_REASON_CODES) | {""}


def load(name: str) -> list:
    p = GOLDEN / name
    return [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else []


def main() -> None:
    A = load("prelabel_A.jsonl")
    B = load("prelabel_B.jsonl")
    if not A or not B:
        raise SystemExit("need prelabel_A.jsonl and prelabel_B.jsonl")
    sample = {r["id"]: r for r in load("sample_220.jsonl")}
    human = {r["id"]: r for r in load("human_labels.jsonl")}   # written by label_cli.py

    B_by_id = {r["id"]: r for r in B}
    out = []
    for a in A:
        b = B_by_id.get(a["id"], {})
        h = human.get(a["id"])
        agree = (a["intent"] == b.get("intent")) and (bool(a["escalate"]) == bool(b.get("escalate")))

        if h:
            rec = {
                "intent": h["intent"], "escalate": bool(h["escalate"]),
                "escalate_reason_code": h.get("escalate_reason_code", ""),
                "reply_must_include": h.get("reply_must_include") or a["reply_must_include"],
                "reply_must_not_include": h.get("reply_must_not_include") or a["reply_must_not_include"],
                "difficulty": h.get("difficulty", a["difficulty"]),
                "labeler": "human_adjudicated",
                "provisional": False,
                "notes": h.get("notes", ""),
            }
        else:
            rec = {
                "intent": a["intent"], "escalate": bool(a["escalate"]),
                "escalate_reason_code": a.get("escalate_reason_code", ""),
                "reply_must_include": a["reply_must_include"],
                "reply_must_not_include": a["reply_must_not_include"],
                "difficulty": a["difficulty"],
                "labeler": "machine_agreed" if agree else "machine_unresolved",
                "provisional": True,
                "notes": a.get("notes", ""),
            }
        s = sample.get(a["id"], {})
        rec.update({
            "id": a["id"], "text": a["text"], "thread_context": a["thread_context"],
            "hard_flags": s.get("hard_flags", []), "is_hard": s.get("is_hard", False),
            "machine_A": {"intent": a["intent"], "escalate": bool(a["escalate"])},
            "machine_B": {"intent": b.get("intent"), "escalate": bool(b.get("escalate", False))},
            "ab_agreed": agree,
        })
        out.append(rec)

    p = GOLDEN / "v1.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    c = Counter(r["labeler"] for r in out)
    prov = sum(1 for r in out if r["provisional"])
    print(f"wrote {p}: {len(out)} rows")
    for k, v in c.most_common():
        print(f"  {k:<22} {v:>4}")
    print(f"\nPROVISIONAL rows (no human has looked): {prov}/{len(out)}")
    if prov:
        print("  -> every downstream number is stamped PROVISIONAL until the human pass runs:")
        print("     python golden/label_cli.py")


if __name__ == "__main__":
    main()
