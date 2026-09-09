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


VALID_REASON_CODES = set(config.ESCALATION_REASON_CODES) | set(config.AUTO_REASON_CODES)


def load(name: str) -> list:
    p = GOLDEN / name
    return [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else []


def normalise_reason(code: str, escalate: bool, fallbacks) -> str:
    """Force the reason code into the enum.

    Pass C (gpt-4o) invented plausible-sounding codes outside the enum -- e.g.
    `content_removal_complex`, `account_merge_request`, `customer_threatening_cancellation`.
    Rather than widening the enum to whatever a model felt like emitting, the code is
    normalised: keep it if valid AND consistent with the escalate decision, otherwise take
    the first valid code from the other passes, otherwise leave it empty. The schema gate
    in validate_golden.py is what surfaced this, and it stays strict.
    """
    def ok(c):
        if c not in VALID_REASON_CODES:
            return False
        return (c in config.ESCALATION_REASON_CODES) if escalate else (c in config.AUTO_REASON_CODES)

    if ok(code):
        return code
    for f in fallbacks:
        if ok(f):
            return f
    return ""


def main() -> None:
    A = load("prelabel_A.jsonl")
    B = load("prelabel_B.jsonl")
    if not A or not B:
        raise SystemExit("need prelabel_A.jsonl and prelabel_B.jsonl")
    sample = {r["id"]: r for r in load("sample_220.jsonl")}
    human = {r["id"]: r for r in load("human_labels.jsonl")}   # written by label_cli.py

    B_by_id = {r["id"]: r for r in B}
    C_by_id = {r["id"]: r for r in load("prelabel_C.jsonl")}   # optional third blind pass
    out = []
    for a in A:
        b = B_by_id.get(a["id"], {})
        c = C_by_id.get(a["id"])
        h = human.get(a["id"])
        agree = (a["intent"] == b.get("intent")) and (bool(a["escalate"]) == bool(b.get("escalate")))

        # 2-of-3 majority where a third blind pass exists. This replaces the previous
        # behaviour on disagreements, which silently defaulted to Pass A -- an arbitrary
        # tiebreak that just meant "whichever pass I happened to run first wins".
        # NOTE: majority of three MODELS is still not human adjudication. Rows resolved
        # this way are labelled `model_majority_3pass` and stay provisional.
        maj_intent = maj_escalate = None
        if c and not h:
            iv = Counter([a["intent"], b.get("intent"), c["intent"]]).most_common(1)[0]
            if iv[1] >= 2:
                maj_intent = iv[0]
            ev = Counter([bool(a["escalate"]), bool(b.get("escalate")), bool(c["escalate"])]).most_common(1)[0]
            if ev[1] >= 2:
                maj_escalate = ev[0]

        if h:
            rec = {
                "intent": h["intent"], "escalate": bool(h["escalate"]),
                "escalate_reason_code": normalise_reason(
                    h.get("escalate_reason_code", ""), bool(h["escalate"]),
                    [a.get("escalate_reason_code", "")]),
                "reply_must_include": h.get("reply_must_include") or a["reply_must_include"],
                "reply_must_not_include": h.get("reply_must_not_include") or a["reply_must_not_include"],
                "difficulty": h.get("difficulty", a["difficulty"]),
                "labeler": "human_adjudicated",
                "provisional": False,
                "notes": h.get("notes", ""),
            }
        else:
            resolved_by_majority = (not agree) and maj_intent is not None
            # Where the majority sides against A, take the majority; the reason code and
            # must/must-not fields come from whichever pass supplied the winning escalate
            # decision, so the row stays internally consistent.
            src = a
            if maj_escalate is not None and bool(a["escalate"]) != maj_escalate:
                src = b if bool(b.get("escalate")) == maj_escalate else (c or a)
            esc_final = maj_escalate if maj_escalate is not None else bool(a["escalate"])
            rec = {
                "intent": maj_intent or a["intent"],
                "escalate": esc_final,
                "escalate_reason_code": normalise_reason(
                    src.get("escalate_reason_code", ""), esc_final,
                    [a.get("escalate_reason_code", ""), b.get("escalate_reason_code", ""),
                     (c or {}).get("escalate_reason_code", "")]),
                "reply_must_include": a["reply_must_include"],
                "reply_must_not_include": a["reply_must_not_include"],
                "difficulty": a["difficulty"],
                "labeler": ("machine_agreed" if agree else
                            "model_majority_3pass" if resolved_by_majority else
                            "machine_unresolved"),
                "provisional": True,       # still true: no human has seen this row
                "notes": a.get("notes", ""),
            }
        s = sample.get(a["id"], {})
        rec.update({
            "id": a["id"], "text": a["text"], "thread_context": a["thread_context"],
            "hard_flags": s.get("hard_flags", []), "is_hard": s.get("is_hard", False),
            "machine_A": {"intent": a["intent"], "escalate": bool(a["escalate"])},
            "machine_B": {"intent": b.get("intent"), "escalate": bool(b.get("escalate", False))},
            "machine_C": ({"intent": c["intent"], "escalate": bool(c["escalate"])} if c else None),
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
