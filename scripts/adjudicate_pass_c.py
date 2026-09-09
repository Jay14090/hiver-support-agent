"""Pass C: a third, blind labelling pass, used to break A/B ties by majority vote.

WHY THIS EXISTS AND WHAT IT IS NOT
----------------------------------
The brief requires a HUMAN adjudication pass over the contested subset. That has not
happened. This is **not a substitute for it** and is never labelled as one: rows resolved
here are marked `model_majority_3pass`, `provisional` stays True, and the submission gate
stays RED on the human item.

What it does buy: currently 76 of 220 rows are `machine_unresolved` -- A and B disagreed
and Pass A's answer was silently used as the tiebreak, which is arbitrary. A third *blind*
pass lets those be settled by 2-of-3 majority instead of by which pass happened to run
first. That is a real methodological improvement over an arbitrary default.

Independence, so the majority vote means something:
  - a THIRD model (gpt-4o -- stronger than both A and B, appropriate for a tiebreak)
  - a THIRD framing (QA-reviewer, rubric-first, decides intent and routing independently)
  - blind: Pass C never sees A's or B's answers
"""
from __future__ import annotations

import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
from src.support_agent import config, llm  # noqa: E402
from src.support_agent.codebook import load_codebook_text  # noqa: E402

GOLDEN = config.GOLDEN
MODEL_C = "gpt-4o"
WORKERS = 8
_lock = threading.Lock()

PASS_C_SYSTEM = """You are a quality-assurance reviewer auditing support-ticket labels.
Work through the message against the rubric below and answer independently. Do not guess
what another reviewer would say -- apply the rubric.

STEP 1 - What is the customer's underlying request?
%s

STEP 2 - Could an automated reply handle this, or must a human?
A human is REQUIRED when any of these hold:
- a specific charge is disputed (double charge, unauthorised, wrong amount)
- money back is requested
- the account may be compromised (hacked, unexpected email change, unauthorised access)
- there is a legal, privacy or press demand
- the customer is cancelling, or threatens to cancel
- the customer is abusive TOWARD A PERSON, or in genuine distress
  (note: swearing about the product is ordinary frustration, NOT distress)
- answering requires private account data
- the customer has already tried the standard fix and it failed
Otherwise an automated reply is acceptable.

Return JSON only:
{"intent": "<intent>", "escalate": true|false,
 "escalate_reason_code": "<code>", "difficulty": "easy|medium|hard",
 "confidence": <0.0-1.0>, "notes": "<under 20 words>"}"""


def _fmt(row: dict) -> str:
    ctx = row.get("thread_context") or []
    s = ""
    if ctx:
        s += "EARLIER TURNS FROM THIS CUSTOMER:\n" + "\n".join(f"- {c[:200]}" for c in ctx[:3]) + "\n\n"
    return s + f'MESSAGE:\n"{row["text"]}"'


def main() -> int:
    rows = [json.loads(l) for l in (GOLDEN / "sample_220.jsonl").open(encoding="utf-8")]
    system = PASS_C_SYSTEM % load_codebook_text()
    print(f"[pass C] blind third pass over {len(rows)} examples ({MODEL_C})")

    done = [0]

    def work(row):
        parsed, _ = llm.complete_json(_fmt(row), system=system, model=MODEL_C,
                                      max_tokens=220, tag="prelabel_C")
        with _lock:
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  ... {done[0]}/{len(rows)}  spend=${llm.spend_so_far():.3f}")
        p = parsed or {}
        return {
            "id": str(row["id"]), "text": row["text"],
            "thread_context": row.get("thread_context") or [],
            "intent": p.get("intent") if p.get("intent") in config.INTENTS else "other",
            "escalate": bool(p.get("escalate", False)),
            "escalate_reason_code": p.get("escalate_reason_code", ""),
            "difficulty": p.get("difficulty", "medium"),
            "notes": str(p.get("notes", ""))[:160],
            "labeler": f"machine_pass_C:{MODEL_C}",
            "parse_failed": parsed is None,
        }

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        out = list(ex.map(work, rows))
    llm.flush_ledger()

    p = GOLDEN / "prelabel_C.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    esc = sum(1 for o in out if o["escalate"])
    fails = sum(1 for o in out if o["parse_failed"])
    print(f"[pass C] wrote {p} (failures {fails}, escalate {esc}/{len(out)} = {esc/len(out):.1%})")

    # ---- three-way agreement summary
    A = {json.loads(l)["id"]: json.loads(l) for l in (GOLDEN / "prelabel_A.jsonl").open(encoding="utf-8")}
    B = {json.loads(l)["id"]: json.loads(l) for l in (GOLDEN / "prelabel_B.jsonl").open(encoding="utf-8")}
    C = {o["id"]: o for o in out}

    unanimous_i = sum(1 for i in A if A[i]["intent"] == B[i]["intent"] == C[i]["intent"])
    majority_i = sum(1 for i in A if Counter([A[i]["intent"], B[i]["intent"], C[i]["intent"]]).most_common(1)[0][1] >= 2)
    none_i = len(A) - majority_i
    unanimous_e = sum(1 for i in A if bool(A[i]["escalate"]) == bool(B[i]["escalate"]) == bool(C[i]["escalate"]))

    from eval.metrics import cohens_kappa

    ids = list(A)
    stats = {
        "n": len(ids), "model_C": MODEL_C,
        "escalate_rate": {"A": round(sum(A[i]['escalate'] for i in ids) / len(ids), 4),
                          "B": round(sum(B[i]['escalate'] for i in ids) / len(ids), 4),
                          "C": round(sum(C[i]['escalate'] for i in ids) / len(ids), 4)},
        "pairwise_kappa_intent": {
            "A_vs_B": round(cohens_kappa([A[i]["intent"] for i in ids], [B[i]["intent"] for i in ids]), 4),
            "A_vs_C": round(cohens_kappa([A[i]["intent"] for i in ids], [C[i]["intent"] for i in ids]), 4),
            "B_vs_C": round(cohens_kappa([B[i]["intent"] for i in ids], [C[i]["intent"] for i in ids]), 4)},
        "pairwise_kappa_escalate": {
            "A_vs_B": round(cohens_kappa([str(A[i]["escalate"]) for i in ids], [str(B[i]["escalate"]) for i in ids]), 4),
            "A_vs_C": round(cohens_kappa([str(A[i]["escalate"]) for i in ids], [str(C[i]["escalate"]) for i in ids]), 4),
            "B_vs_C": round(cohens_kappa([str(B[i]["escalate"]) for i in ids], [str(C[i]["escalate"]) for i in ids]), 4)},
        "intent_unanimous": unanimous_i,
        "intent_majority_available": majority_i,
        "intent_three_way_split": none_i,
        "escalate_unanimous": unanimous_e,
    }
    (GOLDEN / "abc_agreement.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
