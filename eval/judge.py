"""LLM-as-judge for reply quality.

Two design choices that matter more than the rubric wording:

1. **Each axis is scored in its OWN call.** Asking one call for four scores produces a
   halo: a fluent reply gets 4/5 on groundedness because it read nicely. Four calls cost
   4x but measure four things instead of one thing four times.
2. **The judge model differs from the generator** (`gpt-4.1-mini` vs `gpt-4o-mini`).
   Only one provider's key exists on this machine, so this is a weaker separation than a
   different family would be -- which is exactly why the self-preference probe in
   judge_validation.py is mandatory rather than optional, and why its measured bias is
   reported as a limitation instead of being assumed away.

The judge's own trustworthiness is measured, not assumed: see `judge_validation.py` for
human agreement (Spearman, Krippendorff alpha) and the three bias probes.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
from src.support_agent import config, llm  # noqa: E402

WORKERS = 8

RUBRICS = {
    "groundedness": """Score 1-5: is every factual or procedural claim in the reply
supported EITHER by the retrieved historical threads shown to you OR by uncontroversial
general knowledge about Spotify?

5 = every claim traceable to the retrieved history or obviously true of Spotify
4 = supported, with one minor unsupported embellishment
3 = broadly plausible but the specific steps are not evidenced by the history
2 = contains a specific claim the history does not support
1 = INVENTS A POLICY, timeline, entitlement or figure. Any hallucinated policy is a 1.""",

    "actionability": """Score 1-5: does the reply give the customer a concrete next step?

5 = one clear, specific, immediately doable step
4 = a clear step, slightly vague on where or how
3 = generic advice ("try restarting") with no specifics
2 = only asks a diagnostic question, offers nothing to do
1 = no next step at all; pure acknowledgement or deflection""",

    "brand_fit": """Score 1-5: does this read like a real SpotifyCares public reply?
Their house style: short (a tweet), warm, contractions, one step not a numbered list,
usually invites a reply ("Let us know how it goes"), signs off with agent initials, and
never uses corporate apology-speak.

5 = indistinguishable from a real SpotifyCares reply
4 = right register, slightly off in length or sign-off
3 = polite and serviceable but generic; could be any brand
2 = noticeably corporate, robotic, or too long for the medium
1 = wrong register entirely""",

    "safety": """Score 1-5: is this safe to post PUBLICLY from a brand account?

5 = no promises, no invented policy, no request for sensitive data, correctly hands off
    anything account-specific
4 = safe, but slightly over-commits in tone ("we'll get this sorted right away")
3 = vague commitment that could be read as a promise
2 = promises a refund, credit, timeline, or a specific outcome
1 = asks for card details / password / full account details in public, OR guarantees a
    financial outcome. Either is an automatic 1.""",
}

SYSTEM = """You are evaluating a drafted customer-support reply for Spotify's public
support account. Score ONE dimension only. Be strict: 5 means genuinely excellent, not
merely acceptable. Most competent-but-unremarkable replies are a 3.

%s

Return JSON only: {"score": <1-5>, "reason": "<under 20 words>"}"""

SEND_UNEDITED_SYSTEM = """You are an experienced customer-support agent at Spotify. You
are shown a customer message and a drafted public reply.

One question: would you send this reply AS-IS, with no edits?

Say false if you would change anything at all -- wording, safety, accuracy, or tone.
This is the bar for "this saved me work", not "this is a reasonable draft".

Return JSON only: {"send_unedited": true|false, "reason": "<under 20 words>"}"""

_lock = threading.Lock()
_STATS = {"calls": 0, "parse_failures": 0}


def _context_block(item: dict, golden_row: dict | None) -> str:
    s = f'CUSTOMER MESSAGE:\n"{item["text"]}"\n\n'
    if golden_row and golden_row.get("thread_context"):
        s += "EARLIER TURNS:\n" + "\n".join(f"- {c[:160]}" for c in golden_row["thread_context"][:2]) + "\n\n"
    if item.get("retrieved_context"):
        s += "RETRIEVED HISTORICAL THREADS THE DRAFTER WAS GIVEN:\n" + item["retrieved_context"] + "\n\n"
    s += f'DRAFTED REPLY:\n"{item["draft_reply"]}"'
    return s


def score_one(item: dict, axis: str, golden_row=None, model=None) -> dict:
    parsed, _ = llm.complete_json(
        _context_block(item, golden_row),
        system=SYSTEM % RUBRICS[axis],
        model=model or config.JUDGE_MODEL,
        max_tokens=90,
        tag=f"judge_{axis}",
    )
    with _lock:
        _STATS["calls"] += 1
        if parsed is None:
            _STATS["parse_failures"] += 1
    if parsed is None:
        return {"score": None, "reason": "parse failure"}
    try:
        s = int(round(float(parsed.get("score", 0))))
    except (TypeError, ValueError):
        return {"score": None, "reason": "unparseable score"}
    return {"score": max(1, min(5, s)), "reason": str(parsed.get("reason", ""))[:100]}


def send_unedited(item: dict, golden_row=None, model=None) -> dict:
    parsed, _ = llm.complete_json(
        _context_block(item, golden_row), system=SEND_UNEDITED_SYSTEM,
        model=model or config.JUDGE_MODEL, max_tokens=80, tag="judge_send_unedited",
    )
    with _lock:
        _STATS["calls"] += 1
        if parsed is None:
            _STATS["parse_failures"] += 1
    if parsed is None:
        return {"send_unedited": None, "reason": "parse failure"}
    return {"send_unedited": bool(parsed.get("send_unedited")),
            "reason": str(parsed.get("reason", ""))[:100]}


def judge_replies(per_example: list, golden: list, model=None) -> dict:
    """Score every reply on all four axes plus the binary. Returns aggregates + rows."""
    gmap = {g["id"]: g for g in golden}
    jobs = []
    for item in per_example:
        if not (item.get("draft_reply") or "").strip():
            continue
        for axis in config.JUDGE_AXES:
            jobs.append((item, axis))
        jobs.append((item, "_send"))

    out: dict = {}

    def work(job):
        item, axis = job
        g = gmap.get(item["id"])
        if axis == "_send":
            return item["id"], axis, send_unedited(item, g, model)
        return item["id"], axis, score_one(item, axis, g, model)

    print(f"[judge] {len(jobs)} calls over {len(per_example)} replies ({model or config.JUDGE_MODEL})")
    done = [0]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for _id, axis, res in ex.map(work, jobs):
            out.setdefault(_id, {})[axis] = res
            done[0] += 1
            if done[0] % 200 == 0:
                print(f"  ... {done[0]}/{len(jobs)}  spend=${llm.spend_so_far():.3f}")
    llm.flush_ledger()

    rows = []
    for item in per_example:
        r = out.get(item["id"])
        if not r:
            continue
        rows.append({
            "id": item["id"],
            "draft_reply": item["draft_reply"],
            **{a: r.get(a, {}).get("score") for a in config.JUDGE_AXES},
            **{f"{a}_reason": r.get(a, {}).get("reason", "") for a in config.JUDGE_AXES},
            "send_unedited": r.get("_send", {}).get("send_unedited"),
            "send_unedited_reason": r.get("_send", {}).get("reason", ""),
        })

    from eval import metrics as M

    agg = {}
    for a in config.JUDGE_AXES:
        vals = [x[a] for x in rows if x.get(a) is not None]
        ci = M.bootstrap_ci(vals, lambda xs: sum(xs) / len(xs) if xs else 0.0) if vals else None
        agg[a] = {"mean": round(sum(vals) / len(vals), 3) if vals else None, "n": len(vals), "ci": ci}
    sends = [x["send_unedited"] for x in rows if x["send_unedited"] is not None]
    agg["send_unedited"] = {
        "rate": round(sum(sends) / len(sends), 4) if sends else None,
        "n_yes": sum(1 for s in sends if s),
        "n": len(sends),
        "ci": M.bootstrap_ci([1.0 if s else 0.0 for s in sends],
                             lambda xs: sum(xs) / len(xs) if xs else 0.0) if sends else None,
    }
    agg["judge_model"] = model or config.JUDGE_MODEL
    agg["parse_failures"] = _STATS["parse_failures"]
    agg["calls"] = _STATS["calls"]

    (config.REPORTS / "judge_scores.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return {"aggregate": agg, "n_rows": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        item = {"id": "x", "text": "my downloaded songs vanished after the update",
                "draft_reply": "Hey there! Try logging out and back in, then re-download your "
                               "playlist in Settings > Storage. Let us know how it goes /AI"}
        for axis in config.JUDGE_AXES:
            print(f"{axis:<16}", json.dumps(score_one(item, axis)))
        print("send_unedited   ", json.dumps(send_unedited(item)))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
