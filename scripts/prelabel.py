"""Machine pre-labelling for the golden set.

THREE STAGES, kept separate on purpose:

  stratify : intent-only pass over the WHOLE test split. Used only to build sampling
             strata -- sampling weights do not need to be ground truth. Cheap model.
  A        : full label schema over the 220 sampled examples.
  B        : full label schema again, with a DIFFERENT model, a DIFFERENT prompt framing
             (it reasons about escalation first, intent second), the option order
             shuffled, and no access whatsoever to Pass A.

A-vs-B agreement is then a genuine reliability estimate for the machine labels, because
B is not a rerun of A. Everything A and B disagree on, plus a random 20% of what they
agree on, goes into the HUMAN review queue -- and only after a human clears that queue
is `golden/v1.jsonl` allowed to describe itself as adjudicated.

Nothing here may ever be described as "hand-labelled". These are machine labels.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, ".")
from src.support_agent import config, llm  # noqa: E402
from src.support_agent.codebook import load_codebook_text  # noqa: E402

GOLDEN = config.GOLDEN
WORKERS = 8

# --------------------------------------------------------------------------- prompts
STRATIFY_SYSTEM = """Classify a customer message sent to Spotify's public support account
into exactly one intent. Return JSON only:
{"intent": "<intent>", "confidence": <0.0-1.0>}

INTENTS:
%s"""

PASS_A_SYSTEM = """You are labelling a support message for an evaluation set. Work through
it in this order: identify the intent, then decide whether it must go to a human.

%s

ESCALATION POLICY (escalate if ANY of these apply):
- billing_dispute: a specific charge is wrong (double charge, unauthorised, wrong amount)
- suspected_account_compromise: hacked, unauthorised access, unexpected email change
- refund_request: asks for money back
- legal_privacy_data_request: GDPR/data/legal/press
- explicit_cancellation_threat: says they will cancel or is cancelling
- abusive_or_distressed_language: abuse, threats, or genuine distress
- requires_account_specific_pii: cannot be answered without looking at their account
- troubleshooting_already_failed: thread context shows they already tried the standard fix
Otherwise it is auto-handleable.

Return JSON only:
{"intent": "<intent>",
 "escalate": true|false,
 "escalate_reason_code": "<one of the codes above, or known_troubleshooting_flow /
                          informational_answer / chitchat_or_praise when not escalating>",
 "reply_must_include": ["<2-3 things any acceptable reply MUST do, each under 8 words>"],
 "reply_must_not_include": ["<1-3 things a reply MUST NOT do, each under 8 words>"],
 "difficulty": "easy|medium|hard",
 "notes": "<under 20 words>"}

`reply_must_include` describes REQUIRED CONTENT, not wording -- e.g. "acknowledge the
double charge", "point to the refund process". `reply_must_not_include` captures traps --
e.g. "promise a specific refund amount", "ask for card details".
Mark `difficulty` hard if the message is sarcastic, negated, multi-intent, very short, or
depends on thread context."""

PASS_B_SYSTEM = """You are a senior support operations reviewer. For the customer message
below, decide FIRST whether a human agent must handle it, and only then what it is about.

Question 1 -- Does this need a human?
A human is required when: money is disputed, a refund is asked for, the account may be
compromised, there is a legal or privacy demand, the customer is cancelling or
threatening to, the customer is abusive or in real distress, answering needs private
account data, or the customer has already tried the standard fix and it failed.
Otherwise an automated reply is acceptable.

Question 2 -- What is the customer's underlying request?
%s

Return JSON only:
{"escalate": true|false,
 "escalate_reason_code": "<code>",
 "intent": "<intent>",
 "reply_must_include": ["..."],
 "reply_must_not_include": ["..."],
 "difficulty": "easy|medium|hard",
 "notes": "<under 20 words>"}"""

_lock = threading.Lock()


def _fmt_msg(row: dict) -> str:
    ctx = row.get("thread_context") or []
    s = ""
    if len(ctx):
        s += "EARLIER TURNS FROM THIS CUSTOMER:\n" + "\n".join(f"- {c[:200]}" for c in ctx[:3]) + "\n\n"
    return s + f'MESSAGE:\n"{row["text"]}"'


def _run(rows, system, model, tag, max_tokens=300, shuffle_intents=False):
    out = [None] * len(rows)
    done = [0]

    def work(i_row):
        i, row = i_row
        sysmsg = system
        if shuffle_intents:
            # Order-shuffle the intent list per item, so B cannot inherit A's positional bias.
            rnd = random.Random(config.RANDOM_STATE + i)
            order = config.INTENTS[:-1]
            rnd.shuffle(order)
            cb = load_codebook_text()
            lines = cb.split("\n")
            by_intent = {}
            cur = None
            for ln in lines:
                if ln.startswith("- "):
                    cur = ln[2:].split(":")[0]
                    by_intent[cur] = [ln]
                elif cur:
                    by_intent[cur].append(ln)
            reordered = "\n".join("\n".join(by_intent.get(k, [f"- {k}"])) for k in order + ["other"])
            sysmsg = system.replace(load_codebook_text(), reordered) if load_codebook_text() in system else system
            if sysmsg == system:
                sysmsg = system  # codebook not embedded verbatim; harmless
        parsed, raw = llm.complete_json(
            _fmt_msg(row), system=sysmsg, model=model, max_tokens=max_tokens, tag=tag
        )
        with _lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  ... {done[0]}/{len(rows)}  spend=${llm.spend_so_far():.3f}")
        return i, (parsed or {"_parse_failed": True})

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, res in ex.map(work, list(enumerate(rows))):
            out[i] = res
    llm.flush_ledger()
    return out


# --------------------------------------------------------------------------- stages
def stage_stratify() -> None:
    df = pd.read_json(config.DATA_PROCESSED / "test.jsonl", lines=True)
    rows = df.to_dict("records")
    print(f"[prelabel] stratify pass over {len(rows)} test messages ({config.LABELER_A_MODEL})")
    sysmsg = STRATIFY_SYSTEM % "\n".join(f"- {i}" for i in config.INTENTS)
    res = _run(rows, sysmsg, config.LABELER_A_MODEL, "prelabel_stratify", max_tokens=60)

    out, fails = [], 0
    for row, r in zip(rows, res):
        intent = r.get("intent")
        if intent not in config.INTENTS:
            fails += 1
            intent = "other"
        out.append({"id": str(row["id"]), "text": row["text"],
                    "thread_context": row.get("thread_context") or [],
                    "intent": intent, "confidence": r.get("confidence", 0.0)})
    p = GOLDEN / "test_intents.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    from collections import Counter

    c = Counter(o["intent"] for o in out)
    print(f"[prelabel] wrote {p}  (parse/invalid failures: {fails})")
    for k, v in c.most_common():
        print(f"   {k:<32} {v:>5}  ({v/len(out):5.1%})")


def stage_a_full() -> None:
    """Pass A over the ENTIRE test split.

    Originally the stratification pass was a cheap intent-only prompt and Pass A was a
    separate, richer prompt over the sampled 220. That was a real design bug: the two
    procedures disagreed enough that a sample stratified to hit a floor of 12 per intent
    ended up with 4-8 examples for three intents once Pass A relabelled it. Stratifying
    on the SAME procedure that produces the final labels removes the inconsistency, and
    costs one pass over 1,456 rows (~$0.40).
    """
    df = pd.read_json(config.DATA_PROCESSED / "test.jsonl", lines=True)
    rows = df.to_dict("records")
    print(f"[prelabel] FULL pass A over {len(rows)} test messages ({config.LABELER_A_MODEL})")
    system = PASS_A_SYSTEM % load_codebook_text()
    res = _run(rows, system, config.LABELER_A_MODEL, "prelabel_A", max_tokens=320)

    out, fails = [], 0
    for row, r in zip(rows, res):
        if r.get("_parse_failed") or r.get("intent") not in config.INTENTS:
            fails += 1
        out.append({
            "id": str(row["id"]), "text": row["text"],
            "thread_context": row.get("thread_context") or [],
            "intent": r.get("intent") if r.get("intent") in config.INTENTS else "other",
            "escalate": bool(r.get("escalate", False)),
            "escalate_reason_code": r.get("escalate_reason_code", ""),
            "reply_must_include": (r.get("reply_must_include") or [])[:3],
            "reply_must_not_include": (r.get("reply_must_not_include") or [])[:3],
            "difficulty": r.get("difficulty", "medium"),
            "notes": str(r.get("notes", ""))[:160],
            "labeler": f"machine_pass_A:{config.LABELER_A_MODEL}",
            "parse_failed": bool(r.get("_parse_failed")),
        })
    p = GOLDEN / "prelabel_A_full.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(o["intent"] for o in out)
    esc = sum(1 for o in out if o["escalate"])
    print(f"[prelabel] wrote {p} (failures {fails}, escalate {esc}/{len(out)} = {esc/len(out):.1%})")
    for k, v in c.most_common():
        print(f"   {k:<32} {v:>5}  ({v/len(out):5.1%})")


def _stage_full(stage: str) -> None:
    sample = [json.loads(l) for l in (GOLDEN / "sample_220.jsonl").open(encoding="utf-8")]
    cb = load_codebook_text()
    if stage == "A":
        # Project the full Pass-A run onto the sampled ids -- no new calls, and it
        # guarantees the golden labels are literally the ones used for stratification.
        full = {json.loads(l)["id"]: json.loads(l)
                for l in (GOLDEN / "prelabel_A_full.jsonl").open(encoding="utf-8")}
        out = [full[r["id"]] for r in sample]
        p = GOLDEN / "prelabel_A.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        esc = sum(1 for o in out if o["escalate"])
        print(f"[prelabel] wrote {p} from prelabel_A_full ({len(out)} rows, "
              f"escalate {esc}/{len(out)} = {esc/len(out):.1%})")
        return
    else:
        system = PASS_B_SYSTEM % cb
        model = config.LABELER_B_MODEL
        order = list(range(len(sample)))
        random.Random(config.RANDOM_STATE + 7).shuffle(order)
        rows = [sample[i] for i in order]


    print(f"[prelabel] pass {stage} over {len(rows)} examples ({model})")
    res = _run(rows, system, model, f"prelabel_{stage}", max_tokens=320, shuffle_intents=(stage == "B"))

    out = [None] * len(sample)
    fails = 0
    for pos, r in enumerate(res):
        orig_i = order[pos]
        row = sample[orig_i]
        if r.get("_parse_failed") or r.get("intent") not in config.INTENTS:
            fails += 1
        out[orig_i] = {
            "id": str(row["id"]),
            "text": row["text"],
            "thread_context": row.get("thread_context") or [],
            "intent": r.get("intent") if r.get("intent") in config.INTENTS else "other",
            "escalate": bool(r.get("escalate", False)),
            "escalate_reason_code": r.get("escalate_reason_code", ""),
            "reply_must_include": r.get("reply_must_include", [])[:3],
            "reply_must_not_include": r.get("reply_must_not_include", [])[:3],
            "difficulty": r.get("difficulty", "medium"),
            "notes": str(r.get("notes", ""))[:160],
            "labeler": f"machine_pass_{stage}:{model}",
            "parse_failed": bool(r.get("_parse_failed")),
        }
    p = GOLDEN / f"prelabel_{stage}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    esc = sum(1 for o in out if o["escalate"])
    print(f"[prelabel] wrote {p}  (failures: {fails}, escalate: {esc}/{len(out)} = {esc/len(out):.1%})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stratify", "A-full", "A", "B"], required=True)
    a = ap.parse_args()
    if a.stage == "stratify":
        stage_stratify()
    elif a.stage == "A-full":
        stage_a_full()
    else:
        _stage_full(a.stage)
    print(f"[prelabel] cumulative spend ${llm.spend_so_far():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
