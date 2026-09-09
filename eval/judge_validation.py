"""Is the judge trustworthy? Measured, not assumed.

Two halves:

  --report  agreement between the judge and a HUMAN on the same rubric, over 60-80
            replies: Spearman rho, Krippendorff's alpha, exact-match rate, within-+/-1
            rate. All four, because they fail differently: rho is blind to a constant
            offset, exact-match is brutal on a 5-point scale, and alpha is the only
            chance-corrected one.

  --probes  three bias experiments, each with a number attached:
              position     : swap A/B order on 40 pairs, report the flip rate
              length       : correlate score with reply token count, report rho
              self-pref    : judge scores its OWN model's output vs another model's on
                             identical inputs, report the delta

If agreement is mediocre (rho < 0.6) that is REPORTED PLAINLY along with what it does to
every judge-based conclusion. A weak judge honestly reported scores better than a strong
one that was never checked.
"""
from __future__ import annotations

import argparse
import json
import random
import sys

sys.path.insert(0, ".")
from eval import judge, metrics as M  # noqa: E402
from src.support_agent import config, llm  # noqa: E402

HUMAN_SCORES = config.REPORTS / "human_judge_scores.jsonl"
OUT = config.REPORTS / "judge_validation.json"


# --------------------------------------------------------------------------- agreement
def report_agreement() -> dict:
    """Compare judge scores against human scores on the same replies and axes."""
    jpath = config.REPORTS / "judge_scores.jsonl"
    if not jpath.exists():
        return {"error": "no judge_scores.jsonl -- run the eval with --judge first"}
    machine = {json.loads(l)["id"]: json.loads(l) for l in jpath.open(encoding="utf-8")}

    if not HUMAN_SCORES.exists():
        return {
            "status": "NOT_DONE",
            "human_scored": 0,
            "required": config.HUMAN_JUDGE_N,
            "message": (
                "No human judge scores yet. Run `python eval/human_judge_cli.py` to score "
                f"{config.HUMAN_JUDGE_N} replies. Until then the judge is UNVALIDATED and "
                "every reply-quality number must be reported as such."
            ),
        }

    human = [json.loads(l) for l in HUMAN_SCORES.open(encoding="utf-8")]
    out = {"n_human_scored": len(human), "axes": {}}
    for axis in config.JUDGE_AXES:
        pairs = [(h[axis], machine[h["id"]][axis]) for h in human
                 if h.get(axis) is not None and h["id"] in machine
                 and machine[h["id"]].get(axis) is not None]
        if len(pairs) < 5:
            out["axes"][axis] = {"n": len(pairs), "note": "too few pairs to report"}
            continue
        hs = [p[0] for p in pairs]
        ms = [p[1] for p in pairs]
        rho = M.spearman_rho(hs, ms)
        out["axes"][axis] = {
            "n": len(pairs),
            "spearman_rho": round(rho, 4),
            "krippendorff_alpha": round(M.krippendorff_alpha_interval(pairs), 4),
            "exact_match_rate": round(sum(1 for a, b in pairs if a == b) / len(pairs), 4),
            "within_1_rate": round(sum(1 for a, b in pairs if abs(a - b) <= 1) / len(pairs), 4),
            "human_mean": round(sum(hs) / len(hs), 3),
            "judge_mean": round(sum(ms) / len(ms), 3),
            "judge_minus_human": round(sum(ms) / len(ms) - sum(hs) / len(hs), 3),
            "verdict": _verdict(rho),
        }
    # the binary
    b = [(bool(h["send_unedited"]), bool(machine[h["id"]]["send_unedited"])) for h in human
         if h.get("send_unedited") is not None and h["id"] in machine
         and machine[h["id"]].get("send_unedited") is not None]
    if b:
        out["send_unedited"] = {
            "n": len(b),
            "raw_agreement": round(sum(1 for x, y in b if x == y) / len(b), 4),
            "cohens_kappa": round(M.cohens_kappa([str(x) for x, _ in b], [str(y) for _, y in b]), 4),
            "human_yes_rate": round(sum(1 for x, _ in b if x) / len(b), 4),
            "judge_yes_rate": round(sum(1 for _, y in b if y) / len(b), 4),
        }
    rhos = [v["spearman_rho"] for v in out["axes"].values() if "spearman_rho" in v]
    out["overall_mean_rho"] = round(sum(rhos) / len(rhos), 4) if rhos else None
    out["overall_verdict"] = _verdict(out["overall_mean_rho"]) if rhos else "unmeasured"
    return out


def _verdict(rho) -> str:
    if rho is None:
        return "unmeasured"
    if rho >= 0.7:
        return "usable: judge tracks the human well"
    if rho >= 0.6:
        return "acceptable: judge-based deltas are suggestive, not conclusive"
    return ("WEAK: below 0.6. Judge-based reply-quality differences are weak evidence and "
            "must not be reported as findings without human confirmation.")


# --------------------------------------------------------------------------- probes
def probe_position(n: int = 40) -> dict:
    """Pairwise A/B, then the same pair with the order swapped. Flip rate = position bias.

    A perfectly consistent judge flips 0% of the time; 50% would be a coin toss.
    """
    rows = _load_judge_rows()
    if len(rows) < 4:
        return {"error": "not enough judged replies"}
    rng = random.Random(config.RANDOM_STATE)
    pairs = []
    for _ in range(n):
        a, b = rng.sample(rows, 2)
        if a["draft_reply"].strip() and b["draft_reply"].strip():
            pairs.append((a, b))

    SYSTEM = """You compare two drafted support replies to the SAME customer message.
Which is better overall (grounded, actionable, on-brand, safe)?
Return JSON only: {"winner": "A"|"B"}"""

    flips, valid = 0, 0
    for a, b in pairs:
        p1, _ = llm.complete_json(
            f'CUSTOMER MESSAGE:\n"{a.get("text","(see replies)")}"\n\nREPLY A:\n"{a["draft_reply"]}"\n\nREPLY B:\n"{b["draft_reply"]}"',
            system=SYSTEM, model=config.JUDGE_MODEL, max_tokens=40, tag="probe_position")
        p2, _ = llm.complete_json(
            f'CUSTOMER MESSAGE:\n"{a.get("text","(see replies)")}"\n\nREPLY A:\n"{b["draft_reply"]}"\n\nREPLY B:\n"{a["draft_reply"]}"',
            system=SYSTEM, model=config.JUDGE_MODEL, max_tokens=40, tag="probe_position")
        if not p1 or not p2:
            continue
        valid += 1
        w1, w2 = p1.get("winner"), p2.get("winner")
        # Consistent = the SAME underlying reply wins both times, i.e. the letter flips.
        if not ((w1 == "A" and w2 == "B") or (w1 == "B" and w2 == "A")):
            flips += 1
    return {
        "n_pairs": valid,
        "inconsistent_pairs": flips,
        "flip_rate": round(flips / valid, 4) if valid else None,
        "interpretation": ("flip_rate is the share of pairs where swapping the order changed "
                           "the winner. 0.0 = no position bias, 0.5 = coin toss."),
    }


def probe_length() -> dict:
    """Does the judge simply reward longer replies?"""
    rows = _load_judge_rows()
    out = {}
    for axis in config.JUDGE_AXES:
        pairs = [(len(r["draft_reply"].split()), r[axis]) for r in rows if r.get(axis) is not None]
        if len(pairs) < 10:
            out[axis] = {"n": len(pairs), "note": "too few"}
            continue
        rho = M.spearman_rho([p[0] for p in pairs], [p[1] for p in pairs])
        out[axis] = {"n": len(pairs), "spearman_rho_length_vs_score": round(rho, 4),
                     "flag": abs(rho) > 0.4}
    out["interpretation"] = ("rho between reply length in tokens and the axis score. "
                             "|rho| > 0.4 suggests the judge is partly measuring length.")
    return out


def probe_self_preference(n: int = 30) -> dict:
    """Does the judge favour replies written by its own model?

    Generate the same replies with the judge's model and with the generator model, then
    score BOTH with the judge. A positive delta for its own outputs is self-preference.
    This probe is mandatory here because generator and judge share a provider.
    """
    import pandas as pd

    from src.support_agent import draft as D
    from src.support_agent.retrieval import HybridRetriever

    dev = pd.read_json(config.DATA_PROCESSED / "dev.jsonl", lines=True)
    dev = dev.sample(n=min(n, len(dev)), random_state=config.RANDOM_STATE).to_dict("records")
    r = HybridRetriever.from_split("corpus")

    own, other = [], []
    for row in dev:
        hits = r.search(row["text"])
        d_gen = D.draft(row["text"], row.get("thread_context"), hits, model=config.GENERATOR_MODEL)
        d_judge = D.draft(row["text"], row.get("thread_context"), hits, model=config.JUDGE_MODEL)
        for lst, dr in ((other, d_gen), (own, d_judge)):
            item = {"id": f"{row['id']}_{dr.reply[:8]}", "text": row["text"], "draft_reply": dr.reply}
            scores = [judge.score_one(item, ax)["score"] for ax in config.JUDGE_AXES]
            scores = [s for s in scores if s is not None]
            if scores:
                lst.append(sum(scores) / len(scores))

    mo = sum(own) / len(own) if own else None
    mt = sum(other) / len(other) if other else None
    return {
        "n": len(own),
        "judge_model": config.JUDGE_MODEL,
        "generator_model": config.GENERATOR_MODEL,
        "mean_score_on_own_model_output": round(mo, 3) if mo else None,
        "mean_score_on_other_model_output": round(mt, 3) if mt else None,
        "self_preference_delta": round(mo - mt, 3) if (mo and mt) else None,
        "interpretation": ("positive delta = the judge scores its own model's replies higher on "
                           "identical inputs. Both models share a provider, so this is a WEAKER "
                           "separation than a different family would give."),
    }


def _load_judge_rows() -> list:
    p = config.REPORTS / "judge_scores.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.open(encoding="utf-8")]
    # attach the customer message where we can, for the pairwise probe
    gp = config.GOLDEN / "v1.jsonl"
    if gp.exists():
        gmap = {json.loads(l)["id"]: json.loads(l)["text"] for l in gp.open(encoding="utf-8")}
        for r in rows:
            r["text"] = gmap.get(r["id"], "")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="judge vs human agreement")
    ap.add_argument("--probes", action="store_true", help="position / length / self-preference")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    if a.offline:
        llm.set_offline(True)

    existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if a.report:
        existing["human_agreement"] = report_agreement()
        print(json.dumps(existing["human_agreement"], indent=2))
    if a.probes:
        print("[probes] position ...")
        existing.setdefault("bias_probes", {})["position"] = probe_position()
        print("[probes] length ...")
        existing["bias_probes"]["length"] = probe_length()
        print("[probes] self-preference ...")
        existing["bias_probes"]["self_preference"] = probe_self_preference()
        print(json.dumps(existing["bias_probes"], indent=2))
    if not (a.report or a.probes):
        ap.print_help()
        return 0
    OUT.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
