"""Cross-model judge agreement: is the judge STABLE, at least?

This is NOT the human validation the brief requires and it is never presented as such.
Human agreement (Spearman rho / Krippendorff alpha against a person) remains unmeasured,
and `judge_validation.py --report` still returns NOT_DONE until `eval/human_judge_cli.py`
has been run by an actual human.

What this does measure: whether the judge's scores are an artefact of one particular
model. A second, stronger judge (gpt-4o) scores the same 70 replies on the same rubric,
and the two are compared with the same statistics that would be used against a human.

How to read it:
  - HIGH agreement => the rubric is well-posed enough that two models apply it similarly.
    It does NOT mean either agrees with a human; both could share the same blind spot.
  - LOW agreement  => the rubric is ambiguous, and every reply-quality number is shaky
    regardless of what a human would say.
"""
from __future__ import annotations

import json
import random
import sys

sys.path.insert(0, ".")
from eval import judge, metrics as M  # noqa: E402
from src.support_agent import config, llm  # noqa: E402

RATER_B = "gpt-4o"
N = config.HUMAN_JUDGE_N
OUT = config.REPORTS / "judge_cross_model.json"


def main() -> int:
    rows = [json.loads(l) for l in (config.REPORTS / "judge_scores.jsonl").open(encoding="utf-8")]
    golden = [json.loads(l) for l in (config.GOLDEN / "v1.jsonl").open(encoding="utf-8")]
    gmap = {g["id"]: g for g in golden}

    rng = random.Random(config.RANDOM_STATE)
    pool = [r for r in rows if (r.get("draft_reply") or "").strip()]
    rng.shuffle(pool)
    pool = pool[:N]

    items = [{"id": r["id"], "text": gmap.get(r["id"], {}).get("text", ""),
              "draft_reply": r["draft_reply"]} for r in pool]

    print(f"[cross] second judge {RATER_B} scoring {len(items)} replies")
    second = judge.judge_replies(items, golden, model=RATER_B, system_name="cross_gpt4o")
    scores_b = {json.loads(l)["id"]: json.loads(l)
                for l in (config.REPORTS / "judge_scores_cross_gpt4o.jsonl").open(encoding="utf-8")}
    scores_a = {r["id"]: r for r in pool}

    out = {"rater_a": config.JUDGE_MODEL, "rater_b": RATER_B, "n_requested": N,
           "is_human_validation": False,
           "note": ("Model-vs-model agreement. This is NOT the human validation the brief "
                    "requires; Spearman rho against a human remains unmeasured."),
           "axes": {}}
    for axis in config.JUDGE_AXES:
        pairs = [(scores_a[i][axis], scores_b[i][axis]) for i in scores_a
                 if i in scores_b and scores_a[i].get(axis) is not None
                 and scores_b[i].get(axis) is not None]
        if len(pairs) < 5:
            out["axes"][axis] = {"n": len(pairs), "note": "too few pairs"}
            continue
        a_s = [p[0] for p in pairs]
        b_s = [p[1] for p in pairs]
        out["axes"][axis] = {
            "n": len(pairs),
            "spearman_rho": round(M.spearman_rho(a_s, b_s), 4),
            "krippendorff_alpha": round(M.krippendorff_alpha_interval(pairs), 4),
            "exact_match_rate": round(sum(1 for x, y in pairs if x == y) / len(pairs), 4),
            "within_1_rate": round(sum(1 for x, y in pairs if abs(x - y) <= 1) / len(pairs), 4),
            "mean_a": round(sum(a_s) / len(a_s), 3),
            "mean_b": round(sum(b_s) / len(b_s), 3),
            "b_minus_a": round(sum(b_s) / len(b_s) - sum(a_s) / len(a_s), 3),
        }
    b = [(bool(scores_a[i]["send_unedited"]), bool(scores_b[i]["send_unedited"])) for i in scores_a
         if i in scores_b and scores_a[i].get("send_unedited") is not None
         and scores_b[i].get("send_unedited") is not None]
    if b:
        out["send_unedited"] = {
            "n": len(b),
            "raw_agreement": round(sum(1 for x, y in b if x == y) / len(b), 4),
            "cohens_kappa": round(M.cohens_kappa([str(x) for x, _ in b], [str(y) for _, y in b]), 4),
            "yes_rate_a": round(sum(1 for x, _ in b if x) / len(b), 4),
            "yes_rate_b": round(sum(1 for _, y in b if y) / len(b), 4),
        }
    rhos = [v["spearman_rho"] for v in out["axes"].values() if "spearman_rho" in v]
    out["overall_mean_rho"] = round(sum(rhos) / len(rhos), 4) if rhos else None

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}   spend=${llm.spend_so_far():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
