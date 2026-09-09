"""Phase 8: measure the targeted fix properly, with a PAIRED bootstrap.

Comparing two independent 95% CIs is the wrong test here. Both systems are evaluated on
the *same* 220 examples, so the correct question is "resampling examples, how often does
the fix help?" -- a paired bootstrap on the per-example cost difference. That is strictly
more powerful than checking whether two marginal CIs overlap, and it is the difference
between "cannot show an effect" and "there is no effect".

Both numbers are reported: the marginal CIs (which overlap) and the paired CI.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, ".")
from eval.metrics import expected_cost_per_100, routing_metrics  # noqa: E402
from eval.run_eval import evaluate_system, load_golden  # noqa: E402
from src.support_agent import config  # noqa: E402
from src.support_agent.agent import SupportAgent  # noqa: E402


class AgentNoFix(SupportAgent):
    """The agent as it was BEFORE the Phase 8 fix: no intent-corroborated escalation."""

    def handle(self, message, thread_context=None):
        from src.support_agent import classify, draft, retrieval, route

        hits = self.retriever.search(message)
        if self.check_leakage:
            retrieval.assert_no_leakage([h.id for h in hits])
        max_sim = self.retriever.max_similarity(hits)
        ir = classify.classify(message, thread_context, self.codebook)
        dr = draft.draft(message, thread_context, hits)
        rd = route.route(message, thread_context, intent=ir.intent,
                         intent_confidence=ir.confidence, max_retrieval_sim=max_sim,
                         tau=self.tau, sigma=self.sigma,
                         intent_corroboration=False)     # <- the only difference
        from src.support_agent.agent import Decision

        return Decision(
            message=message, intent=ir.intent, intent_confidence=ir.confidence,
            intent_rationale=ir.rationale, draft_reply=dr.reply,
            retrieved_ids=[h.id for h in hits], grounded_in=dr.grounded_in,
            used_retrieval=dr.used_retrieval, max_retrieval_sim=round(max_sim, 4),
            route=rd.route, route_reason_code=rd.reason_code, route_reason=rd.reason,
            route_confidence=round(rd.confidence, 4), route_layer=rd.layer,
        )


def per_example_cost(pe) -> np.ndarray:
    """Cost contributed by each example, so the bootstrap can resample examples."""
    out = []
    for x in pe:
        if x["gold_escalate"] and not x["pred_escalate"]:
            out.append(config.COST_FALSE_AUTO)
        elif (not x["gold_escalate"]) and x["pred_escalate"]:
            out.append(config.COST_FALSE_ESCALATE)
        else:
            out.append(0.0)
    return np.array(out, dtype=float)


def main() -> int:
    golden = load_golden()
    print("[p8] evaluating agent WITHOUT the fix ...")
    before = evaluate_system(AgentNoFix(), golden, "agent_before_p8")
    print("[p8] evaluating agent WITH the fix ...")
    after = evaluate_system(SupportAgent(), golden, "agent_after_p8")

    pb, pa = before.pop("_per_example"), after.pop("_per_example")
    ids_b = [x["id"] for x in pb]
    ids_a = [x["id"] for x in pa]
    assert ids_b == ids_a, "examples must align for a paired test"

    cb, ca = per_example_cost(pb), per_example_cost(pa)
    diff = ca - cb                      # negative = the fix saved cost

    rng = np.random.default_rng(config.RANDOM_STATE)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() * 100 for _ in range(config.BOOTSTRAP_N)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    point = diff.mean() * 100
    n_better = int((boots < 0).mean() * 1000) / 10.0

    rb = routing_metrics([x["gold_escalate"] for x in pb], [x["pred_escalate"] for x in pb])
    ra = routing_metrics([x["gold_escalate"] for x in pa], [x["pred_escalate"] for x in pa])

    changed = [(pb[i]["id"], pb[i]["text"][:90], pb[i]["pred_escalate"], pa[i]["pred_escalate"],
                pa[i]["route_reason_code"], pb[i]["gold_escalate"])
               for i in range(n) if pb[i]["pred_escalate"] != pa[i]["pred_escalate"]]

    meaningful = hi < 0
    res = {
        "n": n,
        "before": {"cost_per_100": rb["expected_cost_per_100"], "missed_escalations": rb["false_auto"],
                   "needless_escalations": rb["false_escalate"], "deflection_rate": rb["deflection_rate"],
                   "escalate_f1": rb["f1"]},
        "after": {"cost_per_100": ra["expected_cost_per_100"], "missed_escalations": ra["false_auto"],
                  "needless_escalations": ra["false_escalate"], "deflection_rate": ra["deflection_rate"],
                  "escalate_f1": ra["f1"]},
        "paired_delta_cost_per_100": {
            "point": round(point, 3), "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
            "share_of_resamples_where_fix_helps": n_better,
            "clears_noise_floor": bool(meaningful),
        },
        "decisions_changed": len(changed),
        "changed_examples": [
            {"id": i, "text": t, "before_escalate": b, "after_escalate": a2,
             "reason_code": rc, "gold_escalate": g, "correct_change": (a2 == g)}
            for i, t, b, a2, rc, g in changed],
    }
    (config.REPORTS / "p8_delta.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "changed_examples"}, indent=2))
    print("\nchanged decisions:")
    for c in res["changed_examples"]:
        print(f"  id={c['id']} {'CORRECT' if c['correct_change'] else 'WRONG  '} "
              f"{c['before_escalate']}->{c['after_escalate']} ({c['reason_code']}) | {c['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
