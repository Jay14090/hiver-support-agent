"""B0 -- the trivial floor. Built BEFORE the agent, on purpose.

Three degenerate policies, because "beats a baseline" means nothing until you say which:

  intent  : always predict the majority class
  reply   : one canned string for every message
  routing : reported TWICE -- always-auto and always-escalate

Always-escalate is not a joke baseline. It is a genuinely strong safety policy: it has
ZERO missed escalations by construction, which is the metric a support org actually
loses sleep over. Its cost is that it deflects nothing. If the agent cannot beat
always-escalate on expected cost, that IS the finding and the report says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

sys.path.insert(0, ".")
from src.support_agent import config  # noqa: E402

CANNED_REPLY = ("Hey there! Sorry about the trouble. Try logging out and back in, and let "
                "us know how it goes /AI")


class B0Trivial:
    name = "b0_trivial"

    def __init__(self, majority_intent: str = "feature_request_or_complaint",
                 routing_policy: str = "always_escalate"):
        self.majority_intent = majority_intent
        self.routing_policy = routing_policy

    def fit(self, rows) -> "B0Trivial":
        """The only 'training' is counting the majority class on the CORPUS split."""
        c = Counter(r.get("intent", "") for r in rows if r.get("intent"))
        if c:
            self.majority_intent = c.most_common(1)[0][0]
        return self

    def handle(self, message: str, thread_context=None) -> dict:
        return {
            "intent": self.majority_intent,
            "intent_confidence": 1.0,     # deliberately overconfident -> awful ECE
            "draft_reply": CANNED_REPLY,
            "retrieved_ids": [],
            "grounded_in": [],
            "used_retrieval": False,
            "max_retrieval_sim": 0.0,
            "route": "escalate" if self.routing_policy == "always_escalate" else "auto",
            "route_reason_code": "low_model_confidence" if self.routing_policy == "always_escalate"
                                 else "informational_answer",
            "route_reason": f"B0 fixed policy: {self.routing_policy}",
            "route_confidence": 1.0,
            "route_layer": "baseline",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if not a.eval:
        ap.print_help()
        return 0

    import pandas as pd

    from eval.run_eval import evaluate_system, load_golden

    golden = load_golden()
    corpus = pd.read_json(config.DATA_PROCESSED / "corpus.jsonl", lines=True).to_dict("records")

    results = {}
    for policy in ("always_escalate", "always_auto"):
        sysm = B0Trivial(routing_policy=policy).fit(corpus)
        results[f"b0_{policy}"] = evaluate_system(sysm, golden, name=f"b0_{policy}", judge=False)
        r = results[f"b0_{policy}"]
        print(f"\n--- b0_{policy} ---")
        print(f"  majority intent : {sysm.majority_intent}")
        print(f"  intent macro-F1 : {r['intent']['macro_f1']['point']:.3f}")
        print(f"  routing cost/100: {r['routing']['expected_cost_per_100']:.1f}")
        print(f"  missed escalations: {r['routing']['false_auto_count']} of {r['routing']['false_auto_of']}")
        print(f"  deflection rate : {r['routing']['deflection_rate']:.1%}")

    out = config.REPORTS / "results_b0.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
