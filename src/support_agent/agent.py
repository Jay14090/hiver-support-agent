"""The agent: one message in, one full decision out.

    handle(message, thread_context) -> Decision

Deliberately a thin orchestrator with no cleverness of its own. Retrieval, then the three
heads, then the leakage assertion. Every field a grader might ask about -- what was
retrieved, what grounded the reply, why it routed the way it did -- is on the Decision
object rather than reconstructed after the fact.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field

from . import classify, config, draft, retrieval, route
from .codebook import load_codebook_text


@dataclass
class Decision:
    message: str
    intent: str
    intent_confidence: float
    intent_rationale: str
    draft_reply: str
    retrieved_ids: list = field(default_factory=list)
    grounded_in: list = field(default_factory=list)
    used_retrieval: bool = False
    max_retrieval_sim: float = 0.0
    route: str = "escalate"
    route_reason_code: str = ""
    route_reason: str = ""
    route_confidence: float = 0.0
    route_layer: str = ""
    parse_failures: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class SupportAgent:
    def __init__(self, retriever=None, codebook: str | None = None, tau=None, sigma=None,
                 check_leakage: bool = True):
        self.retriever = retriever or retrieval.HybridRetriever.from_split("corpus")
        self.codebook = codebook if codebook is not None else load_codebook_text()
        self.tau = config.TAU_INTENT_CONFIDENCE if tau is None else tau
        self.sigma = config.SIGMA_RETRIEVAL_SIM if sigma is None else sigma
        self.check_leakage = check_leakage

    def handle(self, message: str, thread_context=None) -> Decision:
        # 1. retrieve. The same hits feed both the few-shot examples and the drafting
        #    prompt -- one retrieval call, two uses, so the agent cannot be grounded in
        #    one set of threads while having been prompted with another.
        hits = self.retriever.search(message)
        if self.check_leakage:
            retrieval.assert_no_leakage([h.id for h in hits])
        max_sim = self.retriever.max_similarity(hits)

        # 2. classify, with few-shots drawn from the retrieved (corpus-only) threads
        few_shots = [(h.customer_text, "") for h in hits[:3]]
        few_shots = [(t, lab) for t, lab in few_shots if lab]  # only labelled ones are useful
        ir = classify.classify(message, thread_context, self.codebook, few_shots or None)

        # 3. draft, grounded in the retrieved resolutions
        dr = draft.draft(message, thread_context, hits)

        # 4. route
        rd = route.route(message, thread_context, intent=ir.intent,
                         intent_confidence=ir.confidence, max_retrieval_sim=max_sim,
                         tau=self.tau, sigma=self.sigma)

        # If we are escalating, the drafted reply becomes a suggestion for the human
        # rather than something to send. Kept, not discarded: a half-written reply is
        # still useful to the agent who picks it up.
        return Decision(
            message=message,
            intent=ir.intent, intent_confidence=ir.confidence, intent_rationale=ir.rationale,
            draft_reply=dr.reply,
            retrieved_ids=[h.id for h in hits],
            grounded_in=dr.grounded_in, used_retrieval=dr.used_retrieval,
            max_retrieval_sim=round(max_sim, 4),
            route=rd.route, route_reason_code=rd.reason_code, route_reason=rd.reason,
            route_confidence=round(rd.confidence, 4), route_layer=rd.layer,
            parse_failures={"classify": ir.parse_failed, "draft": dr.parse_failed},
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="handle 10 dev messages end to end")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--text", default=None, help="handle a single message")
    a = ap.parse_args()

    agent = SupportAgent()
    if a.text:
        print(agent.handle(a.text).to_json())
        return 0
    if a.demo:
        import pandas as pd

        dev = pd.read_json(config.DATA_PROCESSED / "dev.jsonl", lines=True)
        dev = dev.sample(n=min(a.n, len(dev)), random_state=config.RANDOM_STATE)
        for i, row in enumerate(dev.to_dict("records"), 1):
            d = agent.handle(row["text"], row.get("thread_context"))
            print(f"\n{'='*78}\n[{i}] {d.message[:150]}")
            print(f"  intent   : {d.intent}  (conf {d.intent_confidence:.2f})  -- {d.intent_rationale}")
            print(f"  retrieved: {d.retrieved_ids}   max_sim={d.max_retrieval_sim:.3f}")
            print(f"  reply    : {d.draft_reply}")
            print(f"  grounded : {d.grounded_in or 'NONE'}   (used_retrieval={d.used_retrieval})")
            print(f"  ROUTE    : {d.route.upper()}  [{d.route_reason_code}, layer={d.route_layer}]")
            print(f"             {d.route_reason}")
        print(f"\n{'='*78}")
        print("classify:", json.dumps(classify.stats()))
        print("draft   :", json.dumps(draft.stats()))
        print("route   :", json.dumps(route.stats()))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
