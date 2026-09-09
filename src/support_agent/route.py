"""Auto-handle vs escalate, with a stated reason. Deliberately hybrid, in three layers.

  1. DETERMINISTIC RULES first. Abuse, legal/privacy, explicit refund demands and
     suspected account compromise are safety-critical. They must not depend on whether a
     model felt agreeable this morning, and I must be able to point at the exact line
     that fired. A model that is right 97% of the time is not good enough for the class
     of case where being wrong is a headline.
  2. THE MODEL for the judgement calls -- "has this person already tried the standard
     fix", "is this frustration or genuine distress". Rules are bad at these.
  3. CONFIDENCE GATES last. Escalate if intent confidence < tau or max retrieval
     similarity < sigma. This is the humility layer: not knowing is itself a reason to
     involve a human. tau and sigma are tuned ON DEV ONLY.

Every decision carries an enum reason code AND a human sentence, because the brief
requires a stated reason and an enum alone is not one.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass

from . import config, llm

# --------------------------------------------------------------------------- layer 1
# Ordered: the first match wins, most severe first. Kept narrow and high-precision --
# a rule that fires on "this is fraud, I'm so mad at this playlist" would be worse than
# no rule at all, so each pattern is anchored to concrete, checkable language.
DETERMINISTIC_RULES = [
    ("legal_privacy_data_request",
     r"\b(gdpr|data protection|subject access request|right to be forgotten|"
     r"my lawyer|legal action|sue you|solicitor|attorney|small claims|"
     r"ombudsman|trading standards|press enquiry)\b"),
    ("suspected_account_compromise",
     r"\b(hacked|hacker|compromised|someone (else )?(is |has )?(using|accessing|on) my account|"
     r"unauthorized (access|login)|unauthorised (access|login)|"
     r"changed my (email|password) without|didn'?t (change|authorize) )\b"),
    ("billing_dispute",
     r"\b(charged (me )?(twice|two times|3 times|three times)|double[- ]charg|"
     r"unauthorized charge|unauthorised charge|chargeback|dispute (the|this) charge|"
     r"took money (from|out of) my (account|bank)|charged after (i )?cancel)\b"),
    ("refund_request",
     r"\b(refund me|want (a|my) refund|need (a|my) refund|give me my money back|"
     r"money back|reimburse)\b"),
    ("abusive_or_distressed_language",
     r"\b(kill myself|suicid|self harm|i want to die)\b"),
]
_RULES = [(code, re.compile(pat, re.I)) for code, pat in DETERMINISTIC_RULES]

HUMAN_SENTENCE = {
    "billing_dispute": "A specific charge is disputed, so a human needs to see the account and the payment record.",
    "suspected_account_compromise": "Possible account compromise - security cases go straight to a human.",
    "refund_request": "The customer is asking for money back; only a human can authorise that.",
    "legal_privacy_data_request": "Legal or data-protection request, which must be handled by a person.",
    "explicit_cancellation_threat": "The customer is cancelling or threatening to, so retention should handle it.",
    "abusive_or_distressed_language": "The message contains abusive or distressed language and needs a human.",
    "requires_account_specific_pii": "Answering needs private account details that cannot be discussed publicly.",
    "troubleshooting_already_failed": "They have already tried the standard fix, so the scripted reply would waste their time.",
    "low_model_confidence": "The classifier was not confident enough about the intent to answer automatically.",
    "no_retrieval_support": "No similar resolved thread was found, so any reply would be ungrounded.",
    "known_troubleshooting_flow": "This matches a known troubleshooting flow with good historical support.",
    "informational_answer": "This is a straightforward informational answer supported by past replies.",
    "chitchat_or_praise": "Praise or chit-chat, which needs no human attention.",
}

ROUTER_SYSTEM = """You decide whether a customer support message must be handled by a
HUMAN or can be answered AUTOMATICALLY. Deterministic safety rules have already run and
did not fire, so judge the remaining, subtler cases.

Escalate to a human when:
- the customer has ALREADY tried the standard fix and it failed (check the thread context)
- answering would require looking at their private account data
- they are explicitly cancelling or threatening to cancel
- they are abusive, or genuinely distressed rather than merely annoyed

Otherwise auto-handle.

Return JSON only:
{"route": "auto"|"escalate", "reason_code": "<code>", "confidence": <0.0-1.0>}

reason_code when escalating: requires_account_specific_pii | troubleshooting_already_failed |
explicit_cancellation_threat | abusive_or_distressed_language
reason_code when auto: known_troubleshooting_flow | informational_answer | chitchat_or_praise"""

_STATS = {"calls": 0, "parse_failures": 0, "rule_fired": 0, "gate_fired": 0, "model_decided": 0}


@dataclass
class RouteDecision:
    route: str                 # "auto" | "escalate"
    reason_code: str
    reason: str                # the human sentence -- always populated
    confidence: float
    layer: str                 # which layer decided: rule | model | gate


def check_rules(text: str, context=None) -> tuple:
    """Layer 1. Returns (reason_code, matched_span) or (None, None)."""
    blob = " ".join([text] + list(context or []))
    for code, rx in _RULES:
        m = rx.search(blob)
        if m:
            return code, m.group(0)
    return None, None


def route(message: str, thread_context=None, intent: str = "", intent_confidence: float = 1.0,
          max_retrieval_sim: float = 1.0, tau: float | None = None, sigma: float | None = None,
          model: str | None = None, use_model: bool = True) -> RouteDecision:
    tau = config.TAU_INTENT_CONFIDENCE if tau is None else tau
    sigma = config.SIGMA_RETRIEVAL_SIM if sigma is None else sigma

    # ---- layer 1: deterministic safety rules
    code, span = check_rules(message, thread_context)
    if code:
        _STATS["rule_fired"] += 1
        return RouteDecision("escalate", code,
                             f"{HUMAN_SENTENCE[code]} (matched: \"{span}\")", 1.0, "rule")

    # ---- layer 2: the model, for judgement calls
    model_route, model_code, model_conf = "auto", "informational_answer", 0.5
    if use_model:
        _STATS["calls"] += 1
        ctx = ""
        if thread_context:
            ctx = "EARLIER TURNS FROM THIS CUSTOMER:\n" + "\n".join(f"- {c[:200]}" for c in thread_context[:3]) + "\n\n"
        parsed, _ = llm.complete_json(
            f"{ctx}MESSAGE:\n\"{message}\"\n\nClassified intent: {intent or 'unknown'}",
            system=ROUTER_SYSTEM, model=model or config.GENERATOR_MODEL, max_tokens=100, tag="route",
        )
        if parsed is None:
            # A parse failure must fail SAFE: escalate rather than silently auto-handling.
            _STATS["parse_failures"] += 1
            return RouteDecision("escalate", "low_model_confidence",
                                 "The router returned unparseable output, so this defaults to a human.",
                                 0.0, "model")
        model_route = "escalate" if str(parsed.get("route", "")).lower() == "escalate" else "auto"
        model_code = parsed.get("reason_code") or ("requires_account_specific_pii" if model_route == "escalate"
                                                   else "informational_answer")
        try:
            model_conf = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            model_conf = 0.5
        if model_route == "escalate":
            _STATS["model_decided"] += 1
            return RouteDecision("escalate", model_code,
                                 HUMAN_SENTENCE.get(model_code, "A human should review this."),
                                 model_conf, "model")

    # ---- layer 3: confidence gates. Not knowing is itself a reason to escalate.
    if intent_confidence < tau:
        _STATS["gate_fired"] += 1
        return RouteDecision("escalate", "low_model_confidence",
                             f"{HUMAN_SENTENCE['low_model_confidence']} "
                             f"(confidence {intent_confidence:.2f} < tau {tau:.2f})",
                             intent_confidence, "gate")
    if max_retrieval_sim < sigma:
        _STATS["gate_fired"] += 1
        return RouteDecision("escalate", "no_retrieval_support",
                             f"{HUMAN_SENTENCE['no_retrieval_support']} "
                             f"(max similarity {max_retrieval_sim:.2f} < sigma {sigma:.2f})",
                             max_retrieval_sim, "gate")

    _STATS["model_decided"] += 1
    return RouteDecision("auto", model_code, HUMAN_SENTENCE.get(model_code, "Handled automatically."),
                         model_conf, "model")


def stats() -> dict:
    return dict(_STATS)


def reset_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        cases = [
            ("I was charged twice this month, refund me now", []),
            ("my songs keep skipping on wifi", []),
            ("I already tried reinstalling like you said and it still doesn't work", ["my music won't play"]),
            ("my account got hacked, someone changed my email", []),
            ("love you guys, best app ever 💚", []),
        ]
        for text, ctx in cases:
            d = route(text, ctx, intent="playback_streaming_issue", intent_confidence=0.8, max_retrieval_sim=0.6)
            print(json.dumps({"msg": text[:55], "route": d.route, "code": d.reason_code,
                              "layer": d.layer, "reason": d.reason}, ensure_ascii=False))
        print(json.dumps(stats(), indent=2))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
