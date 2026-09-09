"""Intent classification head.

Prompt = codebook (definitions + tie-break rules) + 3 few-shot examples pulled
dynamically from the CORPUS split via retrieval + the message. Structured JSON out.

The parse-failure rate is counted and REPORTED. It is real reliability data about a
system that returns free text, and hiding it would make the pipeline look more robust
than it is.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from . import config, llm

SYSTEM = """You classify customer messages sent to Spotify's public support account
(@SpotifyCares) into exactly one intent.

Return JSON only:
{"intent": "<one of the listed intents>", "confidence": <0.0-1.0>, "rationale": "<under 15 words>"}

Rules:
- Pick the intent describing what the customer WANTS RESOLVED, not their emotion.
- If a message covers two intents, pick the one a support agent would act on first.
- `confidence` is your genuine probability of being right. Use the full range: if the
  message is ambiguous or very short, say so with a low number. Do not default to 0.9.
- Use `other` only when nothing else fits."""

_STATS = {"calls": 0, "parse_failures": 0, "invalid_intent": 0}


@dataclass
class IntentResult:
    intent: str
    confidence: float
    rationale: str
    parse_failed: bool = False


def build_prompt(message: str, thread_context=None, codebook: str = "", few_shots=None) -> str:
    parts = [f"INTENTS AND DEFINITIONS:\n{codebook}\n" if codebook else
             "INTENTS:\n" + "\n".join(f"- {i}" for i in config.INTENTS) + "\n"]
    if few_shots:
        parts.append("EXAMPLES OF REAL MESSAGES AND THEIR INTENT:")
        for t, lab in few_shots:
            parts.append(f'- "{t[:200]}" -> {lab}')
        parts.append("")
    if thread_context:
        parts.append("EARLIER TURNS FROM THIS CUSTOMER (context):")
        parts.extend(f"- {c[:200]}" for c in thread_context[:3])
        parts.append("")
    parts.append(f'MESSAGE TO CLASSIFY:\n"{message}"')
    return "\n".join(parts)


def classify(message: str, thread_context=None, codebook: str = "", few_shots=None,
             model: str | None = None) -> IntentResult:
    prompt = build_prompt(message, thread_context, codebook, few_shots)
    _STATS["calls"] += 1
    parsed, raw = llm.complete_json(
        prompt, system=SYSTEM, model=model or config.GENERATOR_MODEL,
        max_tokens=120, tag="classify",
    )
    if parsed is None:
        # Counted, never hidden. Falls back to the safest possible answer: `other` with
        # zero confidence, which the router's tau gate will then escalate.
        _STATS["parse_failures"] += 1
        return IntentResult("other", 0.0, "parse failure", parse_failed=True)

    intent = str(parsed.get("intent", "")).strip()
    if intent not in config.INTENTS:
        _STATS["invalid_intent"] += 1
        intent = "other"
    try:
        conf = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return IntentResult(intent, max(0.0, min(1.0, conf)), str(parsed.get("rationale", ""))[:120])


def stats() -> dict:
    n = max(1, _STATS["calls"])
    return {**_STATS,
            "parse_failure_rate": round(_STATS["parse_failures"] / n, 4),
            "invalid_intent_rate": round(_STATS["invalid_intent"] / n, 4)}


def reset_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--text", default="my premium songs keep skipping even on wifi, reinstalled twice")
    a = ap.parse_args()
    if a.smoke:
        from .codebook import load_codebook_text

        r = classify(a.text, codebook=load_codebook_text())
        print(json.dumps({"intent": r.intent, "confidence": r.confidence,
                          "rationale": r.rationale, "stats": stats()}, indent=2))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
