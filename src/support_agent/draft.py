"""Grounded reply drafting.

"Grounded" has to be VERIFIABLE, not asserted. So the model must cite which retrieved
thread(s) it drew on, and the citation is carried through into the Decision and checked
by the judge. A reply that cites nothing is treated as ungrounded.

The hard constraints are in the system prompt because they are safety rules, not style
preferences: inventing a refund policy in public is the single most expensive thing this
system could do.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field

from . import config, llm

# Derived from reading real SpotifyCares replies in the corpus, not invented:
# short, warm, lowercase-friendly, uses "Hey there!", signs off with agent initials,
# almost always ends by inviting the customer to report back.
BRAND_VOICE = """SpotifyCares house style, derived from their real public replies:
- short and warm; 1-2 sentences; contractions; occasional light exclamation
- opens with a greeting like "Hey there!" or "Hi!" when the thread is new
- gives ONE concrete next step, not a numbered list
- usually closes by inviting a reply: "Let us know how it goes" / "Keep us posted"
- signs off with two agent initials, e.g. /JT  (use /AI for this system)
- no corporate jargon, no "we sincerely apologise for the inconvenience\""""

SYSTEM = f"""You draft a public reply from Spotify's support account (@SpotifyCares) to a
customer tweet. You are given historical threads showing how this brand actually resolved
similar issues. Ground your reply in those.

{BRAND_VOICE}

HARD CONSTRAINTS -- breaking any of these makes the reply unusable:
- NEVER invent a policy, a timeline, or a fact that is not in the retrieved history or
  general public knowledge about Spotify.
- NEVER promise a refund, a credit, a specific amount, or a specific timeframe.
- NEVER ask for card numbers, passwords, full account details, or any sensitive personal
  data in a PUBLIC reply.
- If the retrieved history does not actually support an answer, do NOT guess. Say what
  you can do and hand off to a human.
- Maximum {config.REPLY_MAX_CHARS} characters. This is a tweet.

Return JSON only:
{{"reply": "<the reply text>", "grounded_in": ["<id of each retrieved thread you used>"],
  "used_retrieval": true|false}}

`grounded_in` must contain the ids of the retrieved threads you actually drew on. If you
answered from general knowledge rather than the retrieved history, return an empty list
and set used_retrieval to false. Do not claim grounding you did not use."""

_STATS = {"calls": 0, "parse_failures": 0, "over_length": 0, "ungrounded": 0}


@dataclass
class DraftResult:
    reply: str
    grounded_in: list = field(default_factory=list)
    used_retrieval: bool = False
    truncated: bool = False
    parse_failed: bool = False


def build_prompt(message: str, thread_context, hits) -> str:
    parts = []
    if hits:
        parts.append("HISTORICAL THREADS -- how this brand resolved similar issues:")
        for h in hits:
            parts.append(
                f"[id {h.id}] (similarity {h.dense_sim:.2f}, {h.created_at[:10]})\n"
                f"  customer: {h.customer_text[:240]}\n"
                f"  SpotifyCares resolved with: {h.resolution_reply[:300]}"
            )
        parts.append("")
    else:
        parts.append("HISTORICAL THREADS: none retrieved. You have no grounding -- hand off.\n")
    if thread_context:
        parts.append("EARLIER TURNS FROM THIS CUSTOMER:")
        parts.extend(f"- {c[:200]}" for c in thread_context[:3])
        parts.append("")
    parts.append(f'CUSTOMER MESSAGE TO REPLY TO:\n"{message}"')
    return "\n".join(parts)


def _truncate(reply: str, limit: int = config.REPLY_MAX_CHARS) -> tuple[str, bool]:
    """Cut on a sentence boundary where possible; the medium is a tweet."""
    if len(reply) <= limit:
        return reply, False
    cut = reply[:limit]
    m = list(re.finditer(r"[.!?]\s", cut))
    if m and m[-1].end() > limit * 0.6:
        return cut[: m[-1].end()].rstrip(), True
    return cut[: limit - 1].rstrip() + "…", True


def draft(message: str, thread_context=None, hits=None, model: str | None = None) -> DraftResult:
    hits = hits or []
    _STATS["calls"] += 1
    parsed, raw = llm.complete_json(
        build_prompt(message, thread_context, hits), system=SYSTEM,
        model=model or config.GENERATOR_MODEL, max_tokens=300, tag="draft",
    )
    if parsed is None:
        _STATS["parse_failures"] += 1
        # A safe, honest fallback rather than a fabricated answer.
        return DraftResult(
            "Hey there! We want to get this sorted properly - we're passing you to a "
            "teammate who can dig into it. /AI",
            [], False, parse_failed=True,
        )

    reply = str(parsed.get("reply", "")).strip()
    reply, truncated = _truncate(reply)
    if truncated:
        _STATS["over_length"] += 1

    valid_ids = {h.id for h in hits}
    grounded = [str(g) for g in (parsed.get("grounded_in") or []) if str(g) in valid_ids]
    if not grounded:
        _STATS["ungrounded"] += 1
    return DraftResult(reply, grounded, bool(parsed.get("used_retrieval")) and bool(grounded), truncated)


def stats() -> dict:
    n = max(1, _STATS["calls"])
    return {**_STATS,
            "parse_failure_rate": round(_STATS["parse_failures"] / n, 4),
            "over_length_rate": round(_STATS["over_length"] / n, 4),
            "ungrounded_rate": round(_STATS["ungrounded"] / n, 4)}


def reset_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--text", default="my downloaded songs vanished after the update, android")
    a = ap.parse_args()
    if a.smoke:
        from .retrieval import HybridRetriever

        r = HybridRetriever.from_split("corpus")
        hits = r.search(a.text)
        d = draft(a.text, hits=hits)
        print(json.dumps({"reply": d.reply, "chars": len(d.reply), "grounded_in": d.grounded_in,
                          "used_retrieval": d.used_retrieval,
                          "retrieved": [h.id for h in hits], "stats": stats()}, indent=2))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
