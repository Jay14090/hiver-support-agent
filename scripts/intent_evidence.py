"""Evidence check: does every intent in config.INTENTS actually exist in the corpus?

The induced clusters only isolated ~6 intents, because the topic mass is wildly uneven
(content-missing alone is over half the sample) and the embedding structure is weak
(silhouette ~0.04). That is NOT evidence the other intents are absent -- it is evidence
that k-means on unbalanced short text finds the big blobs.

So each remaining intent gets a targeted keyword probe over the corpus split: how many
messages match, and what do they look like. An intent that cannot be evidenced this way
does not belong in the taxonomy. Output feeds golden/CODEBOOK.md.
"""
from __future__ import annotations

import random
import re
import sys

import pandas as pd

sys.path.insert(0, ".")
from src.support_agent import config  # noqa: E402

PROBES = {
    "playback_streaming_issue": r"(won'?t|wont|not|stopped|keeps?) (play|playing|load|loading)|skip(s|ping)|buffer|no sound|cuts? out|stutter|downloaded.*(gone|disappear)|offline.*(not|won'?t)",
    "account_access_login": r"(can'?t|cant|unable to) (log ?in|login|sign ?in|access my account)|password reset|locked out|hacked|someone (else )?(is )?(using|on) my account|email (already )?in use",
    "billing_charge_dispute": r"charged (me )?(twice|again|two)|double charge|unauthorized charge|why (was i|am i) (being )?charged|took money|wrong amount|billed twice",
    "subscription_cancel_refund": r"cancel (my )?(subscription|premium|account)|want a refund|refund me|unsubscribe|stop (charging|billing|my subscription)",
    "plan_or_premium_management": r"student (discount|verification|premium)|family plan|duo plan|upgrade to premium|premium (isn'?t|not) active|change my plan|remove.*from my family",
    "content_missing_or_metadata": r"(isn'?t|not|no longer) (on|available on) spotify|can'?t find.*(album|song|artist)|missing (album|song|track)|why (isn'?t|is)n'?t.*on spotify|add.*(album|artist)",
    "device_integration_issue": r"(car|bluetooth|alexa|google home|sonos|chromecast|ps4|playstation|xbox|apple watch|carplay|android auto)",
    "app_bug_or_crash": r"crash(es|ing|ed)?|freez(es|ing)|app (keeps )?closing|force clos|won'?t open|black screen|glitch|bug",
    "feature_request_or_complaint": r"(please|pls) (add|make|bring|give us)|i wish|why (don'?t|dont) you|feature request|would be (great|nice) if|bring back",
    "praise_or_chitchat": r"^(?!.*(problem|issue|help|fix|broken|can'?t)).{0,200}(love (you|spotify)|thank you|best (app|service)|you'?re the best|amazing|❤|💚)",
}


def main() -> None:
    random.seed(config.RANDOM_STATE)
    df = pd.read_json(config.DATA_PROCESSED / "corpus.jsonl", lines=True)
    texts = df["text"].tolist()
    n = len(texts)
    print(f"corpus split: {n:,} messages\n")
    covered = set()
    for intent, pat in PROBES.items():
        rx = re.compile(pat, re.I)
        hits = [t for t in texts if rx.search(t)]
        covered.update(id(h) for h in hits)
        print(f"=== {intent}: {len(hits):,} matches ({len(hits)/n:.1%})")
        for t in random.sample(hits, min(4, len(hits))):
            print(f"    * {t[:130]}")
        print()
    missing = [i for i in config.INTENTS if i not in PROBES and i != "other"]
    if missing:
        print(f"!! intents with no probe defined: {missing}")


if __name__ == "__main__":
    main()
