"""Every constant in the system lives here.

One file, no magic numbers scattered across modules -- so that in a live walkthrough
I can answer "where does that threshold come from?" by opening one file.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
GOLDEN = ROOT / "golden"
CACHE_DIR = ROOT / "cache" / "llm"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
COST_LEDGER = ROOT / "cost_ledger.json"

for _p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, CACHE_DIR, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

TWCS_CSV = DATA_RAW / "twcs.csv"

# --------------------------------------------------------------------------- brand
# D2: SpotifyCares. D3 fallback chain fires if the Phase-1 filters undershoot.
BRAND = os.environ.get("BRAND", "SpotifyCares")
BRAND_FALLBACKS = ["AppleSupport", "Delta"]
MIN_USABLE_THREADS = 2_000          # D3 trigger
MIN_RESOLUTION_RATE = 0.15          # D3 trigger

# --------------------------------------------------------------------------- splits
# D5: time-based split on the ROOT tweet timestamp. Random splits leak near-duplicate
# future threads into the retrieval index and silently inflate reply quality.
SPLIT_CORPUS = 0.70
SPLIT_DEV = 0.15
SPLIT_TEST = 0.15
RANDOM_STATE = 20260909            # fixed everywhere; recorded in SAMPLING_NOTE.md

# --------------------------------------------------------------------------- models
# D6: cheap model for bulk generation. D7: a *different* model as judge.
# Only OPENAI_API_KEY exists on this machine (see ASSUMPTIONS.md [A-05]), so the judge
# is a different model from the same family, and the self-preference probe is mandatory.
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1-mini")
LABELER_A_MODEL = os.environ.get("LABELER_A_MODEL", "gpt-4o-mini")
LABELER_B_MODEL = os.environ.get("LABELER_B_MODEL", "gpt-4.1-mini")

# USD per 1M tokens (input, output). Published OpenAI list prices.
MODEL_PRICES = {
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4.1-mini": (0.400, 1.600),
    "gpt-4.1-nano": (0.100, 0.400),
    "gpt-4o": (2.500, 10.000),
}
CHEAPEST_MODEL = "gpt-4.1-nano"    # switched to at 80% of budget

BUDGET_USD_CAP = 25.0
BUDGET_WARN_FRACTION = 0.80

# Determinism: temperature 0 everywhere. Same command twice -> identical numbers.
TEMPERATURE = 0.0
MAX_TOKENS_DEFAULT = 400

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------------------------------- taxonomy
# P2 induces these from clustering; the list is then edited BY HAND, because the
# clustering was weak (silhouette ~0.04 at every k, HDBSCAN found 0 clusters / 100%
# noise) and k=8 merged intents that route differently. What the clusters DID establish,
# and what changed my starting list:
#   - content-missing is the single dominant topic (3 of 8 clusters, 52% of the sample):
#     "why isn't <artist> on Spotify", missing albums, metadata gaps. Kept as one intent.
#   - feature requests are the second mass (playlist tooling + "where's the Apple Watch
#     app"), so device-specific app requests fold into feature_request, not device_integration.
#   - a whole cluster was "I paid but Premium isn't active" / student verification /
#     upgrade problems. That is NOT a charge dispute and NOT a cancellation, so
#     `plan_management_family_duo` was widened to `plan_or_premium_management`.
INTENTS = [
    "playback_streaming_issue",
    "account_access_login",
    "billing_charge_dispute",
    "subscription_cancel_refund",
    "plan_or_premium_management",
    "content_missing_or_metadata",
    "device_integration_issue",
    "app_bug_or_crash",
    "feature_request_or_complaint",
    "praise_or_chitchat",
    "other",
]

# --------------------------------------------------------------------------- routing
# The escalation policy is MY invention -- Spotify's real one is unobservable.
# Router "accuracy" therefore measures agreement with me. Said plainly in the report.
ESCALATION_REASON_CODES = [
    "billing_dispute",
    "suspected_account_compromise",
    "refund_request",
    "legal_privacy_data_request",
    "explicit_cancellation_threat",
    "abusive_or_distressed_language",
    "requires_account_specific_pii",
    "troubleshooting_already_failed",
    "low_model_confidence",
    "no_retrieval_support",
]
AUTO_REASON_CODES = [
    "known_troubleshooting_flow",
    "informational_answer",
    "chitchat_or_praise",
]

# D10: asymmetric cost. Letting a should-escalate through is 10x worse than a
# needless escalation. A router is a decision system; raw accuracy is near-meaningless.
COST_FALSE_AUTO = 10.0   # missed escalation
COST_FALSE_ESCALATE = 1.0

# Confidence gates. TUNED ON DEV ONLY (P5.5) -- these are the defaults it starts from.
TAU_INTENT_CONFIDENCE = 0.55   # below this -> escalate (low_model_confidence)
SIGMA_RETRIEVAL_SIM = 0.30     # max retrieval sim below this -> escalate (no_retrieval_support)

# --------------------------------------------------------------------------- retrieval
RRF_K = 60           # reciprocal rank fusion constant, the standard value
RETRIEVE_TOP_N = 8   # fused candidates before rerank
RETRIEVE_TOP_K = 3   # passed into the drafting prompt

# --------------------------------------------------------------------------- generation
REPLY_MAX_CHARS = 280   # match the medium; constraint-fitting is part of the task

# --------------------------------------------------------------------------- eval
GOLDEN_TARGET_N = 220        # D9
GOLDEN_MIN_PER_INTENT = 12   # so per-class F1 is not pure noise
HARD_CASE_QUOTA = 0.15
BOOTSTRAP_N = 1000
BOOTSTRAP_CI = 0.95
JUDGE_AXES = ["groundedness", "actionability", "brand_fit", "safety"]
HUMAN_JUDGE_N = 70           # inside the brief's 60-80 band
