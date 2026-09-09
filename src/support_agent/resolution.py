"""Is a brand reply *resolution-bearing*, or just a handoff?

This filter matters more than it looks. Roughly half of what a brand posts publicly is
"we've sent you a DM" / "sorry to hear that!" -- pure routing with zero content. You
cannot ground a drafted reply in a thread whose only brand turn is "please DM us", so
threads without at least one resolution-bearing reply are dropped.

Deliberately a TRANSPARENT KEYWORD + HEURISTIC SCORER, not an LLM call:
  - I have to defend every decision it makes in a live interview.
  - It runs over ~800k rows in seconds and costs nothing.
  - It is deterministic, so the data funnel is reproducible.

!! This filter is a MAJOR source of survivorship bias. I evaluate on the subset of
   problems the brand chose to answer publicly and substantively. That is written into
   REPORT.md's "misleading numbers" section. !!
"""
from __future__ import annotations

import re

# --------------------------------------------------------- negative: pure handoff/ack
HANDOFF_PATTERNS = [
    r"\bsent you a (dm|direct message)\b",
    r"\b(shoot|send|drop) us a (dm|direct message)\b",
    # "Can you DM us your account's email address?" is Spotify's single most common
    # handoff and was slipping through: it names no DM noun, it uses DM as a verb.
    r"\b(can|could|would) you dm us\b",
    r"\bdm us your\b",
    r"\bsend us a dm\b",
    r"\bwe'?ll take a look backstage\b",
    r"\bfollow (us )?(and|then) dm\b",
    r"\bslide into (our|the) dms\b",
    r"\bcheck your (dms|inbox)\b",
    r"\bwe('| a)?re following you\b",
    r"\blet'?s (take|continue) this (to|in) (dms?|private)\b",
    r"\bhead over to (our )?dms\b",
]
ACK_ONLY_PATTERNS = [
    r"^\s*(so )?sorry (to hear|about) (that|this)",
    r"^\s*(oh no|aw|ah|yikes|eek)\b",
    r"^\s*thanks? (for|so much for) (reaching out|letting us know|the )",
    r"^\s*we('| a)?re (on it|here)\b",
    r"^\s*hey there[!.]?\s*$",
]

# --------------------------------------------------------- positive: actionable content
# 1. Troubleshooting verbs -- the single strongest signal of a real answer.
#    Written as regexes, not literals, because support agents write these in every
#    inflection: "clear your cache" / "clearing the cache" / "cleared your cache".
#    A literal-substring list silently misses the gerund forms, which are common in
#    "Try clearing X" -- by far the most frequent shape of a Spotify troubleshooting reply.
TROUBLESHOOT_VERBS = [
    r"\btry(ing)?\b",
    r"\brestart(ing|ed)?\b",
    r"\bre-?install(ing|ed)?\b",
    r"\bun-?install(ing|ed)?\b",
    r"\blog(ging|ged)? (out|back in)\b",
    r"\bsign(ing|ed)? (out|back in)\b",
    r"\bclear(ing|ed|s)?\s+(the\s+|your\s+|out\s+)?cache\b",
    r"\bupdat(e|ing|ed)\s+(the\s+app|to\b|your\b)",
    r"\btoggl(e|ing|ed)\b",
    r"\b(switch|turn)(ing|ed)?\s+(it\s+)?(off|on)\b",
    r"\breboot(ing|ed)?\b",
    r"\bhead(ing)? (to|over)\b",
    r"\bgo to\b",
    r"\btap(ping|ped)?\b",
    r"\bclick(ing|ed)?\b",
    r"\bselect(ing|ed)?\b",
    r"\bopen(ing|ed)? (settings|the app)\b",
    r"\bcheck(ing|ed)? your\b",
    r"\bmake sure\b",
    r"\bensur(e|ing)\b",
    r"\b(dis|en)abl(e|ing|ed)\b",
    r"\bre-?connect(ing|ed)?\b",
    r"\bre-?add(ing|ed)?\b",
    r"\bdownload(ing|ed)? (it |them )?again\b",
    r"\boffline mode\b",
    r"\b(hard|force) clos(e|ing|ed)\b",
    r"\bswip(e|ing)\b",
    r"\bdelet(e|ing) (and|the app)\b",
]
# 2. A link. Scored in TWO tiers, because a bare link is genuinely ambiguous here.
#    Every Spotify link in twcs is shortened through t.co, so the destination is
#    unknowable from the text: https://t.co/ldFdZRiNAt turns out to be the image
#    attached to their "DM us" reply, while https://t.co/kiXUjgr1xh is a real support
#    article. A shortened link therefore earns +1 (weak evidence), and only an
#    explicitly named support domain earns +2. The handoff penalty cancels the +1 on
#    the DM-with-an-image case, which is exactly the behaviour I want.
STRONG_LINK_PATTERNS = [r"<URL>", r"spotify\.com/(support|help)", r"support\.spotify", r"community\.spotify"]
WEAK_LINK_PATTERNS = [r"https?://\S+", r"\bt\.co/\w+"]
# 2b. The canonical terminal answers for a feature request. These ARE resolutions --
#     "vote for the idea" is genuinely how Spotify closes a feature-request thread --
#     and missing them was under-counting the feature_request intent specifically.
FEATURE_REQUEST_RESOLUTION_PATTERNS = [
    r"\b(add|cast) your vote\b",
    r"\bvote for (the|this) (official )?idea\b",
    r"\bofficial idea\b",
    r"\bpassed (your |this |that )?(feedback|idea|suggestion)s? (on )?to our (devs|team)\b",
    r"\byour feedback('s| has| is)? been (noted|passed|shared)\b",
]
# 3. Policy / process explanation.
POLICY_PATTERNS = [
    r"\b(refunds?|charges?|billing|subscription|plan|trial|payment)\b.{0,60}\b(policy|process|works|handled|processed|takes|within|business days)\b",
    r"\byou can (cancel|change|switch|update|manage|remove|add)\b",
    r"\b(premium|family|duo|student) (plan|members?|accounts?)\b.{0,60}\b(need|must|can|allows?|requires?)\b",
    r"\bit (can )?takes? (up to )?\d+",
    r"\bonce (you|your)\b.{0,50}\b(will|should)\b",
    r"\bthis (happens|occurs) (when|because|if)\b",
    r"\bthe reason\b",
]
# 4. A confirmed action taken by the agent.
CONFIRMED_ACTION_PATTERNS = [
    r"\bwe'?ve (now )?(fixed|resolved|updated|removed|added|refreshed|reset|restored|passed (this|it) on)\b",
    r"\b(this|that|it) (is|should be) (now )?(fixed|resolved|sorted|working|back up)\b",
    r"\bwe'?ve (logged|reported|escalated|flagged) (this|it)\b",
    r"\bour team (is|has) (aware|looking|fixed|working)\b",
    r"\ball (is )?(fixed|sorted|good) now\b",
]

_VERBS = [re.compile(p, re.I) for p in TROUBLESHOOT_VERBS]
_HANDOFF = [re.compile(p, re.I) for p in HANDOFF_PATTERNS]
_ACK = [re.compile(p, re.I) for p in ACK_ONLY_PATTERNS]
_STRONG_LINK = [re.compile(p, re.I) for p in STRONG_LINK_PATTERNS]
_WEAK_LINK = [re.compile(p, re.I) for p in WEAK_LINK_PATTERNS]
_FEATURE = [re.compile(p, re.I) for p in FEATURE_REQUEST_RESOLUTION_PATTERNS]
_POLICY = [re.compile(p, re.I) for p in POLICY_PATTERNS]
_ACTION = [re.compile(p, re.I) for p in CONFIRMED_ACTION_PATTERNS]

SCORE_THRESHOLD = 2  # tuned by eyeballing 100 replies; see SAMPLING_NOTE.md


def score_reply(text: str) -> tuple[int, list]:
    """Return (score, matched_signal_names). Score >= SCORE_THRESHOLD == resolution-bearing.

    Additive rather than a single rule so that a reply which both hands off AND gives a
    step ("try clearing your cache, and DM us if that fails") still counts -- it does
    contain a resolution.
    """
    t = (text or "").strip()
    if not t:
        return 0, []
    low = t.lower()
    signals: list = []
    score = 0

    verb_hits = [p.pattern for p in _VERBS if p.search(t)]
    if verb_hits:
        score += 2 if len(verb_hits) >= 2 else 1
        signals.append(f"troubleshoot:{len(verb_hits)}")

    if any(p.search(t) for p in _STRONG_LINK):
        score += 2
        signals.append("support_link")
    elif any(p.search(t) for p in _WEAK_LINK):
        score += 1
        signals.append("shortened_link")
    if any(p.search(t) for p in _FEATURE):
        score += 2
        signals.append("feature_request_resolution")
    if any(p.search(t) for p in _POLICY):
        score += 2
        signals.append("policy_explanation")
    if any(p.search(t) for p in _ACTION):
        score += 2
        signals.append("confirmed_action")

    # Length is weak evidence of substance, worth exactly one point and no more.
    if len(t) >= 120:
        score += 1
        signals.append("substantive_length")

    # Penalties. A handoff is not a resolution; a pure acknowledgement is not either.
    if any(p.search(t) for p in _HANDOFF):
        score -= 2
        signals.append("handoff")
    if any(p.search(t) for p in _ACK) and len(t) < 90:
        score -= 1
        signals.append("ack_only")

    return score, signals


def is_resolution_bearing(text: str) -> bool:
    return score_reply(text)[0] >= SCORE_THRESHOLD


def thread_resolution_score(brand_texts) -> tuple[int, list]:
    """Best score across a thread's brand turns -- one good reply is enough."""
    best, best_sig = 0, []
    for t in brand_texts:
        s, sig = score_reply(t)
        if s > best:
            best, best_sig = s, sig
    return best, best_sig
