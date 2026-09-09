"""Normalisation, PII scrubbing, and quality filters.

Order matters and is deliberate:
  1. strip the brand @mention (it is a constant, so it is pure noise for the classifier)
  2. detach the agent signature (^JT, /RM, -KM) -- Spotify agents sign with initials.
     Stored separately rather than deleted: it is a genuine feature (which agent
     handled it) and a nice detail for the report.
  3. PII scrub -- URLs, emails, phones, long digit runs, other @handles.
  4. punctuation/whitespace normalisation, emoji KEPT (they carry sentiment signal).

The PII scrub runs before any text can leave this machine. tests/test_clean.py asserts
that no email or phone pattern survives it.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

# --------------------------------------------------------------------------- patterns
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
# Phone: international, grouped, or a bare 10+ digit run. Deliberately greedy -- a
# false positive costs us a token, a false negative leaks a real phone number.
PHONE_RE = re.compile(r"(?:\+?\d[\d\-\.\(\)\s]{7,}\d)")
HANDLE_RE = re.compile(r"@[A-Za-z0-9_]{1,15}")
LONGNUM_RE = re.compile(r"\b\d{5,}\b")
# Agent signature: ^JT  /RM  -KM  ~AB  ^ Jess  at the END of a brand reply.
SIGNATURE_RE = re.compile(r"[\^/~\-]\s*([A-Z][A-Za-z]{0,8})\s*$")
REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")
WS_RE = re.compile(r"\s+")

# Cheap English heuristic. langdetect is a heavy dep for a filter this coarse, and a
# character-class ratio catches the actual failure mode here: CJK/Cyrillic/Arabic
# tweets from Spotify's global audience. Cost: some romanised non-English survives.
NON_LATIN_RE = re.compile(
    r"[Ѐ-ӿ֐-׿؀-ۿ฀-๿"
    r"぀-ヿ㐀-䶿一-鿿가-힯]"
)
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")

# The script check above is necessary but NOT sufficient, and I only found that out by
# reading the induced clusters: one entire cluster (176/3000 messages) was romanised
# Indonesian -- "min sy mau beli spotify premium, bisa pake kartu debit gak??" -- which
# is pure Latin script and sailed through. Function words are the cheap, transparent
# discriminator: genuine English support tweets almost always contain at least one, and
# other Latin-script languages contain none of these. Preferred over adding langdetect
# as a dependency for a filter this coarse, and unlike langdetect it stays explainable.
ENGLISH_FUNCTION_WORDS = {
    "the", "is", "it", "my", "me", "i", "you", "your", "and", "but", "not", "no",
    "to", "of", "in", "on", "for", "with", "this", "that", "have", "has", "had",
    "do", "does", "did", "can", "cant", "cannot", "wont", "will", "would", "should",
    "why", "how", "what", "when", "where", "im", "ive", "its", "dont", "doesnt",
    "isnt", "are", "was", "were", "be", "been", "get", "got", "there", "just",
    "still", "any", "all", "please", "help", "thanks", "thank", "keep", "keeps",
    "work", "works", "working", "an", "a", "at", "if", "or", "so", "up", "out",
    # Added after measuring: the first list dropped "Y'all being DDoS'd? Having trouble
    # connecting", "we need full catalogue" and "where's Taylor Swifts new album" --
    # all unambiguously English. Checked that none of these collide with the Indonesian
    # / French / Polish vocabulary that the filter is meant to catch.
    "we", "us", "our", "they", "them", "their", "he", "she", "his", "her", "him",
    "being", "having", "doing", "getting", "need", "needs", "want", "wants",
    "new", "other", "some", "than", "then", "about", "from", "by", "as", "am",
    "every", "each", "much", "many", "more", "most", "very", "too", "now", "back",
    "again", "since", "after", "before", "yall", "trouble", "issue", "problem",
}
MIN_ENGLISH_FUNCTION_WORDS = 1

MIN_CUSTOMER_TOKENS = 4


def strip_brand_mention(text: str, brand: str) -> str:
    return re.sub(rf"@{re.escape(brand)}\b", " ", text, flags=re.I)


def extract_signature(text: str) -> tuple[str, str]:
    """Return (text_without_signature, signature_or_empty)."""
    m = SIGNATURE_RE.search(text.rstrip())
    if not m:
        return text, ""
    sig = m.group(1)
    # Guard: "-Spotify" or a real word is not an agent's initials.
    if len(sig) > 4 and not sig.isupper():
        return text, ""
    return text[: m.start()].rstrip(), sig


def scrub_pii(text: str) -> str:
    """Replace anything personally identifying with a stable placeholder.

    URLs go first: a URL can contain an @ and would otherwise be half-eaten by the
    handle rule, leaving a fragment behind.
    """
    text = URL_RE.sub(" <URL> ", text)
    text = EMAIL_RE.sub(" <EMAIL> ", text)
    text = PHONE_RE.sub(" <PHONE> ", text)
    text = HANDLE_RE.sub(" <USER> ", text)
    text = LONGNUM_RE.sub(" <NUM> ", text)
    return text


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    text = REPEAT_PUNCT_RE.sub(r"\1\1", text)   # "help!!!!!" -> "help!!"
    text = WS_RE.sub(" ", text)
    return text.strip()


def clean_customer(text: str, brand: str) -> str:
    return normalise(scrub_pii(strip_brand_mention(text, brand)))


def clean_brand(text: str, brand: str) -> tuple[str, str]:
    """Brand replies additionally have their agent signature detached."""
    body, sig = extract_signature(strip_brand_mention(text, brand))
    return normalise(scrub_pii(body)), sig


# --------------------------------------------------------------------------- filters
def is_probably_english(text: str) -> bool:
    """Two gates: script, then English function words.

    Cost of the second gate: a very short but genuinely English message with no function
    word ("premium broken again") is dropped. Measured on the corpus before shipping --
    see ASSUMPTIONS.md [A-10] for the drop rate and the false-positive spot check.
    """
    letters = LATIN_LETTER_RE.findall(text)
    non_latin = NON_LATIN_RE.findall(text)
    if len(non_latin) >= 3 and len(non_latin) > len(letters) * 0.2:
        return False
    if len(letters) < 8:
        return False
    raw_words = {w.replace("'", "") for w in re.findall(r"[a-z']+", text.lower())}
    # Also test the de-suffixed form so possessives and contractions match:
    # "where's" -> "wheres" -> "where". Cheap stemming, one rule, no library.
    words = raw_words | {w[:-1] for w in raw_words if w.endswith("s") and len(w) > 2}
    return len(words & ENGLISH_FUNCTION_WORDS) >= MIN_ENGLISH_FUNCTION_WORDS


def token_count(text: str) -> int:
    return len([t for t in re.split(r"\s+", text.strip()) if t and t not in ("<URL>", "<USER>")])


def dedupe_key(text: str) -> str:
    """Normalised hash for near-duplicate detection.

    Twitter support data is full of copy-paste spam and bot-retried complaints. Lower,
    strip punctuation, collapse whitespace, hash. Catches exact and cosmetic duplicates;
    genuine paraphrases survive, which is the intended sensitivity.
    """
    t = re.sub(r"[^a-z0-9 ]", "", text.lower())
    t = WS_RE.sub(" ", t).strip()
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def passes_quality(customer_text: str) -> tuple[bool, str]:
    """Return (keep, reason_if_dropped) so the funnel can report WHY rows were lost."""
    if not is_probably_english(customer_text):
        return False, "non_english"
    if token_count(customer_text) < MIN_CUSTOMER_TOKENS:
        return False, "too_short"
    return True, ""
