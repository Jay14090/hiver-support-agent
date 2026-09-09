"""Cleaning + PII. The PII tests are the ones that matter: nothing personal leaves the box."""
import re

from src.support_agent.clean import (
    EMAIL_RE,
    PHONE_RE,
    clean_brand,
    clean_customer,
    dedupe_key,
    extract_signature,
    is_probably_english,
    normalise,
    passes_quality,
    scrub_pii,
    strip_brand_mention,
    token_count,
)

BRAND = "SpotifyCares"


# --------------------------------------------------------------------- PII (critical)
def test_pii_email_never_survives():
    for s in ["mail me at jay.doe+spam@example.co.uk please", "JOHN_SMITH99@gmail.com", "a@b.io"]:
        out = scrub_pii(s)
        assert not EMAIL_RE.search(out), f"email leaked from: {s!r} -> {out!r}"
        assert "<EMAIL>" in out


def test_pii_phone_never_survives():
    for s in ["call +44 7700 900123 now", "my number is (555) 123-4567", "0123456789012"]:
        out = scrub_pii(s)
        assert not PHONE_RE.search(out), f"phone leaked from: {s!r} -> {out!r}"


def test_pii_url_is_replaced_before_handle_rule_eats_it():
    # A URL containing an @ must be consumed whole, not shredded into a fragment.
    out = scrub_pii("see https://x.com/@spotify/status/123 ok")
    assert "<URL>" in out and "x.com" not in out


def test_pii_other_handles_are_masked_and_long_numbers_go():
    out = scrub_pii("@someuser my order 9876543210 is late")
    assert "<USER>" in out and "someuser" not in out
    assert "9876543210" not in out


def test_pii_scrub_is_idempotent():
    once = scrub_pii("mail a@b.com or call 555-123-4567")
    assert scrub_pii(once) == once, "re-scrubbing must not corrupt placeholders"


# --------------------------------------------------------------------- brand mention
def test_brand_mention_is_stripped_case_insensitively():
    assert "spotifycares" not in strip_brand_mention("@SpotifyCares help", BRAND).lower()
    assert "spotifycares" not in strip_brand_mention("hey @spotifycares!", BRAND).lower()


# --------------------------------------------------------------------- agent signature
def test_signature_is_detached_not_deleted():
    body, sig = extract_signature("Try clearing your cache and restart /RM")
    assert sig == "RM" and "/RM" not in body and body.endswith("restart")
    for raw, want in [("all sorted now ^JT", "JT"), ("done -KM", "KM"), ("hope that helps ~AB", "AB")]:
        assert extract_signature(raw)[1] == want


def test_signature_rule_does_not_eat_real_words():
    body, sig = extract_signature("we love you too - Spotify family")
    assert sig == "", "a trailing sentence must not be mistaken for initials"


def test_clean_brand_returns_body_and_signature():
    body, sig = clean_brand("@user Try a reinstall, that usually does it! ^JT", BRAND)
    assert sig == "JT" and "JT" not in body and "<USER>" in body


# --------------------------------------------------------------------- normalisation
def test_emoji_survive_because_they_carry_sentiment():
    out = clean_customer("this is awful 😡😡 fix it", BRAND)
    assert "😡" in out


def test_repeated_punctuation_is_collapsed_and_entities_decoded():
    assert normalise("help!!!!!!") == "help!!"
    assert normalise("rock &amp; roll") == "rock & roll"
    assert normalise("a   b\n\nc") == "a b c"


# --------------------------------------------------------------------- quality filters
def test_non_english_is_dropped():
    assert not is_probably_english("音楽が再生されません助けてください")
    assert not is_probably_english("Здравствуйте не работает музыка")
    assert is_probably_english("my music will not play at all today")


def test_short_messages_are_dropped():
    keep, why = passes_quality("still broken")
    assert not keep and why == "too_short"
    keep, why = passes_quality("my premium songs keep skipping badly")
    assert keep and why == ""


def test_token_count_ignores_placeholders():
    assert token_count("hey <USER> check <URL>") == 2


def test_dedupe_key_matches_cosmetic_variants_only():
    a = dedupe_key("My music KEEPS skipping!!!")
    b = dedupe_key("my music keeps skipping")
    c = dedupe_key("my podcasts keep skipping")
    assert a == b, "case and punctuation must not defeat dedupe"
    assert a != c, "a genuinely different complaint must survive"
