"""The resolution-bearing filter decides what data exists. It gets its own tests."""
from src.support_agent.resolution import is_resolution_bearing, score_reply, thread_resolution_score

HANDOFFS = [
    "We've sent you a DM, let's chat there!",
    "Sorry to hear that! Please follow us and DM us so we can look into it.",
    "Oh no! We're on it.",
    "Thanks for reaching out! Check your DMs :)",
    "Hey there!",
]

RESOLUTIONS = [
    "Try logging out of all devices, then clear the cache in Settings > Storage and log back in.",
    "You can cancel anytime from your account page - your Premium stays active until the end of the billing period.",
    "We've now fixed this on our end, give the app a restart and it should be back.",
    "This happens when Offline Mode is on. Head to Settings and toggle it off, then reconnect.",
    "Full steps are here <URL> - work through those and let us know.",
]


def test_pure_handoffs_are_not_resolutions():
    for t in HANDOFFS:
        assert not is_resolution_bearing(t), f"handoff scored as a resolution: {t!r}"


def test_substantive_replies_are_resolutions():
    for t in RESOLUTIONS:
        assert is_resolution_bearing(t), f"real resolution was rejected: {t!r}"


def test_handoff_plus_a_real_step_still_counts():
    # Additive scoring on purpose: this reply does contain a usable resolution.
    t = "Try clearing your cache and restarting the app first - if it still fails, DM us."
    assert is_resolution_bearing(t)


def test_length_alone_is_not_enough():
    waffle = ("Thanks so much for getting in touch with us today, we really do appreciate "
              "you taking the time to reach out and we're sorry about this whole situation.")
    assert not is_resolution_bearing(waffle), "long empty sympathy must not pass"


def test_signals_are_reported_for_auditability():
    score, signals = score_reply("Try a reinstall and clear the cache, details here <URL>")
    assert score >= 2
    assert any(s.startswith("troubleshoot") for s in signals) and "support_link" in signals


def test_thread_score_takes_the_best_brand_turn():
    # One good reply anywhere in the thread is enough to keep it.
    best, _ = thread_resolution_score(["Oh no!", "Try clearing your cache in Settings > Storage"])
    assert best >= 2


def test_empty_and_none_are_safe():
    assert score_reply("") == (0, [])
    assert score_reply(None) == (0, [])
