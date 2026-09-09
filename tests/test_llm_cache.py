"""Cache round-trip + offline behaviour. This is the test that protects `make demo`."""
import json
import pytest
from src.support_agent import llm


def test_cache_key_is_stable_and_sensitive():
    a = llm.cache_key("m", "sys", "hello", 0.0, 10)
    b = llm.cache_key("m", "sys", "hello", 0.0, 10)
    c = llm.cache_key("m", "sys", "hello!", 0.0, 10)
    d = llm.cache_key("m", "sys", "hello", 0.5, 10)
    assert a == b, "same inputs must give the same key or the cache never hits"
    assert a != c and a != d, "any input change must invalidate the entry"


def test_cache_round_trip_returns_without_network(tmp_path, monkeypatch):
    key = llm.cache_key("test-model", None, "PING", 0.0, 5)
    llm.cache_put(key, {"text": "PONG", "model": "test-model"})
    # Offline: a hit must still work, proving replay needs no key.
    llm.set_offline(True)
    try:
        out = llm.complete("PING", model="test-model", temperature=0.0, max_tokens=5, tag="test")
        assert out == "PONG"
    finally:
        llm.set_offline(False)
        (llm._cache_path(key)).unlink(missing_ok=True)


def test_offline_miss_raises_loudly():
    llm.set_offline(True)
    try:
        with pytest.raises(llm.CacheMiss):
            llm.complete("a prompt that is definitely not cached xyzzy-42", model="test-model", tag="test")
    finally:
        llm.set_offline(False)


def test_json_extraction_handles_fences_and_prose():
    assert llm._try_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm._try_json('Sure! {"a": 2} hope that helps') == {"a": 2}
    assert llm._try_json("no json here") is None
    assert llm._try_json("[1,2,3]") is None  # top-level array is not a valid response shape


def test_price_matches_published_rates():
    # 1M input tokens of gpt-4o-mini is $0.15
    assert abs(llm.price("gpt-4o-mini", 1_000_000, 0) - 0.150) < 1e-9
    assert abs(llm.price("gpt-4o-mini", 0, 1_000_000) - 0.600) < 1e-9
