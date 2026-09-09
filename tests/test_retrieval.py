"""Retrieval tests, with the leakage assertion as the one that matters most.

These run against a tiny synthetic corpus so they are fast and deterministic; the real
index is exercised by `agent --demo`. Dense embedding is monkeypatched out where the test
is about fusion or leakage rather than about semantics, so the suite needs no model
download and no network.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.support_agent import config, retrieval


@pytest.fixture
def tiny_corpus(monkeypatch):
    rows = [
        {"id": "c1", "text": "my songs keep skipping on wifi", "resolution_reply": "Try clearing the cache",
         "created_at": "2017-10-01T00:00:00+0000", "res_score": 3},
        {"id": "c2", "text": "i was charged twice for premium", "resolution_reply": "Check your account page",
         "created_at": "2017-10-05T00:00:00+0000", "res_score": 4},
        {"id": "c3", "text": "cannot log into my account", "resolution_reply": "Reset via the login page",
         "created_at": "2017-10-09T00:00:00+0000", "res_score": 2},
    ]
    df = pd.DataFrame(rows)

    # Deterministic fake embeddings: one-hot per row, so cosine is exactly predictable.
    def fake_encode(texts, cache_name=None, batch_size=128):
        out = np.zeros((len(texts), 3), dtype="float32")
        for i, t in enumerate(texts):
            if "skip" in t or "wifi" in t:
                out[i, 0] = 1.0
            elif "charg" in t or "premium" in t:
                out[i, 1] = 1.0
            else:
                out[i, 2] = 1.0
        return out

    monkeypatch.setattr(retrieval.embed, "encode", fake_encode)
    return retrieval.HybridRetriever(df)


def test_search_returns_full_threads_not_just_matches(tiny_corpus):
    hits = tiny_corpus.search("charged twice premium", top_k=1)
    assert hits and hits[0].id == "c2"
    # The resolution is the point of retrieving -- it must come back with the match.
    assert hits[0].resolution_reply == "Check your account page"


def test_rrf_fuses_both_channels(tiny_corpus):
    hits = tiny_corpus.search("my songs keep skipping on wifi", top_k=3)
    top = hits[0]
    assert top.id == "c1"
    assert top.bm25_rank is not None and top.dense_rank is not None, "both channels should contribute"
    assert top.rrf > 0


def test_top_k_is_respected(tiny_corpus):
    assert len(tiny_corpus.search("account", top_k=2)) == 2
    assert len(tiny_corpus.search("account", top_k=1)) == 1


def test_max_similarity_feeds_the_sigma_gate(tiny_corpus):
    hits = tiny_corpus.search("my songs keep skipping on wifi", top_k=3)
    assert tiny_corpus.max_similarity(hits) == pytest.approx(1.0)
    assert tiny_corpus.max_similarity([]) == 0.0, "no hits must read as zero support, not crash"


def test_results_are_deterministic(tiny_corpus):
    a = [h.id for h in tiny_corpus.search("premium charge", top_k=3)]
    b = [h.id for h in tiny_corpus.search("premium charge", top_k=3)]
    assert a == b


# ------------------------------------------------------------------ the important one
def test_leakage_assertion_fires_on_a_dev_or_test_id(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path)
    retrieval._LEAK_CACHE.clear()
    (tmp_path / "dev.jsonl").write_text(json.dumps({"id": "d1"}) + "\n", encoding="utf-8")
    (tmp_path / "test.jsonl").write_text(json.dumps({"id": "t1"}) + "\n", encoding="utf-8")

    retrieval.assert_no_leakage(["c1", "c2"])          # corpus ids: fine
    with pytest.raises(AssertionError, match="LEAKAGE"):
        retrieval.assert_no_leakage(["c1", "t1"])      # a test id must fail loudly
    with pytest.raises(AssertionError, match="LEAKAGE"):
        retrieval.assert_no_leakage(["d1"])
    retrieval._LEAK_CACHE.clear()


def test_real_index_contains_no_dev_or_test_ids():
    """Belt and braces against the real split files: the corpus index must be disjoint."""
    p = config.DATA_PROCESSED
    if not (p / "corpus.jsonl").exists():
        pytest.skip("data not built")
    ids = {}
    for s in ("corpus", "dev", "test"):
        ids[s] = {json.loads(l)["id"] for l in (p / f"{s}.jsonl").open(encoding="utf-8")}
    assert not (ids["corpus"] & ids["dev"]), "corpus overlaps dev"
    assert not (ids["corpus"] & ids["test"]), "corpus overlaps test"
    assert not (ids["dev"] & ids["test"]), "dev overlaps test"
