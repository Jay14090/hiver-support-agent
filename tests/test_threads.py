"""One test per named trap in threads.py. These are the tests I most want to be asked about."""
from src.support_agent.threads import (
    MAX_DEPTH,
    _norm_id,
    build_index,
    find_roots,
    reconstruct,
    split_children,
)


def row(tid, author, inbound, text, resp="", parent="", ts="Tue Oct 31 22:10:47 +0000 2017"):
    return {
        "tweet_id": tid,
        "author_id": author,
        "inbound": str(inbound),
        "created_at": ts,
        "text": text,
        "response_tweet_id": resp,
        "in_response_to_tweet_id": parent,
    }


# --------------------------------------------------------------- T1 comma-separated ids
def test_t1_response_tweet_id_can_be_a_comma_list():
    assert split_children("2,3,4") == ["2", "3", "4"]
    assert split_children("2, 3 ,4") == ["2", "3", "4"]
    assert split_children("") == []
    assert split_children(None) == []


def test_t1_comma_children_do_not_become_orphans():
    rows = [
        row("1", "cust", True, "help", resp="2,3"),
        row("2", "SpotifyCares", False, "try this"),
        row("3", "SpotifyCares", False, "or this"),
    ]
    turns, children = build_index(rows)
    assert set(children["1"]) == {"2", "3"}, "both children must be linked from the comma list"


# --------------------------------------------------------------- T2 float contamination
def test_t2_float_ids_are_normalised():
    assert _norm_id("123.0") == "123"
    assert _norm_id(123.0) == "123"
    assert _norm_id("nan") == ""
    assert _norm_id(None) == ""
    assert _norm_id("  456 ") == "456"


def test_t2_float_ids_still_join_correctly():
    rows = [row("1", "cust", True, "help", resp="2.0"), row("2.0", "SpotifyCares", False, "hi")]
    turns, children = build_index(rows)
    assert "2" in turns and children["1"] == ["2"]


# --------------------------------------------------------------- T3 cycles
def test_t3_self_reference_does_not_create_an_edge():
    rows = [row("1", "cust", True, "help", resp="1", parent="1")]
    _, children = build_index(rows)
    assert children.get("1", []) == []


def test_t3_cycle_terminates():
    # 1 -> 2 -> 3 -> 1 : an unguarded DFS would spin forever.
    rows = [
        row("1", "cust", True, "a", resp="2"),
        row("2", "SpotifyCares", False, "b", resp="3", parent="1"),
        row("3", "cust", True, "c", resp="1", parent="2"),
    ]
    threads = reconstruct(rows, brand="SpotifyCares")
    assert len(threads) == 1
    ids = [t.tweet_id for t in threads[0].turns]
    assert len(ids) == len(set(ids)), "a turn must never repeat inside one thread"


def test_t3_depth_is_capped():
    rows = [row("1", "cust", True, "t0", resp="2")]
    for i in range(2, 60):
        rows.append(row(str(i), "cust" if i % 2 else "SpotifyCares", i % 2 == 1, f"t{i}",
                        resp=str(i + 1), parent=str(i - 1)))
    threads = reconstruct(rows, brand="SpotifyCares")
    assert threads and len(threads[0].turns) <= MAX_DEPTH


# --------------------------------------------------------------- T4 roots
def test_t4_root_is_an_inbound_tweet_with_no_parent_in_the_data():
    rows = [
        row("1", "cust", True, "help", resp="2"),
        row("2", "SpotifyCares", False, "hi", parent="1"),
    ]
    turns, _ = build_index(rows)
    assert find_roots(rows, turns) == ["1"]


def test_t4_outbound_tweet_is_never_a_root():
    rows = [row("9", "SpotifyCares", False, "we are back online")]
    turns, _ = build_index(rows)
    assert find_roots(rows, turns) == []


def test_t4_parent_outside_the_sample_still_counts_as_a_root():
    # in_response_to points at 999 which is not in the file -> 1 is still a usable root.
    rows = [
        row("1", "cust", True, "follow up", resp="2", parent="999"),
        row("2", "SpotifyCares", False, "sure", parent="1"),
    ]
    turns, _ = build_index(rows)
    assert find_roots(rows, turns) == ["1"]


# --------------------------------------------------------------- T5 branch selection
def test_t5_branch_with_the_brand_reply_wins_over_a_longer_branch_without_it():
    rows = [
        row("1", "cust", True, "my music skips", resp="2,3"),
        row("2", "SpotifyCares", False, "clear your cache", parent="1"),   # short, brand
        row("3", "randomuser", True, "same here lol", resp="4", parent="1"),
        row("4", "anotheruser", True, "me too", resp="5", parent="3"),
        row("5", "yetanother", True, "+1", parent="4"),                     # long, no brand
    ]
    t = reconstruct(rows, brand="SpotifyCares")[0]
    assert [x.tweet_id for x in t.turns] == ["1", "2"]


def test_t5_longest_brand_branch_wins_among_brand_branches():
    rows = [
        row("1", "cust", True, "help", resp="2,4"),
        row("2", "SpotifyCares", False, "step one", parent="1"),
        row("4", "SpotifyCares", False, "step one alt", resp="5", parent="1"),
        row("5", "cust", True, "still broken", resp="6", parent="4"),
        row("6", "SpotifyCares", False, "step two", parent="5"),
    ]
    t = reconstruct(rows, brand="SpotifyCares")[0]
    assert [x.tweet_id for x in t.turns] == ["1", "4", "5", "6"]


def test_reconstruct_drops_unanswered_customer_tweets():
    rows = [row("1", "cust", True, "hello?")]
    assert reconstruct(rows, brand="SpotifyCares") == []


def test_reconstruct_is_deterministic():
    rows = [
        row("1", "cust", True, "help", resp="3,2"),
        row("2", "SpotifyCares", False, "a", parent="1"),
        row("3", "SpotifyCares", False, "b", parent="1"),
    ]
    a = [t.tweet_id for t in reconstruct(rows, brand="SpotifyCares")[0].turns]
    b = [t.tweet_id for t in reconstruct(rows, brand="SpotifyCares")[0].turns]
    assert a == b
