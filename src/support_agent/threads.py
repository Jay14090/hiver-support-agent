"""Reconstruct conversation threads from the flat twcs.csv edge list.

twcs.csv is NOT threaded. Each row carries `in_response_to_tweet_id` (a single parent)
and `response_tweet_id` (which can be a COMMA-SEPARATED LIST of children). Rebuilding
threads correctly is the first place this dataset quietly punishes carelessness, so
each trap below has a named test in tests/test_threads.py.

Traps handled here:
  T1  `response_tweet_id` may be "123,124,125" -- splitting naively orphans children.
  T2  pandas reads the id columns as float64 with NaN unless dtype is forced to str,
      which turns "123" into "123.0" and breaks every join.
  T3  cycles and self-references exist in the data; an unguarded walk hangs forever.
  T4  a thread is the maximal chain rooted at an INBOUND tweet with no parent.
  T5  threads branch. We take the branch containing the brand's replies; if several
      qualify, the longest one.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

MAX_DEPTH = 20  # cap: the longest real support threads are far shorter than this


@dataclass
class Turn:
    tweet_id: str
    author_id: str
    inbound: bool
    created_at: str
    text: str


@dataclass
class Thread:
    root_id: str
    turns: list = field(default_factory=list)

    @property
    def created_at(self) -> str:
        return self.turns[0].created_at if self.turns else ""

    @property
    def customer_id(self) -> str:
        return self.turns[0].author_id if self.turns else ""

    def brand_turns(self, brand: str) -> list:
        return [t for t in self.turns if t.author_id == brand]

    def first_customer_text(self) -> str:
        return self.turns[0].text if self.turns else ""


def _norm_id(v) -> str:
    """T2: normalise an id cell to a clean string, or '' for missing.

    Handles the float contamination ('123.0' -> '123') that pandas introduces when a
    column with NaNs is inferred as float64 despite dtype=str on read.
    """
    if v is None:
        return ""
    s = str(v).strip()
    if s in ("", "nan", "NaN", "None", "<NA>"):
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def split_children(v) -> list:
    """T1: `response_tweet_id` is sometimes a comma-separated list of child ids."""
    s = _norm_id(v)
    if not s:
        return []
    return [c for c in (_norm_id(x) for x in s.split(",")) if c]


def build_index(rows: Iterable[dict]) -> tuple[dict, dict]:
    """Return (turns_by_id, children_by_parent).

    Children come from BOTH directions -- the child's `in_response_to_tweet_id` and the
    parent's `response_tweet_id` -- because the dataset populates them inconsistently.
    Using only one side loses threads.
    """
    turns: dict = {}
    children: dict = defaultdict(list)
    seen_edges: set = set()

    for r in rows:
        tid = _norm_id(r.get("tweet_id"))
        if not tid:
            continue
        turns[tid] = Turn(
            tweet_id=tid,
            author_id=str(r.get("author_id", "")).strip(),
            inbound=str(r.get("inbound", "")).strip().lower() in ("true", "1", "yes"),
            created_at=str(r.get("created_at", "")).strip(),
            text=str(r.get("text", "")),
        )
        parent = _norm_id(r.get("in_response_to_tweet_id"))
        if parent and parent != tid:  # T3: drop self-references at the edge level
            if (parent, tid) not in seen_edges:
                children[parent].append(tid)
                seen_edges.add((parent, tid))
        for c in split_children(r.get("response_tweet_id")):
            if c != tid and (tid, c) not in seen_edges:
                children[tid].append(c)
                seen_edges.add((tid, c))

    # Drop edges pointing at tweets that are not in the file (the dataset is a sample).
    for p in list(children):
        children[p] = [c for c in children[p] if c in turns]
    return turns, dict(children)


def find_roots(rows: Iterable[dict], turns: dict) -> list:
    """T4: roots are inbound tweets whose parent is absent from the dataset."""
    has_parent = set()
    for r in rows:
        tid = _norm_id(r.get("tweet_id"))
        parent = _norm_id(r.get("in_response_to_tweet_id"))
        if tid and parent and parent in turns and parent != tid:
            has_parent.add(tid)
    return [tid for tid, t in turns.items() if t.inbound and tid not in has_parent]


def _longest_brand_chain(root: str, turns: dict, children: dict, brand: Optional[str]) -> list:
    """T5 + T3: DFS for the best branch, with a visited set and a depth cap.

    "Best" = contains a brand turn if any branch does (so we keep the branch with the
    actual resolution), then longest. Ties break on the earliest child id for
    determinism -- the same input must always give the same thread.
    """
    best: list = []
    best_score = (-1, -1)

    stack = [(root, [root], {root})]
    while stack:
        node, path, seen = stack.pop()
        kids = [c for c in children.get(node, []) if c not in seen]
        if not kids or len(path) >= MAX_DEPTH:
            has_brand = 1 if (brand and any(turns[n].author_id == brand for n in path)) else 0
            score = (has_brand, len(path))
            if score > best_score:
                best_score, best = score, path
            continue
        for c in sorted(kids):
            stack.append((c, path + [c], seen | {c}))
    return best


def reconstruct(rows: list, brand: Optional[str] = None) -> list:
    """Flat rows -> list of Thread, one per root, best branch only."""
    turns, children = build_index(rows)
    out = []
    for root in find_roots(rows, turns):
        chain = _longest_brand_chain(root, turns, children, brand)
        if len(chain) < 2:  # a customer tweet with no reply teaches us nothing
            continue
        out.append(Thread(root_id=root, turns=[turns[n] for n in chain]))
    return out
