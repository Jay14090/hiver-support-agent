"""Hybrid retrieval over historical resolved threads.

BM25 (lexical) + MiniLM cosine (semantic), fused with Reciprocal Rank Fusion, then
reranked by a cheap LLM-free heuristic. Returns the FULL thread -- customer turn plus
the brand's resolution -- because the resolution is the entire point of retrieving.

Why hybrid: support text is full of exact tokens that lexical search nails and dense
search blurs (`Duo`, `Hulu`, `403`, `Autoplay`) and full of paraphrase that dense search
nails and BM25 misses ("won't play" / "no sound" / "stuck buffering"). RRF needs no
score normalisation between the two, which is why it is the standard choice.

HARD RULE: the index contains the CORPUS split only. `assert_no_leakage()` fails loudly
if a retrieved id ever appears in dev or test. Leakage here would silently inflate every
downstream number and would be invisible in the metrics.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, embed


@dataclass
class Retrieved:
    id: str
    customer_text: str
    resolution_reply: str
    created_at: str
    res_score: int
    bm25_rank: int | None
    dense_rank: int | None
    dense_sim: float
    rrf: float
    final: float

    def as_citation(self) -> str:
        return f"[{self.id}]"


def _tok(s: str) -> list:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


class HybridRetriever:
    def __init__(self, corpus: pd.DataFrame):
        self.df = corpus.reset_index(drop=True)
        self.ids = self.df["id"].astype(str).tolist()
        self.texts = self.df["text"].tolist()

        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([_tok(t) for t in self.texts])
        self.X = embed.encode(self.texts, cache_name="corpus")
        # Recency in years, for the rerank tiebreak. Product eras matter in support data:
        # a 2017 fix for an old app version is worse than a newer equivalent answer.
        ts = pd.to_datetime(self.df["created_at"], errors="coerce", utc=True)
        span = (ts.max() - ts.min()).total_seconds() or 1.0
        self.recency = ((ts - ts.min()).dt.total_seconds() / span).fillna(0.5).to_numpy()

    @classmethod
    def from_split(cls, split: str = "corpus") -> "HybridRetriever":
        p = config.DATA_PROCESSED / f"{split}.jsonl"
        return cls(pd.read_json(p, lines=True))

    def search(self, query: str, top_n: int | None = None, top_k: int | None = None) -> list:
        top_n = top_n or config.RETRIEVE_TOP_N
        top_k = top_k or config.RETRIEVE_TOP_K

        # --- lexical
        bm_scores = self.bm25.get_scores(_tok(query))
        bm_order = np.argsort(-bm_scores)[: top_n * 3]
        bm_rank = {int(i): r for r, i in enumerate(bm_order)}

        # --- dense
        q = embed.encode([query], cache_name=None)
        sims = embed.cosine_sim(q, self.X)[0]
        d_order = np.argsort(-sims)[: top_n * 3]
        d_rank = {int(i): r for r, i in enumerate(d_order)}

        # --- Reciprocal Rank Fusion: 1/(k+rank). No score normalisation needed, which is
        #     the whole reason to use RRF rather than a weighted sum of incomparable scores.
        cand = set(bm_rank) | set(d_rank)
        fused = []
        for i in cand:
            s = 0.0
            if i in bm_rank:
                s += 1.0 / (config.RRF_K + bm_rank[i])
            if i in d_rank:
                s += 1.0 / (config.RRF_K + d_rank[i])
            fused.append((i, s))
        fused.sort(key=lambda x: -x[1])
        fused = fused[:top_n]

        # --- LLM-free rerank: fusion score, then resolution quality, then recency.
        #     Weights are deliberately blunt; anything cleverer I could not defend live.
        out = []
        for i, rrf in fused:
            row = self.df.iloc[i]
            res = float(row.get("res_score", 0))
            final = rrf * 1000 + 0.35 * min(res, 6) + 0.25 * float(self.recency[i])
            out.append(
                Retrieved(
                    id=str(row["id"]),
                    customer_text=row["text"],
                    resolution_reply=row.get("resolution_reply", ""),
                    created_at=str(row.get("created_at", "")),
                    res_score=int(res),
                    bm25_rank=bm_rank.get(i),
                    dense_rank=d_rank.get(i),
                    dense_sim=float(sims[i]),
                    rrf=float(rrf),
                    final=float(final),
                )
            )
        out.sort(key=lambda r: -r.final)
        return out[:top_k]

    def max_similarity(self, hits: list) -> float:
        """Feeds the sigma retrieval gate in route.py. 0.0 when nothing was retrieved."""
        return max((h.dense_sim for h in hits), default=0.0)


_LEAK_CACHE: dict = {}


def assert_no_leakage(retrieved_ids, split_names=("dev", "test")) -> None:
    """Fail loudly if a retrieved id belongs to an evaluation split.

    Called from agent.py on every decision. Cheap (a set lookup) and it is the only
    thing standing between me and a silently inflated result.
    """
    if not _LEAK_CACHE:
        for s in split_names:
            p = config.DATA_PROCESSED / f"{s}.jsonl"
            if p.exists():
                ids = {str(json.loads(line)["id"]) for line in p.open(encoding="utf-8")}
                _LEAK_CACHE[s] = ids
    for s, ids in _LEAK_CACHE.items():
        bad = {str(i) for i in retrieved_ids} & ids
        if bad:
            raise AssertionError(
                f"RETRIEVAL LEAKAGE: {len(bad)} retrieved id(s) are in the {s} split: "
                f"{sorted(bad)[:5]}. The index must contain the corpus split only."
            )
