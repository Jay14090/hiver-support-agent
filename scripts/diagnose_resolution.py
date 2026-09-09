"""Diagnostic: is the 12.2% resolution rate real, or is my scorer under-firing?

Prints a random sample of brand replies at each score band so I can eyeball whether
score==1 replies are genuinely non-resolutions. Run before taking the D3 fallback --
switching brands to escape a bug in my own filter would be the wrong lesson.
"""
from __future__ import annotations

import random
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, ".")
from src.support_agent import config  # noqa: E402
from src.support_agent.ingest import _brand_neighbourhood, load_raw  # noqa: E402
from src.support_agent.resolution import score_reply  # noqa: E402
from src.support_agent.threads import reconstruct  # noqa: E402

random.seed(config.RANDOM_STATE)


def main(brand: str, show: int = 6) -> None:
    df = load_raw()
    n_brand = int((df["author_id"] == brand).sum())
    sub = _brand_neighbourhood(df, brand)
    del df
    threads = reconstruct(sub.to_dict("records"), brand=brand)
    threads = [t for t in threads if t.brand_turns(brand)]

    bands: dict = {}
    scores = []
    for t in threads:
        best, best_txt = -99, ""
        for x in t.brand_turns(brand):
            s, _ = score_reply(x.text)
            if s > best:
                best, best_txt = s, x.text
        scores.append(best)
        bands.setdefault(best, []).append(best_txt)

    c = Counter(scores)
    total = len(scores)
    print(f"\n===== {brand} =====")
    print(f"brand tweets: {n_brand:,}   threads with a brand reply: {total:,}")
    print("score distribution (best reply per thread):")
    cum = 0
    for s in sorted(c, reverse=True):
        cum += c[s]
        print(f"  score {s:>3}: {c[s]:>6,}  ({c[s]/total:6.1%})   cumulative>={s}: {cum/total:6.1%}")

    for s in (3, 2, 1, 0):
        if s not in bands:
            continue
        print(f"\n--- sample of threads whose BEST brand reply scores {s} ---")
        for txt in random.sample(bands[s], min(show, len(bands[s]))):
            print(f"  * {txt[:200]!r}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "SpotifyCares")
