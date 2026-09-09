"""Prune the committed LLM cache to what `make demo` actually replays.

The full cache is ~88MB because it includes one entry per bulk-labelling call: 6,791
corpus labels, 1,455 dev labels, ~3,100 golden pre-labels and 1,456 stratification calls.
None of those are replayed by `make demo` -- their *outputs* are already committed as data
(`data/processed/corpus.jsonl`, `golden/*.jsonl`), so the demo path never needs to make
them again.

What `make demo` DOES replay: classify / draft / route (the agent's 220 decisions) and
judge_* / judge_compliance (reply scoring). Those stay.

Consequence, stated in the README: `make full` will re-issue the labelling calls against
a live API rather than replaying them. That is what `make full` is for.

Run with --dry-run first; it prints exactly what would go.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

sys.path.insert(0, ".")
from src.support_agent import config  # noqa: E402

# Tags whose cache entries `make demo` replays. Everything else is prunable.
KEEP_TAGS = {
    "classify", "draft", "route",
    "judge_groundedness", "judge_actionability", "judge_brand_fit", "judge_safety",
    "judge_send_unedited", "judge_compliance",
    "test",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    files = list(config.CACHE_DIR.glob("*.json"))
    keep_n = keep_b = drop_n = drop_b = 0
    tags = Counter()
    to_drop = []
    for f in files:
        size = f.stat().st_size
        try:
            tag = json.loads(f.read_text(encoding="utf-8")).get("tag", "")
        except (json.JSONDecodeError, OSError):
            tag = ""
        tags[tag] += 1
        if tag in KEEP_TAGS:
            keep_n += 1
            keep_b += size
        else:
            drop_n += 1
            drop_b += size
            to_drop.append(f)

    print("cache entries by tag:")
    for t, n in tags.most_common():
        mark = "KEEP" if t in KEEP_TAGS else "drop"
        print(f"  {mark}  {t or '(none)':<24} {n:>6}")
    print(f"\nkeep: {keep_n:>6} files, {keep_b/1024**2:>7.1f} MB")
    print(f"drop: {drop_n:>6} files, {drop_b/1024**2:>7.1f} MB")

    if a.dry_run:
        print("\n(dry run -- nothing deleted)")
        return 0
    for f in to_drop:
        f.unlink()
    print(f"\ndeleted {drop_n} cache entries; {keep_n} remain for `make demo`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
