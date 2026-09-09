"""Every figure in the report, regenerated from committed artifacts.

Rule: a figure may only read from `data/interim/funnel.json`, `reports/results.json`
or another committed file. No figure is allowed to recompute a metric -- if a number
appears in a chart it must already exist in results.json, so the report and the plots
can never disagree.
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")  # headless: `make demo` must work over SSH / in CI
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
from src.support_agent import config  # noqa: E402

INK = "#1d1d1f"
ACCENT = "#1DB954"   # Spotify green -- the brand under study
MUTED = "#9aa0a6"
WARN = "#d1495b"


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def fig_data_funnel() -> None:
    """2.8M tweets -> ... -> splits. One image that shows I understand my own data."""
    f = json.loads((config.DATA_INTERIM / "funnel.json").read_text(encoding="utf-8"))
    stages = [
        ("All tweets in twcs.csv", f["total_tweets"]),
        (f"Tweets by {f['brand']}", f["brand_tweets"]),
        ("Brand conversation neighbourhood", f["neighbourhood_tweets"]),
        ("Threads reconstructed", f["threads_reconstructed"]),
        ("Resolution-bearing", f["threads_resolution_bearing"]),
        ("Clean / English / deduped", f["threads_clean"]),
    ]
    labels = [s[0] for s in stages]
    vals = [s[1] for s in stages]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [MUTED, MUTED, MUTED, ACCENT, WARN, ACCENT]
    bars = ax.barh(range(len(vals)), vals, color=colors, height=0.62)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("count (log scale)")
    ax.set_title(
        f"Data funnel: {f['brand']}   "
        f"(resolution-bearing rate {f['resolution_rate']:.1%})",
        color=INK, fontsize=12, loc="left", pad=12,
    )
    for i, (b, v) in enumerate(zip(bars, vals)):
        pct = v / vals[0] * 100
        ax.text(v * 1.15, i, f"{v:,}  ({pct:.2f}% of all)", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(right=vals[0] * 12)
    _style(ax)

    drop = f.get("dropped", {})
    note = (
        f"splits (time-based): corpus {f['split_corpus']:,} / dev {f['split_dev']:,} / test {f['split_test']:,}"
        f"\ncleaning drops: " + ", ".join(f"{k} {v}" for k, v in drop.items())
        + "\nthe resolution-bearing filter is the main survivorship-bias source (see REPORT.md)"
    )
    fig.text(0.012, -0.06, note, fontsize=8, color=MUTED, va="top")
    fig.tight_layout()
    out = config.FIGURES / "data_funnel.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figures] wrote {out}")


REGISTRY = {"data_funnel": fig_data_funnel}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="generate a single figure by name")
    a = ap.parse_args()
    names = [a.only] if a.only else list(REGISTRY)
    made, skipped = 0, []
    for n in names:
        try:
            REGISTRY[n]()
            made += 1
        except FileNotFoundError as e:
            # Not a silent failure: say which artifact is missing and keep going.
            skipped.append(f"{n} (missing input: {e})")
    for s in skipped:
        print(f"[figures] SKIPPED {s}")
    print(f"[figures] {made} figure(s) written, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
