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


def fig_taxonomy() -> None:
    """Induced clusters -> hand-edited intents, plus the intents clustering never found.

    The point of this figure is honesty about provenance: it shows at a glance which
    intents have cluster support and which rest on manual evidence.
    """
    from src.support_agent.taxonomy import CLUSTER_TO_INTENT, NOT_ISOLATED_BY_CLUSTERING

    names = json.loads((config.DATA_INTERIM / "cluster_names.json").read_text(encoding="utf-8"))
    clusters = names["clusters"]
    rows = []
    for cid, c in clusters.items():
        rows.append((c["name"], c["size"], CLUSTER_TO_INTENT.get(c["name"], "other")))
    rows.sort(key=lambda r: (r[2], -r[1]))

    intents = sorted({r[2] for r in rows}) + NOT_ISOLATED_BY_CLUSTERING
    y_intent = {name: i for i, name in enumerate(intents)}

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    palette = ["#1DB954", "#4c9be8", "#f2b134", "#d1495b", "#8e7cc3", "#5bc0be"]
    for i, (cname, size, intent) in enumerate(rows):
        y0, y1 = i, y_intent[intent]
        col = palette[y_intent[intent] % len(palette)]
        ax.plot([0, 1], [y0, y1], "-", color=col, lw=1 + size / 130, alpha=0.55, solid_capstyle="round")
        ax.text(-0.02, y0, f"{cname}  ({size})", ha="right", va="center", fontsize=8.5, color=INK)
    for name, y in y_intent.items():
        never = name in NOT_ISOLATED_BY_CLUSTERING
        ax.text(1.02, y, name + ("   (no cluster support)" if never else ""),
                ha="left", va="center", fontsize=8.5,
                color=WARN if never else INK, style="italic" if never else "normal")

    ax.set_xlim(-0.55, 1.75)
    ax.set_ylim(-1, max(len(rows), len(intents)) + 0.5)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(
        f"Induced clusters ({names['method']}) -> hand-edited intents\n"
        f"HDBSCAN found 0 clusters (100% noise); KMeans silhouette ~0.04 at every k",
        loc="left", fontsize=11, color=INK, pad=14,
    )
    fig.text(0.01, 0.015,
             "red = intent confirmed by keyword probe + manual reading, NOT by cluster structure "
             "(scripts/intent_evidence.py)", fontsize=8, color=MUTED)
    fig.tight_layout()
    out = config.FIGURES / "taxonomy.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figures] wrote {out}")


def _results() -> dict:
    p = config.REPORTS / "results.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def fig_confusion() -> None:
    """Confusion matrix for the agent, row-normalised so rare classes stay readable."""
    import numpy as np

    d = _results()
    r = d["systems"].get("agent") or list(d["systems"].values())[0]
    labels = r["intent"]["labels"]
    m = np.array(r["intent"]["confusion_matrix"], dtype=float)
    present = [i for i, l in enumerate(labels) if m[i].sum() > 0]
    m, labels = m[np.ix_(present, present)], [labels[i] for i in present]
    norm = m / np.clip(m.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    im = ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i, j]:
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=7.5,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold (support in parentheses on the diagonal)")
    ax.set_title(f"{r['system']}: intent confusion (counts, shaded by row share)",
                 loc="left", fontsize=11, pad=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label="share of the gold row")
    fig.tight_layout()
    out = config.FIGURES / "confusion_matrix.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figures] wrote {out}")


def fig_calibration() -> None:
    """Reliability diagram. The router consumes confidence, so its calibration is load-bearing."""
    d = _results()
    r = d["systems"].get("agent") or list(d["systems"].values())[0]
    bins = [b for b in r["intent"]["reliability_bins"] if b["n"] > 0]
    xs = [b["mean_confidence"] for b in bins]
    ys = [b["accuracy"] for b in bins]
    ns = [b["n"] for b in bins]

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1, label="perfect calibration")
    ax.scatter(xs, ys, s=[max(28, n * 7) for n in ns], color=ACCENT, alpha=0.85, zorder=3,
               label="observed (area = n)")
    ax.plot(xs, ys, "-", color=ACCENT, lw=1.2, alpha=0.6)
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(7, -11),
                    fontsize=7.5, color=MUTED)
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Intent calibration - ECE = {r['intent']['ece']:.3f}\n"
                 f"points below the line = overconfident",
                 loc="left", fontsize=11, pad=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    out = config.FIGURES / "calibration.png"
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"[figures] wrote {out}")


def fig_deflection() -> None:
    """Deflection vs safety, the trade-off a support-tooling company actually buys."""
    d = _results()
    systems = d["systems"]
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    palette = {"agent": ACCENT, "b1_simple": "#4c9be8", "b0_always_escalate": WARN,
               "b0_always_auto": "#f2b134"}
    for name, r in systems.items():
        c = r["routing"]
        x = c["deflection_rate"] * 100
        y = c["false_auto_count"]
        col = palette.get(name, MUTED)
        ax.scatter([x], [y], s=150, color=col, zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(f"{name}\ncost/100 = {c['expected_cost_per_100']:.0f}",
                    (x, y), textcoords="offset points", xytext=(9, 6), fontsize=8, color=INK)
    ax.set_xlabel("deflection rate (% handled with no human)  ->  cheaper")
    ax.set_ylabel("missed escalations (raw count)  ->  more dangerous")
    ax.set_title("The only trade-off that matters: deflection vs missed escalations\n"
                 "bottom-right is better; always-escalate sits at (0, 0)",
                 loc="left", fontsize=11, pad=10)
    ax.set_xlim(-4, 104)
    _style(ax)
    fig.tight_layout()
    out = config.FIGURES / "deflection_curve.png"
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"[figures] wrote {out}")


def fig_baseline_table() -> None:
    """The headline table as an image, so the report and the figure cannot disagree."""
    d = _results()
    systems = d["systems"]
    cols = ["system", "macro-F1 (95% CI)", "cost/100", "missed esc.", "deflection", "grounded"]
    rows = []
    for name, r in systems.items():
        f1 = r["intent"]["macro_f1"]
        c = r["routing"]
        rows.append([
            name,
            f"{f1['point']:.3f} [{f1['lo']:.3f}, {f1['hi']:.3f}]",
            f"{c['expected_cost_per_100']:.1f}",
            f"{c['false_auto_count']} of {c['false_auto_of']}",
            f"{c['deflection_rate']:.1%}",
            f"{r['reply_checks']['grounded_rate']:.0%}",
        ])
    fig, ax = plt.subplots(figsize=(11, 1.0 + 0.5 * len(rows)))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.55)
    for j in range(len(cols)):
        t[0, j].set_facecolor("#eef6ef")
        t[0, j].set_text_props(weight="bold")
    for i, row in enumerate(rows, start=1):
        if row[0] == "agent":
            for j in range(len(cols)):
                t[i, j].set_facecolor("#f4fbf5")
    banner = d["status"]["banner"]
    ax.set_title(f"Headline results\n{DIM_NOTE if False else banner}", loc="left",
                 fontsize=9.5, color=WARN if d["status"]["is_provisional"] else INK, pad=16)
    fig.tight_layout()
    out = config.FIGURES / "baseline_table.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figures] wrote {out}")


DIM_NOTE = ""

REGISTRY = {
    "data_funnel": fig_data_funnel,
    "taxonomy": fig_taxonomy,
    "confusion_matrix": fig_confusion,
    "calibration": fig_calibration,
    "deflection_curve": fig_deflection,
    "baseline_table": fig_baseline_table,
}


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
