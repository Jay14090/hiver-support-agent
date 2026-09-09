"""Tune the router's confidence gates (tau, sigma) ON THE DEV SPLIT ONLY.

Tuning on the golden set would make every reported number optimistic by construction --
the thresholds would be fit to the same 220 examples the results are computed on. So dev
(the middle 15% by time) is used, and the golden set is never touched here.

Labels for dev come from the same machine Pass-A procedure as the golden set. That is a
real limitation: the thresholds are tuned against machine labels, so they inherit the
labeller's escalation boundary -- the same boundary that Pass A and Pass B disagreed
about by 7 percentage points. Stated in the report.

Objective: expected cost per 100 messages under the 10:1 asymmetry (D10), not accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from eval.metrics import expected_cost_per_100, routing_metrics  # noqa: E402
from src.support_agent import classify, config, llm, route  # noqa: E402
from src.support_agent.codebook import load_codebook_text  # noqa: E402
from src.support_agent.retrieval import HybridRetriever  # noqa: E402

TAUS = [0.0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
SIGMAS = [0.0, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
CACHE = config.DATA_INTERIM / "dev_router_inputs.json"
_lock = threading.Lock()


def collect_dev_inputs(n: int) -> list:
    """Run classify + retrieve + the rule/model router layers ONCE per dev message.

    tau and sigma are applied afterwards, arithmetically -- so a 12x9 grid costs zero
    extra API calls instead of 108 full passes over the dev set.
    """
    if CACHE.exists():
        rows = json.loads(CACHE.read_text(encoding="utf-8"))
        if len(rows) >= n:
            print(f"[tune] reusing {len(rows)} cached dev inputs")
            return rows[:n]

    dev = pd.read_json(config.DATA_PROCESSED / "dev.jsonl", lines=True)
    labels_p = config.GOLDEN / "dev_labels.jsonl"
    if not labels_p.exists():
        raise SystemExit("need golden/dev_labels.jsonl -- run scripts/label_dev.py first")
    lab = {json.loads(l)["id"]: json.loads(l) for l in labels_p.open(encoding="utf-8")}

    dev = dev[dev["id"].astype(str).isin(lab)].head(n)
    rows = dev.to_dict("records")
    r = HybridRetriever.from_split("corpus")
    cb = load_codebook_text()
    print(f"[tune] collecting router inputs for {len(rows)} dev messages")

    done = [0]

    def work(row):
        hits = r.search(row["text"])
        ir = classify.classify(row["text"], row.get("thread_context"), cb)
        # Layers 1 and 2 only; the gates are applied later during the sweep.
        rd = route.route(row["text"], row.get("thread_context"), intent=ir.intent,
                         intent_confidence=1.0, max_retrieval_sim=1.0)
        with _lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  ... {done[0]}/{len(rows)}  spend=${llm.spend_so_far():.3f}")
        g = lab[str(row["id"])]
        return {
            "id": str(row["id"]),
            "gold_escalate": bool(g["escalate"]),
            "gold_intent": g["intent"],
            "pred_intent": ir.intent,
            "intent_confidence": ir.confidence,
            "max_sim": r.max_similarity(hits),
            "pre_gate_escalate": rd.route == "escalate",
            "pre_gate_layer": rd.layer,
        }

    out = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(work, rows))
    llm.flush_ledger()
    CACHE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def apply_gates(rows, tau: float, sigma: float) -> list:
    """The gates are pure arithmetic on top of the cached pre-gate decisions."""
    return [r["pre_gate_escalate"] or r["intent_confidence"] < tau or r["max_sim"] < sigma
            for r in rows]


def sweep(rows) -> dict:
    gold = [r["gold_escalate"] for r in rows]
    grid = np.zeros((len(TAUS), len(SIGMAS)))
    best = None
    for i, tau in enumerate(TAUS):
        for j, sigma in enumerate(SIGMAS):
            pred = apply_gates(rows, tau, sigma)
            cost = expected_cost_per_100(gold, pred)
            grid[i, j] = cost
            rm = routing_metrics(gold, pred)
            cand = {"tau": tau, "sigma": sigma, "cost_per_100": round(cost, 3),
                    "deflection_rate": rm["deflection_rate"],
                    "false_auto": rm["false_auto"], "false_escalate": rm["false_escalate"],
                    "escalate_f1": rm["f1"]}
            if best is None or cost < best["cost_per_100"]:
                best = cand
    return {"grid": grid.tolist(), "taus": TAUS, "sigmas": SIGMAS, "best": best,
            "n_dev": len(rows), "gold_escalate_rate": round(sum(gold) / len(gold), 4)}


def plot(res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.array(res["grid"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    im = ax1.imshow(grid, cmap="RdYlGn_r", aspect="auto")
    ax1.set_xticks(range(len(res["sigmas"])), [f"{s:.2f}" for s in res["sigmas"]], fontsize=8)
    ax1.set_yticks(range(len(res["taus"])), [f"{t:.2f}" for t in res["taus"]], fontsize=8)
    ax1.set_xlabel("sigma (min retrieval similarity)")
    ax1.set_ylabel("tau (min intent confidence)")
    bi = res["taus"].index(res["best"]["tau"])
    bj = res["sigmas"].index(res["best"]["sigma"])
    ax1.plot(bj, bi, "o", ms=15, mfc="none", mec="black", mew=2.2)
    ax1.set_title(f"Expected cost per 100 messages (DEV, n={res['n_dev']})\n"
                  f"best: tau={res['best']['tau']}, sigma={res['best']['sigma']} "
                  f"-> {res['best']['cost_per_100']:.1f}", loc="left", fontsize=10.5)
    fig.colorbar(im, ax=ax1, label="cost / 100 (lower is better)")

    # The trade-off curve at the chosen sigma
    rows = json.loads(CACHE.read_text(encoding="utf-8"))
    gold = [r["gold_escalate"] for r in rows]
    defl, missed = [], []
    for tau in res["taus"]:
        pred = apply_gates(rows, tau, res["best"]["sigma"])
        rm = routing_metrics(gold, pred)
        defl.append(rm["deflection_rate"] * 100)
        missed.append(rm["false_auto"])
    ax2.plot(defl, missed, "-o", color="#1DB954", ms=5)
    for t, x, y in zip(res["taus"], defl, missed):
        ax2.annotate(f"{t:.2f}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax2.set_xlabel("deflection rate (%)")
    ax2.set_ylabel("missed escalations (count)")
    ax2.set_title(f"Safety/deflection trade-off as tau moves (sigma={res['best']['sigma']})",
                  loc="left", fontsize=10.5)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    out = config.FIGURES / "threshold_sweep.png"
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"[tune] wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="dev messages to tune on")
    a = ap.parse_args()
    rows = collect_dev_inputs(a.n)
    res = sweep(rows)
    (config.REPORTS / "threshold_sweep.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    plot(res)
    print(json.dumps(res["best"], indent=2))
    print(f"\n-> set TAU_INTENT_CONFIDENCE = {res['best']['tau']} and "
          f"SIGMA_RETRIEVAL_SIM = {res['best']['sigma']} in config.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
