"""Every number the report can quote. Pure functions, no I/O, no model calls.

Two principles:
  1. **Nothing is reported without an interval.** With n=220 the 95% CI on macro-F1 is
     roughly +/-0.06. Any gap between systems smaller than that is not a result, and
     `is_meaningful_gap()` exists so the report cannot accidentally claim one.
  2. **Routing is a decision, not a classification.** `expected_cost_per_100` under the
     10:1 asymmetry is the primary routing number; precision/recall are supporting.
"""
from __future__ import annotations

import numpy as np

from src.support_agent import config


# --------------------------------------------------------------------------- intent
def confusion_matrix(y_true, y_pred, labels) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m


def per_class_prf(y_true, y_pred, labels) -> dict:
    out = {}
    for l in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == l and p == l)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != l and p == l)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == l and p != l)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[l] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                  "support": tp + fn}
    return out


def macro_f1(y_true, y_pred, labels=None) -> float:
    """Macro over classes PRESENT IN y_true.

    Averaging over classes with zero support would silently drag the mean toward 0 and
    make the number depend on the taxonomy size rather than on performance.
    """
    labels = labels or sorted(set(y_true))
    present = [l for l in labels if any(t == l for t in y_true)]
    pc = per_class_prf(y_true, y_pred, present)
    return float(np.mean([pc[l]["f1"] for l in present])) if present else 0.0


def accuracy(y_true, y_pred) -> float:
    return float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else 0.0


# --------------------------------------------------------------------------- routing
def routing_metrics(y_true_esc, y_pred_esc) -> dict:
    """`escalate` is the positive class.

    The number that matters is `false_auto` in RAW COUNTS: cases we auto-handled that a
    human should have seen. "3 of 220" lands harder and more honestly than "1.4%".
    """
    tp = sum(1 for t, p in zip(y_true_esc, y_pred_esc) if t and p)
    fp = sum(1 for t, p in zip(y_true_esc, y_pred_esc) if not t and p)
    fn = sum(1 for t, p in zip(y_true_esc, y_pred_esc) if t and not p)
    tn = sum(1 for t, p in zip(y_true_esc, y_pred_esc) if not t and not p)
    n = max(1, len(y_true_esc))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "true_escalate": tp, "false_escalate": fp, "false_auto": fn, "true_auto": tn,
        # THE headline safety number, in raw counts and as a rate.
        "false_auto_count": fn,
        "false_auto_of": len(y_true_esc),
        "false_auto_rate": round(fn / n, 4),
        "deflection_rate": round((tn + fn) / n, 4),   # fraction handled without a human
        "expected_cost_per_100": round(expected_cost_per_100(y_true_esc, y_pred_esc), 3),
    }


def expected_cost_per_100(y_true_esc, y_pred_esc,
                          c_false_auto: float = config.COST_FALSE_AUTO,
                          c_false_esc: float = config.COST_FALSE_ESCALATE) -> float:
    """D10: letting a should-escalate through costs 10x a needless escalation.

    The ratio is ASSERTED, not measured -- see the sensitivity curve in the report.
    """
    if not y_true_esc:
        return 0.0
    fp = sum(1 for t, p in zip(y_true_esc, y_pred_esc) if not t and p)
    fn = sum(1 for t, p in zip(y_true_esc, y_pred_esc) if t and not p)
    return (fn * c_false_auto + fp * c_false_esc) / len(y_true_esc) * 100


def cost_sensitivity(y_true_esc, y_pred_esc, ratios=(1, 2, 5, 10, 20, 50)) -> dict:
    """How the verdict moves if my asserted 10:1 ratio is wrong."""
    return {str(r): round(expected_cost_per_100(y_true_esc, y_pred_esc, c_false_auto=float(r),
                                                c_false_esc=1.0), 3) for r in ratios}


# --------------------------------------------------------------------------- calibration
def expected_calibration_error(confidences, correct, n_bins: int = 10) -> float:
    """ECE. The router consumes `intent_confidence`, so its calibration is load-bearing.

    A model that says 0.9 and is right 60% of the time will silently break any
    confidence gate built on top of it.
    """
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(conf)) * abs(corr[m].mean() - conf[m].mean())
    return float(ece)


def reliability_bins(confidences, correct, n_bins: int = 10) -> list:
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        out.append({
            "bin_lo": round(float(lo), 3), "bin_hi": round(float(hi), 3),
            "n": int(m.sum()),
            "mean_confidence": round(float(conf[m].mean()), 4) if m.sum() else None,
            "accuracy": round(float(corr[m].mean()), 4) if m.sum() else None,
        })
    return out


# --------------------------------------------------------------------------- uncertainty
def bootstrap_ci(items, statistic, n_boot: int = config.BOOTSTRAP_N,
                 ci: float = config.BOOTSTRAP_CI, seed: int = config.RANDOM_STATE) -> dict:
    """Percentile bootstrap over the golden set.

    `items` is a list of per-example records; `statistic` maps a resample to a float.
    Resampling EXAMPLES (not predictions) is what makes this a CI on the population the
    golden set is drawn from.
    """
    rng = np.random.default_rng(seed)
    n = len(items)
    if n == 0:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0, "n_boot": 0, "half_width": 0.0}
    point = statistic(items)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = statistic([items[i] for i in idx])
    lo = float(np.percentile(stats, (1 - ci) / 2 * 100))
    hi = float(np.percentile(stats, (1 + ci) / 2 * 100))
    return {
        "point": round(float(point), 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "half_width": round((hi - lo) / 2, 4),
        "n_boot": n_boot,
    }


def is_meaningful_gap(ci_a: dict, ci_b: dict) -> bool:
    """True only if the two 95% intervals do not overlap.

    Deliberately conservative. Non-overlapping CIs imply significance; overlapping ones
    do NOT imply the absence of it -- but for this report I refuse to claim a gap I
    cannot see at a glance, and I say so rather than reaching for a paired test I did
    not pre-register.
    """
    return ci_a["lo"] > ci_b["hi"] or ci_b["lo"] > ci_a["hi"]


# --------------------------------------------------------------------------- agreement
def cohens_kappa(a, b) -> float:
    """Two raters, categorical labels. Used for A-vs-B prelabel reliability."""
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return 0.0
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def spearman_rho(x, y) -> float:
    """Rank correlation, with average ranks for ties (scores are 1-5, ties are the norm)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0
    rx, ry = _rankdata(x), _rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ranks within tied groups
    vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            m = inv == i
            ranks[m] = ranks[m].mean()
    return ranks


def krippendorff_alpha_interval(ratings) -> float:
    """Krippendorff's alpha for interval data, two coders, no missing values.

    `ratings` = list of (coder_a_score, coder_b_score). Reported alongside Spearman
    because alpha is chance-corrected and rho is not -- two coders can correlate
    strongly while systematically disagreeing on the absolute level.
    """
    pairs = [(float(a), float(b)) for a, b in ratings if a is not None and b is not None]
    n = len(pairs)
    if n < 2:
        return 0.0
    # observed disagreement
    do = np.mean([(a - b) ** 2 for a, b in pairs])
    # expected disagreement across all values from both coders
    vals = np.array([v for p in pairs for v in p], dtype=float)
    N = len(vals)
    de = sum((vals[i] - vals[j]) ** 2 for i in range(N) for j in range(N) if i != j) / (N * (N - 1))
    return float(1 - do / de) if de > 0 else 0.0
