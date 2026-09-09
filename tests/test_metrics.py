"""Metrics are the product here, so they get tested against hand-computed values."""
import numpy as np
import pytest

from eval import metrics as M
from src.support_agent import config


# --------------------------------------------------------------------------- intent
def test_per_class_prf_hand_computed():
    y = ["a", "a", "b", "b", "c"]
    p = ["a", "b", "b", "b", "a"]
    r = M.per_class_prf(y, p, ["a", "b", "c"])
    # a: tp=1 fp=1 fn=1 -> P=.5 R=.5 F1=.5
    assert r["a"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    # b: tp=2 fp=1 fn=0 -> P=2/3 R=1
    assert r["b"]["recall"] == 1.0 and abs(r["b"]["precision"] - 0.6667) < 1e-3
    # c: never predicted -> all zero, support 1
    assert r["c"]["f1"] == 0.0 and r["c"]["support"] == 1


def test_macro_f1_ignores_classes_with_no_support():
    # `zzz` has no support in y_true, so including it must not drag the mean down.
    y = ["a", "a", "b"]
    p = ["a", "a", "b"]
    assert M.macro_f1(y, p, ["a", "b", "zzz"]) == 1.0


def test_confusion_matrix_shape_and_counts():
    m = M.confusion_matrix(["a", "b", "a"], ["a", "a", "b"], ["a", "b"])
    assert m.shape == (2, 2)
    assert m[0, 0] == 1 and m[0, 1] == 1 and m[1, 0] == 1


# --------------------------------------------------------------------------- routing
def test_routing_metrics_hand_computed():
    #            TP     FN      FP      TN
    truth = [True,  True,  False, False]
    pred = [True,  False, True,  False]
    r = M.routing_metrics(truth, pred)
    assert r["true_escalate"] == 1 and r["false_auto"] == 1
    assert r["false_escalate"] == 1 and r["true_auto"] == 1
    assert r["precision"] == 0.5 and r["recall"] == 0.5
    assert r["false_auto_count"] == 1 and r["false_auto_of"] == 4


def test_expected_cost_uses_the_10_to_1_asymmetry():
    # one missed escalation among 10 == 10 cost units == 100 per 100 messages
    truth = [True] + [False] * 9
    pred = [False] * 10
    assert M.expected_cost_per_100(truth, pred) == pytest.approx(100.0)
    # one needless escalation among 10 == 1 cost unit == 10 per 100
    truth2 = [False] * 10
    pred2 = [True] + [False] * 9
    assert M.expected_cost_per_100(truth2, pred2) == pytest.approx(10.0)


def test_always_escalate_is_a_strong_safety_baseline():
    """The degenerate always-escalate policy must score ZERO missed escalations.

    If the agent cannot beat this on expected cost, that is the finding, and the report
    has to say so rather than quietly omitting the comparison.
    """
    truth = [True, False, True, False, False]
    always_esc = [True] * 5
    r = M.routing_metrics(truth, always_esc)
    assert r["false_auto"] == 0
    assert r["deflection_rate"] == 0.0
    assert r["expected_cost_per_100"] == pytest.approx(3 / 5 * 100)  # 3 needless escalations


def test_cost_sensitivity_is_monotonic_in_the_ratio():
    truth = [True, False, False]
    pred = [False, True, False]
    s = M.cost_sensitivity(truth, pred)
    vals = [s[k] for k in ("1", "2", "5", "10", "20", "50")]
    assert vals == sorted(vals), "raising the false-auto penalty cannot lower total cost"


# --------------------------------------------------------------------------- calibration
def test_ece_is_zero_for_a_perfectly_calibrated_model():
    conf = [0.95] * 100
    correct = [1] * 95 + [0] * 5
    assert M.expected_calibration_error(conf, correct) < 0.02


def test_ece_catches_overconfidence():
    conf = [0.99] * 100
    correct = [1] * 50 + [0] * 50   # says 99%, right 50% of the time
    assert M.expected_calibration_error(conf, correct) > 0.4


def test_reliability_bins_cover_everything():
    bins = M.reliability_bins([0.05, 0.5, 0.95], [1, 0, 1], n_bins=10)
    assert sum(b["n"] for b in bins) == 3


# --------------------------------------------------------------------------- uncertainty
def test_bootstrap_ci_brackets_the_point_estimate():
    items = [1] * 70 + [0] * 30
    ci = M.bootstrap_ci(items, lambda xs: float(np.mean(xs)), n_boot=400)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert abs(ci["point"] - 0.70) < 1e-9
    assert 0.02 < ci["half_width"] < 0.12   # n=100 -> roughly +/-0.09


def test_bootstrap_is_deterministic_given_the_seed():
    items = list(range(50))
    f = lambda xs: float(np.mean(xs))  # noqa: E731
    assert M.bootstrap_ci(items, f, n_boot=200) == M.bootstrap_ci(items, f, n_boot=200)


def test_meaningful_gap_requires_non_overlapping_intervals():
    a = {"point": 0.80, "lo": 0.75, "hi": 0.85}
    b = {"point": 0.60, "lo": 0.55, "hi": 0.65}
    c = {"point": 0.78, "lo": 0.72, "hi": 0.84}
    assert M.is_meaningful_gap(a, b)
    assert not M.is_meaningful_gap(a, c), "overlapping CIs must never be called a result"


def test_empty_input_does_not_crash():
    assert M.bootstrap_ci([], lambda xs: 0.0)["point"] == 0.0
    assert M.macro_f1([], []) == 0.0
    assert M.expected_cost_per_100([], []) == 0.0


# --------------------------------------------------------------------------- agreement
def test_cohens_kappa_endpoints():
    assert M.cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
    # chance-level agreement -> kappa near 0
    a = ["a", "b"] * 50
    b = ["a", "a", "b", "b"] * 25
    assert abs(M.cohens_kappa(a, b)) < 0.15


def test_spearman_handles_ties_and_direction():
    assert M.spearman_rho([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert M.spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert abs(M.spearman_rho([1, 1, 2, 2], [3, 3, 5, 5]) - 1.0) < 1e-9


def test_krippendorff_alpha_endpoints():
    assert M.krippendorff_alpha_interval([(3, 3), (4, 4), (5, 5), (1, 1)]) == pytest.approx(1.0)
    # systematic +2 offset: correlates perfectly but agrees poorly -> alpha well below rho
    ratings = [(1, 3), (2, 4), (3, 5), (1, 3), (2, 4)]
    alpha = M.krippendorff_alpha_interval(ratings)
    rho = M.spearman_rho([r[0] for r in ratings], [r[1] for r in ratings])
    assert rho > 0.9 and alpha < rho, "alpha must punish a constant offset that rho ignores"
