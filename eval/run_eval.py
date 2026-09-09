"""The single command that produces every number in the report.

    python -m eval.run_eval --system {b0,b1,agent,all} --split golden

Rules this file enforces so the report cannot drift from reality:
  * Every headline metric carries a bootstrap 95% CI.
  * While ANY golden row is still `provisional` (no human has adjudicated it), every
    result is stamped PROVISIONAL and the stamp is written into results.json.
  * Systems are compared only through `is_meaningful_gap()`, so a difference smaller
    than the CI can never be written up as a result.
  * Determinism: temperature 0, fixed seeds, cache on. Same command twice -> byte
    identical numbers (tests/test_determinism.py asserts it).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")
from eval import metrics as M  # noqa: E402
from src.support_agent import classify, config, draft, llm, route  # noqa: E402

RESULTS = config.REPORTS / "results.json"


# --------------------------------------------------------------------------- data
def load_golden(path=None) -> list:
    p = path or (config.GOLDEN / "v1.jsonl")
    if not p.exists():
        raise SystemExit(f"missing {p} -- run scripts/build_golden.py")
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def provisional_status(golden: list) -> dict:
    n_prov = sum(1 for g in golden if g.get("provisional"))
    n_human = sum(1 for g in golden if g.get("labeler") == "human_adjudicated")
    return {
        "is_provisional": n_prov > 0,
        "provisional_rows": n_prov,
        "human_adjudicated_rows": n_human,
        "total_rows": len(golden),
        "banner": (
            f"PROVISIONAL - {n_prov}/{len(golden)} golden rows have NOT been human-adjudicated. "
            f"These numbers are measured against machine labels."
            if n_prov else
            f"Human-adjudicated: {n_human}/{len(golden)} rows reviewed by a human."
        ),
    }


# --------------------------------------------------------------------------- checks
def check_must_include(reply: str, must: list, must_not: list) -> dict:
    """Deterministic, free, and far more defensible than an LLM similarity score.

    Token-overlap matching: a `must_include` item like "acknowledge the double charge" is
    satisfied if a decent share of its content words appear in the reply. Crude, and
    deliberately so -- it is a floor check, not a semantic judgement, and the LLM judge
    covers the semantics separately.
    """
    import re

    stop = {"the", "a", "an", "to", "of", "for", "and", "or", "is", "in", "on", "with", "that"}

    def words(s):
        return {w for w in re.findall(r"[a-z]+", (s or "").lower()) if w not in stop and len(w) > 2}

    rw = words(reply)
    inc_hits = []
    for item in must or []:
        iw = words(item)
        hit = bool(iw) and len(iw & rw) / len(iw) >= 0.5
        inc_hits.append(hit)
    exc_hits = []
    for item in must_not or []:
        iw = words(item)
        hit = bool(iw) and len(iw & rw) / len(iw) >= 0.6   # stricter: avoid false alarms
        exc_hits.append(hit)
    return {
        "must_include_satisfied": sum(inc_hits),
        "must_include_total": len(inc_hits),
        "must_include_rate": (sum(inc_hits) / len(inc_hits)) if inc_hits else None,
        "must_not_violated": sum(exc_hits),
        "must_not_total": len(exc_hits),
    }


# --------------------------------------------------------------------------- core
def evaluate_system(system, golden: list, name: str, judge: bool = False,
                    judge_n: int | None = None) -> dict:
    """Run `system.handle()` over the golden set and compute everything."""
    classify.reset_stats()
    draft.reset_stats()
    route.reset_stats()

    per_example = []
    for g in golden:
        out = system.handle(g["text"], g.get("thread_context"))
        if not isinstance(out, dict):          # the agent returns a Decision dataclass
            from dataclasses import asdict

            out = asdict(out)
        checks = check_must_include(out.get("draft_reply", ""),
                                    g.get("reply_must_include"), g.get("reply_must_not_include"))
        per_example.append({
            "id": g["id"],
            "text": g["text"],
            "gold_intent": g["intent"],
            "pred_intent": out.get("intent", "other"),
            "intent_correct": out.get("intent") == g["intent"],
            "intent_confidence": float(out.get("intent_confidence", 0.0)),
            "gold_escalate": bool(g["escalate"]),
            "pred_escalate": out.get("route") == "escalate",
            "route_reason_code": out.get("route_reason_code", ""),
            "route_layer": out.get("route_layer", ""),
            "draft_reply": out.get("draft_reply", ""),
            "reply_chars": len(out.get("draft_reply", "") or ""),
            "retrieved_ids": out.get("retrieved_ids", []),
            "grounded_in": out.get("grounded_in", []),
            "used_retrieval": bool(out.get("used_retrieval")),
            "max_retrieval_sim": float(out.get("max_retrieval_sim", 0.0)),
            "difficulty": g.get("difficulty", "medium"),
            "is_hard": bool(g.get("is_hard")),
            "gold_labeler": g.get("labeler", ""),
            **checks,
        })

    res = compute_metrics(per_example, name)

    if judge:
        from eval.judge import judge_replies

        subset = per_example if judge_n is None else per_example[:judge_n]
        res["reply_quality"] = judge_replies(subset, golden)

    res["pipeline_reliability"] = {
        "classify": classify.stats(),
        "draft": draft.stats(),
        "route": route.stats(),
    }
    res["_per_example"] = per_example
    return res


def compute_metrics(pe: list, name: str) -> dict:
    labels = config.INTENTS
    y_true = [x["gold_intent"] for x in pe]
    y_pred = [x["pred_intent"] for x in pe]
    e_true = [x["gold_escalate"] for x in pe]
    e_pred = [x["pred_escalate"] for x in pe]

    macro_ci = M.bootstrap_ci(pe, lambda xs: M.macro_f1([x["gold_intent"] for x in xs],
                                                        [x["pred_intent"] for x in xs], labels))
    acc_ci = M.bootstrap_ci(pe, lambda xs: M.accuracy([x["gold_intent"] for x in xs],
                                                      [x["pred_intent"] for x in xs]))
    cost_ci = M.bootstrap_ci(pe, lambda xs: M.expected_cost_per_100(
        [x["gold_escalate"] for x in xs], [x["pred_escalate"] for x in xs]))
    esc_f1_ci = M.bootstrap_ci(pe, lambda xs: M.routing_metrics(
        [x["gold_escalate"] for x in xs], [x["pred_escalate"] for x in xs])["f1"])
    defl_ci = M.bootstrap_ci(pe, lambda xs: M.routing_metrics(
        [x["gold_escalate"] for x in xs], [x["pred_escalate"] for x in xs])["deflection_rate"])

    conf = [x["intent_confidence"] for x in pe]
    corr = [1.0 if x["intent_correct"] else 0.0 for x in pe]

    routing = M.routing_metrics(e_true, e_pred)
    routing["expected_cost_per_100_ci"] = cost_ci
    routing["f1_ci"] = esc_f1_ci
    routing["deflection_rate_ci"] = defl_ci
    routing["cost_sensitivity"] = M.cost_sensitivity(e_true, e_pred)

    hard = [x for x in pe if x["is_hard"]]
    easy = [x for x in pe if not x["is_hard"]]

    mi = [x["must_include_rate"] for x in pe if x["must_include_rate"] is not None]
    return {
        "system": name,
        "n": len(pe),
        "intent": {
            "macro_f1": macro_ci,
            "accuracy": acc_ci,
            "per_class": M.per_class_prf(y_true, y_pred, labels),
            "confusion_matrix": M.confusion_matrix(y_true, y_pred, labels).tolist(),
            "labels": labels,
            "ece": round(M.expected_calibration_error(conf, corr), 4),
            "reliability_bins": M.reliability_bins(conf, corr),
        },
        "routing": routing,
        "reply_checks": {
            "must_include_rate_mean": round(sum(mi) / len(mi), 4) if mi else None,
            "must_not_violations": sum(x["must_not_violated"] for x in pe),
            "mean_reply_chars": round(sum(x["reply_chars"] for x in pe) / max(1, len(pe)), 1),
            "empty_replies": sum(1 for x in pe if not x["draft_reply"].strip()),
            "grounded_rate": round(sum(1 for x in pe if x["grounded_in"]) / max(1, len(pe)), 4),
        },
        "slices": {
            "hard": {"n": len(hard),
                     "accuracy": round(M.accuracy([x["gold_intent"] for x in hard],
                                                  [x["pred_intent"] for x in hard]), 4) if hard else None,
                     "missed_escalations": sum(1 for x in hard if x["gold_escalate"] and not x["pred_escalate"])},
            "easy": {"n": len(easy),
                     "accuracy": round(M.accuracy([x["gold_intent"] for x in easy],
                                                  [x["pred_intent"] for x in easy]), 4) if easy else None,
                     "missed_escalations": sum(1 for x in easy if x["gold_escalate"] and not x["pred_escalate"])},
        },
    }


# --------------------------------------------------------------------------- report
def headline_table(all_results: dict, status: dict) -> str:
    rows = []
    hdr = (f"{'system':<22} {'macro-F1 (95% CI)':<26} {'cost/100':<20} "
           f"{'missed esc':<12} {'deflection':<12}")
    rows.append(hdr)
    rows.append("-" * len(hdr))
    for name, r in all_results.items():
        if not isinstance(r, dict) or "intent" not in r:
            continue
        f1 = r["intent"]["macro_f1"]
        c = r["routing"]
        rows.append(
            f"{name:<22} "
            f"{f1['point']:.3f} [{f1['lo']:.3f},{f1['hi']:.3f}]  "
            f"{c['expected_cost_per_100']:>7.1f} [{c['expected_cost_per_100_ci']['lo']:.0f},"
            f"{c['expected_cost_per_100_ci']['hi']:.0f}]  "
            f"{c['false_auto_count']:>3} of {c['false_auto_of']:<5} "
            f"{c['deflection_rate']:>9.1%}"
        )
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="all", choices=["b0", "b1", "agent", "all"])
    ap.add_argument("--split", default="golden")
    ap.add_argument("--judge", action="store_true", help="also run the LLM reply judge")
    ap.add_argument("--judge-n", type=int, default=None)
    ap.add_argument("--offline", action="store_true", help="replay cache only; no API key needed")
    ap.add_argument("--headline", action="store_true", help="print the table from results.json")
    a = ap.parse_args()

    if a.offline:
        llm.set_offline(True)

    if a.headline:
        if not RESULTS.exists():
            print("no reports/results.json yet -- run the eval first")
            return 1
        d = json.loads(RESULTS.read_text(encoding="utf-8"))
        print("\n" + "=" * 78)
        print(d["status"]["banner"])
        print("=" * 78)
        print(headline_table(d["systems"], d["status"]))
        return 0

    golden = load_golden()
    status = provisional_status(golden)
    print("=" * 78)
    print(status["banner"])
    print("=" * 78)

    import pandas as pd

    corpus = pd.read_json(config.DATA_PROCESSED / "corpus.jsonl", lines=True).to_dict("records")

    systems = {}
    want = ["b0", "b1", "agent"] if a.system == "all" else [a.system]

    if "b0" in want:
        from baselines.b0_trivial import B0Trivial

        for policy in ("always_escalate", "always_auto"):
            s = B0Trivial(routing_policy=policy).fit(corpus)
            print(f"\n[eval] b0_{policy} ...")
            systems[f"b0_{policy}"] = evaluate_system(s, golden, f"b0_{policy}")

    if "b1" in want:
        from baselines.b1_simple import B1Simple

        print("\n[eval] b1_simple (fitting TF-IDF + LogReg) ...")
        s = B1Simple().fit(corpus)
        systems["b1_simple"] = evaluate_system(s, golden, "b1_simple")

    if "agent" in want:
        from src.support_agent.agent import SupportAgent

        print("\n[eval] agent ...")
        s = SupportAgent()
        systems["agent"] = evaluate_system(s, golden, "agent", judge=a.judge, judge_n=a.judge_n)

    # ---- pairwise comparisons, gated on non-overlapping CIs
    comparisons = []
    names = list(systems)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = systems[names[i]], systems[names[j]]
            fa, fb = A["intent"]["macro_f1"], B["intent"]["macro_f1"]
            comparisons.append({
                "a": names[i], "b": names[j], "metric": "intent_macro_f1",
                "a_point": fa["point"], "b_point": fb["point"],
                "gap": round(fa["point"] - fb["point"], 4),
                "meaningful": M.is_meaningful_gap(fa, fb),
                "note": ("CIs do not overlap" if M.is_meaningful_gap(fa, fb)
                         else "CIs OVERLAP -- this gap is NOT a result and must not be claimed"),
            })

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "config": {
            "brand": config.BRAND, "generator_model": config.GENERATOR_MODEL,
            "judge_model": config.JUDGE_MODEL, "tau": config.TAU_INTENT_CONFIDENCE,
            "sigma": config.SIGMA_RETRIEVAL_SIM, "cost_false_auto": config.COST_FALSE_AUTO,
            "bootstrap_n": config.BOOTSTRAP_N, "random_state": config.RANDOM_STATE,
        },
        "systems": systems,
        "comparisons": comparisons,
        "spend_usd": round(llm.spend_so_far(), 4),
    }
    # errors go to their own file; results.json stays readable
    errs = []
    for name, r in systems.items():
        for x in r.pop("_per_example", []):
            if (not x["intent_correct"]) or (x["gold_escalate"] != x["pred_escalate"]):
                errs.append({"system": name, **x})
    (config.REPORTS / "errors.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in errs), encoding="utf-8")

    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print(headline_table(systems, status))
    print("=" * 78)
    print(f"\nwrote {RESULTS}  ({len(errs)} errors -> reports/errors.jsonl)")
    for c in comparisons:
        if not c["meaningful"]:
            print(f"  NOTE {c['a']} vs {c['b']}: gap {c['gap']:+.3f} -- {c['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
