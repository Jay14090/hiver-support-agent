"""Measure the resolution-bearing filter instead of just asserting it works.

The filter decides what data exists, so an unmeasured filter is an unmeasured dataset.
This draws a RANDOM sample from the population the filter actually runs on (all threads
with >=1 brand reply), has an independent model adjudicate each reply against a written
definition, and reports precision / recall / F1 / Cohen's kappa of my heuristic against
that adjudication.

Honesty note: the adjudicator is a MODEL, not a human. This is a machine-vs-machine
agreement statistic, and it is described as exactly that in the report. It bounds how
much of the filter's behaviour is defensible; it is not ground truth.
"""
from __future__ import annotations

import json
import random
import sys

sys.path.insert(0, ".")
from src.support_agent import config, llm  # noqa: E402
from src.support_agent.ingest import _brand_neighbourhood, load_raw  # noqa: E402
from src.support_agent.resolution import score_reply  # noqa: E402
from src.support_agent.threads import reconstruct  # noqa: E402

N = 100
OUT = config.REPORTS / "resolution_filter_validation.json"

SYSTEM = """You are auditing a customer-support dataset. For a single public reply from
a brand's support account, decide whether it is RESOLUTION-BEARING.

RESOLUTION-BEARING = the reply contains actionable content that could help the customer:
  - concrete troubleshooting steps to perform, OR
  - a link to a support/help article or an official idea/community page, OR
  - an explanation of a policy, a process, or why something is happening, OR
  - confirmation that the brand has taken an action that resolves the issue.

NOT RESOLUTION-BEARING = the reply is purely:
  - a handoff ("DM us", "we've sent you a DM", "we'll take a look backstage"), OR
  - an acknowledgement or apology with no content ("sorry to hear that!", "oh no!"), OR
  - an information-gathering question with no suggested action, OR
  - generic chit-chat.

A reply that gives a real step AND also asks them to DM is still RESOLUTION-BEARING.

Return JSON only: {"resolution_bearing": true|false, "why": "<10 words>"}"""


def main() -> None:
    random.seed(config.RANDOM_STATE)
    df = load_raw()
    sub = _brand_neighbourhood(df, config.BRAND)
    del df
    threads = reconstruct(sub.to_dict("records"), brand=config.BRAND)
    threads = [t for t in threads if t.brand_turns(config.BRAND)]

    # Take each thread's best-scoring brand reply -- the one the filter's decision rests on.
    population = []
    for t in threads:
        best_s, best_t = -99, ""
        for x in t.brand_turns(config.BRAND):
            s, _ = score_reply(x.text)
            if s > best_s:
                best_s, best_t = s, x.text
        population.append((best_t, best_s))

    sample = random.sample(population, N)
    rows, tp = [], 0
    fp = fn = tn = 0
    for i, (text, score) in enumerate(sample, 1):
        mine = score >= 2
        parsed, raw = llm.complete_json(
            f"Brand reply:\n\"\"\"{text}\"\"\"",
            system=SYSTEM,
            model=config.JUDGE_MODEL,
            max_tokens=80,
            tag="resolution_filter_validation",
        )
        theirs = bool(parsed.get("resolution_bearing")) if parsed else None
        if theirs is None:
            print(f"[{i}] PARSE FAIL -- counted as a disagreement")
            theirs = not mine
        rows.append({"text": text, "heuristic_score": score, "heuristic": mine,
                     "adjudicator": theirs, "why": (parsed or {}).get("why", "")})
        if mine and theirs:
            tp += 1
        elif mine and not theirs:
            fp += 1
        elif not mine and theirs:
            fn += 1
        else:
            tn += 1
        if i % 20 == 0:
            print(f"  ... {i}/{N}")

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    agree = (tp + tn) / N
    # Cohen's kappa for two binary raters
    p_yes_a, p_yes_b = (tp + fp) / N, (tp + fn) / N
    pe = p_yes_a * p_yes_b + (1 - p_yes_a) * (1 - p_yes_b)
    kappa = (agree - pe) / max(1e-9, 1 - pe)

    res = {
        "n": N,
        "adjudicator_model": config.JUDGE_MODEL,
        "adjudicator_is_human": False,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "raw_agreement": round(agree, 3),
        "cohens_kappa": round(kappa, 3),
        "heuristic_positive_rate": round((tp + fp) / N, 3),
        "adjudicator_positive_rate": round((tp + fn) / N, 3),
        "examples": rows,
    }
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
