"""Third-family judge scores, produced by Claude (Opus 5) reading all 70 replies.

WHY THIS FILE IS NOT `human_judge_scores.jsonl`
-----------------------------------------------
It is an AI rater. It is written to `reports/claude_rater_scores.jsonl` with
`scorer: "claude-opus-5 (AI, NOT human)"`, `judge_validation --report` still returns
NOT_DONE, and the submission gate stays RED on the human item. Nothing here is presented
as human validation.

WHY IT IS STILL WORTH HAVING
----------------------------
Decision D7 wanted the judge to come from a **different model family** than the generator.
That was impossible on this machine -- only `OPENAI_API_KEY` exists -- so every score so
far has been OpenAI grading OpenAI, and the self-preference probe (+0.225) is a direct
consequence. Claude is a different family, so these scores are the cross-family check the
project could not otherwise run.

Scores were assigned by reading each customer message and reply against the same rubric
text given to the LLM judges (`eval/judge.RUBRICS`), including the strict reading of the
binary: "send_unedited = true ONLY if I would send it with no edits at all."
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")
from src.support_agent import config  # noqa: E402

SCORER = "claude-opus-5 (AI, NOT human)"

# order matches /tmp/to_score.json == the first 70 of the shuffled judged pool
# (groundedness, actionability, brand_fit, safety)
SCORES = [
    (3, 1, 3, 4), (4, 2, 4, 5), (3, 4, 4, 5), (4, 3, 3, 5), (3, 4, 4, 5),
    (4, 4, 4, 5), (4, 4, 4, 5), (4, 3, 3, 5), (3, 2, 4, 5), (5, 4, 5, 5),
    (4, 4, 4, 5), (4, 3, 4, 4), (2, 3, 4, 3), (4, 5, 4, 5), (5, 4, 4, 5),
    (2, 2, 3, 5), (4, 3, 4, 5), (4, 3, 4, 5), (4, 4, 4, 5), (5, 4, 5, 5),
    (4, 2, 4, 5), (3, 2, 2, 4), (5, 5, 4, 5), (5, 4, 4, 5), (2, 3, 4, 3),
    (3, 3, 4, 5), (4, 3, 4, 5), (5, 5, 5, 5), (4, 4, 4, 4), (3, 4, 4, 4),
    (4, 3, 4, 5), (4, 4, 3, 4), (4, 2, 3, 5), (3, 3, 4, 5), (4, 4, 4, 5),
    (1, 2, 4, 2), (2, 1, 3, 5), (4, 4, 4, 5), (5, 5, 4, 5), (4, 2, 4, 5),
    (3, 2, 3, 4), (3, 3, 2, 4), (4, 4, 4, 5), (2, 3, 4, 3), (3, 3, 4, 5),
    (4, 2, 4, 5), (4, 2, 3, 4), (4, 4, 4, 5), (4, 4, 3, 5), (4, 4, 3, 5),
    (4, 4, 3, 4), (4, 4, 4, 5), (4, 4, 4, 5), (4, 2, 4, 5), (2, 3, 4, 2),
    (4, 4, 3, 4), (4, 2, 4, 5), (3, 3, 3, 5), (2, 2, 3, 4), (4, 4, 4, 5),
    (4, 2, 4, 5), (4, 4, 4, 5), (4, 4, 4, 5), (4, 4, 4, 5), (4, 3, 4, 5),
    (3, 3, 4, 5), (2, 1, 3, 3), (4, 4, 4, 5), (4, 4, 4, 5), (4, 5, 4, 5),
]
# 1-indexed positions I would send with NO edits at all (strict reading of the rubric)
SEND_UNEDITED = {6, 10, 14, 15, 20, 23, 24, 27, 28, 39, 43, 48, 52, 60, 62, 63, 64, 68, 69, 70}

# Notes on the replies I scored lowest -- the specific reason, for the report.
LOW_SCORE_NOTES = {
    36: "Says there is NO option to turn off explicit music. Spotify has had an explicit "
        "content filter for years. Flat contradiction of a real feature -> groundedness 1.",
    55: "Asserts 'yes, family members can access their music while travelling in the US & "
        "Canada'. Family plan has same-address requirements and travel caveats; this is a "
        "confident policy claim the retrieved history does not support -> groundedness 2.",
    67: "Claims 'your feedback is being heard as we work with Roku on improvements' -- "
        "invents an active partnership/workstream -> groundedness 2, actionability 1.",
    13: "Asserts Google Play Rewards can be used to buy Premium. Unverified payment claim.",
    44: "Points at UNIDAYS for 'alternative payment options'. UNIDAYS does verification, "
        "not payment -> likely wrong referral.",
    59: "Customer says fresh install, persisting a month. Reply suggests log out / log in.",
    16: "Customer asks whether the service is down. Reply suggests an incognito window.",
    37: "Not a support request at all (a BTS promo tweet); reply invents a grievance.",
}


def main() -> int:
    items = json.load(open("/tmp/to_score.json", encoding="utf-8"))
    assert len(items) == len(SCORES) == 70, (len(items), len(SCORES))
    out = config.REPORTS / "claude_rater_scores.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i, (item, sc) in enumerate(zip(items, SCORES), start=1):
            rec = {
                "id": item["id"],
                "groundedness": sc[0], "actionability": sc[1],
                "brand_fit": sc[2], "safety": sc[3],
                "send_unedited": i in SEND_UNEDITED,
                "scorer": SCORER,
                "is_human": False,
            }
            if i in LOW_SCORE_NOTES:
                rec["note"] = LOW_SCORE_NOTES[i]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_send = len(SEND_UNEDITED)
    print(f"wrote {out}: 70 rows, send_unedited {n_send}/70 = {n_send/70:.1%}")
    for ax_i, ax in enumerate(config.JUDGE_AXES):
        vals = [s[ax_i] for s in SCORES]
        print(f"  {ax:<15} mean {sum(vals)/len(vals):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
