"""B1 -- the real bar. Classical NLP, no LLM anywhere.

  intent  : TF-IDF (word 1-2 grams + char 3-5 grams) -> LogisticRegression
  reply   : BM25 over historical customer messages, take the top-1 nearest thread, and
            COPY that thread's brand reply verbatim
  routing : hand-written keyword rules straight from the escalation policy

The reply baseline is the interesting one. Copying a real human reply from a similar past
thread is a *strong* grounded-reply baseline: it is guaranteed to be real brand voice,
guaranteed to contain no hallucinated policy, and it costs nothing. It should embarrass a
careless LLM implementation, and if it beats the agent on groundedness that is one of the
most interesting things this project could report -- so it is reported, not buried.

Training labels: B1 is fit on the CORPUS split, whose labels come from the same machine
pass. That means B1 learns to imitate the labeller, which is a real limitation and is
stated in the report -- it is not a classifier trained on human ground truth.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

sys.path.insert(0, ".")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.support_agent import config  # noqa: E402
from src.support_agent.route import DETERMINISTIC_RULES  # noqa: E402

# Keyword router: the escalation policy written out as regexes, nothing learned.
KEYWORD_RULES = list(DETERMINISTIC_RULES) + [
    ("explicit_cancellation_threat",
     r"\b(cancel(ling|ing)? my (subscription|premium|account)|unsubscribe|"
     r"switching to (apple|tidal|deezer|youtube)|done with (you|spotify))\b"),
    ("requires_account_specific_pii",
     r"\b(my account|my email|my username|my payment method|my card|my invoice|"
     r"my billing)\b"),
    ("troubleshooting_already_failed",
     r"\b(already tried|tried (that|this|everything|all of that)|still (not|doesn'?t|won'?t)|"
     r"none of (that|this) work|reinstalled? (twice|again|multiple))\b"),
]
_KRULES = [(c, re.compile(p, re.I)) for c, p in KEYWORD_RULES]


class B1Simple:
    name = "b1_simple"

    def __init__(self):
        self.clf = None
        self.vec = None
        self.bm25 = None
        self.corpus_rows = None

    # ------------------------------------------------------------------ fit
    def fit(self, rows) -> "B1Simple":
        from rank_bm25 import BM25Okapi
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion

        self.corpus_rows = rows
        texts = [r["text"] for r in rows]
        labels = [r.get("intent", "other") for r in rows]

        # Word n-grams catch topic; char n-grams catch the misspellings and elongations
        # ("skippinggg", "cancle") that are everywhere in tweets and destroy word features.
        self.vec = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                     strip_accents="unicode")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                     sublinear_tf=True)),
        ])
        X = self.vec.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                                      random_state=config.RANDOM_STATE)
        self.clf.fit(X, labels)

        self.bm25 = BM25Okapi([self._tok(t) for t in texts])
        return self

    @staticmethod
    def _tok(s: str) -> list:
        return re.findall(r"[a-z0-9]+", (s or "").lower())

    # ------------------------------------------------------------------ predict
    def handle(self, message: str, thread_context=None) -> dict:
        Xq = self.vec.transform([message])
        proba = self.clf.predict_proba(Xq)[0]
        k = int(np.argmax(proba))
        intent = self.clf.classes_[k]
        conf = float(proba[k])

        # --- reply: copy the nearest historical resolution verbatim
        scores = self.bm25.get_scores(self._tok(message))
        best = int(np.argmax(scores)) if len(scores) else -1
        if best >= 0 and scores[best] > 0:
            src = self.corpus_rows[best]
            reply = src.get("resolution_reply", "") or ""
            retrieved = [str(src["id"])]
            # BM25 scores are unbounded; squash to a 0-1 pseudo-similarity purely so the
            # sigma gate has something comparable to work with. Not a real cosine.
            sim = float(min(1.0, scores[best] / 25.0))
        else:
            reply, retrieved, sim = "", [], 0.0

        # --- routing: keyword rules only
        blob = " ".join([message] + list(thread_context or []))
        code = next((c for c, rx in _KRULES if rx.search(blob)), None)
        escalate = code is not None

        return {
            "intent": intent,
            "intent_confidence": conf,
            "draft_reply": reply[: config.REPLY_MAX_CHARS],
            "retrieved_ids": retrieved,
            # The copied reply IS the retrieved thread, so grounding is trivially perfect
            # -- which is exactly why this baseline is hard to beat on groundedness.
            "grounded_in": retrieved,
            "used_retrieval": bool(retrieved),
            "max_retrieval_sim": sim,
            "route": "escalate" if escalate else "auto",
            "route_reason_code": code or "known_troubleshooting_flow",
            "route_reason": f"B1 keyword rule matched: {code}" if code else "No escalation keyword matched.",
            "route_confidence": 1.0 if escalate else 0.5,
            "route_layer": "baseline_rules",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if not a.eval:
        ap.print_help()
        return 0

    from eval.run_eval import evaluate_system, load_golden

    corpus = pd.read_json(config.DATA_PROCESSED / "corpus.jsonl", lines=True).to_dict("records")
    if not corpus or "intent" not in corpus[0]:
        print("corpus.jsonl has no intent labels -- run `python scripts/label_corpus.py` first")
        return 1

    print(f"[b1] fitting TF-IDF + LogReg on {len(corpus)} corpus messages")
    sysm = B1Simple().fit(corpus)
    golden = load_golden()
    res = evaluate_system(sysm, golden, name="b1_simple", judge=False)
    print(f"\n--- b1_simple ---")
    print(f"  intent macro-F1   : {res['intent']['macro_f1']['point']:.3f} "
          f"[{res['intent']['macro_f1']['lo']:.3f}, {res['intent']['macro_f1']['hi']:.3f}]")
    print(f"  routing cost/100  : {res['routing']['expected_cost_per_100']:.1f}")
    print(f"  missed escalations: {res['routing']['false_auto_count']} of {res['routing']['false_auto_of']}")
    print(f"  deflection rate   : {res['routing']['deflection_rate']:.1%}")
    out = config.REPORTS / "results_b1.json"
    out.write_text(json.dumps({"b1_simple": res}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
