"""Label the CORPUS split with intents.

Two consumers:
  1. B1's TF-IDF+LogReg classifier needs training labels.
  2. The agent's classifier prompt pulls few-shot examples from retrieved corpus threads,
     which is only useful if those threads carry a label.

Uses the SAME codebook and the same model as Pass A, but asks for intent only -- the
must_include/must_not_include fields are golden-set machinery and are not needed here.
Keeping the definition identical matters: if B1 trained against a different notion of
`device_integration_issue` than the golden set uses, its errors would be measuring prompt
drift rather than model capability.

Honest limitation, repeated in the report: **B1 is trained on machine labels.** It learns
to imitate the labeller, not human ground truth. Its ceiling is the labeller's accuracy.
"""
from __future__ import annotations

import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, ".")
from src.support_agent import config, llm  # noqa: E402
from src.support_agent.codebook import load_codebook_text  # noqa: E402

WORKERS = 8
SYSTEM = """Classify a customer message sent to Spotify's public support account into
exactly one intent.

%s

Rules:
- Label what the customer WANTS RESOLVED, not their emotion.
- If it covers two intents, pick the one a support agent would act on first.
- Use `other` only when nothing else fits.

Return JSON only: {"intent": "<intent>", "confidence": <0.0-1.0>}"""

_lock = threading.Lock()


def main() -> None:
    src = config.DATA_PROCESSED / "corpus.jsonl"
    df = pd.read_json(src, lines=True)
    if "intent" in df.columns and df["intent"].notna().all():
        print("corpus already labelled; delete the `intent` column to redo")
        return
    rows = df.to_dict("records")
    system = SYSTEM % load_codebook_text()
    print(f"[label_corpus] labelling {len(rows)} corpus messages with {config.LABELER_A_MODEL}")

    done = [0]

    def work(i_row):
        i, row = i_row
        ctx = row.get("thread_context") or []
        prompt = ""
        if len(ctx):
            prompt += "EARLIER TURNS:\n" + "\n".join(f"- {c[:180]}" for c in ctx[:2]) + "\n\n"
        prompt += f'MESSAGE:\n"{row["text"]}"'
        parsed, _ = llm.complete_json(prompt, system=system, model=config.LABELER_A_MODEL,
                                      max_tokens=60, tag="label_corpus")
        with _lock:
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"  ... {done[0]}/{len(rows)}  spend=${llm.spend_so_far():.3f}")
        intent = (parsed or {}).get("intent")
        conf = (parsed or {}).get("confidence", 0.0)
        return i, (intent if intent in config.INTENTS else "other",
                   float(conf) if isinstance(conf, (int, float)) else 0.0,
                   parsed is None)

    out = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, res in ex.map(work, list(enumerate(rows))):
            out[i] = res
    llm.flush_ledger()

    df["intent"] = [o[0] for o in out]
    df["intent_confidence"] = [o[1] for o in out]
    fails = sum(1 for o in out if o[2])
    df.to_json(src, orient="records", lines=True, force_ascii=False)

    c = Counter(df["intent"])
    print(f"[label_corpus] wrote {src} (parse failures: {fails})")
    for k, v in c.most_common():
        print(f"   {k:<32} {v:>5}  ({v/len(df):5.1%})")
    print(f"[label_corpus] spend ${llm.spend_so_far():.4f}")


if __name__ == "__main__":
    main()
