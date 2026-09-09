"""Label the DEV split with the same Pass-A procedure, for threshold tuning.

Dev is where tau and sigma are fit. Golden is never touched by tuning, so the reported
numbers are not optimised against the set they are measured on.
"""
import json, sys, threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
sys.path.insert(0, ".")
from src.support_agent import config, llm
from src.support_agent.codebook import load_codebook_text
from scripts.prelabel import PASS_A_SYSTEM, _fmt_msg

_lock = threading.Lock()

def main(n=400):
    df = pd.read_json(config.DATA_PROCESSED / "dev.jsonl", lines=True).head(n)
    rows = df.to_dict("records")
    system = PASS_A_SYSTEM % load_codebook_text()
    print(f"[label_dev] labelling {len(rows)} dev messages with {config.LABELER_A_MODEL}")
    done = [0]
    def work(row):
        parsed, _ = llm.complete_json(_fmt_msg(row), system=system,
                                      model=config.LABELER_A_MODEL, max_tokens=320, tag="label_dev")
        with _lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  ... {done[0]}/{len(rows)}  spend=${llm.spend_so_far():.3f}")
        p = parsed or {}
        return {"id": str(row["id"]), "text": row["text"],
                "intent": p.get("intent") if p.get("intent") in config.INTENTS else "other",
                "escalate": bool(p.get("escalate", False)),
                "escalate_reason_code": p.get("escalate_reason_code", ""),
                "difficulty": p.get("difficulty", "medium")}
    with ThreadPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(work, rows))
    llm.flush_ledger()
    p = config.GOLDEN / "dev_labels.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    esc = sum(1 for o in out if o["escalate"])
    print(f"[label_dev] wrote {p} ({len(out)} rows, escalate {esc} = {esc/len(out):.1%})")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
