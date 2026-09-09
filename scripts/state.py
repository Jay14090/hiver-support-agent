"""Tiny helper to mark tasks done/skipped in STATE.json and append to PROGRESS.md."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def mark(task_id, status="done", note="", artifacts=None):
    p = ROOT / "STATE.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    hit = False
    for t in s["tasks"]:
        if t["id"] == task_id:
            t["status"] = status
            if artifacts: t["artifacts"] = artifacts
            s["current_phase"] = t["phase"]
            hit = True
    if not hit: raise SystemExit(f"no such task {task_id}")
    p.write_text(json.dumps(s, indent=2), encoding="utf-8")
    line = f"- [{time.strftime('%Y-%m-%d %H:%M')}] {task_id} {status.upper()} - {note}\n"
    with (ROOT / "PROGRESS.md").open("a", encoding="utf-8") as f: f.write(line)
    print(line.strip())

def summary():
    s = json.loads((ROOT/"STATE.json").read_text(encoding="utf-8"))
    from collections import Counter
    c = Counter(t["status"] for t in s["tasks"])
    nxt = next((t for t in s["tasks"] if t["status"] not in ("done","skipped")), None)
    print(json.dumps({"counts": dict(c), "next": nxt["id"] if nxt else None,
                      "next_title": nxt["title"] if nxt else None}, indent=2))

if __name__ == "__main__":
    if sys.argv[1] == "--summary": summary()
    else: mark(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "done", sys.argv[3] if len(sys.argv)>3 else "")
