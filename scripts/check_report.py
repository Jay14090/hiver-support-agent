"""Gates for the written deliverables (P9.1, P9.3)."""
import re, sys, argparse
sys.path.insert(0, ".")
from src.support_agent import config

def check_misleading():
    p = config.REPORTS / "REPORT.md"
    if not p.exists():
        print("FAIL: reports/REPORT.md missing"); return 1
    t = p.read_text(encoding="utf-8")
    m = re.search(r"##\s*1\..*?misleading.*?\n(.*?)(?=\n## )", t, re.S | re.I)
    if not m:
        print("FAIL: no 'What is misleading about my headline number?' section"); return 1
    items = re.findall(r"^\s*\d+\.\s+\*\*", m.group(1), re.M)
    print(f"misleading-numbers section: {len(items)} numbered items")
    if len(items) < 8:
        print(f"FAIL: need >= 8 items, found {len(items)}"); return 1
    if t.index(m.group(0)) > t.index("## 2."):
        print("FAIL: misleading section must come FIRST"); return 1
    print("OK")
    return 0

def check_decisions():
    p = config.REPORTS / "DECISION_LOG.md"
    if not p.exists():
        print("FAIL: DECISION_LOG.md missing"); return 1
    t = p.read_text(encoding="utf-8")
    entries = re.findall(r"^###\s+D\d+\.", t, re.M)
    print(f"DECISION_LOG: {len(entries)} entries")
    missing = []
    for block in re.split(r"^### ", t, flags=re.M)[1:]:
        head = block.split("\n")[0]
        for field in ("**Decision.**", "**Alternatives", "**Why", "change my mind"):
            if field not in block:
                missing.append(f"{head[:40]}: missing {field}")
    if missing:
        print("FAIL:"); [print("  -", x) for x in missing[:10]]; return 1
    if not (10 <= len(entries) <= 15):
        print(f"FAIL: need 10-15 entries, found {len(entries)}"); return 1
    print("OK")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--misleading", action="store_true")
    ap.add_argument("--decisions", action="store_true")
    a = ap.parse_args()
    rc = 0
    if a.decisions: rc |= check_decisions()
    if a.misleading: rc |= check_misleading()
    sys.exit(rc)
