"""P7.3 gate: FAILURE_ANALYSIS.md must have 5 modes, each fully specified."""
import re, sys
sys.path.insert(0, ".")
from src.support_agent import config

p = config.REPORTS / "FAILURE_ANALYSIS.md"
if not p.exists():
    print("FAIL: reports/FAILURE_ANALYSIS.md missing"); raise SystemExit(1)
t = p.read_text(encoding="utf-8")
blocks = re.split(r"^## Mode \d+", t, flags=re.M)[1:]
print(f"FAILURE_ANALYSIS.md: {len(blocks)} modes")
errs = []
if len(blocks) != 5:
    errs.append(f"need exactly 5 modes, found {len(blocks)}")
for i, b in enumerate(blocks, 1):
    head = b.split("\n")[0].strip()
    if not re.search(r"\*\*Frequency", b): errs.append(f"Mode {i} ({head[:34]}): no **Frequency")
    if not re.search(r"\*\*Hypothesis", b): errs.append(f"Mode {i}: no **Hypothesis")
    if not re.search(r"\*\*(Cheap )?[Tt]est", b): errs.append(f"Mode {i}: no **Test")
    n_ex = len(re.findall(r'^\s*[->]\s*[`"]', b, re.M)) + len(re.findall(r"^> ", b, re.M))
    if n_ex < 2: errs.append(f"Mode {i}: needs 2 verbatim examples, found {n_ex}")
    if not re.search(r"\bid[= ]", b): errs.append(f"Mode {i}: examples must cite ids")
if errs:
    print("FAIL:"); [print("  -", e) for e in errs]; raise SystemExit(1)
print("OK")
