"""P2.5 gate: every intent in config.INTENTS must have a codebook entry with a definition."""
import sys
sys.path.insert(0, ".")
from src.support_agent.codebook import missing_intents, parse_codebook
from src.support_agent import config

cb = parse_codebook()
missing = missing_intents()
if missing:
    print(f"FAIL: intents with no codebook entry: {missing}")
    raise SystemExit(1)
bad = [i for i, e in cb.items() if not e["definition"]]
if bad:
    print(f"FAIL: codebook entries with no definition: {bad}")
    raise SystemExit(1)
no_tie = [i for i, e in cb.items() if not e["tie_breaks"] and i != "other"]
print(f"OK: {len(cb)}/{len(config.INTENTS)} intents documented")
for i, e in cb.items():
    print(f"  {i:<32} def={len(e['definition']):>3}ch  tie_breaks={len(e['tie_breaks'])}  positives={len(e['positives'])}")
if no_tie:
    print(f"NOTE: no tie-break rule for: {no_tie}")
