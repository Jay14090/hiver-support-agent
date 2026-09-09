"""P2.4 gate: taxonomy size stays in the 8-12 intents + other band."""
import sys
sys.path.insert(0, ".")
from src.support_agent.config import INTENTS
n = len(INTENTS)
assert "other" in INTENTS, "taxonomy must include `other`"
assert 9 <= n <= 13, f"taxonomy has {n} intents, expected 8-12 plus other"
print(f"OK: {n} intents ({n-1} + other): {INTENTS}")
