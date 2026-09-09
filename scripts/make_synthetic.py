"""Last-resort synthetic stand-in for twcs.csv so the pipeline stays runnable offline.

Every row is stamped SYNTHETIC and `ingest.py` refuses to report headline numbers
from it without printing a NOT REAL RESULTS banner. This exists to keep the build
loop unblocked, not to produce evidence.
"""
from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(20260909)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "twcs.csv"

CUSTOMER = [
    "SYNTHETIC my songs keep skipping on wifi, already reinstalled twice",
    "SYNTHETIC i was charged twice for premium this month, can you refund",
    "SYNTHETIC cant log into my account it says email already in use",
    "SYNTHETIC how do i remove someone from my family plan",
    "SYNTHETIC the app crashes every time i open a playlist on android",
    "SYNTHETIC an album is missing from an artist page, where did it go",
    "SYNTHETIC spotify wont connect to my car bluetooth anymore",
    "SYNTHETIC i want to cancel my subscription immediately",
    "SYNTHETIC please add a sleep timer for podcasts on desktop",
    "SYNTHETIC love the new wrapped feature, amazing work team",
]
BRAND = [
    "SYNTHETIC Hey there! Try clearing the app cache in Settings > Storage, then restart. Let us know how it goes /JT",
    "SYNTHETIC We hear you. Head to your account page and select Manage Plan to review recent charges /RM",
    "SYNTHETIC Thanks for reaching out! A quick log out and back in on all devices usually sorts this /KM",
    "SYNTHETIC We've sent you a DM, let's continue there!",
]


def main() -> None:
    rows = []
    tid = 1
    t0 = datetime(2017, 10, 1)
    for i in range(4000):
        ts = (t0 + timedelta(minutes=7 * i)).strftime("%a %b %d %H:%M:%S +0000 %Y")
        cust_id = f"synthuser{i}"
        c_id, r_id = str(tid), str(tid + 1)
        tid += 2
        rows.append([c_id, cust_id, "True", ts, random.choice(CUSTOMER), r_id, ""])
        rows.append([r_id, "SpotifyCares", "False", ts, random.choice(BRAND), "", c_id])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"]
        )
        w.writerows(rows)
    print(f"[synthetic] wrote {len(rows)} SYNTHETIC rows to {OUT} -- NOT REAL RESULTS")


if __name__ == "__main__":
    main()
