"""twcs.csv -> threads -> brand slice -> resolution filter -> cleaned -> time-based splits.

Every stage records how many rows it dropped and why, so `--verify` can print an honest
data funnel. The funnel figure is one image that shows I understand my own data.

Memory note: twcs.csv is ~2.8M rows / ~500MB. Building parent/child dicts over the whole
file would cost several GB, so instead the brand's conversational neighbourhood is grown
iteratively with vectorised `isin` passes (see `_brand_neighbourhood`) and only that
subset -- O(100k) rows -- is threaded. Same result, a fraction of the memory.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import config
from .clean import clean_brand, clean_customer, dedupe_key, passes_quality
from .resolution import thread_resolution_score
from .threads import _norm_id, reconstruct, split_children

COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

FUNNEL_PATH = config.DATA_INTERIM / "funnel.json"
THREADS_PATH = config.DATA_INTERIM / "threads.parquet"


def load_raw(nrows: int | None = None) -> pd.DataFrame:
    """Read the CSV with EVERY column forced to str.

    Without dtype=str pandas infers the id columns as float64 (because of NaNs) and
    turns "123" into "123.0", which breaks every join downstream. This is trap T2 in
    threads.py and it is the single most common way to get this dataset wrong.
    """
    if not config.TWCS_CSV.exists():
        raise SystemExit(f"missing {config.TWCS_CSV} -- run `bash scripts/get_data.sh` first")
    df = pd.read_csv(
        config.TWCS_CSV,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        nrows=nrows,
        encoding="utf-8",
        on_bad_lines="warn",
    )
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"schema mismatch, missing columns: {missing}. Got: {list(df.columns)}")
    return df[COLUMNS]


def _brand_neighbourhood(df: pd.DataFrame, brand: str, max_hops: int = 6) -> pd.DataFrame:
    """Grow the set of tweets reachable from the brand's own tweets, `max_hops` deep.

    Cheaper than indexing all 2.8M rows: each hop is two vectorised `isin` passes.
    6 hops comfortably covers real support threads (MAX_DEPTH is 20 but the median
    thread is 2-4 turns; the depth cap in threads.py catches anything longer).
    """
    tid = df["tweet_id"].map(_norm_id)
    parent = df["in_response_to_tweet_id"].map(_norm_id)
    df = df.assign(_tid=tid, _parent=parent)

    frontier = set(df.loc[df["author_id"] == brand, "_tid"])
    if not frontier:
        return df.iloc[0:0]
    keep = set(frontier)

    for _ in range(max_hops):
        # up: parents of the frontier;  down: rows replying to the frontier
        up = set(df.loc[df["_tid"].isin(frontier), "_parent"]) - {""}
        down = set(df.loc[df["_parent"].isin(frontier), "_tid"])
        # also the comma-list children the frontier rows point at
        side: set = set()
        for v in df.loc[df["_tid"].isin(frontier), "response_tweet_id"]:
            side.update(split_children(v))
        new = (up | down | side) - keep
        if not new:
            break
        keep |= new
        frontier = new
    return df[df["_tid"].isin(keep)].drop(columns=["_tid", "_parent"])


def _parse_ts(s: pd.Series) -> pd.Series:
    """twcs timestamps look like 'Tue Oct 31 22:10:47 +0000 2017'."""
    return pd.to_datetime(s, format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)


def build(brand: str | None = None, nrows: int | None = None) -> dict:
    brand = brand or config.BRAND
    funnel: dict = {"brand": brand}
    synthetic = False

    print(f"[ingest] loading {config.TWCS_CSV.name} ...")
    df = load_raw(nrows=nrows)
    funnel["total_tweets"] = len(df)
    if df["text"].str.contains("SYNTHETIC", regex=False).any():
        synthetic = True
        print("[ingest] !! SYNTHETIC DATA DETECTED -- these are NOT REAL RESULTS !!")
    funnel["synthetic"] = synthetic

    funnel["brand_tweets"] = int((df["author_id"] == brand).sum())
    print(f"[ingest] {funnel['total_tweets']:,} tweets, {funnel['brand_tweets']:,} by {brand}")
    if funnel["brand_tweets"] == 0:
        raise SystemExit(f"brand {brand!r} not present in the data")

    print("[ingest] growing brand conversational neighbourhood ...")
    sub = _brand_neighbourhood(df, brand)
    funnel["neighbourhood_tweets"] = len(sub)
    del df

    print(f"[ingest] threading {len(sub):,} rows ...")
    rows = sub.to_dict("records")
    threads = reconstruct(rows, brand=brand)
    funnel["threads_reconstructed"] = len(threads)

    # ---- brand slice: the brand must actually speak in the thread
    threads = [t for t in threads if t.brand_turns(brand)]
    funnel["threads_with_brand_reply"] = len(threads)

    # ---- resolution-bearing filter (survivorship bias source -- see REPORT.md)
    kept, res_scores = [], []
    for t in threads:
        raw_brand = [x.text for x in t.brand_turns(brand)]
        score, signals = thread_resolution_score(raw_brand)
        res_scores.append(score)
        if score >= 2:
            t.res_score, t.res_signals = score, signals
            kept.append(t)
    funnel["threads_resolution_bearing"] = len(kept)
    funnel["resolution_rate"] = round(len(kept) / max(1, len(threads)), 4)
    print(
        f"[ingest] resolution-bearing: {len(kept):,}/{len(threads):,} "
        f"= {funnel['resolution_rate']:.1%}"
    )

    # ---- D3 fallback check, run BEFORE any further work
    d3_ok = len(kept) >= config.MIN_USABLE_THREADS and funnel["resolution_rate"] >= config.MIN_RESOLUTION_RATE
    funnel["d3_check"] = {
        "usable_threads": len(kept),
        "min_required": config.MIN_USABLE_THREADS,
        "resolution_rate": funnel["resolution_rate"],
        "min_rate_required": config.MIN_RESOLUTION_RATE,
        "passed": bool(d3_ok),
    }
    if not d3_ok:
        print(
            f"[ingest] !! D3 TRIGGERED for {brand}: {len(kept)} threads @ "
            f"{funnel['resolution_rate']:.1%}. Fall back to {config.BRAND_FALLBACKS}."
        )

    # ---- clean + quality filters
    records, drop_reasons, seen = [], {"non_english": 0, "too_short": 0, "duplicate": 0}, set()
    for t in kept:
        cust_raw = t.turns[0].text
        cust = clean_customer(cust_raw, brand)
        ok, why = passes_quality(cust)
        if not ok:
            drop_reasons[why] += 1
            continue
        k = dedupe_key(cust)
        if k in seen:
            drop_reasons["duplicate"] += 1
            continue
        seen.add(k)

        brand_texts, sigs = [], []
        for x in t.brand_turns(brand):
            b, sg = clean_brand(x.text, brand)
            if b:
                brand_texts.append(b)
                if sg:
                    sigs.append(sg)
        if not brand_texts:
            continue

        # Context = every turn before the brand's first reply, cleaned.
        ctx = []
        for x in t.turns[1:]:
            if x.author_id == brand:
                break
            ctx.append(clean_customer(x.text, brand))

        records.append(
            {
                "id": t.root_id,
                "created_at": t.turns[0].created_at,
                "customer_id": t.customer_id,
                "text": cust,
                "thread_context": ctx,
                "brand_replies": brand_texts,
                "resolution_reply": max(brand_texts, key=len),
                "agent_signature": sigs[0] if sigs else "",
                "res_score": getattr(t, "res_score", 0),
                "res_signals": getattr(t, "res_signals", []),
                "n_turns": len(t.turns),
            }
        )
    funnel["dropped"] = drop_reasons
    funnel["threads_clean"] = len(records)
    print(f"[ingest] after cleaning/dedupe: {len(records):,} (dropped {drop_reasons})")

    # ---- D5 time-based split on the ROOT tweet timestamp
    out = pd.DataFrame(records)
    out["ts"] = _parse_ts(out["created_at"])
    n_bad_ts = int(out["ts"].isna().sum())
    funnel["unparseable_timestamps"] = n_bad_ts
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    n = len(out)
    i_corpus = int(n * config.SPLIT_CORPUS)
    i_dev = int(n * (config.SPLIT_CORPUS + config.SPLIT_DEV))
    parts = {"corpus": out.iloc[:i_corpus], "dev": out.iloc[i_corpus:i_dev], "test": out.iloc[i_dev:]}

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        p = part.copy()
        p["created_at"] = p["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        p = p.drop(columns=["ts"])
        path = config.DATA_PROCESSED / f"{name}.jsonl"
        p.to_json(path, orient="records", lines=True, force_ascii=False)
        funnel[f"split_{name}"] = len(p)
        funnel[f"split_{name}_date_range"] = [
            str(part["ts"].min()) if len(part) else None,
            str(part["ts"].max()) if len(part) else None,
        ]
        print(f"[ingest] wrote {path.name}: {len(p):,} rows")

    out.drop(columns=["ts"]).to_parquet(THREADS_PATH, index=False)
    FUNNEL_PATH.write_text(json.dumps(funnel, indent=2), encoding="utf-8")
    return funnel


def print_funnel(f: dict) -> None:
    if f.get("synthetic"):
        print("\n" + "!" * 68)
        print("!! SYNTHETIC STAND-IN DATA -- EVERY NUMBER BELOW IS NOT A REAL RESULT !!")
        print("!" * 68)
    rows = [
        ("all tweets in twcs.csv", f.get("total_tweets")),
        (f"  tweets authored by {f.get('brand')}", f.get("brand_tweets")),
        ("  brand conversational neighbourhood", f.get("neighbourhood_tweets")),
        ("threads reconstructed", f.get("threads_reconstructed")),
        ("  with >=1 brand reply", f.get("threads_with_brand_reply")),
        ("  resolution-bearing (survivorship filter)", f.get("threads_resolution_bearing")),
        ("  clean + english + deduped", f.get("threads_clean")),
        ("split: corpus (oldest 70%)", f.get("split_corpus")),
        ("split: dev (next 15%)", f.get("split_dev")),
        ("split: test (newest 15%)", f.get("split_test")),
    ]
    print("\n=== DATA FUNNEL ===")
    for label, v in rows:
        print(f"{label:<48} {v:>12,}" if isinstance(v, int) else f"{label:<48} {v}")
    print(f"\nresolution-bearing rate: {f.get('resolution_rate', 0):.1%}")
    print(f"dropped in cleaning:     {f.get('dropped')}")
    d3 = f.get("d3_check", {})
    print(f"D3 brand-viability check: {'PASS' if d3.get('passed') else 'FAIL -> fall back'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="twcs ingest pipeline")
    ap.add_argument("--build", action="store_true", help="run the full pipeline")
    ap.add_argument("--verify", action="store_true", help="print the data funnel")
    ap.add_argument("--verify-schema", action="store_true", help="check the CSV header only")
    ap.add_argument("--brand-stats", action="store_true", help="brand slice + D3 check")
    ap.add_argument("--brand", default=None)
    ap.add_argument("--nrows", type=int, default=None, help="subsample for a fast smoke run")
    a = ap.parse_args()

    if a.verify_schema:
        df = load_raw(nrows=5)
        print("schema OK:", list(df.columns))
        print(df.head(2).to_string())
        return 0

    if a.build:
        f = build(brand=a.brand, nrows=a.nrows)
        print_funnel(f)
        return 0

    if a.brand_stats or a.verify:
        if not FUNNEL_PATH.exists():
            print("no funnel yet -- run `--build` first")
            return 1
        f = json.loads(FUNNEL_PATH.read_text(encoding="utf-8"))
        print_funnel(f)
        if a.brand_stats:
            print(json.dumps(f.get("d3_check"), indent=2))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
