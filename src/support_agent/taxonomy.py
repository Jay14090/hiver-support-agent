"""Induce the intent taxonomy FROM THE DATA, rather than inventing it at a whiteboard.

Pipeline:
  1. embed 3,000 sampled customer messages from the CORPUS split (never dev/test)
  2. cluster -- HDBSCAN if it produces a sane result, else KMeans swept over k
  3. for each cluster, show the LLM 20 messages nearest the centroid + 10 random members
     and ask for a name, a one-line definition, and two boundary rules
  4. I then edit those names by hand and merge/split by ROUTING outcome

Step 4 matters: clusters are a starting point, not the answer. Two clusters a support
agent would route identically should be one intent; one cluster that mixes routing
outcomes must be split. The hand-edited mapping lives in `CLUSTER_TO_INTENT` below.
"""
from __future__ import annotations

import argparse
import json
import random
import sys

import numpy as np
import pandas as pd

from . import config, embed, llm

SAMPLE_N = 3000
K_RANGE = range(6, 21)
NAMES_PATH = config.DATA_INTERIM / "cluster_names.json"
ASSIGN_PATH = config.DATA_INTERIM / "cluster_assignments.parquet"
EMB_PATH = config.DATA_INTERIM / "taxonomy_emb.npy"

NAMING_SYSTEM = """You are analysing clusters of real customer messages sent to a music
streaming service's public support account. You will see messages from ONE cluster.

Name the cluster as a support INTENT: what the customer wants, not how they feel.
Use lower_snake_case, 2-4 words, in the style of `billing_charge_dispute`.

Return JSON only:
{"name": "...", "definition": "<one line, what belongs here>",
 "boundary_rules": ["<rule 1>", "<rule 2>"],
 "routing": "auto|escalate|mixed",
 "coherent": true|false}

`routing` = how a support agent would usually handle these.
`coherent` = false if the cluster is clearly a mix of unrelated intents."""


# ---------------------------------------------------------------- the hand-edited map
# Step 4 of the pipeline, done by reading the clusters rather than trusting their names.
# Keyed by the induced cluster name so it survives a re-run that renumbers clusters.
# Two rules drove every merge: (a) merge clusters a support agent would route
# identically, (b) split anything that mixes routing outcomes.
CLUSTER_TO_INTENT = {
    # Three separate clusters, one intent. "why isn't <artist> on Spotify", "where's the
    # new album", "so many songs aren't on Spotify" are the same request with different
    # phrasing, and a support agent answers all three the same way. Together they are
    # ~50% of the sampled corpus -- the dominant topic in SpotifyCares' public mentions.
    "content_availability_issue": "content_missing_or_metadata",
    "album_availability_request": "content_missing_or_metadata",
    "music_feature_requests": "content_missing_or_metadata",
    # Product/market asks. "make an Apple Watch app", "launch in India", "fix playlist
    # ordering" are all "something that does not exist yet", routed identically.
    "service_availability_request": "feature_request_or_complaint",
    "app_update_request": "feature_request_or_complaint",
    "playlist_management_requests": "feature_request_or_complaint",
    # Student verification / premium-not-active / upgrade problems. This cluster is why
    # `plan_management_family_duo` was widened to `plan_or_premium_management`.
    "account_upgrade_issue": "plan_or_premium_management",
    "music_playback_issues": "playback_streaming_issue",
    # Genuinely low-coherence: "you ok spotify?", "I have an issue I can't find an answer
    # to". Vague by nature, so it maps to `other` rather than being forced into a topic.
    "service_access_issue": "other",
}

# Intents that exist in the data but were NOT isolated by clustering. Confirmed instead
# by targeted keyword probes (scripts/intent_evidence.py) plus reading. Recorded here so
# the claim "induced from the data" is not overstated: k-means on unbalanced short text
# finds the big blobs, and silence about a class is not evidence of its absence.
NOT_ISOLATED_BY_CLUSTERING = [
    "account_access_login",
    "billing_charge_dispute",
    "subscription_cancel_refund",
    "device_integration_issue",
    "app_bug_or_crash",
    "praise_or_chitchat",
]


def load_corpus() -> pd.DataFrame:
    p = config.DATA_PROCESSED / "corpus.jsonl"
    if not p.exists():
        raise SystemExit("no corpus.jsonl -- run `python -m src.support_agent.ingest --build`")
    return pd.read_json(p, lines=True)


def induce(sample_n: int = SAMPLE_N) -> dict:
    df = load_corpus()
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.choice(len(df), size=min(sample_n, len(df)), replace=False)
    sample = df.iloc[idx].reset_index(drop=True)
    texts = sample["text"].tolist()

    print(f"[taxonomy] embedding {len(texts)} corpus messages")
    X = embed.encode(texts, cache_name="taxonomy")
    np.save(EMB_PATH, X)

    labels, method, sweep = _cluster(X)
    sample["cluster"] = labels
    sample[["id", "text", "cluster"]].to_parquet(ASSIGN_PATH, index=False)

    names = _name_clusters(sample, X, labels)
    out = {"method": method, "sweep": sweep, "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
           "clusters": names}
    NAMES_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[taxonomy] wrote {NAMES_PATH}")
    return out


def _cluster(X: np.ndarray):
    """HDBSCAN first (it finds natural density structure and marks noise), else KMeans.

    HDBSCAN is accepted only if it leaves <40% of points as noise and finds >=6 clusters
    -- on short, lexically similar tweets it often collapses to one blob plus noise,
    which is worse than nothing. The sweep is saved either way so the choice is auditable.
    """
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.metrics import silhouette_score

    sweep = {}
    try:
        h = HDBSCAN(min_cluster_size=40, min_samples=10, metric="euclidean")
        hl = h.fit_predict(X)
        n_clusters = len(set(hl)) - (1 if -1 in hl else 0)
        noise = float((hl == -1).mean())
        sweep["hdbscan"] = {"n_clusters": int(n_clusters), "noise_fraction": round(noise, 3)}
        print(f"[taxonomy] HDBSCAN: {n_clusters} clusters, {noise:.1%} noise")
        if n_clusters >= 6 and noise < 0.40:
            return hl, "hdbscan", sweep
        print("[taxonomy] HDBSCAN rejected (too few clusters or too much noise) -> KMeans sweep")
    except Exception as e:  # noqa: BLE001 -- reported, never swallowed
        print(f"[taxonomy] HDBSCAN unavailable/failed ({e}) -> KMeans sweep")
        sweep["hdbscan"] = {"error": str(e)}

    scores = {}
    best_k, best_s, best_labels = None, -1.0, None
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        lab = km.fit_predict(X)
        s = float(silhouette_score(X, lab, sample_size=2000, random_state=config.RANDOM_STATE))
        scores[k] = round(s, 4)
        print(f"[taxonomy]   k={k:>2}  silhouette={s:.4f}")
        if s > best_s:
            best_k, best_s, best_labels = k, s, lab
    sweep["kmeans_silhouette"] = scores

    # Silhouette on short-text embeddings is monotonically noisy and tends to favour tiny
    # k. I take the best k INSIDE the 8-14 band the brief targets, and record that this
    # was a judgement call rather than a pure argmax.
    band = {k: v for k, v in scores.items() if 8 <= k <= 14}
    chosen_k = max(band, key=band.get)
    sweep["argmax_k"] = best_k
    sweep["chosen_k"] = chosen_k
    sweep["note"] = ("silhouette argmax was k=%d; chose k=%d as the best value inside the "
                     "8-14 band, because the target taxonomy is 8-12 intents + other and "
                     "very small k merges intents that route differently." % (best_k, chosen_k))
    if chosen_k != best_k:
        print(f"[taxonomy] silhouette argmax k={best_k}; using k={chosen_k} (8-14 band)")
        km = KMeans(n_clusters=chosen_k, random_state=config.RANDOM_STATE, n_init=10)
        best_labels = km.fit_predict(X)
    _plot_sweep(scores, chosen_k, best_k)
    return best_labels, f"kmeans_k{chosen_k}", sweep


def _plot_sweep(scores: dict, chosen: int, argmax: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ks, ss = list(scores), list(scores.values())
    ax.plot(ks, ss, "-o", color="#9aa0a6", ms=4)
    ax.plot([chosen], [scores[chosen]], "o", color="#1DB954", ms=11, label=f"chosen k={chosen}")
    ax.plot([argmax], [scores[argmax]], "o", color="#d1495b", ms=8, label=f"silhouette argmax k={argmax}")
    ax.axvspan(8, 14, color="#1DB954", alpha=0.07, label="target band 8-14")
    ax.set_xlabel("k")
    ax.set_ylabel("silhouette")
    ax.set_title("KMeans sweep for intent induction", loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = config.FIGURES / "cluster_sweep.png"
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"[taxonomy] wrote {out}")


def _name_clusters(sample: pd.DataFrame, X: np.ndarray, labels: np.ndarray) -> dict:
    random.seed(config.RANDOM_STATE)
    out = {}
    for c in sorted(set(labels)):
        if c == -1:
            continue
        mask = labels == c
        members = sample.loc[mask, "text"].tolist()
        vecs = X[mask]
        centroid = vecs.mean(axis=0)
        near = np.argsort(-(vecs @ centroid))[:20]
        reps = [members[i] for i in near]
        rest = [m for i, m in enumerate(members) if i not in set(near.tolist())]
        rnd = random.sample(rest, min(10, len(rest)))

        prompt = (
            f"Cluster {c} has {int(mask.sum())} messages.\n\n"
            "20 messages closest to the cluster centre:\n"
            + "\n".join(f"- {t[:220]}" for t in reps)
            + "\n\n10 random members:\n"
            + "\n".join(f"- {t[:220]}" for t in rnd)
        )
        parsed, raw = llm.complete_json(
            prompt, system=NAMING_SYSTEM, model=config.GENERATOR_MODEL, max_tokens=300, tag="taxonomy_naming"
        )
        if parsed is None:
            print(f"[taxonomy] cluster {c}: naming parse FAILED, recorded as unnamed")
            parsed = {"name": f"unnamed_{c}", "definition": "", "boundary_rules": [], "coherent": False}
        parsed["size"] = int(mask.sum())
        parsed["examples"] = reps[:5]
        out[str(c)] = parsed
        print(f"[taxonomy] cluster {c:>2} n={int(mask.sum()):>4} -> {parsed.get('name')} "
              f"(routing={parsed.get('routing')}, coherent={parsed.get('coherent')})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--induce", action="store_true")
    ap.add_argument("--sample", type=int, default=SAMPLE_N)
    ap.add_argument("--show", action="store_true", help="print the saved cluster names")
    a = ap.parse_args()
    if a.induce:
        induce(a.sample)
        return 0
    if a.show:
        print(NAMES_PATH.read_text(encoding="utf-8"))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
