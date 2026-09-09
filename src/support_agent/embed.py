"""Sentence embeddings, cached to disk.

Shared by taxonomy induction (Phase 2) and dense retrieval (Phase 5) so both see the
identical vector space. Embeddings are cached by (model, sha1 of the text list) because
encoding the corpus takes minutes on CPU and gets re-run constantly during development.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np

from . import config

_model = None
_model_lock = threading.Lock()


def get_model():
    """Lazy singleton -- importing sentence_transformers costs ~5s and pulls in torch.

    Locked because the eval runs 8 worker threads: without it, every thread saw
    `_model is None` at once and loaded its own copy of the model (8x the RAM and 8x
    the startup cost, visible as eight "loading" lines in the logs).
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:            # re-check: another thread may have won the race
                from sentence_transformers import SentenceTransformer

                print(f"[embed] loading {config.EMBED_MODEL} (first run downloads ~90MB)")
                _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def _key(texts: list, model_name: str) -> str:
    h = hashlib.sha1()
    h.update(model_name.encode())
    h.update(str(len(texts)).encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def encode(texts: list, cache_name: str | None = None, batch_size: int = 128) -> np.ndarray:
    """Return L2-normalised float32 embeddings. Normalised so cosine == dot product."""
    texts = [t if isinstance(t, str) else "" for t in texts]
    name = cache_name or _key(texts, config.EMBED_MODEL)
    path: Path = config.DATA_INTERIM / f"emb_{name}.npy"
    if path.exists():
        arr = np.load(path)
        if arr.shape[0] == len(texts):
            return arr
        print(f"[embed] cache {path.name} has {arr.shape[0]} rows, need {len(texts)} -- re-encoding")

    m = get_model()
    arr = m.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        # Progress bar only for bulk encoding. Single-query retrieval calls this on every
        # decision, and a progress bar per query makes the demo output unreadable.
        show_progress_bar=len(texts) > 64,
    ).astype("float32")
    np.save(path, arr)
    return arr


def cosine_sim(query_vecs: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Both sides are already L2-normalised, so a dot product IS cosine similarity.

    A numpy matmul over O(10k) vectors answers in milliseconds. This is exactly why
    there is no vector database in this repo (see NOT_BUILDING.md).
    """
    if query_vecs.ndim == 1:
        query_vecs = query_vecs[None, :]
    return query_vecs @ matrix.T
