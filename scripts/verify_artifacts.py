"""Integrity check for the four index artifact files before loading them.

Usage:  python scripts/verify_artifacts.py [artifacts_dir]
Default artifacts_dir: ./artifacts (or ARTIFACTS_DIR).
Exits non-zero on the first failed check.
"""
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

art = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ARTIFACTS_DIR", "artifacts"))
version = "support_kb_v1"

meta = json.load(open(art / f"index_meta_{version}.json"))
assert meta["collection"] == version, meta["collection"]
n, dim = meta["n_chunks"], meta["embedding_dim"]
print(f"meta: {n} chunks, dim {dim}, model {meta['embedding']['model']}")

payload = pd.read_parquet(art / f"index_payload_{version}.parquet")
assert len(payload) == n, f"payload rows {len(payload)} != {n}"
assert payload["chunk_id"].is_unique
missing = {"text", "doc_id", "chunk_id", "source", "title", "intent"} - set(payload.columns)
assert not missing, f"payload missing columns: {missing}"
print(f"payload: {len(payload)} rows, sources {payload['source'].value_counts().to_dict()}")

vecs = np.load(art / f"embeddings_{version}.npy", mmap_mode="r")
assert vecs.shape == (n, dim), f"embeddings shape {vecs.shape} != ({n}, {dim})"
assert vecs.dtype == np.float32, vecs.dtype
sample = np.asarray(vecs[:: max(1, n // 500)])
assert not np.isnan(sample).any(), "NaN values in embeddings"
norms = np.linalg.norm(sample, axis=1)
assert np.allclose(norms, 1.0, atol=1e-2), f"vectors not normalized (sample norms {norms.min():.3f}..{norms.max():.3f})"
print(f"embeddings: {vecs.shape} float32, sampled norms ok")

with open(art / f"bm25_corpus_{version}.pkl", "rb") as f:
    bm = pickle.load(f)
assert bm["version"] == version, bm["version"]
assert len(bm["tokenized"]) == n, f"bm25 corpus {len(bm['tokenized'])} != {n}"
tok = re.compile(bm["tokenizer_regex"])
for i in (0, n // 2, n - 1):
    assert bm["tokenized"][i] == tok.findall(payload.iloc[i]["text"].lower()), f"bm25 row {i} not aligned with payload"
print(f"bm25: {len(bm['tokenized'])} docs, aligned with payload")

print("all artifact checks passed")
