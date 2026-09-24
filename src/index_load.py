"""Load the prebuilt index artifacts into the Qdrant server (run once per index version).
Usage: python -m src.index_load
Requires the four artifact files in ARTIFACTS_DIR (default ./artifacts)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

PAYLOAD_COLS = ["text", "doc_id", "chunk_id", "position", "source",
                "category", "intent", "title"]


def main():
    from qdrant_client import QdrantClient, models

    cfg = load_config()
    art = Path(cfg["env"]["artifacts_dir"])
    version = cfg["version"]

    meta = json.load(open(art / f"index_meta_{version}.json"))
    vecs = np.load(art / f"embeddings_{version}.npy")
    payload = pd.read_parquet(art / f"index_payload_{version}.parquet")
    assert meta["n_chunks"] == len(payload) == len(vecs)
    print(f"artifacts OK: {len(payload)} chunks, dim {vecs.shape[1]}")

    client = QdrantClient(url=cfg["env"]["qdrant_url"])
    if client.collection_exists(version):
        count = client.get_collection(version).points_count
        if count == len(payload):
            print(f"collection {version} already loaded ({count} points), nothing to do")
            return
        raise SystemExit(
            f"collection {version} exists with {count} points (expected {len(payload)}). "
            "Delete it or bump the version; never overwrite in place.")

    client.create_collection(version, vectors_config=models.VectorParams(
        size=vecs.shape[1], distance=models.Distance.COSINE))
    for start in range(0, len(payload), 512):
        end = min(start + 512, len(payload))
        b = payload.iloc[start:end]
        client.upsert(version, points=models.Batch(
            ids=list(range(start, end)),
            vectors=vecs[start:end].tolist(),
            payloads=b[PAYLOAD_COLS].where(b[PAYLOAD_COLS].notna(), None)
                     .to_dict("records")))
        if start % (512 * 20) == 0:
            print(f"upserted {end}/{len(payload)}")
    count = client.get_collection(version).points_count
    assert count == len(payload)
    print(f"done: collection {version} with {count} points")


if __name__ == "__main__":
    main()
