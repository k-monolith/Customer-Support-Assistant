"""Hybrid retrieval: dense search in Qdrant plus BM25, merged with Reciprocal Rank Fusion."""
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


def rrf(rankings, k=60):
    scores = {}
    for ranked in rankings:
        for rank, pid in enumerate(ranked, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True), scores


class HybridRetriever:
    def __init__(self, cfg: dict):
        from qdrant_client import QdrantClient
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        art = Path(cfg["env"]["artifacts_dir"])
        version = cfg["version"]
        self.cfg = cfg["retrieval"]
        self.collection = version

        self.payload = pd.read_parquet(art / f"index_payload_{version}.parquet")
        with open(art / f"bm25_corpus_{version}.pkl", "rb") as f:
            bm25_data = pickle.load(f)
        assert bm25_data["version"] == version
        assert len(bm25_data["tokenized"]) == len(self.payload)
        self._token_re = re.compile(bm25_data["tokenizer_regex"])
        self.bm25 = BM25Okapi(bm25_data["tokenized"])

        self.client = QdrantClient(url=cfg["env"]["qdrant_url"])
        count = self.client.get_collection(self.collection).points_count
        assert count == len(self.payload), (
            f"Qdrant collection {self.collection} has {count} points, payload has "
            f"{len(self.payload)}. Run `python -m src.index_load` first.")

        self.embedder = SentenceTransformer(cfg["embedding"]["model"])
        self.query_prompt = cfg["embedding"]["query_prompt_name"]

    def _encode_query(self, q: str) -> np.ndarray:
        return self.embedder.encode([q], prompt_name=self.query_prompt,
                                    normalize_embeddings=True)[0]

    def _bm25_tokenize(self, text: str):
        return self._token_re.findall(text.lower())

    def retrieve(self, query: str) -> list[dict]:
        c = self.cfg
        qv = self._encode_query(query)
        dense = [p.id for p in self.client.query_points(
            self.collection, query=qv.tolist(), limit=c["dense_top_k"]).points]
        scores = self.bm25.get_scores(self._bm25_tokenize(query))
        bm = np.argsort(scores)[::-1][:c["bm25_top_k"]].tolist()
        fused_ids, fused_scores = rrf([dense, bm], k=c["rrf_k"])
        d_rank = {pid: r for r, pid in enumerate(dense, 1)}
        b_rank = {pid: r for r, pid in enumerate(bm, 1)}
        out = []
        for pid in fused_ids[:c["fused_candidates"]]:
            row = self.payload.iloc[pid]
            out.append({
                "point_id": int(pid), "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"], "text": row["text"],
                "source": row["source"], "title": row["title"],
                "intent": row["intent"], "url": row.get("url"),
                "rrf_score": fused_scores[pid],
                "dense_rank": d_rank.get(pid), "bm25_rank": b_rank.get(pid),
            })
        return out
