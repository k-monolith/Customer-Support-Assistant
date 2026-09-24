"""Cross-encoder reranking of the fused retrieval candidates."""
import numpy as np


class Reranker:
    def __init__(self, cfg: dict):
        from sentence_transformers import CrossEncoder
        rc = cfg["rerank"]
        self.model = CrossEncoder(rc["model"], max_length=rc["max_length"])
        self.batch_size = rc["batch_size"]
        self.keep_top_n = rc["keep_top_n"]

    def rerank(self, query: str, candidates: list[dict]):
        scores = self.model.predict([(query, c["text"]) for c in candidates],
                                    batch_size=self.batch_size)
        order = np.argsort(scores)[::-1]
        full = []
        for pos, i in enumerate(order):
            c = dict(candidates[i])
            c["rerank_score"] = float(scores[i])
            c["rerank_pos"] = pos
            full.append(c)
        return full, full[:self.keep_top_n]
