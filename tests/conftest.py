"""API tests run against a stubbed pipeline: same orchestration code
(SupportPipeline), fake retrieval/rerank/generation, no model downloads."""
import pytest
from fastapi.testclient import TestClient

from src.answer import SupportPipeline
from src.api.main import STATE, app
from src.config import load_config
from src.guards import NoopMasker

FAKE_CHUNKS = [
    {"point_id": i, "chunk_id": f"c{i}", "doc_id": f"d{i}",
     "text": f"To cancel an order, open Settings and choose Orders. (chunk {i})",
     "source": "wixqa_kb", "title": f"Article {i}", "intent": None,
     "url": f"https://example.com/a{i}", "rrf_score": 1.0 / (i + 1),
     "dense_rank": i + 1, "bm25_rank": None}
    for i in range(60)
]


class FakeRetriever:
    def retrieve(self, query):
        return [dict(c) for c in FAKE_CHUNKS]


class FakeReranker:
    keep_top_n = 6

    def rerank(self, query, candidates):
        full = []
        for pos, c in enumerate(candidates):
            c = dict(c)
            c["rerank_score"] = 10.0 - pos
            c["rerank_pos"] = pos
            full.append(c)
        return full, full[:self.keep_top_n]


class FakeGenerator:
    """Scripted per-question behavior; grounding check always SUPPORTED."""
    model = "fake-model"

    def __init__(self):
        self.abstain = ("I don't have enough information to answer this. "
                        "Let me connect you with a support agent.")

    def generate(self, system, user, max_tokens=None):
        if "One word:" in user:
            return {"text": "SUPPORTED", "usage": {}, "gen_seconds": 0.0,
                    "model": self.model}
        if "weather" in user.lower():
            return {"text": self.abstain, "usage": {}, "gen_seconds": 0.0,
                    "model": self.model}
        return {"text": "Open Settings and choose Orders to cancel [1][2].",
                "usage": {"total_tokens": 42}, "gen_seconds": 0.0,
                "model": self.model}


@pytest.fixture()
def client():
    cfg = load_config()
    STATE["pipeline"] = SupportPipeline(
        cfg, retriever=FakeRetriever(), reranker=FakeReranker(),
        generator=FakeGenerator(), masker=NoopMasker())
    with TestClient(app) as c:
        yield c
    STATE.pop("pipeline", None)
