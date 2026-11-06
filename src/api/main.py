"""FastAPI service: /ask, /feedback, /health, plus the chat UI at /.
Models load once at startup. The pipeline is provided through a dependency so
tests can substitute a stub without touching the endpoints."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import load_config
from ..tracing import make_tracer

STATE: dict = {}


def build_pipeline(cfg):
    from ..answer import SupportPipeline
    from ..generate import Generator
    from ..guards import NoopMasker, PIIMasker
    from ..rerank import Reranker
    from ..retrieval import HybridRetriever

    masker = PIIMasker() if cfg["guards"]["pii_masking"] else NoopMasker()
    return SupportPipeline(cfg,
                           retriever=HybridRetriever(cfg),
                           reranker=Reranker(cfg),
                           generator=Generator(cfg),
                           masker=masker)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    STATE["cfg"] = cfg
    STATE["tracer"] = make_tracer(cfg)
    if "pipeline" not in STATE:          # tests pre-populate a stub
        STATE["pipeline"] = build_pipeline(cfg)
    yield


app = FastAPI(title="Support Assistant", lifespan=lifespan)


def get_pipeline():
    return STATE["pipeline"]


class Ask(BaseModel):
    question: str
    session_id: str | None = None


class Feedback(BaseModel):
    request_id: str
    helpful: bool
    comment: str | None = None


@app.post("/ask")
def ask(req: Ask, pipeline=Depends(get_pipeline)):
    result = pipeline.answer(req.question)
    STATE["tracer"].trace_request(result)
    return {"request_id": result["request_id"],
            "answer": result["answer"],
            "citations": result["citations"],
            "escalate": result["escalate"],
            "fallback_links": result.get("fallback_links"),
            "latency_s": result["trace"]["stages"].get("total_s")}


@app.post("/feedback")
def feedback(fb: Feedback):
    STATE["tracer"].attach_feedback(fb.request_id, fb.helpful, fb.comment)
    return {"stored": True}


@app.get("/health")
def health():
    cfg = STATE["cfg"]
    return {"status": "ok",
            "kb_version": cfg["version"],
            "chunk_config_hash": cfg["chunking"]["config_hash"],
            "prompt_version": "answer_v1",
            "generation_model": cfg["generation"]["model"],
            "commit": os.environ.get("GIT_COMMIT", "unknown")}


@app.get("/")
def ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
