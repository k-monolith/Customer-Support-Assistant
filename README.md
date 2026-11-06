# RAG Customer Support Assistant

A retrieval-augmented question answering service for customer support. It answers
questions from a help-center knowledge base, cites the passages it used, and hands
the conversation to a human agent when the knowledge base does not contain the
answer. The full stack runs locally with Docker Compose.

## Features

- **Hybrid retrieval**: dense search (Qwen3-Embedding-0.6B vectors in Qdrant) and
  BM25 keyword search, merged with Reciprocal Rank Fusion.
- **Reranking**: a cross-encoder (BAAI/bge-reranker-v2-m3) reorders the 60 fused
  candidates and keeps the best 6.
- **Cited answers**: Llama 3.1 8B (served by Ollama) answers only from the numbered
  passages and cites them, for example `[1][3]`.
- **Guardrails**: PII masking with Microsoft Presidio before anything is logged, a
  prompt-injection filter, a grounding check on every answer, and a fixed fallback
  message with escalation when a check fails or the answer is not in the passages.
- **API and web interface**: FastAPI endpoints (`/ask`, `/feedback`, `/health`) and a
  chat page with citation links, escalation notices and feedback buttons.
- **Tracing (optional)**: one Langfuse trace per request with timings for each stage.
- **Versioning**: the index version, chunking configuration hash and prompt version
  are reported by `/health`; all settings live in `config/pipeline.yaml`.

## Architecture

```
question
  -> input guards (PII masking, injection filter)
  -> dense top-50 + BM25 top-50 -> Reciprocal Rank Fusion (60 candidates)
  -> cross-encoder reranking -> top 6 passages
  -> answer generation with citations
  -> output guards (grounding check, PII scan)
  -> answer with citations, or fallback message + escalation
```

## Data

- Knowledge base: 6,221 help-center articles from [WixQA](https://huggingface.co/datasets/Wix/WixQA)
  and 26,870 customer-support responses from the
  [Bitext dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset),
  split into 38,782 chunks of about 400 tokens.
- Evaluation questions: WixQA expert-written and simulated sets (400 questions with
  reference articles) and 2,401 Bitext queries in clean, colloquial and typo variants.

## Evaluation

Evaluated with GPU notebooks on Kaggle.

| Metric | Result |
|---|---|
| Hit rate@5, Bitext queries (clean / colloquial / typos) | 0.996 / 1.000 / 0.993 |
| MRR, Bitext typo queries | 0.974 |
| Faithfulness (RAGAS, LLM judge) | 0.89 |
| Relevant article among the 60 retrieval candidates (WixQA) | 91% |

## Project structure

```
config/pipeline.yaml      all tunable settings
prompts/answer_v1.txt     versioned answer prompt
src/retrieval.py          dense + BM25 retrieval, RRF fusion
src/rerank.py             cross-encoder reranking
src/generate.py           client for any OpenAI-compatible endpoint (Ollama by default)
src/guards.py             PII masking, injection filter, citation parsing
src/answer.py             end-to-end answer pipeline
src/tracing.py            optional Langfuse tracing
src/index_load.py         loads the index artifacts into Qdrant
src/api/main.py           FastAPI application
src/api/static/index.html chat interface
tests/                    API and guard tests
```

## Getting started

### 1. Index artifacts

The index files are not stored in git because of their size. Place these four files
in `./artifacts/`:

- `embeddings_support_kb_v1.npy` (about 152 MB)
- `index_payload_support_kb_v1.parquet` (about 15 MB)
- `bm25_corpus_support_kb_v1.pkl`
- `index_meta_support_kb_v1.json`

### 2. Run with Docker (Windows, macOS, Linux)

Requires Docker Desktop (or Docker Engine with Compose).

```bash
docker compose up -d qdrant ollama
docker compose exec ollama ollama pull llama3.1:8b     # once, about 5 GB
docker compose build api
docker compose run --rm api python scripts/verify_artifacts.py /app/artifacts  # integrity check
docker compose run --rm api python -m src.index_load   # once per index version
docker compose up -d api
```

Open http://localhost:8000 for the chat page. API documentation is at `/docs`.

### 3. Run without Docker (development)

Requires Python 3.11+, [Ollama](https://ollama.com) and a Qdrant server
(`docker run -d -p 6333:6333 qdrant/qdrant`).

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
ollama pull llama3.1:8b
python scripts/verify_artifacts.py      # integrity check on ./artifacts
python -m src.index_load
uvicorn src.api.main:app --reload
```

## Configuration

Settings are read from `config/pipeline.yaml`. Endpoints and credentials come from
environment variables only; see `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `GEN_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible generation endpoint |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server |
| `ARTIFACTS_DIR` | `./artifacts` | index artifact folder |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | unset | enable tracing when all three are set |

## Tests

```bash
pip install -r requirements-test.txt
pytest -q
```

The tests run the real answer pipeline with stubbed retrieval and generation, so no
models or services are needed. They cover a normal question with citations, an
out-of-scope question (fallback and escalation), an injection attempt, the feedback
endpoint, the health endpoint and the chat page. They also run on every push through
GitHub Actions.

## License

Code is released under the MIT License. The datasets and models keep their own
licenses; no dataset content or model weights are included in this repository.
