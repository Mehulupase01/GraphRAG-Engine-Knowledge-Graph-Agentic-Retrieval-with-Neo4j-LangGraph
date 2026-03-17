# GraphRAG Engine

Production-grade GraphRAG for the EU AI Act, GDPR, and Digital Services Act using Neo4j, agentic retrieval, and provenance-grounded answer generation.

## What this repo builds

- A corpus ingestion pipeline for regulatory PDFs with article-aware chunking and stable provenance.
- Entity and relationship extraction with canonicalization and resumable JSONL artifacts.
- Neo4j graph construction with community labels and hybrid search indexes.
- Hybrid retrieval across graph traversal, vector similarity, and lexical scoring.
- A LangGraph-style agent loop for query analysis, evidence sufficiency checks, rewrites, and multi-hop retrieval.
- FastAPI endpoints, a Streamlit chat plus ops dashboard, CLI workflows, Docker Compose, and evaluation harnesses.

## Repo layout

```text
src/graphrag_engine/
  agent/        Query planning and orchestration
  api/          FastAPI service
  cli/          Operator CLI
  common/       Config, logging, models, providers, storage helpers
  evaluation/   Baseline vs GraphRAG benchmarking
  extraction/   Entity and relation extraction
  generation/   Grounded answer synthesis
  graph/        Neo4j load + community detection
  ingestion/    PDF parsing and chunking
  retrieval/    Hybrid retrieval and rank fusion
dashboard/      Streamlit app
configs/        Default benchmark fixture and configuration
data/           Raw corpora, processed artifacts, and cache
tests/          Unit and integration coverage
```

## Quick start

1. Copy `.env.example` to `.env`.
2. Put the EU AI Act, GDPR, and DSA PDFs in `data/raw/`.
3. Create and use the Conda environment:

```powershell
conda activate RAGenv
python -m pip install -e ".[dev,local]"
```

4. Build artifacts and run the local stack:

```powershell
graphrag-engine doctor
graphrag-engine ingest
graphrag-engine extract
graphrag-engine build-graph
graphrag-engine reindex
graphrag-engine run-eval
docker compose up --build
```

## Model backends

Set `GRAPH_RAG_MODEL_BACKEND` in `.env`:

- `auto`: prefer OpenAI if credentials/base URL are configured, otherwise prefer local transformers if available, otherwise fall back to deterministic heuristics.
- `openai`: use the OpenAI SDK. This also works with OpenAI-compatible local servers if you set `GRAPH_RAG_OPENAI_BASE_URL`.
- `local`: use a local `transformers` chat model plus a `sentence-transformers` embedding model.
- `heuristic`: no external model calls; useful for smoke tests.

Local defaults are already configured for a Qwen-based setup:

```env
GRAPH_RAG_MODEL_BACKEND=local
GRAPH_RAG_LOCAL_CHAT_MODEL=Qwen/Qwen2.5-1.5B-Instruct
GRAPH_RAG_LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GRAPH_RAG_LOCAL_DEVICE=auto
```

The first time you use the `local` backend, Hugging Face will download the selected models into your local cache. That can take a while on the first run.

External OpenAI-compatible usage is also supported:

```env
GRAPH_RAG_MODEL_BACKEND=openai
GRAPH_RAG_OPENAI_API_KEY=your-key
GRAPH_RAG_OPENAI_BASE_URL=
```

## Development notes

- The code is written to degrade gracefully when optional dependencies are missing. Core tests use the standard library fallback layer and can run before the full dependency set is installed.
- Neo4j is the primary persistence target. If the driver is unavailable, graph artifacts are still produced locally so ingestion, extraction, and evaluation can continue.
- Retrieval uses provider embeddings when available and degrades to deterministic embeddings when they are not.
- On the host machine, `GRAPH_RAG_NEO4J_URI` should stay `bolt://localhost:7687`. Docker overrides that to `bolt://neo4j:7687` automatically for container-to-container communication.
- `RAGenv` is the recommended interpreter for this project because it now supports both external and local-model execution.
