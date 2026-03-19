# User Guide

## Before You Start

Make sure you have:

- the `RAGenv` Conda environment
- the three legal PDFs in `data/raw/`
- Docker Desktop running if you want the full container stack

## Main Ways To Use The Project

### 1. Dashboard

Open:

- `http://localhost:8501`

Use this when you want:

- interactive question answering
- benchmark summaries
- graph and corpus exploration
- in-app documentation

### 2. API

Open:

- `http://localhost:8000/docs`

Use this when you want:

- programmable access
- integration with other applications
- health and readiness checks

### 3. CLI

Use this when you want to run the pipeline directly:

```powershell
graphrag-engine ingest
graphrag-engine extract
graphrag-engine build-graph
graphrag-engine query "What does Article 6 require for high-risk AI systems?"
graphrag-engine run-eval
```

## Recommended First Workflow

1. Open the dashboard home page
2. Check that all three PDFs are present
3. Confirm graph stats and evaluation outputs are visible
4. Go to Chat and run a known sample question
5. Open Corpus Explorer to inspect the evidence chunks
6. Review Ops to inspect benchmark and graph quality signals

## How To Read Answers

Every answer should be interpreted alongside:

- citations
- article references
- page spans
- score breakdowns
- graph paths

The text answer matters, but the evidence trail matters more.

## When To Use Baseline vs Hybrid

- `baseline`: useful for comparison and debugging
- `hybrid`: the normal GraphRAG mode and the one you should trust more

## Common Debugging Hints

- Wrong article: inspect Corpus Explorer and score breakdowns
- Weak answer: compare hybrid vs baseline in Chat
- Missing data: rebuild ingestion, extraction, and graph artifacts
- Slow response: local generation can be slow, especially on larger questions or when comparing modes
