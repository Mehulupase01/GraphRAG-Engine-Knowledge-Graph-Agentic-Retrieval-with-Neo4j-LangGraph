# User Guide

## Before You Start

Make sure you have:

- the `RAGenv` Conda environment
- the three legal PDFs in `data/raw/`
- local model artifacts available through `doctor`
- Docker Desktop running if you want the full container stack

## Main Ways To Use The Project

### 1. Dashboard

Open:

- `http://localhost:8501`

Use this when you want:

- interactive question answering
- mode comparison
- path inspection
- benchmark summaries
- graph and corpus exploration
- in-app documentation

### 2. API

Open:

- `http://localhost:8000/docs`

Use this when you want:

- programmable access
- health and readiness checks
- integration with other applications

### 3. CLI

Use this when you want to run the pipeline directly:

```powershell
graphrag-engine ingest
graphrag-engine extract
graphrag-engine build-graph
graphrag-engine query "What does Article 6 require for high-risk AI systems?" --mode adaptive
graphrag-engine path-cache-stats
graphrag-engine run-eval
```

## Recommended First Workflow

1. Open the dashboard home page
2. Confirm the three PDFs are present
3. Check graph counts and the latest benchmark snapshot
4. Go to Chat and run a sample question in `adaptive` mode
5. Open Path Explorer to inspect how the route was selected
6. Open Corpus Explorer to inspect the underlying source chunks
7. Review Ops to inspect benchmark, artifacts, and cache posture

## Retrieval Modes

- `baseline`: simple reference mode for comparison
- `hybrid`: strong fixed-mode GraphRAG retrieval
- `path_hybrid`: path-aware exploratory retrieval without cache reuse
- `path_cache`: path-aware retrieval with persistent cache reuse
- `adaptive`: compares candidate evidence packs and selects the strongest route

## How To Read Answers

Every answer should be interpreted alongside:

- citations
- article references
- page spans
- score breakdowns
- graph paths
- the agent trace
- the adaptive route metadata, when present

The answer text matters, but the evidence trail matters more.

## Best Pages For Different Tasks

- `Home`: overall health and benchmark posture
- `Chat`: fastest way to test grounded legal questions
- `Corpus Explorer`: inspect exact evidence chunks
- `Path Explorer`: inspect path retrieval, cache behavior, and route arbitration
- `Ops`: review graph health, evaluation results, and artifact status
- `Project Guide`: understand what the system is and how to run it

## Common Debugging Hints

- Wrong article: inspect the top citations and article alignment in Chat or Path Explorer
- Weak answer: compare `adaptive` with `baseline`
- Slow response: `path_hybrid` and local generation are the most expensive paths
- Cache confusion: run `graphrag-engine path-cache-stats`
- Cache reset needed: run `graphrag-engine clear-path-cache`
