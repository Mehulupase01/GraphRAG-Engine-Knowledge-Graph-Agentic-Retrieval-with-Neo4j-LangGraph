# Architecture Guide

## High-Level Flow

The system is split into four major layers:

1. offline knowledge build
2. online GraphRAG retrieval
3. PathCacheRAG retrieval and arbitration
4. product and operations surfaces

## Offline Knowledge Build

The offline pipeline converts the legal PDFs into reusable knowledge artifacts:

1. PDF parsing
2. heading and article detection
3. chunk creation with provenance
4. entity and relation extraction
5. canonicalization and graph-ready artifact generation
6. Neo4j load and community detection

The key outputs are:

- `documents.jsonl`
- `chunks.jsonl`
- `entities.jsonl`
- `relations.jsonl`
- `graph_catalog.json`
- `load_stats.json`

## Online GraphRAG Runtime

The GraphRAG baseline path:

1. analyzes the question
2. computes vector similarity
3. computes lexical overlap
4. computes metadata and article alignment
5. computes graph/entity signals
6. fuses ranking signals
7. performs evidence sufficiency checks
8. generates a grounded answer with citations

## PathCacheRAG Runtime

The PathCacheRAG extension adds:

1. seed entity and article anchor detection
2. path enumeration across the graph
3. path scoring and chunk alignment
4. path evidence packaging
5. persistent path caching
6. adaptive arbitration between candidate retrieval modes

This means the system can compare:

- `baseline`
- `hybrid`
- `path_hybrid`
- `path_cache`
- `adaptive`

## Adaptive Retrieval Design

Adaptive mode does not just route by fixed rules. It now works as a lightweight arbiter:

1. infer candidate retrieval modes from question structure
2. run the most relevant candidate modes
3. score the returned evidence packs
4. choose the strongest route
5. record the reasoning in the response trace

This makes the final route inspectable in both the API response and the Path Explorer page.

## Storage Design

The project uses:

- local processed artifacts on disk for reproducibility
- Neo4j as the graph database
- local Hugging Face caches for model artifacts
- persisted path cache entries under `data/processed/path_cache/`

## Application Surfaces

The project exposes the system through:

- CLI for ingestion, extraction, graph build, querying, cache control, and evaluation
- FastAPI for machine-to-machine access
- Streamlit for user-facing and operator-facing workflows
- Docker Compose for local orchestration

## Dashboard Page Roles

- `Home`: command center, benchmark summary, posture, and app navigation
- `Chat`: grounded Q&A with mode comparison
- `Ops`: jobs, graph health, evaluation, artifacts, and cache posture
- `Corpus Explorer`: source chunk and graph context inspection
- `Path Explorer`: path-centric evidence and adaptive routing inspection
- `Project Guide`: in-app documentation and release guidance
