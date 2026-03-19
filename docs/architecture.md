# Architecture Guide

## High-Level Flow

The system is split into three major layers:

1. Offline knowledge build
2. Online retrieval and answer generation
3. Product and operations surfaces

## Offline Knowledge Build

The offline pipeline converts legal PDFs into structured knowledge:

1. PDF parsing
2. Heading and article detection
3. Chunk creation with provenance
4. Entity and relation extraction
5. Canonicalization and graph-ready artifact generation
6. Neo4j load and community detection

The key outputs are:

- `documents.jsonl`
- `chunks.jsonl`
- `entities.jsonl`
- `relations.jsonl`
- `graph_catalog.json`
- graph load statistics

## Online Runtime

At query time, the application:

1. analyzes the question
2. computes vector similarity
3. computes lexical overlap
4. computes graph/entity/path signals
5. fuses these ranking signals
6. checks evidence sufficiency
7. generates a grounded answer with citations

## Storage Design

The project uses:

- local processed artifacts on disk for repeatable pipelines
- Neo4j as the primary graph persistence layer
- Hugging Face local cache for model artifacts

The design intentionally keeps the first version lean and understandable while still being serious enough for production-style engineering.

## Application Surfaces

The project has four main operator surfaces:

- CLI for ingestion, extraction, graph building, querying, and evaluation
- FastAPI for machine-to-machine access
- Streamlit for user-facing and operator-facing workflows
- Docker Compose for local orchestration

## Dashboard Page Roles

- `Home`: command-center overview, readiness, benchmark and corpus summary
- `Chat`: grounded question-answering with evidence inspection
- `Ops`: graph health, evaluation outputs, artifacts, and job visibility
- `Corpus Explorer`: inspect the source chunks and graph context directly
- `Project Guide`: in-app product, architecture, and operations documentation
