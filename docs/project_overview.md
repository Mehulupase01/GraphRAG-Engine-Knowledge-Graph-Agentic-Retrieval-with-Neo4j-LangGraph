# Project Overview

## What This Project Is

This repository is a local-first regulatory intelligence platform built around:

- the EU AI Act
- GDPR
- the Digital Services Act

It starts as a strong GraphRAG system, then extends that base with a PathCacheRAG branch that adds path-centric retrieval, persistent path caching, and adaptive route selection.

## What It Is For

This project is designed to be useful in multiple ways:

- a flagship portfolio project for retrieval and graph engineering
- a serious end-to-end AI product build
- a legal assistant that can answer with evidence and provenance
- an evaluation environment for comparing retrieval strategies
- a learning environment for ingestion, extraction, graph modeling, APIs, dashboards, and deployment

## Core Product Promise

The application can:

- ingest official legal PDFs
- preserve article-level provenance
- extract entities and relations
- build a Neo4j-backed knowledge graph
- answer questions through multiple retrieval modes
- expose answers with citations, graph paths, and traces
- compare fixed-mode and adaptive retrieval strategies through a real benchmark

## What Makes PathCacheRAG Different

Plain RAG retrieves chunks.

GraphRAG retrieves chunks plus graph signals.

PathCacheRAG goes one step further and treats graph paths as first-class evidence. It can:

- enumerate candidate legal paths
- score them against article anchors and document hints
- cache reusable path evidence packs
- compare hybrid and path-aware evidence before choosing a final route

## Current Validated Snapshot

At the current validated branch snapshot:

- `3` regulations are ingested
- `1319` chunks are indexed
- `557` canonical entities are present
- `3408` relations are present
- `147` graph communities are detected
- `54` benchmark cases are stored
- `21` tests are passing in the validated environment

## Honest Constraints

This is a strong project, but there are still real constraints:

- local generation is slower than hosted frontier APIs
- adaptive mode can improve answer quality while costing more total latency
- external providers need real keys for live validation
- internet-facing deployment still needs stronger auth and security hardening than local-first usage
