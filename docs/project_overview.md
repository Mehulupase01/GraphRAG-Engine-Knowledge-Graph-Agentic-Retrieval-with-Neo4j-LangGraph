# Project Overview

## What This Project Is

F-GEN-1 GraphRAG Engine is a local-first regulatory intelligence platform built around three EU legal corpora:

- The EU AI Act
- GDPR
- The Digital Services Act

The project is not a toy chatbot or a plain vector-search demo. It is a production-style GraphRAG system that parses raw legal PDFs, converts them into graph-aware knowledge artifacts, and answers questions with traceable evidence.

## What It Is For

This system is designed for:

- portfolio-grade demonstration of GraphRAG engineering
- experimenting with multi-hop retrieval on legal corpora
- building a regulatory assistant that can show provenance
- comparing baseline RAG against graph-augmented retrieval
- learning how ingestion, extraction, graph modeling, retrieval, evaluation, and product UX fit together

## Core Product Promise

The application should be able to:

- ingest official legal texts from raw PDFs
- preserve article-level provenance and page references
- extract legal entities and relationships
- build a Neo4j-backed knowledge graph
- combine lexical, vector, and graph retrieval
- answer questions with grounded evidence and citations
- expose the system through CLI, API, and an interactive dashboard

## What Makes It Different From Plain RAG

Plain RAG often retrieves semantically similar text chunks but struggles when the answer depends on connected legal concepts or cross-references.

This project improves on that by combining:

- article-aware chunking
- extracted entity and relation structure
- graph traversal over related concepts
- hybrid rank fusion
- agentic evidence checks and query rewrites

## Current Constraints

The project is strong and working, but there are still honest limitations:

- the local benchmark uses a lightweight evaluation path to stay practical on consumer hardware
- external provider backends are implemented, but not all are live-validated without user-supplied keys
- security hardening is optional/configurable and should be tightened before a real internet-facing deployment
