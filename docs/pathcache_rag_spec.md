# PathCacheRAG Branch Spec

## Purpose

This branch evolves the current GraphRAG system into a more novel retrieval architecture built around:

- path-centric retrieval
- path-grounded prompting
- persistent subgraph or path caching
- stronger visual evidence exploration

The goal is not to discard the current GraphRAG engine, but to extend it with a retrieval style that is better aligned with multi-hop legal reasoning.

## Why This Branch Exists

The current system already supports:

- chunk-aware retrieval
- lexical and semantic ranking
- graph traversal signals
- grounded answers with citations

PathCacheRAG pushes the design further by making graph paths the primary evidence object rather than treating them only as a side signal.

## High-Level Idea

Instead of:

1. retrieve chunks
2. rank chunks
3. answer from chunks

PathCacheRAG aims to do:

1. identify seed entities and article anchors
2. expand and score legal graph paths
3. turn the highest-value paths into evidence packs
4. cache recurring path bundles for repeated legal query patterns
5. generate answers from path-grounded evidence

## Retrieval Modes

The branch should ultimately support these modes:

- `baseline`
- `hybrid`
- `path_hybrid`
- `path_cache`

## Phase Plan

### Phase 0

- create and isolate the feature branch
- lock the architecture and evaluation goals
- identify benchmark questions that stress multi-hop legal reasoning

### Phase 1

- implement path-aware retrieval
- enumerate candidate legal paths from matched seed entities
- rank paths and convert them into chunk-aligned evidence
- surface path metadata through traces

### Phase 2

- add path textualization and path-grounded prompt packaging
- compare chunk-only vs path-only vs mixed prompting

### Phase 3

- add persistent path or subgraph caching
- define cache keys, cache metadata, and cache invalidation behavior
- record cache hit and miss metrics

### Phase 4

- extend the agent workflow to route between hybrid and path-aware modes
- add path-aware fallback behavior

### Phase 5

- extend evaluation to compare baseline, GraphRAG, PathRAG, and PathCacheRAG
- add latency and cache-efficiency metrics

### Phase 6

- add a Path Explorer dashboard page
- show retrieved paths, supporting chunks, and cache behavior visually

### Phase 7

- optimize path pruning and traversal cost
- add stronger tests for ranking stability and cache correctness

### Phase 8

- finalize documentation, benchmarks, and release narrative

## Success Criteria

This branch is successful when it can demonstrate:

- better multi-hop retrieval quality than the current GraphRAG mode
- stable article-specific precision on legal queries
- measurable repeated-query speedup through caching
- more interpretable evidence paths in the dashboard

## Current Status

The first implementation slice in this branch has already added:

- `path_hybrid` retrieval mode
- `path_cache` retrieval mode
- persistent cache entries under `data/processed/path_cache/`
- retrieval trace metadata for path count and cache hit status
- chat UI support for the new retrieval modes

This is the foundation, not the finished branch.
