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
- `adaptive`

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

The current branch now includes:

- `path_hybrid` retrieval mode
- `path_cache` retrieval mode
- `adaptive` retrieval mode that compares candidate evidence packs before selecting a final route
- persistent cache entries under `data/processed/path_cache/`
- cache schema versioning and invalidation-safe cache keys
- retrieval trace metadata for:
  - path count
  - cache hit status
  - cache lookup latency
  - path enumeration latency
  - total retrieval latency
- adaptive routing metadata for:
  - heuristic preselection
  - candidate arbitration scores
  - selected retrieval mode
- chat UI support for the new retrieval modes
- a dedicated `Path Explorer` dashboard page with adaptive route inspection
- cache diagnostics in the CLI, API, and ops dashboard
- evaluation reports that now include per-mode latency and cache-hit metrics

## Current Benchmark Snapshot

Latest validated PathCacheRAG branch evaluation:

- `baseline average_score = 0.3491`
- `adaptive average_score = 0.3704`
- `hybrid average_score = 0.3658`
- `path_cache average_score = 0.3556`
- `path_hybrid average_score = 0.3556`

Key operational takeaways from that snapshot:

- `adaptive` is now the strongest overall retrieval mode on this branch
- `adaptive` outperforms both baseline and fixed-mode hybrid on the current benchmark
- `path_cache` is above baseline on overall score and preserves the strongest repeated-query speed story
- `path_cache` cache hit rate is `1.0` in the warmed benchmark path
- `path_cache` average latency is much lower than `path_hybrid`
- `path_hybrid` remains the more expensive exploratory mode
- `adaptive` raises quality by arbitration, but its end-to-end latency is higher than fixed-mode hybrid because it evaluates multiple candidate evidence packs before answer generation

## Operator Commands

Useful branch-specific commands:

- `graphrag-engine doctor`
- `graphrag-engine path-cache-stats`
- `graphrag-engine clear-path-cache`
- `graphrag-engine query "..." --mode adaptive`
- `graphrag-engine query "..." --mode path_hybrid`
- `graphrag-engine query "..." --mode path_cache`
- `graphrag-engine run-eval`

## Remaining Improvement Opportunities

The branch is already meaningful, but the next research-quality improvements would be:

- smarter path pruning for broad entities like `AI System`
- better path textualization for low-resource local models
- adaptive routing that learns from past benchmark deltas instead of using only rule-based arbitration
- stronger legal-path scoring for cross-regulation questions
- optional persistence of adaptive routing analytics for offline policy tuning
