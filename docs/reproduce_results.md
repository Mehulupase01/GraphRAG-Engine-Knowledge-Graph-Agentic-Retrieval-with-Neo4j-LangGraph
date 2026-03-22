# Reproduce Results

This guide explains how to reproduce the metrics quoted in the README for the current GraphRAG / PathCacheRAG branch snapshot.

It answers a simple question:

> Are these numbers real, and how do I verify them myself?

Short answer: yes. The README metrics come from generated artifacts under `data/processed/` plus validation commands run in the project environment.

## What This Guide Covers

This file shows how to reproduce:

- graph-scale metrics
- corpus distribution numbers
- benchmark scores across all retrieval modes
- path cache statistics
- test and compile validation
- runtime doctor output
- a representative adaptive query check

## Prerequisites

Use the recommended project environment:

```powershell
conda activate RAGenv
python -m pip install -e ".[dev,local]"
```

The raw source PDFs should already exist in:

```text
data/raw/
```

## 1. Rebuild The Core Artifacts

To regenerate the project outputs from scratch:

```powershell
graphrag-engine ingest
graphrag-engine extract
graphrag-engine build-graph
graphrag-engine reindex
graphrag-engine run-eval
```

These commands regenerate the main artifacts under:

- `data/processed/ingestion/`
- `data/processed/extraction/`
- `data/processed/graph/`
- `data/processed/evaluation/`
- `data/processed/path_cache/`

## 2. Reproduce The Graph Metrics

The README graph counts come from:

```text
data/processed/graph/load_stats.json
```

Inspect it directly:

```powershell
Get-Content data/processed/graph/load_stats.json
```

## 3. Reproduce The Corpus Distribution Numbers

The regulation chunk counts come from:

```text
data/processed/graph/graph_catalog.json
```

Print chunk counts by document:

```powershell
python -c "import json, pathlib, collections; p=pathlib.Path('data/processed/graph/graph_catalog.json'); data=json.loads(p.read_text(encoding='utf-8')); docs={d['document_id']:d['name'] for d in data.get('documents', [])}; counts=collections.Counter(docs.get(c['document_id'], 'Unknown') for c in data.get('chunks', [])); print(counts)"
```

## 4. Reproduce The Entity-Type Distribution

Also from `graph_catalog.json`:

```powershell
python -c "import json, pathlib, collections; p=pathlib.Path('data/processed/graph/graph_catalog.json'); data=json.loads(p.read_text(encoding='utf-8')); counts=collections.Counter(e.get('entity_type', 'unknown') for e in data.get('entities', [])); print(counts.most_common(10))"
```

## 5. Reproduce The Benchmark Numbers

The current benchmark snapshot comes from:

```text
data/processed/evaluation/eval_2fca346db6561871.json
```

Print the aggregate scores:

```powershell
python -c "import json, pathlib; p=pathlib.Path('data/processed/evaluation/eval_2fca346db6561871.json'); data=json.loads(p.read_text(encoding='utf-8')); print(json.dumps(data['aggregate_scores'], indent=2))"
```

This reproduces the current table for:

- `baseline`
- `hybrid`
- `path_hybrid`
- `path_cache`
- `adaptive`

## 6. Reproduce Path Cache Statistics

Inspect the persisted cache:

```powershell
graphrag-engine path-cache-stats
```

This verifies:

- entry count
- cache root
- schema version
- total cache size

## 7. Reproduce The Test Count

Run:

```powershell
python -m unittest discover -s tests
```

This prints the executed test count and whether they passed.

## 8. Reproduce The Compile Check

Run:

```powershell
python -m compileall dashboard src tests
```

## 9. Reproduce The Runtime Doctor Output

Run:

```powershell
python -m graphrag_engine.cli.main doctor
```

This is useful for validating:

- backend selection
- local model availability
- visible raw files
- processed directories
- path cache posture

## 10. Reproduce A Representative Adaptive Query

Use the branch's headline query:

```powershell
python -m graphrag_engine.cli.main query "What does Article 6 require for high-risk AI systems?" --mode adaptive --top-k 6
```

What to inspect in the result:

- whether top citations come from the AI Act
- whether `Article 6` appears in the evidence
- whether the response trace includes a `route` event
- whether candidate mode scores are present for adaptive arbitration

## 11. Reproduce Health Checks

Start the local stack or the API directly, then run:

```powershell
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
curl http://localhost:8000/v1/system/status
```
