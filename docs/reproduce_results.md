# Reproduce Results

This guide explains how to reproduce the key results quoted in the README.

It is meant to answer a simple question clearly:

> Are the numbers in this repository real, and how can I verify them myself?

Short answer: yes. The README metrics come from generated artifacts inside `data/processed/` plus local validation commands run in the project environment.

## What This Guide Covers

This file shows how to reproduce:

- graph-scale metrics
- corpus distribution numbers
- benchmark scores and deltas
- entity-type distribution summaries
- test and compile validation
- runtime doctor output
- a representative end-to-end query check

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

Expected documents:

- AI Act
- GDPR
- Digital Services Act

## 1. Rebuild The Core Artifacts

If you want to regenerate the data from scratch instead of only inspecting the saved outputs, run:

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

## 2. Reproduce The Graph Metrics In README

The README reports graph counts such as:

- documents
- chunks
- entities
- relations
- communities
- whether Neo4j was used during graph load

These come from:

```text
data/processed/graph/load_stats.json
```

### Inspect the raw file

```powershell
Get-Content data/processed/graph/load_stats.json
```

### Print a clean summary

```powershell
python -c "import json, pathlib; p=pathlib.Path('data/processed/graph/load_stats.json'); data=json.loads(p.read_text(encoding='utf-8')); print(json.dumps(data, indent=2))"
```

## 3. Reproduce The Corpus Distribution Numbers

The README pie chart showing chunk distribution by regulation comes from:

```text
data/processed/graph/graph_catalog.json
```

### Print chunk counts by document

```powershell
python -c "import json, pathlib, collections; p=pathlib.Path('data/processed/graph/graph_catalog.json'); data=json.loads(p.read_text(encoding='utf-8')); docs={d['document_id']:d['name'] for d in data.get('documents', [])}; counts=collections.Counter(docs.get(c['document_id'], 'Unknown') for c in data.get('chunks', [])); print(json.dumps(counts, indent=2))"
```

That is the source for values like:

- AI Act chunks
- GDPR chunks
- Digital Services Act chunks

## 4. Reproduce The Entity-Type Distribution Table

The README table for top extracted entity groups also comes from:

```text
data/processed/graph/graph_catalog.json
```

### Print the top entity types

```powershell
python -c "import json, pathlib, collections; p=pathlib.Path('data/processed/graph/graph_catalog.json'); data=json.loads(p.read_text(encoding='utf-8')); counts=collections.Counter(e.get('entity_type', 'unknown') for e in data.get('entities', [])); print(json.dumps(counts.most_common(12), indent=2))"
```

## 5. Reproduce The Benchmark Numbers In README

The README benchmark table is based on the saved evaluation output:

```text
data/processed/evaluation/eval_2fca346db6561871.json
```

If you rerun evaluation, the run ID may be different. In that case, inspect the latest file under:

```text
data/processed/evaluation/
```

### Inspect the stored benchmark file

```powershell
Get-Content data/processed/evaluation/eval_2fca346db6561871.json
```

### Print aggregate scores only

```powershell
python -c "import json, pathlib; p=pathlib.Path('data/processed/evaluation/eval_2fca346db6561871.json'); data=json.loads(p.read_text(encoding='utf-8')); print(json.dumps(data['aggregate_scores'], indent=2))"
```

### Compute the percentage deltas shown in README

```powershell
python -c "import json, pathlib; p=pathlib.Path('data/processed/evaluation/eval_2fca346db6561871.json'); data=json.loads(p.read_text(encoding='utf-8')); b=data['aggregate_scores']['baseline']; g=data['aggregate_scores']['graphrag']; out={k: round(((g[k]-b[k]) / b[k])*100, 2) if b[k] else None for k in b}; print(json.dumps(out, indent=2))"
```

This reproduces values such as:

- average score delta
- context precision delta
- answer relevancy delta
- multi-hop accuracy delta

## 6. Reproduce The Test Counts

The README quotes the validated test pass count from the real project environment.

Run:

```powershell
python -m unittest discover -s tests
```

This prints the total number of executed tests and whether they passed.

If you want the provider-specific regression test that recently fixed CI:

```powershell
python -m unittest tests.test_providers
```

## 7. Reproduce The Compile Check

The README also references compile validation across the repo.

Run:

```powershell
python -m compileall dashboard src tests
```

This checks that the Python sources compile cleanly.

## 8. Reproduce The Runtime Doctor Output

The runtime doctor is useful for verifying:

- active backend
- visible raw files
- processed directories
- model backend configuration
- provider metadata

Run:

```powershell
python -m graphrag_engine.cli.main doctor
```

This is especially useful when validating:

- local backend availability
- embedding model wiring
- whether the right project data directory is being used

## 9. Reproduce A Representative End-To-End Query

One of the practical quality checks used during development was a representative legal query:

```text
What does Article 6 require for high-risk AI systems?
```

### Via CLI

```powershell
python -m graphrag_engine.cli.main query "What does Article 6 require for high-risk AI systems?" --mode hybrid --top-k 4
```

### Via API

First start the API:

```powershell
python -m uvicorn graphrag_engine.api.app:app --host 127.0.0.1 --port 8000
```

Then send the query:

```powershell
$body = @{ question = 'What does Article 6 require for high-risk AI systems?'; retrieval_mode = 'hybrid'; top_k = 4; debug = $true } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/v1/query' -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 6
```

What you should inspect in the response:

- whether the top citations come from the **AI Act**
- whether the answer is anchored to **Article 6**
- whether the citation list includes page spans and score breakdowns
- whether graph paths are present

## 10. Reproduce The Health Checks

Start the local stack or the API directly, then run:

```powershell
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
curl http://localhost:8000/v1/system/status
```

If `GRAPH_RAG_API_KEY` is configured, add:

```powershell
curl -H "X-API-Key: your-key" http://localhost:8000/v1/system/status
```

These endpoints help confirm:

- whether the API is alive
- whether artifacts are present
- whether Neo4j is reachable
- which provider/backend is currently active

## 11. Mapping README Claims To Their Sources

| README claim | Source |
| --- | --- |
| Graph counts | `data/processed/graph/load_stats.json` |
| Corpus chunk distribution | `data/processed/graph/graph_catalog.json` |
| Top entity-type distribution | `data/processed/graph/graph_catalog.json` |
| Benchmark scores and deltas | `data/processed/evaluation/eval_2fca346db6561871.json` |
| Test pass count | `python -m unittest discover -s tests` |
| Compile validation | `python -m compileall dashboard src tests` |
| Backend/provider snapshot | `python -m graphrag_engine.cli.main doctor` |
| End-to-end legal answer example | CLI query or `/v1/query` API call |

## 12. What Is Real Vs What Is Snapshot Documentation

This is the important nuance:

- the README metrics are **real**
- they are based on **actual generated artifacts** and **actual local validation commands**
- but they are still a **snapshot in time**, not live-updating values

So if you rebuild the graph or rerun evaluation later, some numbers may change. That is expected.

## 13. Reviewer Shortcut

If someone wants the shortest path to verify that the README is grounded in real outputs, these are the most important commands:

```powershell
python -m unittest discover -s tests
python -m compileall dashboard src tests
python -m graphrag_engine.cli.main doctor
python -c "import json, pathlib; print(json.dumps(json.loads(pathlib.Path('data/processed/graph/load_stats.json').read_text(encoding='utf-8')), indent=2))"
python -c "import json, pathlib; data=json.loads(pathlib.Path('data/processed/evaluation/eval_2fca346db6561871.json').read_text(encoding='utf-8')); print(json.dumps(data['aggregate_scores'], indent=2))"
python -m graphrag_engine.cli.main query \"What does Article 6 require for high-risk AI systems?\" --mode hybrid --top-k 4
```

That is usually enough to confirm that the project has real artifacts, real tests, real evaluation output, and a real end-to-end query path.
