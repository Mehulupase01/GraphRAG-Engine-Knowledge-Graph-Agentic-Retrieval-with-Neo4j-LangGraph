# Operations Runbook

## Startup

### Full Stack

```powershell
docker compose up --build
```

### Local App Only

```powershell
python -m uvicorn graphrag_engine.api.app:app --host 127.0.0.1 --port 8000
streamlit run dashboard/Home.py
```

### Health Checks

```powershell
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
curl http://localhost:8000/v1/system/status
```

## Rebuild The Knowledge Base

```powershell
graphrag-engine ingest
graphrag-engine extract
graphrag-engine build-graph
graphrag-engine reindex
graphrag-engine run-eval
```

## PathCacheRAG Operator Commands

```powershell
graphrag-engine doctor
graphrag-engine query "What does Article 6 require for high-risk AI systems?" --mode adaptive
graphrag-engine path-cache-stats
graphrag-engine clear-path-cache
```

Use these when you need to:

- confirm local model readiness
- inspect cache size and schema version
- reset persisted path artifacts
- reproduce adaptive route behavior

## Monitoring Signals

Watch these first:

- readiness endpoint status
- graph artifact counts
- evaluation regressions
- retrieval quality on representative questions
- path cache hit rate
- adaptive route choices on known sample prompts

## Production Readiness Notes

- change the default Neo4j password before any shared deployment
- set `GRAPH_RAG_API_KEY` if you want API-level protection
- keep `data/` mounted as a volume so processed artifacts survive restarts
- use `doctor` after any backend change
- preserve the benchmark artifact that matches the branch you plan to ship

## Known Practical Limits

- local model generation is slower than hosted APIs
- adaptive mode can improve quality while adding route-arbitration overhead
- benchmark loops are intentionally lightweight enough to be practical on consumer hardware
- corpus and graph quality still benefit from periodic extraction refinement
