# Operations Runbook

## Startup

### Full Stack

```powershell
docker compose up --build
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
graphrag-engine run-eval
```

## Production Readiness Notes

- Change the default Neo4j password before any serious deployment
- Set `GRAPH_RAG_API_KEY` if you want API-level protection
- Keep `data/` mounted as a volume so processed artifacts survive restarts
- Use local mode for development and controlled demos
- Use stronger external backends for higher-stakes final validation when keys are available

## Monitoring Signals

Watch these first:

- readiness endpoint status
- graph artifact counts
- benchmark regressions
- retrieval quality on representative questions

## Known Practical Limits

- local model generation is slower than hosted frontier APIs
- benchmark loops are intentionally simplified in local mode
- corpus and graph quality still benefit from periodic extraction refinement
