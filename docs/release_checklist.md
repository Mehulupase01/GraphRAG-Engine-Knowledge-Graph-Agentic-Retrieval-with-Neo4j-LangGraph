# GraphRAG Engine Release Checklist

Use this checklist before tagging a milestone as production-ready.

## Data And Graph

- Confirm the three source PDFs in `data/raw/` are the intended corpus versions.
- Run `graphrag-engine ingest`, `extract`, and `build-graph` without errors.
- Verify the latest graph load stats show non-zero documents, chunks, entities, and relations.
- Check a few representative questions and confirm citations point to the right regulation and article.

## Model Backends

- Verify the configured backend in `.env` is the one you intend to ship.
- For external backends, confirm the corresponding API key is present and valid.
- For local backends, confirm model files are already cached and the target device is available.
- Run `graphrag-engine doctor` and inspect the reported provider metadata.

## API And Dashboard

- Start the stack with `docker compose up --build`.
- Confirm Neo4j passes its container health check.
- Confirm `http://localhost:8000/health/live` returns `ok`.
- Confirm `http://localhost:8000/health/ready` reports `ready` and shows artifact counts.
- Confirm the Streamlit dashboard loads, the chat page answers a known query, and the ops page shows the latest evaluation file.

## Quality Gates

- Run `python -m unittest discover -s tests`.
- Run `python -m compileall dashboard src tests`.
- Run `graphrag-engine run-eval` and compare GraphRAG against baseline.
- Review the latest evaluation JSON for regressions or obviously bad retrieval patterns.

## Deployment Hygiene

- Replace default secrets like the Neo4j password before any shared deployment.
- Review `.env` and remove unused keys.
- Ensure logs do not expose secrets or raw API keys.
- Confirm Docker volumes and artifact paths are correct for the target machine.
- Commit updated README and ops notes alongside code changes.
