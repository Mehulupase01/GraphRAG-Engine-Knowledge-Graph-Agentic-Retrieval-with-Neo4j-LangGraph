# GraphRAG / PathCacheRAG Release Checklist

Use this checklist before calling the current branch release-ready.

## Data And Graph

- Confirm the three source PDFs in `data/raw/` are the intended corpus versions.
- Run `graphrag-engine ingest`, `extract`, and `build-graph` without errors.
- Verify the latest graph load stats show non-zero documents, chunks, entities, and relations.
- Check a few representative article-specific questions and confirm citations point to the correct regulation and article.

## Model Backends

- Verify the configured backend in `.env` is the one you intend to ship.
- For external backends, confirm the corresponding API key is present and valid.
- For local backends, confirm model files are already cached and the target device is available.
- Run `graphrag-engine doctor` and inspect the reported provider metadata.

## PathCacheRAG-Specific Checks

- Run `graphrag-engine query "What does Article 6 require for high-risk AI systems?" --mode adaptive`.
- Confirm the response trace includes a `route` event.
- Confirm the Path Explorer page can show the selected route and candidate scores.
- Run `graphrag-engine path-cache-stats` and verify the cache root and schema version.
- If needed, run `graphrag-engine clear-path-cache` and confirm cache repopulation works on repeated queries.

## API And Dashboard

- Start the stack with `docker compose up --build`.
- Confirm Neo4j passes its container health check.
- Confirm `http://localhost:8000/health/live` returns `ok`.
- Confirm `http://localhost:8000/health/ready` reports `ready` and shows artifact counts.
- Confirm the Streamlit dashboard loads and the `Home`, `Chat`, `Ops`, `Corpus Explorer`, `Path Explorer`, and `Project Guide` pages all work.

## Quality Gates

- Run `python -m unittest discover -s tests`.
- Run `python -m compileall dashboard src tests`.
- Run `graphrag-engine run-eval`.
- Review the latest evaluation JSON for regressions or obviously bad retrieval patterns.
- Confirm the benchmark snapshot quoted in the README matches the saved artifact you intend to ship.

## Deployment Hygiene

- Replace default secrets like the Neo4j password before any shared deployment.
- Review `.env` and remove unused keys.
- Ensure logs do not expose secrets or raw API keys.
- Confirm Docker volumes and artifact paths are correct for the target machine.
- Commit updated README and branch docs alongside code changes.
