from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graphrag_engine.common.artifacts import read_json, read_jsonl
from graphrag_engine.common.compat import BaseModel, Field
from graphrag_engine.common.models import IngestionJobRecord, QueryRequest
from graphrag_engine.retrieval.path_cache import PathCacheStore
from graphrag_engine.runtime import GraphRAGRuntime

try:
    from fastapi import Depends, FastAPI, Header, HTTPException
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None
    Depends = None
    Header = None
    HTTPException = RuntimeError

try:
    from neo4j import GraphDatabase  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    GraphDatabase = None


class IngestionRequest(BaseModel):
    source_paths: list[str] = Field(default_factory=list)


def _artifact_counts(runtime: GraphRAGRuntime) -> dict[str, int]:
    graph_catalog = runtime.settings.processed_data_path / "graph" / "graph_catalog.json"
    if not graph_catalog.exists():
        return {"documents": 0, "chunks": 0, "entities": 0, "relations": 0}
    payload = json.loads(graph_catalog.read_text(encoding="utf-8"))
    return {
        "documents": len(payload.get("documents", [])),
        "chunks": len(payload.get("chunks", [])),
        "entities": len(payload.get("entities", [])),
        "relations": len(payload.get("relations", [])),
    }


def _path_cache_counts(runtime: GraphRAGRuntime) -> dict[str, int]:
    return PathCacheStore(runtime.settings).stats()


def _adaptive_route_stats(runtime: GraphRAGRuntime) -> dict[str, Any]:
    path = runtime.settings.processed_data_path / "analytics" / "adaptive_routes.jsonl"
    events = read_jsonl(path)
    if not events:
        return {"events": 0, "selected_modes": {}, "cache_hit_rate": 0.0, "avg_latency_ms": 0.0}
    selected_modes: dict[str, int] = {}
    cache_hits = 0
    total_latency = 0.0
    for event in events:
        mode = str(event.get("selected_mode", "unknown"))
        selected_modes[mode] = selected_modes.get(mode, 0) + 1
        if bool(event.get("cache_hit")):
            cache_hits += 1
        total_latency += float(event.get("total_latency_ms", 0.0))
    return {
        "events": len(events),
        "selected_modes": selected_modes,
        "cache_hit_rate": round(cache_hits / max(len(events), 1), 4),
        "avg_latency_ms": round(total_latency / max(len(events), 1), 2),
        "path": str(path),
    }


def _neo4j_status(runtime: GraphRAGRuntime) -> dict[str, Any]:
    if GraphDatabase is None:
        return {"available": False, "reason": "neo4j driver not installed"}
    driver = None
    try:
        driver = GraphDatabase.driver(
            runtime.settings.neo4j_uri,
            auth=(runtime.settings.neo4j_user, runtime.settings.neo4j_password),
        )
        with driver.session() as session:
            result = session.run("RETURN 1 AS ready").single()
        return {"available": bool(result and result.get("ready") == 1)}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    finally:
        if driver is not None:
            driver.close()


def _runtime_warnings(runtime: GraphRAGRuntime) -> list[str]:
    warnings: list[str] = []
    if runtime.settings.neo4j_password.strip() in {"", "change-me-now", "neo4j"}:
        warnings.append("Default Neo4j password is still configured.")
    if not runtime.settings.api_key.strip():
        warnings.append("API key protection is disabled for /v1 endpoints.")
    if runtime.settings.model_backend in {"anthropic", "gemini", "openai"}:
        provider_name = runtime.provider.provider_name
        if provider_name != runtime.settings.model_backend:
            warnings.append(
                f"Configured backend '{runtime.settings.model_backend}' fell back to '{provider_name}'."
            )
    return warnings


def create_app(runtime: GraphRAGRuntime | None = None):
    if FastAPI is None:  # pragma: no cover - optional dependency
        raise RuntimeError("fastapi is not installed")

    runtime = runtime or GraphRAGRuntime()
    app = FastAPI(title="GraphRAG Engine", version="0.1.0")

    def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        expected = runtime.settings.api_key.strip()
        if not expected:
            return
        if x_api_key != expected:
            raise HTTPException(status_code=401, detail="Missing or invalid API key")

    @app.get("/health/live")
    def health_live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def health_ready() -> dict[str, Any]:
        provider = runtime.provider.describe()
        artifacts = _artifact_counts(runtime)
        neo4j = _neo4j_status(runtime)
        return {
            "status": "ready" if artifacts["chunks"] > 0 else "degraded",
            "provider": provider,
            "model_backend": runtime.settings.model_backend,
            "artifacts": artifacts,
            "path_cache": _path_cache_counts(runtime),
            "adaptive_routes": _adaptive_route_stats(runtime),
            "neo4j": neo4j,
            "warnings": _runtime_warnings(runtime),
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return health_ready()

    @app.post("/v1/ingestion/jobs")
    def create_ingestion_job(payload: IngestionRequest, _: None = Depends(require_api_key)) -> dict:
        paths = [Path(item).resolve() for item in payload.source_paths] if payload.source_paths else None
        return runtime.ingestion.ingest(paths).model_dump()

    @app.get("/v1/ingestion/jobs/{job_id}")
    def get_ingestion_job(job_id: str, _: None = Depends(require_api_key)) -> dict:
        job_path = runtime.settings.processed_data_path / "jobs" / f"{job_id}.json"
        if not job_path.exists():
            raise HTTPException(status_code=404, detail="Ingestion job not found")
        return IngestionJobRecord.model_validate(read_json(job_path)).model_dump()

    @app.post("/v1/query")
    def query(payload: QueryRequest, _: None = Depends(require_api_key)) -> dict:
        return runtime.build_agent().run(payload).model_dump()

    @app.post("/v1/evaluations/run")
    def run_evaluation(_: None = Depends(require_api_key)) -> dict:
        return runtime.build_evaluator().run().model_dump()

    @app.get("/v1/evaluations/{run_id}")
    def get_evaluation(run_id: str, _: None = Depends(require_api_key)) -> dict:
        path = runtime.settings.processed_data_path / "evaluation" / f"{run_id}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        return read_json(path)

    @app.get("/v1/system/status")
    def system_status(_: None = Depends(require_api_key)) -> dict[str, Any]:
        evaluation_root = runtime.settings.processed_data_path / "evaluation"
        evaluations = sorted(evaluation_root.glob("*.json"), reverse=True)
        latest_evaluation = evaluations[0].name if evaluations else None
        return {
            "provider": runtime.provider.describe(),
            "model_backend": runtime.settings.model_backend,
            "raw_files": sorted(path.name for path in runtime.settings.raw_data_path.glob("*") if path.is_file()),
            "artifact_counts": _artifact_counts(runtime),
            "path_cache": _path_cache_counts(runtime),
            "adaptive_routes": _adaptive_route_stats(runtime),
            "neo4j": _neo4j_status(runtime),
            "latest_evaluation": latest_evaluation,
            "warnings": _runtime_warnings(runtime),
        }

    return app


app = create_app() if FastAPI is not None else None
