from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graphrag_engine.common.artifacts import read_json
from graphrag_engine.common.compat import BaseModel, Field
from graphrag_engine.common.models import IngestionJobRecord, QueryRequest
from graphrag_engine.runtime import GraphRAGRuntime

try:
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None
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


def create_app(runtime: GraphRAGRuntime | None = None):
    if FastAPI is None:  # pragma: no cover - optional dependency
        raise RuntimeError("fastapi is not installed")

    runtime = runtime or GraphRAGRuntime()
    app = FastAPI(title="GraphRAG Engine", version="0.1.0")

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
            "neo4j": neo4j,
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return health_ready()

    @app.post("/v1/ingestion/jobs")
    def create_ingestion_job(payload: IngestionRequest) -> dict:
        paths = [Path(item).resolve() for item in payload.source_paths] if payload.source_paths else None
        return runtime.ingestion.ingest(paths).model_dump()

    @app.get("/v1/ingestion/jobs/{job_id}")
    def get_ingestion_job(job_id: str) -> dict:
        job_path = runtime.settings.processed_data_path / "jobs" / f"{job_id}.json"
        if not job_path.exists():
            raise HTTPException(status_code=404, detail="Ingestion job not found")
        return IngestionJobRecord.model_validate(read_json(job_path)).model_dump()

    @app.post("/v1/query")
    def query(payload: QueryRequest) -> dict:
        return runtime.build_agent().run(payload).model_dump()

    @app.post("/v1/evaluations/run")
    def run_evaluation() -> dict:
        return runtime.build_evaluator().run().model_dump()

    @app.get("/v1/evaluations/{run_id}")
    def get_evaluation(run_id: str) -> dict:
        path = runtime.settings.processed_data_path / "evaluation" / f"{run_id}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        return read_json(path)

    @app.get("/v1/system/status")
    def system_status() -> dict[str, Any]:
        evaluation_root = runtime.settings.processed_data_path / "evaluation"
        evaluations = sorted(evaluation_root.glob("*.json"), reverse=True)
        latest_evaluation = evaluations[0].name if evaluations else None
        return {
            "provider": runtime.provider.describe(),
            "model_backend": runtime.settings.model_backend,
            "raw_files": sorted(path.name for path in runtime.settings.raw_data_path.glob("*") if path.is_file()),
            "artifact_counts": _artifact_counts(runtime),
            "neo4j": _neo4j_status(runtime),
            "latest_evaluation": latest_evaluation,
        }

    return app


app = create_app() if FastAPI is not None else None
