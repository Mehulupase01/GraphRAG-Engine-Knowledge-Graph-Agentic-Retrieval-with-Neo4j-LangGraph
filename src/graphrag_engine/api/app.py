from __future__ import annotations

from pathlib import Path

from graphrag_engine.common.artifacts import read_json
from graphrag_engine.common.compat import BaseModel, Field
from graphrag_engine.common.models import IngestionJobRecord, QueryRequest
from graphrag_engine.runtime import GraphRAGRuntime

try:
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None
    HTTPException = RuntimeError


class IngestionRequest(BaseModel):
    source_paths: list[str] = Field(default_factory=list)


def create_app():
    if FastAPI is None:  # pragma: no cover - optional dependency
        raise RuntimeError("fastapi is not installed")

    runtime = GraphRAGRuntime()
    app = FastAPI(title="GraphRAG Engine", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        provider = runtime.provider.describe()
        return {
            "status": "ok",
            "provider": provider["provider"],
            "model_backend": runtime.settings.model_backend,
        }

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

    return app


app = create_app() if FastAPI is not None else None
