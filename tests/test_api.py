from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from graphrag_engine.api.app import create_app
from graphrag_engine.common.settings import Settings
from graphrag_engine.runtime import GraphRAGRuntime

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - optional dependency
    TestClient = None


@unittest.skipIf(TestClient is None, "fastapi test client is unavailable")
class ApiTests(unittest.TestCase):
    def test_health_and_status_endpoints(self) -> None:
        tmp = Path.cwd() / "data" / "cache" / "test_api_case"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = Settings(
                data_dir=str(tmp),
                model_backend="heuristic",
                neo4j_uri="bolt://127.0.0.1:9999",
            )
            raw_file = settings.raw_data_path / "sample.pdf"
            raw_file.write_text("placeholder", encoding="utf-8")
            graph_dir = settings.processed_data_path / "graph"
            graph_dir.mkdir(parents=True, exist_ok=True)
            (graph_dir / "graph_catalog.json").write_text(
                json.dumps(
                    {
                        "documents": [{"document_id": "doc-1", "name": "AI Act"}],
                        "chunks": [{"chunk_id": "chunk-1"}],
                        "entities": [{"entity_id": "ent-1"}],
                        "relations": [{"relation_id": "rel-1"}],
                    }
                ),
                encoding="utf-8",
            )

            app = create_app(GraphRAGRuntime(settings))
            client = TestClient(app)

            live = client.get("/health/live")
            self.assertEqual(live.status_code, 200)
            self.assertEqual(live.json()["status"], "ok")

            ready = client.get("/health/ready")
            self.assertEqual(ready.status_code, 200)
            self.assertIn("provider", ready.json())
            self.assertEqual(ready.json()["artifacts"]["chunks"], 1)
            self.assertIn("schema_version", ready.json()["path_cache"])
            self.assertIn("API key protection is disabled for /v1 endpoints.", ready.json()["warnings"])

            status = client.get("/v1/system/status")
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json()["artifact_counts"]["documents"], 1)
            self.assertIn("schema_version", status.json()["path_cache"])
            self.assertIn("sample.pdf", status.json()["raw_files"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_protected_endpoints_require_api_key_when_configured(self) -> None:
        tmp = Path.cwd() / "data" / "cache" / "test_api_auth_case"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = Settings(
                data_dir=str(tmp),
                model_backend="heuristic",
                api_key="secret-token",
                neo4j_uri="bolt://127.0.0.1:9999",
            )
            graph_dir = settings.processed_data_path / "graph"
            graph_dir.mkdir(parents=True, exist_ok=True)
            (graph_dir / "graph_catalog.json").write_text(
                json.dumps({"documents": [], "chunks": [], "entities": [], "relations": []}),
                encoding="utf-8",
            )

            app = create_app(GraphRAGRuntime(settings))
            client = TestClient(app)

            unauthorized = client.get("/v1/system/status")
            self.assertEqual(unauthorized.status_code, 401)

            authorized = client.get("/v1/system/status", headers={"X-API-Key": "secret-token"})
            self.assertEqual(authorized.status_code, 200)

            ready = client.get("/health/ready")
            self.assertEqual(ready.status_code, 200)
            self.assertNotIn("API key protection is disabled for /v1 endpoints.", ready.json()["warnings"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
