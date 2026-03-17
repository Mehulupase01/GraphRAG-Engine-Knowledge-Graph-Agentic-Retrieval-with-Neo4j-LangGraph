from __future__ import annotations

import tempfile
import unittest

from graphrag_engine.common.artifacts import write_json, write_jsonl
from graphrag_engine.common.models import ChunkRecord, DocumentRecord, EntityRecord, QueryRequest, RelationRecord
from graphrag_engine.common.providers import HeuristicLLMProvider
from graphrag_engine.common.settings import Settings
from graphrag_engine.agent.workflow import GraphRAGAgent
from graphrag_engine.retrieval.service import HybridRetriever


class WorkflowTests(unittest.TestCase):
    def test_agent_returns_citations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(data_dir=tmp)
            graph_dir = settings.processed_data_path / "graph"
            document = DocumentRecord(document_id="doc-1", name="AI Act", source_path="x", checksum="y")
            chunk = ChunkRecord(
                chunk_id="chunk-1",
                document_id="doc-1",
                section_id="sec-1",
                article_ref="Article 6",
                text="Article 6 of the AI Act defines high-risk systems and requires providers to assess conformity.",
                text_hash="hash",
            )
            entity_a = EntityRecord(
                entity_id="ent-a",
                canonical_name="AI Act",
                raw_name="AI Act",
                entity_type="regulation",
                source_chunk_id="chunk-1",
            )
            entity_b = EntityRecord(
                entity_id="ent-b",
                canonical_name="High-Risk",
                raw_name="high-risk",
                entity_type="risk_class",
                source_chunk_id="chunk-1",
            )
            relation = RelationRecord(
                relation_id="rel-1",
                subject_entity_id="ent-a",
                object_entity_id="ent-b",
                relation_type="defines",
                source_chunk_id="chunk-1",
            )
            write_json(
                graph_dir / "graph_catalog.json",
                {
                    "documents": [document.model_dump()],
                    "chunks": [chunk.model_dump()],
                    "entities": [entity_a.model_dump(), entity_b.model_dump()],
                    "relations": [relation.model_dump()],
                    "communities": {"ent-a": 0, "ent-b": 0},
                },
            )
            provider = HeuristicLLMProvider(settings)
            agent = GraphRAGAgent(settings, provider, HybridRetriever(settings, provider))
            response = agent.run(QueryRequest(question="What does Article 6 require for high-risk systems?"))
            self.assertTrue(response.citations)
            self.assertIn("Article 6", response.answer)


if __name__ == "__main__":
    unittest.main()

