from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from graphrag_engine.agent.workflow import GraphRAGAgent
from graphrag_engine.common.artifacts import read_jsonl, write_json, write_jsonl
from graphrag_engine.common.models import ChunkRecord, DocumentRecord, EntityRecord, QueryRequest, RelationRecord
from graphrag_engine.common.providers import HeuristicLLMProvider
from graphrag_engine.common.settings import Settings
from graphrag_engine.graph.loader import GraphLoader
from graphrag_engine.retrieval.service import HybridRetriever


class RetrievalTests(unittest.TestCase):
    def test_article_and_domain_hints_prioritize_relevant_regulation(self) -> None:
        tmp = Path.cwd() / "data" / "cache" / "test_retrieval_case"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = Settings(data_dir=str(tmp))
            graph_dir = settings.processed_data_path / "graph"
            ai_act = DocumentRecord(document_id="doc-ai", name="OJ_L_202401689_EN_TXT", source_path="x", checksum="1")
            gdpr = DocumentRecord(document_id="doc-gdpr", name="CELEX_32016R0679_EN_TXT", source_path="y", checksum="2")
            ai_chunk = ChunkRecord(
                chunk_id="chunk-ai-6",
                document_id="doc-ai",
                section_id="sec-ai-6",
                article_ref="Article 6",
                text="Article 6 Classification rules for high-risk AI systems. Providers shall ensure conformity assessment for high-risk AI systems before placing them on the market.",
                text_hash="hash-ai",
                metadata={"section_title": "Article 6"},
            )
            gdpr_chunk = ChunkRecord(
                chunk_id="chunk-gdpr-6",
                document_id="doc-gdpr",
                section_id="sec-gdpr-6",
                article_ref="Article 6",
                text="Article 6 Lawfulness of processing. Processing shall be lawful only if one of the legal bases applies to personal data.",
                text_hash="hash-gdpr",
                metadata={"section_title": "Article 6"},
            )
            write_json(
                graph_dir / "graph_catalog.json",
                {
                    "documents": [ai_act.model_dump(), gdpr.model_dump()],
                    "chunks": [ai_chunk.model_dump(), gdpr_chunk.model_dump()],
                    "entities": [
                        EntityRecord(
                            entity_id="ent-ai-act",
                            canonical_name="AI Act",
                            raw_name="AI Act",
                            entity_type="regulation",
                            source_chunk_id="chunk-ai-6",
                            metadata={"mention_chunk_ids": ["chunk-ai-6"], "document_ids": ["doc-ai"]},
                        ).model_dump(),
                        EntityRecord(
                            entity_id="ent-gdpr",
                            canonical_name="GDPR",
                            raw_name="GDPR",
                            entity_type="regulation",
                            source_chunk_id="chunk-gdpr-6",
                            metadata={"mention_chunk_ids": ["chunk-gdpr-6"], "document_ids": ["doc-gdpr"]},
                        ).model_dump(),
                        EntityRecord(
                            entity_id="ent-article-6",
                            canonical_name="Article 6",
                            raw_name="Article 6",
                            entity_type="article",
                            source_chunk_id="chunk-ai-6",
                            metadata={"mention_chunk_ids": ["chunk-ai-6", "chunk-gdpr-6"]},
                        ).model_dump(),
                        EntityRecord(
                            entity_id="ent-high-risk",
                            canonical_name="High-Risk AI System",
                            raw_name="high-risk AI systems",
                            entity_type="risk_class",
                            source_chunk_id="chunk-ai-6",
                            metadata={"mention_chunk_ids": ["chunk-ai-6"]},
                        ).model_dump(),
                    ],
                    "relations": [
                        RelationRecord(
                            relation_id="rel-ai-1",
                            subject_entity_id="ent-article-6",
                            object_entity_id="ent-high-risk",
                            relation_type="defines",
                            source_chunk_id="chunk-ai-6",
                            confidence=0.9,
                        ).model_dump()
                    ],
                    "communities": {"ent-ai-act": 0, "ent-gdpr": 1, "ent-article-6": 0, "ent-high-risk": 0},
                    "mentions": [
                        {"chunk_id": "chunk-ai-6", "entity_id": "ent-ai-act"},
                        {"chunk_id": "chunk-ai-6", "entity_id": "ent-article-6"},
                        {"chunk_id": "chunk-ai-6", "entity_id": "ent-high-risk"},
                        {"chunk_id": "chunk-gdpr-6", "entity_id": "ent-gdpr"},
                        {"chunk_id": "chunk-gdpr-6", "entity_id": "ent-article-6"},
                    ],
                },
            )
            provider = HeuristicLLMProvider(settings)
            retriever = HybridRetriever(settings, provider)
            hits = retriever.retrieve("What does Article 6 require for high-risk AI systems?", top_k=2)
            self.assertEqual(hits[0].chunk.chunk_id, "chunk-ai-6")
            self.assertEqual(hits[0].document_name, "AI Act")
            self.assertGreater(hits[0].metadata_score, hits[1].metadata_score)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_path_modes_generate_ranked_paths_and_cache_entries(self) -> None:
        tmp = Path.cwd() / "data" / "cache" / "test_path_retrieval_case"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = Settings(data_dir=str(tmp))
            graph_dir = settings.processed_data_path / "graph"
            ai_act = DocumentRecord(document_id="doc-ai", name="OJ_L_202401689_EN_TXT", source_path="x", checksum="1")
            gdpr = DocumentRecord(document_id="doc-gdpr", name="CELEX_32016R0679_EN_TXT", source_path="y", checksum="2")
            ai_chunk = ChunkRecord(
                chunk_id="chunk-ai-6",
                document_id="doc-ai",
                section_id="sec-ai-6",
                article_ref="Article 6",
                text="Article 6 Classification rules for high-risk AI systems. Providers shall ensure conformity assessment for high-risk AI systems before placing them on the market.",
                text_hash="hash-ai",
                metadata={"section_title": "Article 6"},
            )
            gdpr_chunk = ChunkRecord(
                chunk_id="chunk-gdpr-6",
                document_id="doc-gdpr",
                section_id="sec-gdpr-6",
                article_ref="Article 6",
                text="Article 6 Lawfulness of processing. Processing shall be lawful only if one of the legal bases applies to personal data.",
                text_hash="hash-gdpr",
                metadata={"section_title": "Article 6"},
            )
            write_json(
                graph_dir / "graph_catalog.json",
                {
                    "documents": [ai_act.model_dump(), gdpr.model_dump()],
                    "chunks": [ai_chunk.model_dump(), gdpr_chunk.model_dump()],
                    "entities": [
                        EntityRecord(
                            entity_id="ent-ai-act",
                            canonical_name="AI Act",
                            raw_name="AI Act",
                            entity_type="regulation",
                            source_chunk_id="chunk-ai-6",
                            metadata={"mention_chunk_ids": ["chunk-ai-6"], "document_ids": ["doc-ai"]},
                        ).model_dump(),
                        EntityRecord(
                            entity_id="ent-gdpr",
                            canonical_name="GDPR",
                            raw_name="GDPR",
                            entity_type="regulation",
                            source_chunk_id="chunk-gdpr-6",
                            metadata={"mention_chunk_ids": ["chunk-gdpr-6"], "document_ids": ["doc-gdpr"]},
                        ).model_dump(),
                        EntityRecord(
                            entity_id="ent-article-6",
                            canonical_name="Article 6",
                            raw_name="Article 6",
                            entity_type="article",
                            source_chunk_id="chunk-ai-6",
                            metadata={"mention_chunk_ids": ["chunk-ai-6", "chunk-gdpr-6"]},
                        ).model_dump(),
                        EntityRecord(
                            entity_id="ent-high-risk",
                            canonical_name="High-Risk AI System",
                            raw_name="high-risk AI systems",
                            entity_type="risk_class",
                            source_chunk_id="chunk-ai-6",
                            metadata={"mention_chunk_ids": ["chunk-ai-6"]},
                        ).model_dump(),
                    ],
                    "relations": [
                        RelationRecord(
                            relation_id="rel-ai-1",
                            subject_entity_id="ent-article-6",
                            object_entity_id="ent-high-risk",
                            relation_type="defines",
                            source_chunk_id="chunk-ai-6",
                            confidence=0.9,
                        ).model_dump()
                    ],
                    "communities": {"ent-ai-act": 0, "ent-gdpr": 1, "ent-article-6": 0, "ent-high-risk": 0},
                    "mentions": [
                        {"chunk_id": "chunk-ai-6", "entity_id": "ent-ai-act"},
                        {"chunk_id": "chunk-ai-6", "entity_id": "ent-article-6"},
                        {"chunk_id": "chunk-ai-6", "entity_id": "ent-high-risk"},
                        {"chunk_id": "chunk-gdpr-6", "entity_id": "ent-gdpr"},
                        {"chunk_id": "chunk-gdpr-6", "entity_id": "ent-article-6"},
                    ],
                },
            )
            provider = HeuristicLLMProvider(settings)
            retriever = HybridRetriever(settings, provider)

            path_hits = retriever.retrieve("What does Article 6 require for high-risk AI systems?", top_k=2, mode="path_hybrid")
            self.assertEqual(path_hits[0].chunk.chunk_id, "chunk-ai-6")
            self.assertTrue(path_hits[0].graph_paths)
            self.assertEqual(retriever.last_retrieval_meta["mode"], "path_hybrid")
            self.assertGreater(int(retriever.last_retrieval_meta["path_count"]), 0)
            self.assertIn("total_latency_ms", retriever.last_retrieval_meta)
            top_paths = retriever.last_retrieval_meta["top_paths"]
            self.assertEqual(len({item["path_id"] for item in top_paths}), len(top_paths))
            self.assertTrue(any(item.get("supporting_context") for item in top_paths))

            cached_hits = retriever.retrieve("What does Article 6 require for high-risk AI systems?", top_k=2, mode="path_cache")
            self.assertEqual(cached_hits[0].chunk.chunk_id, "chunk-ai-6")
            self.assertFalse(bool(retriever.last_retrieval_meta["cache_hit"]))
            self.assertEqual(int(retriever.last_retrieval_meta["cache_schema_version"]), 2)

            cached_hits_again = retriever.retrieve("What does Article 6 require for high-risk AI systems?", top_k=2, mode="path_cache")
            self.assertEqual(cached_hits_again[0].chunk.chunk_id, "chunk-ai-6")
            self.assertTrue(bool(retriever.last_retrieval_meta["cache_hit"]))
            self.assertTrue(any((settings.processed_data_path / "path_cache").glob("*.json")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_graph_loader_preserves_entity_metadata(self) -> None:
        tmp = Path.cwd() / "data" / "cache" / "test_graph_loader_case"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = Settings(data_dir=str(tmp))
            ingestion_dir = settings.processed_data_path / "ingestion"
            extraction_dir = settings.processed_data_path / "extraction"
            write_jsonl(
                ingestion_dir / "documents.jsonl",
                [DocumentRecord(document_id="doc-ai", name="AI Act", source_path="x", checksum="1").model_dump()],
            )
            write_jsonl(
                ingestion_dir / "chunks.jsonl",
                [
                    ChunkRecord(
                        chunk_id="chunk-ai-6",
                        document_id="doc-ai",
                        section_id="sec-ai-6",
                        article_ref="Article 6",
                        text="Article 6 text",
                        text_hash="hash-ai",
                    ).model_dump()
                ],
            )
            write_jsonl(
                extraction_dir / "entities.jsonl",
                [
                    EntityRecord(
                        entity_id="ent-article-6",
                        canonical_name="Article 6",
                        raw_name="Article 6",
                        entity_type="article",
                        source_chunk_id="chunk-ai-6",
                        metadata={"mention_chunk_ids": ["chunk-ai-6"], "document_ids": ["doc-ai"]},
                    ).model_dump()
                ],
            )
            write_jsonl(extraction_dir / "relations.jsonl", [])
            stats = GraphLoader(settings).build_graph()
            self.assertEqual(stats.entities_loaded, 1)
            graph_catalog = (settings.processed_data_path / "graph" / "graph_catalog.json").read_text(encoding="utf-8")
            self.assertIn("mention_chunk_ids", graph_catalog)
            self.assertIn("document_ids", graph_catalog)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_path_aggregation_prefers_specific_article_chunk_over_repeated_related_paths(self) -> None:
        tmp = Path.cwd() / "data" / "cache" / "test_path_specificity_case"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = Settings(data_dir=str(tmp))
            graph_dir = settings.processed_data_path / "graph"
            ai_act = DocumentRecord(document_id="doc-ai", name="OJ_L_202401689_EN_TXT", source_path="x", checksum="1")
            main_chunk = ChunkRecord(
                chunk_id="chunk-ai-6-main",
                document_id="doc-ai",
                section_id="sec-ai-6-main",
                article_ref="Article 6",
                text="Article 6 Classification rules for high-risk AI systems. Providers shall ensure conformity assessment for high-risk AI systems before placing them on the market.",
                text_hash="hash-main",
                metadata={"section_title": "Article 6"},
            )
            amend_chunk = ChunkRecord(
                chunk_id="chunk-ai-6-amend",
                document_id="doc-ai",
                section_id="sec-ai-6-amend",
                article_ref="Article 6",
                text="Article 6 also empowers the Commission to adopt delegated acts and coordinate amendments with Article 7 and Article 97.",
                text_hash="hash-amend",
                metadata={"section_title": "Article 6"},
            )
            write_json(
                graph_dir / "graph_catalog.json",
                {
                    "documents": [ai_act.model_dump()],
                    "chunks": [main_chunk.model_dump(), amend_chunk.model_dump()],
                    "entities": [
                        EntityRecord(
                            entity_id="ent-ai-act",
                            canonical_name="AI Act",
                            raw_name="AI Act",
                            entity_type="regulation",
                            source_chunk_id="chunk-ai-6-main",
                            metadata={"mention_chunk_ids": ["chunk-ai-6-main", "chunk-ai-6-amend"], "document_ids": ["doc-ai"]},
                        ).model_dump(),
                        EntityRecord(
                            entity_id="ent-article-6",
                            canonical_name="Article 6",
                            raw_name="Article 6",
                            entity_type="article",
                            source_chunk_id="chunk-ai-6-main",
                            metadata={"mention_chunk_ids": ["chunk-ai-6-main", "chunk-ai-6-amend"]},
                        ).model_dump(),
                        EntityRecord(
                            entity_id="ent-high-risk",
                            canonical_name="High-Risk AI System",
                            raw_name="high-risk AI systems",
                            entity_type="risk_class",
                            source_chunk_id="chunk-ai-6-main",
                            metadata={"mention_chunk_ids": ["chunk-ai-6-main"]},
                        ).model_dump(),
                        EntityRecord(
                            entity_id="ent-article-7",
                            canonical_name="Article 7",
                            raw_name="Article 7",
                            entity_type="article",
                            source_chunk_id="chunk-ai-6-amend",
                            metadata={"mention_chunk_ids": ["chunk-ai-6-amend"]},
                        ).model_dump(),
                        EntityRecord(
                            entity_id="ent-article-97",
                            canonical_name="Article 97",
                            raw_name="Article 97",
                            entity_type="article",
                            source_chunk_id="chunk-ai-6-amend",
                            metadata={"mention_chunk_ids": ["chunk-ai-6-amend"]},
                        ).model_dump(),
                    ],
                    "relations": [
                        RelationRecord(
                            relation_id="rel-ai-main",
                            subject_entity_id="ent-article-6",
                            object_entity_id="ent-high-risk",
                            relation_type="requires",
                            source_chunk_id="chunk-ai-6-main",
                            confidence=0.95,
                        ).model_dump(),
                        RelationRecord(
                            relation_id="rel-ai-7",
                            subject_entity_id="ent-article-6",
                            object_entity_id="ent-article-7",
                            relation_type="references",
                            source_chunk_id="chunk-ai-6-amend",
                            confidence=0.95,
                        ).model_dump(),
                        RelationRecord(
                            relation_id="rel-ai-97",
                            subject_entity_id="ent-article-6",
                            object_entity_id="ent-article-97",
                            relation_type="references",
                            source_chunk_id="chunk-ai-6-amend",
                            confidence=0.95,
                        ).model_dump(),
                    ],
                    "communities": {
                        "ent-ai-act": 0,
                        "ent-article-6": 0,
                        "ent-high-risk": 0,
                        "ent-article-7": 0,
                        "ent-article-97": 0,
                    },
                    "mentions": [
                        {"chunk_id": "chunk-ai-6-main", "entity_id": "ent-ai-act"},
                        {"chunk_id": "chunk-ai-6-main", "entity_id": "ent-article-6"},
                        {"chunk_id": "chunk-ai-6-main", "entity_id": "ent-high-risk"},
                        {"chunk_id": "chunk-ai-6-amend", "entity_id": "ent-ai-act"},
                        {"chunk_id": "chunk-ai-6-amend", "entity_id": "ent-article-6"},
                        {"chunk_id": "chunk-ai-6-amend", "entity_id": "ent-article-7"},
                        {"chunk_id": "chunk-ai-6-amend", "entity_id": "ent-article-97"},
                    ],
                },
            )
            provider = HeuristicLLMProvider(settings)
            retriever = HybridRetriever(settings, provider)

            hits = retriever.retrieve("What does Article 6 require for high-risk AI systems?", top_k=2, mode="path_hybrid")
            self.assertEqual(hits[0].chunk.chunk_id, "chunk-ai-6-main")
            self.assertGreater(hits[0].graph_score, 0.0)
            self.assertGreaterEqual(hits[0].text_score, hits[1].text_score)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_adaptive_routing_prefers_path_cache_for_article_questions_and_hybrid_for_simple_lookup(self) -> None:
        tmp = Path.cwd() / "data" / "cache" / "test_adaptive_routing_case"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = Settings(data_dir=str(tmp))
            graph_dir = settings.processed_data_path / "graph"
            ai_act = DocumentRecord(document_id="doc-ai", name="OJ_L_202401689_EN_TXT", source_path="x", checksum="1")
            gdpr = DocumentRecord(document_id="doc-gdpr", name="CELEX_32016R0679_EN_TXT", source_path="y", checksum="2")
            ai_chunk = ChunkRecord(
                chunk_id="chunk-ai-6",
                document_id="doc-ai",
                section_id="sec-ai-6",
                article_ref="Article 6",
                text="Article 6 Classification rules for high-risk AI systems. Providers shall ensure conformity assessment for high-risk AI systems before placing them on the market.",
                text_hash="hash-ai",
                metadata={"section_title": "Article 6"},
            )
            gdpr_chunk = ChunkRecord(
                chunk_id="chunk-gdpr-1",
                document_id="doc-gdpr",
                section_id="sec-gdpr-1",
                article_ref="Article 1",
                text="GDPR protects personal data and establishes rules for processing by controllers and processors.",
                text_hash="hash-gdpr",
                metadata={"section_title": "Article 1"},
            )
            write_json(
                graph_dir / "graph_catalog.json",
                {
                    "documents": [ai_act.model_dump(), gdpr.model_dump()],
                    "chunks": [ai_chunk.model_dump(), gdpr_chunk.model_dump()],
                    "entities": [
                        EntityRecord(
                            entity_id="ent-ai-act",
                            canonical_name="AI Act",
                            raw_name="AI Act",
                            entity_type="regulation",
                            source_chunk_id="chunk-ai-6",
                            metadata={"mention_chunk_ids": ["chunk-ai-6"], "document_ids": ["doc-ai"]},
                        ).model_dump(),
                        EntityRecord(
                            entity_id="ent-gdpr",
                            canonical_name="GDPR",
                            raw_name="GDPR",
                            entity_type="regulation",
                            source_chunk_id="chunk-gdpr-1",
                            metadata={"mention_chunk_ids": ["chunk-gdpr-1"], "document_ids": ["doc-gdpr"]},
                        ).model_dump(),
                        EntityRecord(
                            entity_id="ent-article-6",
                            canonical_name="Article 6",
                            raw_name="Article 6",
                            entity_type="article",
                            source_chunk_id="chunk-ai-6",
                            metadata={"mention_chunk_ids": ["chunk-ai-6"]},
                        ).model_dump(),
                        EntityRecord(
                            entity_id="ent-high-risk",
                            canonical_name="High-Risk AI System",
                            raw_name="high-risk AI systems",
                            entity_type="risk_class",
                            source_chunk_id="chunk-ai-6",
                            metadata={"mention_chunk_ids": ["chunk-ai-6"]},
                        ).model_dump(),
                    ],
                    "relations": [
                        RelationRecord(
                            relation_id="rel-ai-1",
                            subject_entity_id="ent-article-6",
                            object_entity_id="ent-high-risk",
                            relation_type="requires",
                            source_chunk_id="chunk-ai-6",
                            confidence=0.9,
                        ).model_dump()
                    ],
                    "communities": {"ent-ai-act": 0, "ent-gdpr": 1, "ent-article-6": 0, "ent-high-risk": 0},
                    "mentions": [
                        {"chunk_id": "chunk-ai-6", "entity_id": "ent-ai-act"},
                        {"chunk_id": "chunk-ai-6", "entity_id": "ent-article-6"},
                        {"chunk_id": "chunk-ai-6", "entity_id": "ent-high-risk"},
                        {"chunk_id": "chunk-gdpr-1", "entity_id": "ent-gdpr"},
                    ],
                },
            )
            provider = HeuristicLLMProvider(settings)
            retriever = HybridRetriever(settings, provider)
            agent = GraphRAGAgent(settings, provider, retriever)

            adaptive_route = retriever.resolve_mode(
                "What does Article 6 require for high-risk AI systems?",
                requested_mode="adaptive",
            )
            self.assertEqual(adaptive_route["resolved_mode"], "path_cache")
            self.assertFalse(bool(adaptive_route["cache_available"]))
            self.assertIn("path_cache", adaptive_route["candidate_modes"])

            response = agent.run(
                QueryRequest(
                    question="What does Article 6 require for high-risk AI systems?",
                    retrieval_mode="adaptive",
                    top_k=2,
                )
            )
            route_events = [event for event in response.trace if event.get("step") == "route"]
            self.assertTrue(route_events)
            self.assertEqual(route_events[0]["preselected_mode"], "path_cache")
            self.assertEqual(route_events[0]["strategy"], "adaptive_compare")
            self.assertGreaterEqual(len(route_events[0].get("candidate_scores", [])), 2)
            self.assertIn(route_events[0]["resolved_mode"], {"hybrid", "path_cache"})

            warmed_response = agent.run(
                QueryRequest(
                    question="What does Article 6 require for high-risk AI systems?",
                    retrieval_mode="adaptive",
                    top_k=2,
                )
            )
            warmed_route_events = [event for event in warmed_response.trace if event.get("step") == "route"]
            self.assertTrue(warmed_route_events)
            self.assertEqual(warmed_route_events[0]["resolved_mode"], "path_cache")

            adaptive_route_cached = retriever.resolve_mode(
                "What does Article 6 require for high-risk AI systems?",
                requested_mode="adaptive",
            )
            self.assertTrue(bool(adaptive_route_cached["cache_available"]))

            simple_route = retriever.resolve_mode("What is GDPR?", requested_mode="adaptive")
            self.assertEqual(simple_route["resolved_mode"], "hybrid")
            self.assertEqual(simple_route["candidate_modes"], ["hybrid"])

            analytics_rows = read_jsonl(settings.processed_data_path / "analytics" / "adaptive_routes.jsonl")
            self.assertGreaterEqual(len(analytics_rows), 2)
            self.assertEqual(analytics_rows[-1]["selected_mode"], warmed_route_events[0]["resolved_mode"])
            self.assertIn("candidate_scores", analytics_rows[-1])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
