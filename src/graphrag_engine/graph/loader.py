from __future__ import annotations

import logging

from graphrag_engine.common.artifacts import read_json, read_jsonl, write_json
from graphrag_engine.common.models import ChunkRecord, DocumentRecord, EntityRecord, GraphLoadStats, RelationRecord
from graphrag_engine.common.settings import Settings

from .community import detect_communities

try:
    from neo4j import GraphDatabase  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    GraphDatabase = None

LOGGER = logging.getLogger(__name__)


class GraphLoader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_graph(self) -> GraphLoadStats:
        ingestion_dir = self.settings.processed_data_path / "ingestion"
        extraction_dir = self.settings.processed_data_path / "extraction"
        documents = [DocumentRecord.model_validate(row) for row in read_jsonl(ingestion_dir / "documents.jsonl")]
        chunks = [ChunkRecord.model_validate(row) for row in read_jsonl(ingestion_dir / "chunks.jsonl")]
        entities = [EntityRecord.model_validate(row) for row in read_jsonl(extraction_dir / "entities.jsonl")]
        relations = [RelationRecord.model_validate(row) for row in read_jsonl(extraction_dir / "relations.jsonl")]
        communities = detect_communities(entities, relations)

        graph_dir = self.settings.processed_data_path / "graph"
        catalog = {
            "documents": [document.model_dump() for document in documents],
            "chunks": [chunk.model_dump() for chunk in chunks],
            "entities": [
                entity.model_copy(update={"metadata": {"community_id": communities.get(entity.entity_id)}}).model_dump()
                for entity in entities
            ],
            "relations": [relation.model_dump() for relation in relations],
            "communities": communities,
        }
        write_json(graph_dir / "graph_catalog.json", catalog)

        notes: list[str] = []
        used_neo4j = False
        if GraphDatabase is not None:
            try:  # pragma: no cover - requires external service
                driver = GraphDatabase.driver(
                    self.settings.neo4j_uri,
                    auth=(self.settings.neo4j_user, self.settings.neo4j_password),
                )
                with driver.session() as session:
                    session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE")
                    session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
                    session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE")
                    for document in documents:
                        session.run(
                            "MERGE (d:Document {document_id: $document_id}) SET d += $props",
                            document_id=document.document_id,
                            props=document.model_dump(),
                        )
                    for chunk in chunks:
                        session.run(
                            """
                            MERGE (c:Chunk {chunk_id: $chunk_id})
                            SET c += $props
                            WITH c
                            MATCH (d:Document {document_id: $document_id})
                            MERGE (d)-[:HAS_CHUNK]->(c)
                            """,
                            chunk_id=chunk.chunk_id,
                            document_id=chunk.document_id,
                            props=chunk.model_dump(),
                        )
                    for entity in entities:
                        entity_payload = entity.model_copy(
                            update={"metadata": {"community_id": communities.get(entity.entity_id)}}
                        ).model_dump()
                        session.run(
                            "MERGE (e:Entity {entity_id: $entity_id}) SET e += $props",
                            entity_id=entity.entity_id,
                            props=entity_payload,
                        )
                    for relation in relations:
                        session.run(
                            """
                            MATCH (s:Entity {entity_id: $subject}), (o:Entity {entity_id: $object}), (c:Chunk {chunk_id: $chunk_id})
                            MERGE (s)-[r:RELATES {relation_id: $relation_id}]->(o)
                            SET r.relation_type = $relation_type, r.confidence = $confidence, r.evidence = $evidence
                            MERGE (c)-[:SUPPORTS]->(r)
                            """,
                            subject=relation.subject_entity_id,
                            object=relation.object_entity_id,
                            chunk_id=relation.source_chunk_id,
                            relation_id=relation.relation_id,
                            relation_type=relation.relation_type,
                            confidence=relation.confidence,
                            evidence=relation.evidence,
                        )
                used_neo4j = True
                notes.append("Loaded graph into Neo4j.")
            except Exception as exc:
                notes.append(f"Neo4j load skipped: {exc}")
        else:
            notes.append("Neo4j driver unavailable; graph persisted as local catalog only.")

        stats = GraphLoadStats(
            documents_loaded=len(documents),
            chunks_loaded=len(chunks),
            entities_loaded=len(entities),
            relations_loaded=len(relations),
            communities_detected=len(set(communities.values())),
            used_neo4j=used_neo4j,
            notes=notes,
        )
        write_json(graph_dir / "load_stats.json", stats.model_dump())
        LOGGER.info("graph_build_completed", extra=stats.model_dump())
        return stats

    def load_catalog(self) -> dict:
        return read_json(self.settings.processed_data_path / "graph" / "graph_catalog.json")

