from __future__ import annotations

import json
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


def _neo4j_props(payload: dict) -> dict:
    props: dict = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            props[key] = value
            continue
        if isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
            props[key] = [item for item in value if item is not None]
            continue
        props[f"{key}_json"] = json.dumps(value, ensure_ascii=True, sort_keys=True)
    return props


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
        mentions = [
            {"chunk_id": chunk_id, "entity_id": entity.entity_id}
            for entity in entities
            for chunk_id in entity.metadata.get("mention_chunk_ids", [entity.source_chunk_id])
        ]

        graph_dir = self.settings.processed_data_path / "graph"
        catalog = {
            "documents": [document.model_dump() for document in documents],
            "chunks": [chunk.model_dump() for chunk in chunks],
            "entities": [
                entity.model_copy(
                    update={"metadata": {**entity.metadata, "community_id": communities.get(entity.entity_id)}}
                ).model_dump()
                for entity in entities
            ],
            "relations": [relation.model_dump() for relation in relations],
            "communities": communities,
            "mentions": mentions,
        }
        write_json(graph_dir / "graph_catalog.json", catalog)

        notes: list[str] = []
        used_neo4j = False
        if GraphDatabase is not None:
            driver = None
            try:  # pragma: no cover - requires external service
                driver = GraphDatabase.driver(
                    self.settings.neo4j_uri,
                    auth=(self.settings.neo4j_user, self.settings.neo4j_password),
                )
                with driver.session() as session:
                    session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE")
                    session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
                    session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE")
                    session.run(
                        """
                        UNWIND $rows AS row
                        MERGE (d:Document {document_id: row.document_id})
                        SET d += row.props
                        """,
                        rows=[
                            {"document_id": document.document_id, "props": _neo4j_props(document.model_dump())}
                            for document in documents
                        ],
                    )
                    session.run(
                        """
                        UNWIND $rows AS row
                        MERGE (c:Chunk {chunk_id: row.chunk_id})
                        SET c += row.props
                        WITH c, row
                        MATCH (d:Document {document_id: row.document_id})
                        MERGE (d)-[:HAS_CHUNK]->(c)
                        """,
                        rows=[
                            {
                                "chunk_id": chunk.chunk_id,
                                "document_id": chunk.document_id,
                                "props": _neo4j_props(chunk.model_dump()),
                            }
                            for chunk in chunks
                        ],
                    )
                    session.run(
                        """
                        UNWIND $rows AS row
                        MERGE (e:Entity {entity_id: row.entity_id})
                        SET e += row.props
                        """,
                        rows=[
                            {
                                "entity_id": entity.entity_id,
                                "props": _neo4j_props(
                                    {
                                        **entity.model_dump(),
                                        "community_id": communities.get(entity.entity_id),
                                    }
                                ),
                            }
                            for entity in entities
                        ],
                    )
                    session.run(
                        """
                        UNWIND $rows AS row
                        MATCH (s:Entity {entity_id: row.subject})
                        MATCH (o:Entity {entity_id: row.object})
                        MERGE (s)-[r:RELATES {relation_id: row.relation_id}]->(o)
                        SET r.relation_type = row.relation_type,
                            r.confidence = row.confidence,
                            r.evidence = row.evidence,
                            r.source_chunk_id = row.chunk_id
                        """,
                        rows=[
                            {
                                "subject": relation.subject_entity_id,
                                "object": relation.object_entity_id,
                                "chunk_id": relation.source_chunk_id,
                                "relation_id": relation.relation_id,
                                "relation_type": relation.relation_type,
                                "confidence": relation.confidence,
                                "evidence": relation.evidence,
                            }
                            for relation in relations
                        ],
                    )
                    session.run(
                        """
                        UNWIND $rows AS row
                        MATCH (c:Chunk {chunk_id: row.chunk_id})
                        MATCH (e:Entity {entity_id: row.entity_id})
                        MERGE (c)-[:MENTIONS]->(e)
                        """,
                        rows=mentions,
                    )
                used_neo4j = True
                notes.append("Loaded graph into Neo4j.")
            except Exception as exc:
                notes.append(f"Neo4j load skipped: {exc}")
            finally:
                if driver is not None:
                    driver.close()
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
