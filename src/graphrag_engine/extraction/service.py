from __future__ import annotations

import logging
import re

from graphrag_engine.common.artifacts import read_jsonl, write_json, write_jsonl
from graphrag_engine.common.hashing import stable_hash
from graphrag_engine.common.models import ChunkRecord, EntityRecord, RelationRecord
from graphrag_engine.common.providers import LLMProvider
from graphrag_engine.common.settings import Settings

LOGGER = logging.getLogger(__name__)


def canonicalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


class ExtractionService:
    def __init__(self, settings: Settings, provider: LLMProvider) -> None:
        self.settings = settings
        self.provider = provider

    def extract(self) -> dict[str, int]:
        chunks = [
            ChunkRecord.model_validate(row)
            for row in read_jsonl(self.settings.processed_data_path / "ingestion" / "chunks.jsonl")
        ]
        entities: dict[str, EntityRecord] = {}
        relations: dict[str, RelationRecord] = {}
        alias_map: dict[str, str] = {}

        for chunk in chunks:
            structured = self.provider.extract_structured_knowledge(chunk)
            for payload in structured.get("entities", []):
                raw_name = payload.get("raw_name") or payload.get("canonical_name") or "Unknown"
                canonical_name = payload.get("canonical_name") or raw_name
                normalized = canonicalize(canonical_name)
                entity_id = alias_map.get(normalized)
                if entity_id is None:
                    entity_id = stable_hash(f"{payload.get('entity_type', 'concept')}:{normalized}", prefix="ent")
                    alias_map[normalized] = entity_id
                    entities[entity_id] = EntityRecord(
                        entity_id=entity_id,
                        canonical_name=canonical_name,
                        raw_name=raw_name,
                        entity_type=payload.get("entity_type", "concept"),
                        source_chunk_id=chunk.chunk_id,
                        confidence=float(payload.get("confidence", 0.5)),
                        aliases=[raw_name],
                        evidence=list(payload.get("evidence", [])),
                    )
                else:
                    existing = entities[entity_id]
                    if raw_name not in existing.aliases:
                        existing.aliases.append(raw_name)
                    for evidence in payload.get("evidence", []):
                        if evidence not in existing.evidence:
                            existing.evidence.append(evidence)
                    existing.confidence = max(existing.confidence, float(payload.get("confidence", 0.5)))

            for payload in structured.get("relations", []):
                subject_id = alias_map.get(canonicalize(payload.get("subject", "")))
                object_id = alias_map.get(canonicalize(payload.get("object", "")))
                if not subject_id or not object_id:
                    continue
                relation_id = stable_hash(
                    f"{subject_id}:{payload.get('relation_type','references')}:{object_id}:{chunk.chunk_id}",
                    prefix="rel",
                )
                relations[relation_id] = RelationRecord(
                    relation_id=relation_id,
                    subject_entity_id=subject_id,
                    object_entity_id=object_id,
                    relation_type=payload.get("relation_type", "references"),
                    source_chunk_id=chunk.chunk_id,
                    confidence=float(payload.get("confidence", 0.5)),
                    evidence=list(payload.get("evidence", [])),
                )

        extraction_dir = self.settings.processed_data_path / "extraction"
        write_jsonl(extraction_dir / "entities.jsonl", [entity.model_dump() for entity in entities.values()])
        write_jsonl(extraction_dir / "relations.jsonl", [relation.model_dump() for relation in relations.values()])
        write_json(extraction_dir / "aliases.json", alias_map)
        LOGGER.info(
            "extraction_completed",
            extra={"entities": len(entities), "relations": len(relations), "chunks": len(chunks)},
        )
        return {"entities": len(entities), "relations": len(relations), "chunks": len(chunks)}

