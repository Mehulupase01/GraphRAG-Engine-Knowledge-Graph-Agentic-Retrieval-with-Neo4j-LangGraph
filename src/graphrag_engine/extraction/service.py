from __future__ import annotations

import logging
import re
from collections import defaultdict

from graphrag_engine.common.artifacts import read_jsonl, write_json, write_jsonl
from graphrag_engine.common.hashing import stable_hash
from graphrag_engine.common.models import ChunkRecord, DocumentRecord, EntityRecord, RelationRecord
from graphrag_engine.common.providers import HeuristicLLMProvider, LLMProvider
from graphrag_engine.common.settings import Settings

LOGGER = logging.getLogger(__name__)

DOCUMENT_NAME_MAP = {
    "celex_32016r0679_en_txt": "GDPR",
    "celex_32022r2065_en_txt": "Digital Services Act",
    "oj_l_202401689_en_txt": "AI Act",
}

STATIC_ENTITY_RULES: tuple[tuple[str, str, str], ...] = (
    (r"\bAI Act\b|\bRegulation\s*\(EU\)\s*2024/1689\b", "AI Act", "regulation"),
    (r"\bGDPR\b|\bGeneral Data Protection Regulation\b", "GDPR", "regulation"),
    (r"\bDigital Services Act\b|\bDSA\b", "Digital Services Act", "regulation"),
    (r"\bEuropean Commission\b|\bCommission\b", "European Commission", "institution"),
    (r"\bEuropean Data Protection Board\b|\bEDPB\b", "European Data Protection Board", "institution"),
    (r"\bMember States?\b", "Member State", "jurisdiction"),
    (r"\bcontrollers?\b", "Controller", "actor"),
    (r"\bprocessors?\b", "Processor", "actor"),
    (r"\bdata subjects?\b", "Data Subject", "actor"),
    (r"\brecipients?\b", "Recipient", "actor"),
    (r"\bsupervisory authorities?\b", "Supervisory Authority", "actor"),
    (r"\bdata protection officer\b", "Data Protection Officer", "actor"),
    (r"\bproviders?\b", "Provider", "actor"),
    (r"\bdeployers?\b", "Deployer", "actor"),
    (r"\bimporters?\b", "Importer", "actor"),
    (r"\bdistributors?\b", "Distributor", "actor"),
    (r"\bauthori[sz]ed representatives?\b", "Authorized Representative", "actor"),
    (r"\bnotified bod(?:y|ies)\b", "Notified Body", "actor"),
    (r"\bonline platforms?\b", "Online Platform", "service"),
    (r"\bvery large online platforms?\b", "Very Large Online Platform", "service"),
    (r"\bsearch engines?\b", "Search Engine", "service"),
    (r"\bvery large online search engines?\b", "Very Large Online Search Engine", "service"),
    (r"\bhosting services?\b", "Hosting Service", "service"),
    (r"\bintermediary services?\b", "Intermediary Service", "service"),
    (r"\bgeneral-purpose AI models?\b", "General-Purpose AI Model", "ai_concept"),
    (r"\bgeneral-purpose AI systems?\b", "General-Purpose AI System", "ai_concept"),
    (r"\bAI systems?\b", "AI System", "ai_concept"),
    (r"\bhigh-risk AI systems?\b|\bhigh-risk systems?\b", "High-Risk AI System", "risk_class"),
    (r"\bminimal risk\b", "Minimal Risk", "risk_class"),
    (r"\blimited risk\b", "Limited Risk", "risk_class"),
    (r"\bunacceptable risk\b", "Unacceptable Risk", "risk_class"),
    (r"\bprohibited AI practices?\b", "Prohibited AI Practice", "risk_class"),
    (r"\bpersonal data\b", "Personal Data", "data_concept"),
    (r"\bspecial categor(?:y|ies) of personal data\b", "Special Category of Personal Data", "data_concept"),
    (r"\bprofiling\b", "Profiling", "data_concept"),
    (r"\bconsent\b", "Consent", "data_concept"),
    (r"\bdata portability\b", "Data Portability", "right"),
    (r"\bright of access\b|\bright to access\b", "Right of Access", "right"),
    (r"\bright to rectification\b", "Right to Rectification", "right"),
    (r"\bright to erasure\b|\bright to be forgotten\b", "Right to Erasure", "right"),
    (r"\bconformity assessment\b", "Conformity Assessment", "obligation"),
    (r"\brisk management system\b", "Risk Management System", "obligation"),
    (r"\bpost-market monitoring system\b", "Post-Market Monitoring System", "obligation"),
    (r"\btechnical documentation\b", "Technical Documentation", "obligation"),
    (r"\brecord[- ]keeping\b", "Record-Keeping", "obligation"),
    (r"\blogging\b", "Logging", "obligation"),
    (r"\bhuman oversight\b", "Human Oversight", "obligation"),
    (r"\baccuracy\b", "Accuracy", "quality_requirement"),
    (r"\brobustness\b", "Robustness", "quality_requirement"),
    (r"\bcybersecurity\b", "Cybersecurity", "quality_requirement"),
    (r"\btransparency(?: obligations?)?\b", "Transparency", "obligation"),
    (r"\bCE marking\b", "CE Marking", "obligation"),
    (r"\bfundamental rights impact assessment\b", "Fundamental Rights Impact Assessment", "obligation"),
    (r"\bpersonal data breach\b|\bbreach notification\b", "Personal Data Breach", "obligation"),
    (r"\bdata protection by design\b", "Data Protection by Design", "obligation"),
    (r"\bdata protection by default\b", "Data Protection by Default", "obligation"),
)

RELATION_PRIORITY: tuple[tuple[tuple[str, ...], str], ...] = (
    (("shall not", "must not", "prohibited", "forbidden"), "prohibits"),
    (("shall mean", "means", "defines"), "defines"),
    (("applies to", "apply to", "scope"), "applies_to"),
    (("right to", "entitled to"), "grants"),
    (("includes", "consists of", "comprises"), "includes"),
    (("shall", "must", "required", "require", "ensure"), "requires"),
)


def canonicalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


class ExtractionService:
    def __init__(self, settings: Settings, provider: LLMProvider) -> None:
        self.settings = settings
        self.provider = provider
        self.heuristic_provider = HeuristicLLMProvider(settings)

    def extract(self) -> dict[str, int]:
        documents = [
            DocumentRecord.model_validate(row)
            for row in read_jsonl(self.settings.processed_data_path / "ingestion" / "documents.jsonl")
        ]
        chunks = [
            ChunkRecord.model_validate(row)
            for row in read_jsonl(self.settings.processed_data_path / "ingestion" / "chunks.jsonl")
        ]
        document_name_by_id = {document.document_id: self._canonical_document_name(document.name) for document in documents}
        entities: dict[str, EntityRecord] = {}
        relations: dict[str, RelationRecord] = {}
        alias_map: dict[str, str] = {}

        for chunk in chunks:
            structured = self._extract_chunk_knowledge(chunk, document_name_by_id.get(chunk.document_id, chunk.document_id))
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
                        metadata={
                            "mention_chunk_ids": [chunk.chunk_id],
                            "document_ids": [chunk.document_id],
                            **dict(payload.get("metadata", {})),
                        },
                    )
                else:
                    existing = entities[entity_id]
                    if raw_name not in existing.aliases:
                        existing.aliases.append(raw_name)
                    for evidence in payload.get("evidence", []):
                        if evidence not in existing.evidence:
                            existing.evidence.append(evidence)
                    existing.confidence = max(existing.confidence, float(payload.get("confidence", 0.5)))
                    mention_chunk_ids = existing.metadata.setdefault("mention_chunk_ids", [existing.source_chunk_id])
                    if chunk.chunk_id not in mention_chunk_ids:
                        mention_chunk_ids.append(chunk.chunk_id)
                    document_ids = existing.metadata.setdefault("document_ids", [])
                    if chunk.document_id not in document_ids:
                        document_ids.append(chunk.document_id)

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

    def _extract_chunk_knowledge(self, chunk: ChunkRecord, document_name: str) -> dict[str, list[dict[str, object]]]:
        structured = self._heuristic_chunk_knowledge(chunk, document_name)
        if self.provider.provider_name == "openai":
            try:
                llm_payload = self.provider.extract_structured_knowledge(chunk)
                structured["entities"].extend(llm_payload.get("entities", []))
                structured["relations"].extend(llm_payload.get("relations", []))
            except Exception:
                LOGGER.warning("llm_extraction_failed", extra={"chunk_id": chunk.chunk_id})
        return structured

    def _heuristic_chunk_knowledge(self, chunk: ChunkRecord, document_name: str) -> dict[str, list[dict[str, object]]]:
        entities: list[dict[str, object]] = []
        seen_entities: set[tuple[str, str]] = set()

        def add_entity(
            *,
            canonical_name: str,
            entity_type: str,
            raw_name: str | None = None,
            confidence: float = 0.72,
            evidence: list[str] | None = None,
            metadata: dict[str, object] | None = None,
        ) -> None:
            key = (canonicalize(canonical_name), entity_type)
            if not key[0] or key in seen_entities:
                return
            seen_entities.add(key)
            entities.append(
                {
                    "raw_name": raw_name or canonical_name,
                    "canonical_name": canonical_name,
                    "entity_type": entity_type,
                    "confidence": confidence,
                    "evidence": evidence or [raw_name or canonical_name],
                    "metadata": metadata or {},
                }
            )

        add_entity(
            canonical_name=document_name,
            entity_type="regulation",
            confidence=0.96,
            metadata={"derived_from": "document_name"},
        )

        if chunk.article_ref:
            add_entity(
                canonical_name=chunk.article_ref,
                entity_type="article",
                confidence=0.95,
                metadata={"derived_from": "chunk.article_ref"},
            )

        topic = self._extract_chunk_topic(chunk)
        if topic:
            add_entity(
                canonical_name=topic,
                entity_type=self._classify_topic(topic),
                confidence=0.84,
                metadata={"derived_from": "chunk_heading"},
            )

        base_payload = self.heuristic_provider.extract_structured_knowledge(chunk)
        entities.extend(base_payload.get("entities", []))
        seen_entities.update((canonicalize(item["canonical_name"]), item["entity_type"]) for item in entities if item)

        text = chunk.text
        for pattern, canonical_name, entity_type in STATIC_ENTITY_RULES:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                add_entity(
                    canonical_name=canonical_name,
                    entity_type=entity_type,
                    raw_name=match.group(0),
                    confidence=0.78,
                    evidence=[match.group(0)],
                )

        relations = self._build_relations(chunk, document_name, entities)
        relations.extend(base_payload.get("relations", []))
        return {"entities": entities, "relations": self._dedupe_relations(relations)}

    def _build_relations(
        self,
        chunk: ChunkRecord,
        document_name: str,
        entities: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        relations: list[dict[str, object]] = []
        entity_index = {str(item["canonical_name"]): item for item in entities}
        article_entities = [name for name, item in entity_index.items() if item.get("entity_type") == "article"]
        topic_entities = [
            name
            for name, item in entity_index.items()
            if item.get("entity_type") in {"article_topic", "chapter_topic", "section_topic", "topic"}
        ]

        for article_name in article_entities:
            relations.append(
                self._relation_payload(
                    subject=document_name,
                    object_name=article_name,
                    relation_type="contains_article",
                    evidence=[article_name],
                    confidence=0.95,
                )
            )
            for topic_name in topic_entities:
                relations.append(
                    self._relation_payload(
                        subject=article_name,
                        object_name=topic_name,
                        relation_type="covers",
                        evidence=[topic_name],
                        confidence=0.83,
                    )
                )

        sentences = self._split_sentences(chunk.text)
        canonical_names = list(entity_index.keys())
        for sentence in sentences:
            lowered = sentence.lower()
            mentioned = [
                name
                for name in canonical_names
                if canonicalize(name) and canonicalize(name) in canonicalize(lowered)
            ]
            if len(mentioned) < 2:
                continue
            relation_type = self._infer_relation_type(lowered)
            mentioned_types = {name: str(entity_index[name].get("entity_type")) for name in mentioned}
            articles = [name for name in mentioned if mentioned_types[name] == "article"]
            actors = [name for name in mentioned if mentioned_types[name] == "actor"]
            obligations = [
                name
                for name in mentioned
                if mentioned_types[name] in {"obligation", "quality_requirement", "right"}
            ]
            risks = [name for name in mentioned if mentioned_types[name] == "risk_class"]
            others = [name for name in mentioned if mentioned_types[name] not in {"article"}]

            for article_name in articles:
                for other_name in others:
                    relations.append(
                        self._relation_payload(
                            subject=article_name,
                            object_name=other_name,
                            relation_type=relation_type,
                            evidence=[sentence[:240]],
                            confidence=0.68,
                        )
                    )
            for actor_name in actors:
                for target_name in obligations:
                    relations.append(
                        self._relation_payload(
                            subject=actor_name,
                            object_name=target_name,
                            relation_type="requires" if relation_type == "references" else relation_type,
                            evidence=[sentence[:240]],
                            confidence=0.7,
                        )
                    )
            for risk_name in risks:
                for target_name in obligations:
                    relations.append(
                        self._relation_payload(
                            subject=risk_name,
                            object_name=target_name,
                            relation_type="requires" if relation_type == "references" else relation_type,
                            evidence=[sentence[:240]],
                            confidence=0.67,
                        )
                    )

        return relations

    @staticmethod
    def _dedupe_relations(relations: list[dict[str, object]]) -> list[dict[str, object]]:
        unique: dict[tuple[str, str, str], dict[str, object]] = {}
        for relation in relations:
            subject = str(relation.get("subject", ""))
            object_name = str(relation.get("object", ""))
            relation_type = str(relation.get("relation_type", "references"))
            if not subject or not object_name:
                continue
            key = (canonicalize(subject), relation_type, canonicalize(object_name))
            current = unique.get(key)
            if current is None:
                unique[key] = relation
                continue
            current_evidence = set(map(str, current.get("evidence", [])))
            current_evidence.update(map(str, relation.get("evidence", [])))
            current["evidence"] = list(current_evidence)[:5]
            current["confidence"] = max(float(current.get("confidence", 0.5)), float(relation.get("confidence", 0.5)))
        return list(unique.values())

    @staticmethod
    def _relation_payload(
        *,
        subject: str,
        object_name: str,
        relation_type: str,
        evidence: list[str],
        confidence: float,
    ) -> dict[str, object]:
        return {
            "subject": subject,
            "object": object_name,
            "relation_type": relation_type,
            "confidence": confidence,
            "evidence": evidence,
        }

    @staticmethod
    def _canonical_document_name(document_name: str) -> str:
        return DOCUMENT_NAME_MAP.get(canonicalize(document_name), document_name.replace("_", " ").strip())

    @staticmethod
    def _extract_chunk_topic(chunk: ChunkRecord) -> str | None:
        lines = [re.sub(r"\s+", " ", line).strip(" ;:") for line in chunk.text.splitlines() if line.strip()]
        if not lines:
            return None
        if lines[0].startswith("Article ") and len(lines) > 1 and not re.match(r"^[0-9(]", lines[1]):
            return lines[1]
        if re.match(r"^(Chapter|Section|Title|Annex)\b", lines[0], flags=re.IGNORECASE) and len(lines) > 1:
            return lines[1]
        return None

    @staticmethod
    def _classify_topic(topic: str) -> str:
        lowered = topic.lower()
        if lowered.startswith("right to") or lowered.startswith("right of"):
            return "right"
        if any(keyword in lowered for keyword in ("obligation", "requirements", "assessment", "monitoring")):
            return "obligation"
        if any(keyword in lowered for keyword in ("principles", "definitions", "scope", "provisions")):
            return "section_topic"
        return "article_topic"

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        pieces = [
            re.sub(r"\s+", " ", piece).strip()
            for piece in re.split(r"(?<=[.!?;:])\s+(?=[A-Z0-9(])", text)
            if piece.strip()
        ]
        return pieces

    @staticmethod
    def _infer_relation_type(sentence: str) -> str:
        for hints, relation_type in RELATION_PRIORITY:
            if any(hint in sentence for hint in hints):
                return relation_type
        return "references"
