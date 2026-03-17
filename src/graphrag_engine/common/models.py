from __future__ import annotations

from typing import Any

from .compat import BaseModel, ConfigDict, Field


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_id: str
    name: str
    source_path: str
    checksum: str
    page_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SectionRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    section_id: str
    document_id: str
    title: str
    article_ref: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    document_id: str
    section_id: str
    article_ref: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    sequence: int = 0
    text: str = ""
    text_hash: str = ""
    token_estimate: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entity_id: str
    canonical_name: str
    raw_name: str
    entity_type: str
    source_chunk_id: str
    confidence: float = 0.0
    aliases: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relation_id: str
    subject_entity_id: str
    object_entity_id: str
    relation_type: str
    source_chunk_id: str
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphLoadStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    documents_loaded: int = 0
    chunks_loaded: int = 0
    entities_loaded: int = 0
    relations_loaded: int = 0
    communities_detected: int = 0
    used_neo4j: bool = False
    notes: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    document_name: str
    article_ref: str | None = None
    snippet: str = ""
    page_start: int | None = None
    page_end: int | None = None
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class GraphPath(BaseModel):
    model_config = ConfigDict(extra="ignore")

    seed_entity: str
    traversed_entities: list[str] = Field(default_factory=list)
    relation_chain: list[str] = Field(default_factory=list)
    score: float = 0.0


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk: ChunkRecord
    document_name: str
    text_score: float = 0.0
    vector_score: float = 0.0
    graph_score: float = 0.0
    metadata_score: float = 0.0
    fused_score: float = 0.0
    graph_paths: list[GraphPath] = Field(default_factory=list)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    retrieval_mode: str = "hybrid"
    debug: bool = False
    top_k: int = 8


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    graph_paths: list[GraphPath] = Field(default_factory=list)
    retrieved_chunks: list[ChunkRecord] = Field(default_factory=list)
    retrieval_scores: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    fallback_used: bool = False
    trace: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    case_id: str
    question: str
    expected_keywords: list[str] = Field(default_factory=list)
    expected_articles: list[str] = Field(default_factory=list)
    difficulty: str = "medium"
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    case_id: str
    approach: str
    score: float = 0.0
    faithfulness: float = 0.0
    context_precision: float = 0.0
    answer_relevancy: float = 0.0
    multi_hop_accuracy: float = 0.0
    notes: list[str] = Field(default_factory=list)
    response: QueryResponse


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    total_cases: int = 0
    aggregate_scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    cases: list[EvaluationResult] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionJobRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: str
    phase: str
    source_files: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
