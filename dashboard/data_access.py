from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
RAW_ROOT = DATA_ROOT / "raw"
DOCS_ROOT = PROJECT_ROOT / "docs"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphrag_engine.common.settings import Settings
from graphrag_engine.runtime import GraphRAGRuntime

DOCUMENT_NAME_MAP = {
    "celex_32016r0679_en_txt": "GDPR",
    "celex_32022r2065_en_txt": "Digital Services Act",
    "oj_l_202401689_en_txt": "AI Act",
}
SAMPLE_QUESTIONS = [
    "What does Article 6 require for high-risk AI systems?",
    "How do provider obligations in the AI Act connect to conformity assessment?",
    "Which GDPR provision establishes the lawfulness of processing personal data?",
    "How does the Digital Services Act handle transparency and risk assessments for platforms?",
]


def canonical_document_name(name: str) -> str:
    normalized = name.lower().strip()
    return DOCUMENT_NAME_MAP.get(normalized, name.replace("_", " ").strip())


@st.cache_resource(show_spinner=False)
def get_runtime() -> GraphRAGRuntime:
    return GraphRAGRuntime()


@st.cache_data(ttl=10, show_spinner=False)
def load_settings() -> dict[str, Any]:
    return Settings.load().model_dump()


@st.cache_data(ttl=10, show_spinner=False)
def load_graph_catalog() -> dict[str, Any]:
    path = PROCESSED_ROOT / "graph" / "graph_catalog.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=10, show_spinner=False)
def load_graph_stats() -> dict[str, Any]:
    path = PROCESSED_ROOT / "graph" / "load_stats.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=10, show_spinner=False)
def load_jobs() -> list[dict[str, Any]]:
    jobs_root = PROCESSED_ROOT / "jobs"
    jobs = []
    for path in sorted(jobs_root.glob("*.json"), reverse=True):
        jobs.append(json.loads(path.read_text(encoding="utf-8")))
    return jobs


@st.cache_data(ttl=10, show_spinner=False)
def load_evaluations() -> list[dict[str, Any]]:
    evaluation_root = PROCESSED_ROOT / "evaluation"
    results = []
    for path in sorted(evaluation_root.glob("*.json"), reverse=True):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        results.append(payload)
    return results


@st.cache_data(ttl=10, show_spinner=False)
def load_path_cache_entries() -> list[dict[str, Any]]:
    cache_root = PROCESSED_ROOT / "path_cache"
    rows = []
    for path in sorted(cache_root.glob("*.json"), reverse=True):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        rows.append(payload)
    return rows


@st.cache_data(ttl=10, show_spinner=False)
def load_doc(path: str) -> str:
    doc_path = DOCS_ROOT / path
    if not doc_path.exists():
        return ""
    return doc_path.read_text(encoding="utf-8")


def corpus_overview() -> dict[str, Any]:
    catalog = load_graph_catalog()
    documents = catalog.get("documents", [])
    chunks = catalog.get("chunks", [])
    entities = catalog.get("entities", [])
    relations = catalog.get("relations", [])
    communities = catalog.get("communities", {})

    chunk_counts = Counter()
    article_counts = Counter()
    page_spans = Counter()
    doc_name_by_id = _document_name_by_id(documents)
    for chunk in chunks:
        document_name = canonical_document_name(doc_name_by_id.get(chunk.get("document_id", ""), "Unknown"))
        chunk_counts[document_name] += 1
        if chunk.get("article_ref"):
            article_counts[document_name] += 1
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        if page_start and page_end:
            page_spans[document_name] = max(page_spans[document_name], int(page_end) - int(page_start) + 1)

    document_frame = pd.DataFrame(
        [
            {
                "document_name": document_name,
                "chunks": count,
                "article_chunks": article_counts.get(document_name, 0),
                "page_span": page_spans.get(document_name, 0),
            }
            for document_name, count in sorted(chunk_counts.items(), key=lambda item: item[1], reverse=True)
        ]
    )
    entity_frame = pd.DataFrame(
        Counter(entity.get("entity_type", "unknown") for entity in entities).most_common(),
        columns=["entity_type", "count"],
    )
    relation_frame = pd.DataFrame(
        Counter(relation.get("relation_type", "unknown") for relation in relations).most_common(),
        columns=["relation_type", "count"],
    )
    community_frame = pd.DataFrame(
        Counter(communities.values()).most_common(12),
        columns=["community_id", "member_count"],
    )
    regulation_frame = chunk_frame()
    return {
        "documents": documents,
        "chunks": chunks,
        "entities": entities,
        "relations": relations,
        "document_frame": document_frame,
        "entity_frame": entity_frame,
        "relation_frame": relation_frame,
        "community_frame": community_frame,
        "regulation_frame": regulation_frame,
    }


def latest_evaluation_frames() -> tuple[dict[str, Any] | None, pd.DataFrame, pd.DataFrame]:
    evaluations = load_evaluations()
    if not evaluations:
        return None, pd.DataFrame(), pd.DataFrame()

    summary = evaluations[0]
    case_rows: list[dict[str, Any]] = []
    for case in summary.get("cases", []):
        metadata = case.get("response", {}).get("trace", [])
        case_metadata = case.get("metadata", {})
        case_rows.append(
            {
                "case_id": case.get("case_id"),
                "approach": case.get("approach"),
                "score": case.get("score", 0.0),
                "faithfulness": case.get("faithfulness", 0.0),
                "context_precision": case.get("context_precision", 0.0),
                "answer_relevancy": case.get("answer_relevancy", 0.0),
                "multi_hop_accuracy": case.get("multi_hop_accuracy", 0.0),
                "difficulty": _notes_value(case.get("notes", []), prefix="difficulty="),
                "trace_steps": len(metadata),
                "latency_ms": case_metadata.get("latency_ms", 0.0),
                "retrieval_latency_ms": case_metadata.get("retrieval_latency_ms", 0.0),
                "cache_hit": case_metadata.get("cache_hit", False),
            }
        )
    case_frame = pd.DataFrame(case_rows)

    aggregate_rows = []
    for approach, metrics in summary.get("aggregate_scores", {}).items():
        aggregate_rows.append({"approach": approach, **metrics})
    aggregate_frame = pd.DataFrame(aggregate_rows)
    return summary, case_frame, aggregate_frame


def latest_evaluation_delta() -> dict[str, float | None]:
    _, _, aggregate_frame = latest_evaluation_frames()
    if aggregate_frame.empty or "approach" not in aggregate_frame.columns:
        return {"baseline": None, "hybrid": None, "best_mode": None, "best_score": None, "best_delta": None}

    score_lookup = {
        str(row["approach"]).lower(): float(row.get("average_score", 0.0))
        for _, row in aggregate_frame.iterrows()
    }
    baseline = score_lookup.get("baseline")
    non_baseline = {mode: score for mode, score in score_lookup.items() if mode != "baseline"}
    best_mode = None
    best_score = None
    if non_baseline:
        best_mode, best_score = max(non_baseline.items(), key=lambda item: item[1])
    best_delta = None
    if baseline is not None and best_score is not None:
        best_delta = best_score - baseline
    return {
        "baseline": baseline,
        "hybrid": score_lookup.get("hybrid"),
        "best_mode": best_mode,
        "best_score": best_score,
        "best_delta": best_delta,
    }


def path_cache_frame() -> pd.DataFrame:
    rows = []
    for entry in load_path_cache_entries():
        rows.append(
            {
                "cache_key": entry.get("cache_key", ""),
                "retrieval_mode": entry.get("retrieval_mode", ""),
                "cache_schema_version": entry.get("metadata", {}).get("cache_schema_version", ""),
                "question_signature": entry.get("question_signature", ""),
                "path_count": len(entry.get("paths", [])),
                "top_chunk_ids": ", ".join(entry.get("top_chunk_ids", [])[:3]),
            }
        )
    return pd.DataFrame(rows)


def path_cache_stats() -> dict[str, Any]:
    entries = load_path_cache_entries()
    retrieval_modes = Counter(entry.get("retrieval_mode", "unknown") for entry in entries)
    return {
        "entries": len(entries),
        "schema_version": max(
            (
                int(entry.get("metadata", {}).get("cache_schema_version", 0))
                for entry in entries
            ),
            default=0,
        ),
        "retrieval_modes": dict(retrieval_modes),
        "total_size_kb": round(
            sum(Path(entry.get("_path", "")).stat().st_size for entry in entries if entry.get("_path")) / 1024,
            1,
        )
        if entries
        else 0.0,
    }


def path_records_from_response(response) -> list[dict[str, Any]]:
    trace = getattr(response, "trace", [])
    for event in trace:
        if event.get("step") == "retrieve":
            retrieval_meta = event.get("retrieval_meta", {})
            if isinstance(retrieval_meta, dict):
                return retrieval_meta.get("top_paths", [])
    return []


def retrieval_meta_from_response(response) -> dict[str, Any]:
    trace = getattr(response, "trace", [])
    for event in trace:
        if event.get("step") == "retrieve":
            retrieval_meta = event.get("retrieval_meta", {})
            if isinstance(retrieval_meta, dict):
                return retrieval_meta
    return {}


def artifact_frame() -> pd.DataFrame:
    rows = []
    for path in sorted(PROCESSED_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "artifact": str(path.relative_to(PROJECT_ROOT)),
                "folder": path.parent.name,
                "size_kb": round(path.stat().st_size / 1024, 1),
            }
        )
    return pd.DataFrame(rows)


def chunk_frame() -> pd.DataFrame:
    catalog = load_graph_catalog()
    documents = catalog.get("documents", [])
    chunks = catalog.get("chunks", [])
    doc_name_by_id = _document_name_by_id(documents)
    rows = []
    for chunk in chunks:
        document_name = canonical_document_name(doc_name_by_id.get(chunk.get("document_id", ""), "Unknown"))
        preview = " ".join(str(chunk.get("text", "")).split())[:180]
        section_title = str(chunk.get("metadata", {}).get("section_title", ""))
        rows.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "document_name": document_name,
                "article_ref": chunk.get("article_ref") or "",
                "section_title": section_title,
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "preview": preview,
                "label": f"{document_name} | {chunk.get('article_ref') or section_title or 'Section'} | {str(chunk.get('chunk_id', ''))[-6:]}",
            }
        )
    return pd.DataFrame(rows)


def regulation_names() -> list[str]:
    frame = chunk_frame()
    if frame.empty:
        return ["All"]
    return ["All", *sorted(frame["document_name"].dropna().unique().tolist())]


def available_article_refs(regulation: str = "All") -> list[str]:
    frame = chunk_frame()
    if frame.empty:
        return ["All"]
    if regulation != "All":
        frame = frame[frame["document_name"] == regulation]
    refs = sorted(item for item in frame["article_ref"].unique().tolist() if item)
    return ["All", *refs]


def filtered_chunk_frame(
    *,
    regulation: str = "All",
    article_ref: str = "All",
    query: str = "",
    article_only: bool = False,
) -> pd.DataFrame:
    frame = chunk_frame()
    if frame.empty:
        return frame
    if regulation != "All":
        frame = frame[frame["document_name"] == regulation]
    if article_ref != "All":
        frame = frame[frame["article_ref"] == article_ref]
    if article_only:
        frame = frame[frame["article_ref"] != ""]
    normalized_query = query.strip().lower()
    if normalized_query:
        mask = frame["preview"].str.lower().str.contains(normalized_query) | frame["section_title"].str.lower().str.contains(
            normalized_query
        )
        frame = frame[mask]
    return frame.reset_index(drop=True)


def chunk_detail(chunk_id: str) -> dict[str, Any]:
    catalog = load_graph_catalog()
    chunk_by_id = {chunk.get("chunk_id", ""): chunk for chunk in catalog.get("chunks", [])}
    entity_by_id = {entity.get("entity_id", ""): entity for entity in catalog.get("entities", [])}
    relations = catalog.get("relations", [])
    mentions_by_chunk = defaultdict(list)
    for mention in catalog.get("mentions", []):
        mentions_by_chunk[mention.get("chunk_id", "")].append(mention.get("entity_id", ""))

    chunk = chunk_by_id.get(chunk_id, {})
    mentioned_entities = [entity_by_id[entity_id] for entity_id in mentions_by_chunk.get(chunk_id, []) if entity_id in entity_by_id]
    related_entity_ids = {entity.get("entity_id", "") for entity in mentioned_entities}
    related_edges = [
        relation
        for relation in relations
        if relation.get("source_chunk_id") == chunk_id or relation.get("subject_entity_id") in related_entity_ids
    ]
    return {
        "chunk": chunk,
        "entities": mentioned_entities,
        "relations": related_edges[:30],
    }


def project_posture() -> dict[str, Any]:
    settings = load_settings()
    evaluation_summary, _, evaluation_aggregate = latest_evaluation_frames()
    cache_stats = path_cache_stats()
    warnings = []
    if str(settings.get("neo4j_password", "")).strip() in {"", "change-me-now", "neo4j"}:
        warnings.append("Default Neo4j password is still configured.")
    if not str(settings.get("api_key", "")).strip():
        warnings.append("API key protection is disabled for /v1 endpoints.")
    if str(settings.get("model_backend", "")).lower() == "local":
        warnings.append("Local mode is ideal for development, but final enterprise validation still benefits from external models.")
    return {
        "warnings": warnings,
        "evaluation_summary": evaluation_summary,
        "evaluation_aggregate": evaluation_aggregate,
        "runtime": {
            "provider": settings.get("model_backend", "unknown"),
            "chat_model": settings.get("local_chat_model") if settings.get("model_backend") == "local" else settings.get("chat_model"),
        },
        "raw_pdf_count": len(list(RAW_ROOT.glob("*.pdf"))),
        "path_cache_entries": cache_stats["entries"],
    }


def _document_name_by_id(documents: list[dict[str, Any]]) -> dict[str, str]:
    return {document.get("document_id", ""): document.get("name", "Unknown") for document in documents}


def _notes_value(notes: list[str], *, prefix: str) -> str:
    for note in notes:
        if note.startswith(prefix):
            return note.removeprefix(prefix)
    return "unknown"
