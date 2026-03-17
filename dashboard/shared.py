from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
RAW_ROOT = DATA_ROOT / "raw"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
THEME_CSS = """
<style>
    :root {
        --ink: #1d2a39;
        --muted: #5c6b78;
        --accent: #b55a3c;
        --accent-soft: #f8e5d7;
        --teal: #2f7c7d;
        --card: rgba(255, 250, 243, 0.86);
        --border: rgba(29, 42, 57, 0.12);
        --shadow: 0 16px 40px rgba(29, 42, 57, 0.08);
    }
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(247, 214, 180, 0.55), transparent 28%),
            radial-gradient(circle at 85% 8%, rgba(142, 203, 205, 0.30), transparent 24%),
            linear-gradient(180deg, #f8f3ea 0%, #f4efe7 52%, #eef3f6 100%);
        color: var(--ink);
    }
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.8rem;
        max-width: 1300px;
    }
    h1, h2, h3 {
        font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
        color: var(--ink);
        letter-spacing: -0.03em;
    }
    .stMarkdown, .stCaption, .stTextArea label, .stSelectbox label, .stSlider label, .stTabs [data-baseweb="tab"] {
        font-family: "Segoe UI Variable Text", "Trebuchet MS", sans-serif;
    }
    [data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow);
    }
    .hero-shell {
        background: linear-gradient(135deg, rgba(255, 250, 243, 0.94), rgba(255, 240, 224, 0.92));
        border: 1px solid rgba(181, 90, 60, 0.16);
        border-radius: 26px;
        padding: 1.6rem 1.8rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.2rem;
    }
    .hero-kicker {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--teal);
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2.25rem;
        line-height: 1.05;
        margin: 0 0 0.45rem 0;
        font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
        color: var(--ink);
    }
    .hero-copy {
        font-size: 1rem;
        line-height: 1.55;
        color: var(--muted);
        max-width: 760px;
        margin: 0;
    }
    .section-title {
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--teal);
        margin: 1.4rem 0 0.75rem;
    }
    .citation-card {
        background: rgba(255, 250, 243, 0.88);
        border: 1px solid rgba(29, 42, 57, 0.10);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        box-shadow: var(--shadow);
        margin-bottom: 0.9rem;
    }
    .citation-meta {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        color: var(--muted);
        font-size: 0.85rem;
        margin-bottom: 0.55rem;
    }
    .answer-shell {
        background: linear-gradient(135deg, rgba(47, 124, 125, 0.12), rgba(255, 250, 243, 0.92));
        border: 1px solid rgba(47, 124, 125, 0.18);
        border-radius: 24px;
        padding: 1.15rem 1.2rem;
        box-shadow: var(--shadow);
    }
    .pill-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.8rem;
    }
    .pill {
        border-radius: 999px;
        padding: 0.38rem 0.72rem;
        background: rgba(47, 124, 125, 0.1);
        border: 1px solid rgba(47, 124, 125, 0.12);
        color: var(--ink);
        font-size: 0.83rem;
    }
    .status-good {
        color: #1f6b52;
        font-weight: 700;
    }
    .status-warn {
        color: #a46018;
        font-weight: 700;
    }
</style>
"""


def configure_page(title: str, subtitle: str) -> None:
    st.set_page_config(page_title=title, layout="wide")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-kicker">GraphRAG Engine</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-copy">{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_title(label: str) -> None:
    st.markdown(f'<div class="section-title">{label}</div>', unsafe_allow_html=True)


def canonical_document_name(name: str) -> str:
    normalized = name.lower().strip()
    return DOCUMENT_NAME_MAP.get(normalized, name.replace("_", " ").strip())


@st.cache_resource(show_spinner=False)
def get_runtime() -> GraphRAGRuntime:
    return GraphRAGRuntime()


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
    for chunk in chunks:
        document_name = canonical_document_name(_document_name_by_id(documents).get(chunk.get("document_id", ""), "Unknown"))
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
    return {
        "documents": documents,
        "chunks": chunks,
        "entities": entities,
        "relations": relations,
        "document_frame": document_frame,
        "entity_frame": entity_frame,
        "relation_frame": relation_frame,
        "community_frame": community_frame,
    }


def latest_evaluation_frames() -> tuple[dict[str, Any] | None, pd.DataFrame, pd.DataFrame]:
    evaluations = load_evaluations()
    if not evaluations:
        return None, pd.DataFrame(), pd.DataFrame()

    summary = evaluations[0]
    case_rows: list[dict[str, Any]] = []
    for case in summary.get("cases", []):
        metadata = case.get("response", {}).get("trace", [])
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
            }
        )
    case_frame = pd.DataFrame(case_rows)

    aggregate_rows = []
    for approach, metrics in summary.get("aggregate_scores", {}).items():
        aggregate_rows.append(
            {
                "approach": approach,
                **metrics,
            }
        )
    aggregate_frame = pd.DataFrame(aggregate_rows)
    return summary, case_frame, aggregate_frame


def artifact_frame() -> pd.DataFrame:
    rows = []
    for path in sorted(PROCESSED_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "artifact": str(path.relative_to(PROJECT_ROOT)),
                "size_kb": round(path.stat().st_size / 1024, 1),
            }
        )
    return pd.DataFrame(rows)


def citation_card(citation, *, index: int) -> None:
    scores = citation.score_breakdown
    st.markdown(
        f"""
        <div class="citation-card">
            <div class="citation-meta">
                <strong>{index}. {citation.document_name}</strong>
                <span>{citation.article_ref or 'No article reference'}</span>
                <span>Pages {citation.page_start or '?'}-{citation.page_end or '?'}</span>
                <span>Fused score {scores.get('fused', 0.0):.4f}</span>
            </div>
            <div>{citation.snippet}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _document_name_by_id(documents: list[dict[str, Any]]) -> dict[str, str]:
    return {document.get("document_id", ""): document.get("name", "Unknown") for document in documents}


def _notes_value(notes: list[str], *, prefix: str) -> str:
    for note in notes:
        if note.startswith(prefix):
            return note.removeprefix(prefix)
    return "unknown"
