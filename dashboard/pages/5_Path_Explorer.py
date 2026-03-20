from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_ROOT = PROJECT_ROOT / "dashboard"
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from data_access import SAMPLE_QUESTIONS, get_runtime, load_settings, path_records_from_response, retrieval_meta_from_response
from ui import citation_card, configure_page, render_badges, render_card, render_doc_panel, section_title
from graphrag_engine.common.models import QueryRequest, QueryResponse


configure_page(
    "Path Explorer",
    "Inspect PathCacheRAG as a first-class retrieval mode: ranked legal paths, cache hits and misses, supporting chunks, and the grounded answer that emerged from them.",
)

settings = load_settings()


def run_query(question: str, *, retrieval_mode: str, top_k: int) -> QueryResponse:
    runtime = get_runtime()
    return runtime.build_agent().run(
        QueryRequest(question=question.strip(), retrieval_mode=retrieval_mode, top_k=top_k, debug=True)
    )


def _path_frame(response: QueryResponse) -> pd.DataFrame:
    rows = []
    for index, record in enumerate(path_records_from_response(response), start=1):
        rows.append(
            {
                "rank": index,
                "seed_entity": record.get("seed_entity", ""),
                "path": " -> ".join(record.get("traversed_entities", [])),
                "relations": " | ".join(record.get("relation_chain", [])),
                "supporting_chunks": ", ".join(record.get("supporting_chunk_ids", [])[:4]),
                "terminal_chunk_id": record.get("terminal_chunk_id", ""),
                "score": float(record.get("score", 0.0)),
                "depth": record.get("metadata", {}).get("depth", 0),
            }
        )
    return pd.DataFrame(rows)


def _chunk_frame(response: QueryResponse) -> pd.DataFrame:
    rows = []
    for index, chunk in enumerate(response.retrieved_chunks, start=1):
        rows.append(
            {
                "rank": index,
                "chunk_id": chunk.chunk_id,
                "article_ref": chunk.article_ref or "",
                "page_span": f"{chunk.page_start or '?'}-{chunk.page_end or '?'}",
                "preview": " ".join(chunk.text.split())[:220],
            }
        )
    return pd.DataFrame(rows)


with st.sidebar:
    st.subheader("Path session")
    st.write(f"Backend: `{settings['model_backend']}`")
    st.write(
        f"Configured model: `{settings['local_chat_model'] if settings['model_backend'] == 'local' else settings['chat_model']}`"
    )
    retrieval_mode = st.selectbox("Path retrieval mode", ["path_hybrid", "path_cache"], index=0)
    top_k = st.slider("Top-k evidence", min_value=3, max_value=12, value=int(settings["default_retrieval_k"]))
    starter = st.selectbox("Choose a question", ["Custom question", *SAMPLE_QUESTIONS])
    if st.button("Clear explorer state", use_container_width=True):
        st.session_state.pop("path_explorer_response", None)
        st.session_state.pop("path_explorer_question", None)

question_default = starter if starter != "Custom question" else st.session_state.get("path_explorer_question", "")

render_badges(
    [
        f"Mode: {retrieval_mode}",
        f"Top-k: {top_k}",
        "Path-grounded evidence",
        "Cache-aware retrieval",
    ]
)

question = st.text_area(
    "Question",
    value=question_default,
    height=140,
    placeholder="Ask an article-specific or multi-hop legal question to inspect path retrieval.",
)

run_now = st.button("Run path exploration", type="primary", use_container_width=True)

if run_now and question.strip():
    with st.spinner("Enumerating graph paths, checking the cache, and assembling path-grounded evidence..."):
        response = run_query(question.strip(), retrieval_mode=retrieval_mode, top_k=top_k)
    st.session_state["path_explorer_response"] = response
    st.session_state["path_explorer_question"] = question.strip()

response: QueryResponse | None = st.session_state.get("path_explorer_response")

if response is None:
    intro_left, intro_right = st.columns(2, gap="large")
    with intro_left:
        render_card(
            "What this page is for",
            "Use this page when you want to understand why PathCacheRAG chose a legal route, whether it reused cached path evidence, and which chunks backed the final answer.",
        )
    with intro_right:
        render_card(
            "Best questions for this view",
            "Article-specific questions, cross-regulation questions, and obligation chains are the best fit because they benefit most from graph-path retrieval rather than plain chunk ranking.",
        )
    st.info("Run a path-mode question to populate the explorer.")
else:
    retrieval_meta = retrieval_meta_from_response(response)
    path_frame = _path_frame(response)
    chunk_frame = _chunk_frame(response)

    metrics = st.columns(6)
    metrics[0].metric("Ranked paths", int(retrieval_meta.get("path_count", len(path_frame))))
    metrics[1].metric("Cache hit", "Yes" if retrieval_meta.get("cache_hit") else "No")
    metrics[2].metric("Cached entries", int(retrieval_meta.get("cached_entries", 0)))
    metrics[3].metric("Graph paths", len(response.graph_paths))
    metrics[4].metric("Citations", len(response.citations))
    metrics[5].metric("Retrieval latency", f"{float(retrieval_meta.get('total_latency_ms', 0.0)):.1f} ms")

    section_title("Answer alignment")
    answer_left, answer_right = st.columns((1.1, 0.9), gap="large")
    with answer_left:
        render_doc_panel(
            f"""
### Grounded answer

{response.answer}
            """
        )
    with answer_right:
        render_card("Confidence", f"{response.confidence:.2f}")
        render_card("Fallback used", "Yes" if response.fallback_used else "No")
        render_card(
            "Matched entity count",
            str(retrieval_meta.get("matched_entities", 0)),
        )
        render_card(
            "Document hints",
            ", ".join(sorted(retrieval_meta.get("document_hints", {}).keys())) or "None",
        )
        render_card(
            "Cache lookup latency",
            f"{float(retrieval_meta.get('cache_lookup_ms', 0.0)):.1f} ms",
        )
        render_card(
            "Path enumeration latency",
            f"{float(retrieval_meta.get('path_enumeration_ms', 0.0)):.1f} ms",
        )

    tab_paths, tab_chunks, tab_trace = st.tabs(["Ranked Paths", "Supporting Chunks", "Trace"])

    with tab_paths:
        if not path_frame.empty:
            chart = (
                alt.Chart(path_frame.head(12))
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#2f7c7d")
                .encode(
                    x=alt.X("rank:O", title="Path rank"),
                    y=alt.Y("score:Q", title="Path score"),
                    tooltip=["seed_entity", "path", "relations", "supporting_chunks", "terminal_chunk_id", "score"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(path_frame, use_container_width=True, hide_index=True)
        else:
            st.info("No path records were surfaced in the response trace.")

    with tab_chunks:
        if response.citations:
            for index, citation in enumerate(response.citations, start=1):
                citation_card(citation, index=index)
        if not chunk_frame.empty:
            st.dataframe(chunk_frame, use_container_width=True, hide_index=True)
        else:
            st.info("No supporting chunks were returned.")

    with tab_trace:
        trace_left, trace_right = st.columns(2, gap="large")
        with trace_left:
            st.markdown("**Retrieval metadata**")
            st.json(retrieval_meta)
        with trace_right:
            st.markdown("**Agent trace**")
            st.json(response.trace)

        if retrieval_meta.get("cache_key"):
            st.caption(f"Cache key: {retrieval_meta['cache_key']}")
