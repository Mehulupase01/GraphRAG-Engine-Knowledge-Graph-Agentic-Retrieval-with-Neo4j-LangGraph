from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_ROOT = PROJECT_ROOT / "dashboard"
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from data_access import SAMPLE_QUESTIONS, get_runtime, load_settings
from ui import citation_card, configure_page, render_badges, render_card, section_title
from graphrag_engine.common.models import QueryRequest, QueryResponse


configure_page(
    "Analyst Chat Console",
    "Ask grounded regulatory questions, compare GraphRAG with baseline retrieval, and inspect exactly why the system cited each piece of evidence.",
)

settings = load_settings()


def run_query(question: str, *, retrieval_mode: str, top_k: int) -> QueryResponse:
    runtime = get_runtime()
    return runtime.build_agent().run(
        QueryRequest(question=question.strip(), retrieval_mode=retrieval_mode, top_k=top_k, debug=True)
    )


def response_metrics(response: QueryResponse) -> dict[str, float | int | str]:
    return {
        "citations": len(response.citations),
        "graph_paths": len(response.graph_paths),
        "confidence": response.confidence,
        "fallback": "Yes" if response.fallback_used else "No",
    }


def render_response_panel(title: str, response: QueryResponse) -> None:
    metrics = response_metrics(response)
    render_card(
        title,
        f"Confidence {metrics['confidence']:.2f} | Citations {metrics['citations']} | Graph paths {metrics['graph_paths']} | Fallback {metrics['fallback']}",
    )
    st.markdown(f'<div class="answer-shell">{response.answer}</div>', unsafe_allow_html=True)


def render_response_details(response: QueryResponse, *, panel_key: str) -> None:
    metrics = response_metrics(response)
    summary_columns = st.columns(4)
    summary_columns[0].metric("Citations", metrics["citations"])
    summary_columns[1].metric("Graph Paths", metrics["graph_paths"])
    summary_columns[2].metric("Confidence", f"{metrics['confidence']:.2f}")
    summary_columns[3].metric("Fallback", metrics["fallback"])

    section_title("Grounded Answer")
    st.markdown(f'<div class="answer-shell">{response.answer}</div>', unsafe_allow_html=True)

    tab_evidence, tab_scores, tab_trace = st.tabs(["Evidence", "Scores", "Trace"])
    with tab_evidence:
        if response.citations:
            for index, citation in enumerate(response.citations, start=1):
                citation_card(citation, index=index)
        else:
            st.info("No citations were returned.")

    with tab_scores:
        rows = []
        for citation in response.citations:
            rows.append(
                {
                    "document": citation.document_name,
                    "article": citation.article_ref or "None",
                    "chunk_id": citation.chunk_id,
                    **citation.score_breakdown,
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No score breakdowns are available.")

    with tab_trace:
        trace_col, graph_col = st.columns(2, gap="large")
        with trace_col:
            st.markdown("**Trace events**")
            st.json(response.trace)
        with graph_col:
            st.markdown("**Graph paths**")
            st.json([path.model_dump() for path in response.graph_paths[:20]])

    if response.retrieved_chunks:
        with st.expander(f"Retrieved chunks ({len(response.retrieved_chunks)})", expanded=False):
            preview_rows = []
            for chunk in response.retrieved_chunks[:25]:
                preview_rows.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "article_ref": chunk.article_ref or "",
                        "page_span": f"{chunk.page_start or '?'}-{chunk.page_end or '?'}",
                        "preview": " ".join(chunk.text.split())[:220],
                    }
                )
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    st.caption(f"Panel key: {panel_key}")


with st.sidebar:
    st.subheader("Session")
    st.write(f"Backend: `{settings['model_backend']}`")
    st.write(
        f"Configured model: `{settings['local_chat_model'] if settings['model_backend'] == 'local' else settings['chat_model']}`"
    )
    compare_mode = st.checkbox("Compare adaptive vs baseline", value=True)
    retrieval_mode = st.selectbox(
        "Primary retrieval mode",
        ["adaptive", "hybrid", "baseline", "path_hybrid", "path_cache"],
        index=0,
        disabled=compare_mode,
    )
    top_k = st.slider("Top-k evidence", min_value=3, max_value=12, value=int(settings["default_retrieval_k"]))
    st.subheader("Starter prompts")
    starter = st.selectbox("Choose a question", ["Custom question", *SAMPLE_QUESTIONS])
    if st.button("Clear session history", use_container_width=True):
        st.session_state.pop("query_runs", None)
        st.session_state.pop("query_history", None)
        st.session_state.pop("last_question", None)

default_question = starter if starter != "Custom question" else st.session_state.get("last_question", "")

render_badges(
    [
        "Adaptive compare mode" if compare_mode else f"Single mode: {retrieval_mode}",
        f"Top-k: {top_k}",
        "Adaptive routing enabled",
        "Grounded legal QA",
    ]
)

question = st.text_area(
    "Question",
    value=default_question,
    height=140,
    placeholder="Ask about obligations, article scope, cross-regulation interactions, or provenance.",
)

run_now = st.button("Run grounded query", type="primary", use_container_width=True)

if run_now and question.strip():
    requested_modes = ["adaptive", "baseline"] if compare_mode else [retrieval_mode]
    results: dict[str, QueryResponse] = {}
    with st.spinner("Running retrieval, graph traversal, and grounded answer synthesis..."):
        for mode in requested_modes:
            results[mode] = run_query(question.strip(), retrieval_mode=mode, top_k=top_k)
    st.session_state["query_runs"] = results
    st.session_state["last_question"] = question.strip()
    history = st.session_state.setdefault("query_history", [])
    history.insert(
        0,
        {
            "question": question.strip(),
            "modes": ", ".join(requested_modes),
            "top_k": top_k,
            "confidence": ", ".join(f"{mode}:{response.confidence:.2f}" for mode, response in results.items()),
        },
    )
    st.session_state["query_history"] = history[:8]

query_runs: dict[str, QueryResponse] = st.session_state.get("query_runs", {})

if not query_runs:
    section_title("How to use this page")
    intro_left, intro_right = st.columns(2, gap="large")
    with intro_left:
        render_card(
            "Compare mode",
            "Runs the same question through adaptive PathCacheRAG and the baseline path so you can see whether cache-aware graph routing improved the answer.",
        )
    with intro_right:
        render_card(
            "What to inspect",
            "Read the answer, then inspect citations, score breakdowns, retrieved chunks, and graph paths. The answer matters, but the evidence trail matters more.",
        )
    st.info("Run a question to populate the analyst console.")
else:
    section_title("Session Summary")
    history = st.session_state.get("query_history", [])
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)

    modes = list(query_runs.keys())
    if set(modes) == {"adaptive", "baseline"}:
        comparison_columns = st.columns(2, gap="large")
        with comparison_columns[0]:
            render_response_panel("Adaptive PathCacheRAG", query_runs["adaptive"])
        with comparison_columns[1]:
            render_response_panel("Baseline Retrieval", query_runs["baseline"])

        hybrid_conf = query_runs["adaptive"].confidence
        baseline_conf = query_runs["baseline"].confidence
        comparison_metrics = st.columns(4)
        comparison_metrics[0].metric("Adaptive citations", len(query_runs["adaptive"].citations))
        comparison_metrics[1].metric("Baseline citations", len(query_runs["baseline"].citations))
        comparison_metrics[2].metric("Confidence delta", f"{hybrid_conf - baseline_conf:+.2f}")
        comparison_metrics[3].metric(
            "Graph path delta",
            f"{len(query_runs['adaptive'].graph_paths) - len(query_runs['baseline'].graph_paths):+d}",
        )

        compare_tab, hybrid_tab, baseline_tab = st.tabs(["Comparison", "Adaptive Detail", "Baseline Detail"])
        with compare_tab:
            compare_rows = []
            for mode_name, response in query_runs.items():
                compare_rows.append(
                    {
                        "mode": mode_name,
                        "confidence": response.confidence,
                        "citations": len(response.citations),
                        "graph_paths": len(response.graph_paths),
                        "fallback": response.fallback_used,
                    }
                )
            st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)
        with hybrid_tab:
            render_response_details(query_runs["adaptive"], panel_key="adaptive")
        with baseline_tab:
            render_response_details(query_runs["baseline"], panel_key="baseline")
    else:
        mode_name = modes[0]
        render_response_details(query_runs[mode_name], panel_key=mode_name)
