from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_ROOT = PROJECT_ROOT / "dashboard"
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from shared import SAMPLE_QUESTIONS, citation_card, configure_page, get_runtime, section_title
from graphrag_engine.common.models import QueryRequest


configure_page(
    "Analyst Chat Console",
    "Ask grounded regulatory questions, inspect the answer path, and verify why each citation was selected.",
)

runtime = get_runtime()
settings = runtime.settings

with st.sidebar:
    st.subheader("Runtime")
    st.write(f"Backend: `{settings.model_backend}`")
    st.write(f"Chat model: `{settings.local_chat_model if settings.model_backend == 'local' else settings.chat_model}`")
    mode = st.selectbox("Retrieval mode", ["hybrid", "baseline"])
    top_k = st.slider("Top-k evidence", min_value=3, max_value=12, value=settings.default_retrieval_k)
    st.subheader("Starter prompts")
    starter = st.selectbox("Choose a question", ["Custom question", *SAMPLE_QUESTIONS])

default_question = ""
if starter != "Custom question":
    default_question = starter

question = st.text_area("Question", value=default_question, height=140, placeholder="Ask about obligations, article scope, provenance, or cross-regulation duties.")
run_query = st.button("Run grounded query", type="primary", use_container_width=True)

if run_query and question.strip():
    with st.spinner("Running retrieval, graph traversal, and grounded answer synthesis..."):
        response = runtime.build_agent().run(
            QueryRequest(question=question.strip(), retrieval_mode=mode, top_k=top_k, debug=True)
        )
    st.session_state["last_response"] = response
    st.session_state["last_question"] = question.strip()

response = st.session_state.get("last_response")
if response is not None:
    summary_columns = st.columns(4)
    summary_columns[0].metric("Citations", len(response.citations))
    summary_columns[1].metric("Graph Paths", len(response.graph_paths))
    summary_columns[2].metric("Confidence", f"{response.confidence:.2f}")
    summary_columns[3].metric("Fallback", "Yes" if response.fallback_used else "No")

    section_title("Grounded Answer")
    st.markdown(f'<div class="answer-shell">{response.answer}</div>', unsafe_allow_html=True)

    section_title("Evidence Pack")
    for index, citation in enumerate(response.citations, start=1):
        citation_card(citation, index=index)

    score_rows = []
    for citation in response.citations:
        score_rows.append(
            {
                "document": citation.document_name,
                "article": citation.article_ref or "None",
                "chunk_id": citation.chunk_id,
                **citation.score_breakdown,
            }
        )
    if score_rows:
        st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

    with st.expander("Trace and graph paths"):
        trace_col, path_col = st.columns(2, gap="large")
        with trace_col:
            st.markdown("**Trace**")
            st.json(response.trace)
        with path_col:
            st.markdown("**Graph paths**")
            st.json([path.model_dump() for path in response.graph_paths[:20]])
else:
    st.info("Run a question to see grounded answers, scored citations, and the retrieval trace.")
