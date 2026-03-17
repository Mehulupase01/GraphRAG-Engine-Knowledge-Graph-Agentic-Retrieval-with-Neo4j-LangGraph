from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphrag_engine.common.models import QueryRequest
from graphrag_engine.runtime import GraphRAGRuntime


st.title("Chat")
st.caption("Grounded querying over the knowledge graph and corpus artifacts")

runtime = GraphRAGRuntime()
question = st.text_area("Ask a regulatory question", height=120)
mode = st.selectbox("Retrieval mode", ["hybrid", "baseline"])

if st.button("Run query", type="primary") and question.strip():
    response = runtime.build_agent().run(QueryRequest(question=question, retrieval_mode=mode))
    st.subheader("Answer")
    st.write(response.answer)

    st.subheader("Citations")
    for citation in response.citations:
        st.markdown(
            f"**{citation.document_name}** | {citation.article_ref or 'No article ref'} | "
            f"pages {citation.page_start or '?'}-{citation.page_end or '?'}"
        )
        st.code(citation.snippet)

    with st.expander("Trace"):
        st.json(response.trace)
