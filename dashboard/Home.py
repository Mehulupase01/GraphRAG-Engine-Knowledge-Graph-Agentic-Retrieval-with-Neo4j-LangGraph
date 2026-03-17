from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "processed"

st.set_page_config(page_title="GraphRAG Engine", layout="wide")
st.title("GraphRAG Engine")
st.caption("Production-grade GraphRAG for EU regulations")

st.markdown(
    """
    Use the left navigation to switch between:

    - `Chat`: query the GraphRAG engine and inspect grounded citations
    - `Ops`: inspect ingestion jobs, graph stats, and evaluation reports
    """
)

metrics = {
    "documents": 0,
    "chunks": 0,
    "entities": 0,
    "relations": 0,
}

graph_catalog = DATA_ROOT / "graph" / "graph_catalog.json"
if graph_catalog.exists():
    payload = json.loads(graph_catalog.read_text(encoding="utf-8"))
    metrics["documents"] = len(payload.get("documents", []))
    metrics["chunks"] = len(payload.get("chunks", []))
    metrics["entities"] = len(payload.get("entities", []))
    metrics["relations"] = len(payload.get("relations", []))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Documents", metrics["documents"])
col2.metric("Chunks", metrics["chunks"])
col3.metric("Entities", metrics["entities"])
col4.metric("Relations", metrics["relations"])

