from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "processed"

st.title("Ops")
st.caption("Ingestion jobs, graph stats, and evaluation reports")

jobs_root = DATA_ROOT / "jobs"
job_files = sorted(jobs_root.glob("*.json"), reverse=True)
if job_files:
    st.subheader("Ingestion Jobs")
    for job_file in job_files[:5]:
        st.json(json.loads(job_file.read_text(encoding="utf-8")))
else:
    st.info("No ingestion jobs found yet.")

graph_stats = DATA_ROOT / "graph" / "load_stats.json"
if graph_stats.exists():
    st.subheader("Graph Load Stats")
    st.json(json.loads(graph_stats.read_text(encoding="utf-8")))

evaluation_root = DATA_ROOT / "evaluation"
evaluation_files = sorted(evaluation_root.glob("*.json"), reverse=True)
if evaluation_files:
    st.subheader("Evaluation Reports")
    for path in evaluation_files[:3]:
        st.json(json.loads(path.read_text(encoding="utf-8")))

