from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_ROOT = PROJECT_ROOT / "dashboard"
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from shared import artifact_frame, configure_page, corpus_overview, latest_evaluation_frames, load_graph_stats, load_jobs, section_title


configure_page(
    "Operations And Quality",
    "Inspect ingestion runs, graph health, evaluation outcomes, and generated artifacts from the production workspace.",
)

overview = corpus_overview()
graph_stats = load_graph_stats()
jobs = load_jobs()
evaluation_summary, evaluation_cases, evaluation_aggregate = latest_evaluation_frames()

jobs_col, graph_col, eval_col, neo4j_col = st.columns(4)
jobs_col.metric("Recorded Jobs", len(jobs))
graph_col.metric("Communities", graph_stats.get("communities_detected", 0))
eval_col.metric("Evaluation Runs", 1 if evaluation_summary else 0)
neo4j_col.metric("Neo4j", "Connected" if graph_stats.get("used_neo4j") else "Pending")

tab_pipeline, tab_graph, tab_eval, tab_artifacts = st.tabs(
    ["Pipeline", "Knowledge Graph", "Evaluation", "Artifacts"]
)

with tab_pipeline:
    section_title("Recent Ingestion Jobs")
    if jobs:
        st.dataframe(jobs[:8], use_container_width=True)
    else:
        st.info("No persisted ingestion job records were found.")

    section_title("Graph Load Stats")
    if graph_stats:
        st.json(graph_stats)
    else:
        st.info("No graph load stats are available yet.")

with tab_graph:
    graph_left, graph_right = st.columns(2, gap="large")
    with graph_left:
        section_title("Document Coverage")
        document_frame = overview["document_frame"]
        if not document_frame.empty:
            chart = (
                alt.Chart(document_frame)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#2f7c7d")
                .encode(
                    x=alt.X("document_name:N", sort="-y", title="Regulation"),
                    y=alt.Y("article_chunks:Q", title="Article-aware chunks"),
                    tooltip=["document_name", "chunks", "article_chunks", "page_span"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Document coverage will populate after graph build.")

    with graph_right:
        section_title("Entity Communities")
        community_frame = overview["community_frame"]
        if not community_frame.empty:
            chart = (
                alt.Chart(community_frame)
                .mark_circle(size=240, color="#b55a3c")
                .encode(
                    x=alt.X("community_id:O", title="Community"),
                    y=alt.Y("member_count:Q", title="Members"),
                    tooltip=["community_id", "member_count"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Community data will appear after graph construction.")

    section_title("Type Distributions")
    distribution_left, distribution_right = st.columns(2, gap="large")
    with distribution_left:
        entity_frame = overview["entity_frame"].head(12)
        if not entity_frame.empty:
            st.dataframe(entity_frame, use_container_width=True, hide_index=True)
    with distribution_right:
        relation_frame = overview["relation_frame"].head(12)
        if not relation_frame.empty:
            st.dataframe(relation_frame, use_container_width=True, hide_index=True)

with tab_eval:
    section_title("Latest Evaluation Summary")
    if evaluation_summary and not evaluation_aggregate.empty:
        summary_left, summary_right = st.columns((0.8, 1.2), gap="large")
        with summary_left:
            st.dataframe(evaluation_aggregate, use_container_width=True, hide_index=True)
            if evaluation_summary.get("regressions"):
                st.warning("\n".join(evaluation_summary["regressions"]))
            else:
                st.success("No regression was flagged in the latest report.")
        with summary_right:
            if not evaluation_cases.empty:
                chart = (
                    alt.Chart(evaluation_cases)
                    .mark_boxplot(extent="min-max")
                    .encode(
                        x=alt.X("approach:N", title="Approach"),
                        y=alt.Y("score:Q", title="Per-case score"),
                        color=alt.Color("approach:N", legend=None),
                        tooltip=["case_id", "approach", "score", "difficulty"],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart, use_container_width=True)
        if not evaluation_cases.empty:
            st.dataframe(evaluation_cases.head(24), use_container_width=True, hide_index=True)
    else:
        st.info("No evaluation output is available yet. Run the benchmark to populate this tab.")

with tab_artifacts:
    section_title("Processed Artifacts")
    artifacts = artifact_frame()
    if not artifacts.empty:
        st.dataframe(artifacts, use_container_width=True, hide_index=True)
    else:
        st.info("No processed artifacts were found.")
