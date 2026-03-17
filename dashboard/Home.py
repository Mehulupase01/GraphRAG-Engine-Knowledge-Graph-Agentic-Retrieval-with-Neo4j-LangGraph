from __future__ import annotations

import altair as alt
import streamlit as st

from shared import RAW_ROOT, configure_page, corpus_overview, latest_evaluation_frames, load_graph_stats, section_title


configure_page(
    "Flagship EU GraphRAG Workspace",
    "A local-first regulatory intelligence system for the AI Act, GDPR, and Digital Services Act with graph retrieval, provenance, and operator tooling.",
)

overview = corpus_overview()
graph_stats = load_graph_stats()
evaluation_summary, _, evaluation_aggregate = latest_evaluation_frames()

metric_columns = st.columns(4)
metric_columns[0].metric("Corpus Documents", len(overview["documents"]))
metric_columns[1].metric("Indexed Chunks", len(overview["chunks"]))
metric_columns[2].metric("Canonical Entities", len(overview["entities"]))
metric_columns[3].metric("Graph Relations", len(overview["relations"]))

section_title("Pipeline Readiness")
status_columns = st.columns(4)
status_columns[0].metric("Raw PDFs", len(list(RAW_ROOT.glob("*.pdf"))))
status_columns[1].metric("Graph Communities", graph_stats.get("communities_detected", 0))
status_columns[2].metric("Neo4j Load", "Ready" if graph_stats.get("used_neo4j") else "Pending")
status_columns[3].metric("Evaluation Runs", 1 if evaluation_summary else 0)

left, right = st.columns((1.2, 1.0), gap="large")
with left:
    section_title("Corpus Coverage")
    document_frame = overview["document_frame"]
    if not document_frame.empty:
        chart = (
            alt.Chart(document_frame)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("document_name:N", sort="-y", title="Regulation"),
                y=alt.Y("chunks:Q", title="Chunks"),
                color=alt.Color("document_name:N", legend=None),
                tooltip=["document_name", "chunks", "article_chunks", "page_span"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Corpus artifacts will appear here after ingestion and graph build.")

with right:
    section_title("Knowledge Mix")
    entity_frame = overview["entity_frame"].head(10)
    if not entity_frame.empty:
        chart = (
            alt.Chart(entity_frame)
            .mark_arc(innerRadius=58)
            .encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color("entity_type:N", legend=alt.Legend(title="Entity Type")),
                tooltip=["entity_type", "count"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Entity distribution will appear after extraction.")

bottom_left, bottom_right = st.columns((1.0, 1.0), gap="large")
with bottom_left:
    section_title("Relation Profile")
    relation_frame = overview["relation_frame"].head(12)
    if not relation_frame.empty:
        chart = (
            alt.Chart(relation_frame)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#b55a3c")
            .encode(
                x=alt.X("relation_type:N", sort="-y", title="Relation Type"),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["relation_type", "count"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Relation profile will appear after graph construction.")

with bottom_right:
    section_title("Evaluation Snapshot")
    if evaluation_summary and not evaluation_aggregate.empty:
        st.dataframe(
            evaluation_aggregate.rename(columns={"average_score": "average_score"}),
            use_container_width=True,
            hide_index=True,
        )
        regressions = evaluation_summary.get("regressions", [])
        if regressions:
            st.warning("\n".join(regressions))
        else:
            st.success("Latest evaluation shows GraphRAG at or above baseline.")
    else:
        st.info("No evaluation report has been generated yet. Run `graphrag-engine run-eval` to populate this panel.")

if graph_stats.get("notes"):
    section_title("Latest Graph Load Notes")
    for note in graph_stats["notes"]:
        st.write(f"- {note}")
