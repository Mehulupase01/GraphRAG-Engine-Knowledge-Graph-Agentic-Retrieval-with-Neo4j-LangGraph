from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_ROOT = PROJECT_ROOT / "dashboard"
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from data_access import (
    artifact_frame,
    corpus_overview,
    latest_evaluation_delta,
    latest_evaluation_frames,
    load_graph_stats,
    load_jobs,
    path_cache_frame,
    path_cache_stats,
    project_posture,
)
from ui import configure_page, render_card, render_success_banner, render_warning_banner, section_title


configure_page(
    "Operations And Quality",
    "Inspect ingestion runs, graph health, evaluation outcomes, cache posture, processed artifacts, and local deployment readiness from one operator workspace.",
)

overview = corpus_overview()
graph_stats = load_graph_stats()
jobs = load_jobs()
posture = project_posture()
evaluation_summary, evaluation_cases, evaluation_aggregate = latest_evaluation_frames()
evaluation_delta = latest_evaluation_delta()
cache_stats = path_cache_stats()
cache_frame = path_cache_frame()

if posture["warnings"]:
    render_warning_banner(posture["warnings"])
else:
    render_success_banner("This workspace currently shows no obvious local operations warnings.")

metric_columns = st.columns(6)
metric_columns[0].metric("Recorded Jobs", len(jobs))
metric_columns[1].metric("Communities", graph_stats.get("communities_detected", 0))
metric_columns[2].metric("Neo4j", "Connected" if graph_stats.get("used_neo4j") else "Pending")
metric_columns[3].metric("Raw PDFs", posture["raw_pdf_count"])
metric_columns[4].metric("Path cache entries", cache_stats["entries"])
metric_columns[5].metric(
    "Best eval delta",
    f"{evaluation_delta['best_delta']:+.3f}" if evaluation_delta["best_delta"] is not None else "n/a",
)

tabs = st.tabs(
    ["Pipeline", "Knowledge Graph", "Benchmark", "Path Cache", "Artifacts", "Runtime Posture"]
)
tab_pipeline, tab_graph, tab_eval, tab_cache, tab_artifacts, tab_posture = tabs

with tab_pipeline:
    section_title("Recent ingestion and processing jobs")
    if jobs:
        st.dataframe(jobs[:12], use_container_width=True, hide_index=True)
    else:
        st.info("No persisted job records were found.")

    section_title("Graph load stats")
    if graph_stats:
        stats_left, stats_right = st.columns((0.95, 1.05), gap="large")
        with stats_left:
            st.json(graph_stats)
        with stats_right:
            render_card(
                "What this means",
                "These stats describe the latest graph build, including whether Neo4j was used, how many records were loaded, and what notes the loader emitted.",
            )
            render_card(
                "What to check next",
                "If documents or chunks are lower than expected, revisit ingestion artifacts first. If entities or relations look weak, re-run extraction before tuning retrieval.",
            )
    else:
        st.info("No graph load stats are available yet.")

with tab_graph:
    graph_left, graph_right = st.columns(2, gap="large")
    with graph_left:
        section_title("Document coverage")
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
        section_title("Entity communities")
        community_frame = overview["community_frame"]
        if not community_frame.empty:
            chart = (
                alt.Chart(community_frame)
                .mark_circle(size=260, color="#b55a3c")
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

    lower_left, lower_right = st.columns(2, gap="large")
    with lower_left:
        section_title("Entity type distribution")
        entity_frame = overview["entity_frame"].head(15)
        if not entity_frame.empty:
            st.dataframe(entity_frame, use_container_width=True, hide_index=True)
        else:
            st.info("No entity data is available.")
    with lower_right:
        section_title("Relation type distribution")
        relation_frame = overview["relation_frame"].head(15)
        if not relation_frame.empty:
            st.dataframe(relation_frame, use_container_width=True, hide_index=True)
        else:
            st.info("No relation data is available.")

with tab_eval:
    section_title("Latest evaluation summary")
    if evaluation_summary and not evaluation_aggregate.empty:
        summary_left, summary_right = st.columns((0.95, 1.05), gap="large")
        with summary_left:
            st.dataframe(evaluation_aggregate, use_container_width=True, hide_index=True)
            regressions = evaluation_summary.get("regressions", [])
            if regressions:
                render_warning_banner(regressions)
            else:
                render_success_banner("No regression was flagged in the latest report.")
            metadata = evaluation_summary.get("metadata", {})
            if metadata:
                st.caption("Evaluation metadata")
                st.json(metadata)
        with summary_right:
            score_chart = (
                alt.Chart(evaluation_aggregate)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#b55a3c")
                .encode(
                    x=alt.X("approach:N", title="Approach"),
                    y=alt.Y("average_score:Q", title="Average score"),
                    tooltip=["approach", "average_score", "faithfulness", "context_precision", "answer_relevancy", "multi_hop_accuracy"],
                )
                .properties(height=280)
            )
            st.altair_chart(score_chart, use_container_width=True)

            if not evaluation_cases.empty:
                boxplot = (
                    alt.Chart(evaluation_cases)
                    .mark_boxplot(extent="min-max")
                    .encode(
                        x=alt.X("approach:N", title="Approach"),
                        y=alt.Y("score:Q", title="Per-case score"),
                        color=alt.Color("approach:N", legend=None),
                        tooltip=["case_id", "approach", "score", "difficulty"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(boxplot, use_container_width=True)
        if not evaluation_cases.empty:
            section_title("Per-case sample")
            st.dataframe(evaluation_cases.head(30), use_container_width=True, hide_index=True)
    else:
        st.info("No evaluation report has been generated yet.")

with tab_cache:
    cache_left, cache_right = st.columns((0.95, 1.05), gap="large")
    with cache_left:
        section_title("Cache posture")
        render_card(
            "Why this matters",
            "PathCacheRAG stores reusable path evidence packs so repeated legal questions can skip expensive path enumeration and preserve more consistent evidence routing.",
        )
        retrieval_modes = cache_stats.get("retrieval_modes", {})
        if retrieval_modes:
            mode_rows = [
                {"retrieval_mode": mode, "entries": count}
                for mode, count in sorted(retrieval_modes.items(), key=lambda item: item[1], reverse=True)
            ]
            chart = (
                alt.Chart(alt.Data(values=mode_rows))
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#2f7c7d")
                .encode(
                    x=alt.X("retrieval_mode:N", title="Cache mode"),
                    y=alt.Y("entries:Q", title="Entries"),
                    tooltip=["retrieval_mode", "entries"],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No path cache entries have been persisted yet.")

    with cache_right:
        section_title("Cache interpretation")
        render_card(
            "Healthy pattern",
            "You want repeated article-specific or multi-hop questions to produce cache hits, while still allowing fresh retrieval when the question meaning materially changes.",
        )
        render_card(
            "Operator note",
            "A large cache is not automatically better. What matters is whether high-value legal routes are reused and whether cache hits preserve answer quality.",
        )
        st.metric("Cache entries", cache_stats["entries"])
        st.metric("Best evaluated mode", str(evaluation_delta["best_mode"] or "n/a"))

    section_title("Persisted cache entries")
    if not cache_frame.empty:
        st.dataframe(cache_frame.head(40), use_container_width=True, hide_index=True)
    else:
        st.info("No persisted cache artifacts were found.")

with tab_artifacts:
    section_title("Processed artifacts")
    artifacts = artifact_frame()
    if not artifacts.empty:
        st.dataframe(artifacts, use_container_width=True, hide_index=True)
    else:
        st.info("No processed artifacts were found.")

with tab_posture:
    section_title("Operational interpretation")
    posture_left, posture_right = st.columns(2, gap="large")
    with posture_left:
        render_card(
            "Current release posture",
            "This project is local-first and production-structured. It has ingestion, graph build, retrieval, evaluation, API, Docker, and dashboard layers in place, but should still be treated as a controlled deployment until secrets and internet-facing auth are hardened.",
        )
        render_card(
            "Without Anthropic or Gemini keys",
            "That is fine. The local Qwen path remains the main development backend, and external providers stay optional rather than required.",
        )
    with posture_right:
        render_card(
            "Recommended next checks",
            "Change the default Neo4j password, optionally set GRAPH_RAG_API_KEY for the API, re-run evaluation after major retrieval changes, and smoke test the API plus dashboard before sharing beyond localhost.",
        )
        if graph_stats.get("notes"):
            st.caption("Latest graph notes")
            for note in graph_stats["notes"]:
                st.write(f"- {note}")
