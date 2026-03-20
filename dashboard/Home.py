from __future__ import annotations

import altair as alt
import streamlit as st

from data_access import corpus_overview, latest_evaluation_delta, latest_evaluation_frames, load_graph_stats, load_settings, path_cache_stats, project_posture
from ui import configure_page, render_badges, render_card, render_doc_panel, render_success_banner, render_warning_banner, section_title


configure_page(
    "Flagship EU GraphRAG Workspace",
    "A modular regulatory intelligence workspace for the EU AI Act, GDPR, and Digital Services Act with graph retrieval, provenance, evaluation, and operator tooling.",
)

overview = corpus_overview()
graph_stats = load_graph_stats()
settings = load_settings()
posture = project_posture()
evaluation_summary, evaluation_cases, evaluation_aggregate = latest_evaluation_frames()
evaluation_delta = latest_evaluation_delta()
cache_stats = path_cache_stats()

if posture["warnings"]:
    render_warning_banner(posture["warnings"])
else:
    render_success_banner("The workspace is configured with no obvious local runtime warnings.")

render_badges(
    [
        f"Backend: {settings['model_backend']}",
        f"Chat model: {settings['local_chat_model'] if settings['model_backend'] == 'local' else settings['chat_model']}",
        f"Raw PDFs: {posture['raw_pdf_count']}",
        f"Path cache entries: {cache_stats['entries']}",
        "Neo4j ready" if graph_stats.get("used_neo4j") else "Neo4j pending",
    ]
)

metric_columns = st.columns(5)
metric_columns[0].metric("Corpus Documents", len(overview["documents"]))
metric_columns[1].metric("Indexed Chunks", len(overview["chunks"]))
metric_columns[2].metric("Canonical Entities", len(overview["entities"]))
metric_columns[3].metric("Graph Relations", len(overview["relations"]))

delta_value = evaluation_delta["best_delta"]
delta_label = f"{delta_value:+.3f}" if delta_value is not None else "n/a"
metric_columns[4].metric(
    "Best Mode vs Baseline",
    f"{evaluation_delta['best_score']:.3f}" if evaluation_delta["best_score"] is not None else "n/a",
    delta_label,
)

tab_overview, tab_quality, tab_navigate = st.tabs(["Mission Control", "Quality Posture", "How To Use This App"])

with tab_overview:
    left, right = st.columns((1.15, 0.85), gap="large")
    with left:
        section_title("Corpus Coverage")
        document_frame = overview["document_frame"]
        if not document_frame.empty:
            chart = (
                alt.Chart(document_frame)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#2f7c7d")
                .encode(
                    x=alt.X("document_name:N", sort="-y", title="Regulation"),
                    y=alt.Y("chunks:Q", title="Chunks"),
                    tooltip=["document_name", "chunks", "article_chunks", "page_span"],
                )
                .properties(height=330)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Corpus coverage will populate once graph artifacts are available.")

    with right:
        section_title("Workspace Summary")
        render_card(
            "What this project does",
            "It ingests the EU AI Act, GDPR, and DSA, extracts entities and relations, builds a Neo4j knowledge graph, and answers questions with graph-grounded citations.",
        )
        render_card(
            "Why GraphRAG matters here",
            "Legal questions often depend on linked concepts, article references, and multi-hop reasoning. Hybrid retrieval plus graph traversal performs better than plain similarity search alone.",
        )
        render_card(
            "What this app gives you",
            "A guided home page, analyst chat, corpus explorer, path explorer, operational diagnostics, benchmark evidence, and an in-app project manual for learning and demonstration.",
        )

    lower_left, lower_right = st.columns((0.95, 1.05), gap="large")
    with lower_left:
        section_title("Knowledge Mix")
        entity_frame = overview["entity_frame"].head(10)
        if not entity_frame.empty:
            chart = (
                alt.Chart(entity_frame)
                .mark_arc(innerRadius=64)
                .encode(
                    theta=alt.Theta("count:Q"),
                    color=alt.Color("entity_type:N", legend=alt.Legend(title="Entity Type")),
                    tooltip=["entity_type", "count"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Entity distribution will appear after extraction and graph build.")

    with lower_right:
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
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Relation profile will appear after graph construction.")

with tab_quality:
    summary_left, summary_right = st.columns((0.9, 1.1), gap="large")
    with summary_left:
        section_title("Runtime Posture")
        runtime = posture["runtime"]
        render_card("Configured backend", str(runtime.get("provider", "unknown")))
        render_card("Configured chat model", str(runtime.get("chat_model", "unknown")))
        render_card(
            "Graph load status",
            f"Communities detected: {graph_stats.get('communities_detected', 0)}. Neo4j load: {'ready' if graph_stats.get('used_neo4j') else 'pending'}.",
        )
        if graph_stats.get("notes"):
            st.caption("Latest graph-load notes")
            for note in graph_stats["notes"]:
                st.write(f"- {note}")

    with summary_right:
        section_title("Benchmark Snapshot")
        if not evaluation_aggregate.empty:
            st.dataframe(evaluation_aggregate, use_container_width=True, hide_index=True)
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
        else:
            st.info("No evaluation output is available yet.")

        if evaluation_summary and evaluation_summary.get("regressions"):
            render_warning_banner(evaluation_summary["regressions"])
        elif evaluation_summary:
            render_success_banner("Latest evaluation did not flag a GraphRAG regression against baseline.")

with tab_navigate:
    section_title("App Map")
    nav_left, nav_right = st.columns(2, gap="large")
    with nav_left:
        render_card(
            "Chat",
            "Use the analyst chat page to ask grounded questions, compare hybrid GraphRAG with baseline retrieval, inspect citations, and read the agent trace.",
        )
        render_card(
            "Corpus Explorer",
            "Use the explorer to filter chunks by regulation and article, read the exact source text, and inspect linked entities and relations behind each chunk.",
        )
    with nav_right:
        render_card(
            "Ops",
            "Use the operations page to review ingestion jobs, graph load signals, evaluation results, artifact inventories, and runtime posture.",
        )
        render_card(
            "Path Explorer",
            "Use the path explorer to inspect PathCacheRAG retrieval, view ranked legal paths, watch cache hits and misses, and understand how path-centric evidence is assembled.",
        )
        render_card(
            "Project Guide",
            "Use the guide page for a detailed explanation of what this project is, why it exists, how to operate it, and how to release it responsibly.",
        )

    section_title("Project Brief")
    render_doc_panel(
        """
## Start Here

This workspace is designed to be used in a loop:

1. Check the **Home** page for system posture and benchmark status.
2. Use **Chat** to test grounded legal questions.
3. Use **Corpus Explorer** to inspect the exact source evidence.
4. Use **Path Explorer** to inspect path-centric retrieval and cache behavior.
5. Use **Ops** to monitor ingestion, graph build, evaluation artifacts, and cache posture.
6. Use **Project Guide** to understand the architecture, usage, and operations workflow in detail.
        """
    )
