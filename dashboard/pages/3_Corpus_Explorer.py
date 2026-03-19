from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_ROOT = PROJECT_ROOT / "dashboard"
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from data_access import available_article_refs, chunk_detail, filtered_chunk_frame, regulation_names
from ui import configure_page, render_card, render_doc_panel, section_title


configure_page(
    "Corpus Explorer",
    "Browse the ingested EU corpus by regulation and article, inspect exact source chunks, and trace the entities and relations connected to each evidence block.",
)

with st.sidebar:
    st.subheader("Filters")
    regulation = st.selectbox("Regulation", regulation_names())
    article_ref = st.selectbox("Article reference", available_article_refs(regulation))
    article_only = st.checkbox("Only show article-tagged chunks", value=False)
    search_text = st.text_input("Keyword or phrase", placeholder="high-risk, transparency, personal data ...")

frame = filtered_chunk_frame(
    regulation=regulation,
    article_ref=article_ref,
    query=search_text,
    article_only=article_only,
)

summary_columns = st.columns(4)
summary_columns[0].metric("Filtered chunks", len(frame))
summary_columns[1].metric("Regulation", regulation)
summary_columns[2].metric("Article filter", article_ref)
summary_columns[3].metric("Search active", "Yes" if search_text.strip() else "No")

section_title("Chunk browser")
if frame.empty:
    st.info("No chunks matched the current filters.")
else:
    st.dataframe(
        frame[["document_name", "article_ref", "section_title", "page_start", "page_end", "preview"]],
        use_container_width=True,
        hide_index=True,
    )

    options = frame["label"].tolist()
    default_index = 0
    selected_label = st.selectbox("Inspect a chunk", options, index=default_index)
    selected_row = frame.loc[frame["label"] == selected_label].iloc[0]
    detail = chunk_detail(str(selected_row["chunk_id"]))

    detail_left, detail_right = st.columns((1.2, 0.8), gap="large")
    with detail_left:
        section_title("Selected source chunk")
        chunk = detail["chunk"]
        render_doc_panel(
            f"""
### {selected_row['document_name']}

- **Article**: {chunk.get('article_ref') or 'None'}
- **Pages**: {chunk.get('page_start') or '?'} - {chunk.get('page_end') or '?'}
- **Chunk ID**: `{chunk.get('chunk_id', '')}`
- **Section**: {chunk.get('metadata', {}).get('section_title', 'Unknown')}

{chunk.get('text', 'No text available')}
            """
        )

    with detail_right:
        section_title("Interpretation aids")
        render_card(
            "How to use this panel",
            "Read the exact chunk first, then inspect the linked entities and relations to understand why the retriever or graph traversal may select it during question answering.",
        )
        render_card(
            "Best debugging pattern",
            "If a chat answer looks wrong, come here, filter by regulation and article, and confirm whether the system indexed the correct source text and article reference.",
        )

    lower_left, lower_right = st.columns(2, gap="large")
    with lower_left:
        section_title("Mentioned entities")
        entities = detail["entities"]
        if entities:
            entity_rows = []
            for entity in entities:
                entity_rows.append(
                    {
                        "canonical_name": entity.get("canonical_name", ""),
                        "entity_type": entity.get("entity_type", ""),
                        "confidence": entity.get("confidence", 0.0),
                        "aliases": ", ".join(entity.get("aliases", [])[:4]),
                    }
                )
            st.dataframe(pd.DataFrame(entity_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No linked entities were found for this chunk.")

    with lower_right:
        section_title("Related relations")
        relations = detail["relations"]
        if relations:
            relation_rows = []
            for relation in relations:
                relation_rows.append(
                    {
                        "relation_type": relation.get("relation_type", ""),
                        "subject_entity_id": relation.get("subject_entity_id", ""),
                        "object_entity_id": relation.get("object_entity_id", ""),
                        "confidence": relation.get("confidence", 0.0),
                    }
                )
            st.dataframe(pd.DataFrame(relation_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No relations were linked to this chunk.")
