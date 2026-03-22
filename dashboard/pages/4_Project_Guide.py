from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_ROOT = PROJECT_ROOT / "dashboard"
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from data_access import load_doc
from ui import configure_page, render_badges, render_doc_panel, section_title


configure_page(
    "Project Guide",
    "A built-in reference manual for what this GraphRAG and PathCacheRAG system is, why it exists, how to use it, how to reproduce results, and how to operate it responsibly.",
)

render_badges(
    [
        "Project overview",
        "Architecture",
        "PathCacheRAG spec",
        "User guide",
        "Operations runbook",
        "Release checklist",
        "Reproduce results",
    ]
)

section_title("How to use this guide")
render_doc_panel(
    """
Use this page like an internal handbook:

- **Project Overview** explains the product and its purpose.
- **Architecture** explains how the ingestion, graph, retrieval, and agent layers fit together.
- **PathCacheRAG Spec** explains the branch-level retrieval innovation and benchmark posture.
- **User Guide** explains how to use the dashboard, API, and CLI.
- **Operations Runbook** explains how to run and maintain the system.
- **Release Checklist** explains what to verify before you call the project ready.
- **Reproduce Results** explains how to verify the README numbers and benchmark artifacts yourself.
    """
)

tabs = st.tabs(
    [
        "Project Overview",
        "Architecture",
        "PathCacheRAG Spec",
        "User Guide",
        "Operations Runbook",
        "Release Checklist",
        "Reproduce Results",
    ]
)

doc_specs = [
    ("project_overview.md", tabs[0]),
    ("architecture.md", tabs[1]),
    ("pathcache_rag_spec.md", tabs[2]),
    ("user_guide.md", tabs[3]),
    ("operations_runbook.md", tabs[4]),
    ("release_checklist.md", tabs[5]),
    ("reproduce_results.md", tabs[6]),
]

for doc_name, tab in doc_specs:
    with tab:
        content = load_doc(doc_name)
        if content.strip():
            render_doc_panel(content)
        else:
            st.info(f"{doc_name} is not available.")
