from __future__ import annotations

from typing import Iterable

import streamlit as st

THEME_CSS = """
<style>
    :root {
        --ink: #1d2a39;
        --muted: #5c6b78;
        --accent: #b55a3c;
        --teal: #2f7c7d;
        --card: rgba(255, 250, 243, 0.86);
        --border: rgba(29, 42, 57, 0.12);
        --shadow: 0 18px 42px rgba(29, 42, 57, 0.08);
    }
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(247, 214, 180, 0.55), transparent 28%),
            radial-gradient(circle at 85% 8%, rgba(142, 203, 205, 0.30), transparent 24%),
            linear-gradient(180deg, #f8f3ea 0%, #f4efe7 52%, #eef3f6 100%);
        color: var(--ink);
    }
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.8rem;
        max-width: 1320px;
    }
    h1, h2, h3 {
        font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
        color: var(--ink);
        letter-spacing: -0.03em;
    }
    .stMarkdown, .stCaption, .stTextArea label, .stSelectbox label, .stSlider label, .stTabs [data-baseweb="tab"], .stCheckbox label {
        font-family: "Segoe UI Variable Text", "Trebuchet MS", sans-serif;
    }
    [data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow);
    }
    .hero-shell {
        background: linear-gradient(135deg, rgba(255, 250, 243, 0.94), rgba(255, 240, 224, 0.92));
        border: 1px solid rgba(181, 90, 60, 0.16);
        border-radius: 28px;
        padding: 1.6rem 1.8rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.2rem;
    }
    .hero-kicker {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--teal);
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2.35rem;
        line-height: 1.05;
        margin: 0 0 0.45rem 0;
        font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
        color: var(--ink);
    }
    .hero-copy {
        font-size: 1rem;
        line-height: 1.58;
        color: var(--muted);
        max-width: 780px;
        margin: 0;
    }
    .section-title {
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--teal);
        margin: 1.4rem 0 0.75rem;
    }
    .app-card {
        background: rgba(255, 250, 243, 0.88);
        border: 1px solid rgba(29, 42, 57, 0.10);
        border-radius: 22px;
        padding: 1.05rem 1.1rem;
        box-shadow: var(--shadow);
        margin-bottom: 0.95rem;
    }
    .app-card h4 {
        margin: 0 0 0.35rem 0;
        font-size: 1.05rem;
        color: var(--ink);
    }
    .app-card p {
        margin: 0;
        color: var(--muted);
        line-height: 1.55;
        font-size: 0.95rem;
    }
    .callout-warn {
        background: rgba(255, 247, 230, 0.92);
        border-left: 4px solid #c98b2c;
        border-radius: 18px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.85rem;
        color: #704a14;
    }
    .callout-good {
        background: rgba(236, 249, 244, 0.92);
        border-left: 4px solid #2f7c7d;
        border-radius: 18px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.85rem;
        color: #1f5a52;
    }
    .citation-card {
        background: rgba(255, 250, 243, 0.88);
        border: 1px solid rgba(29, 42, 57, 0.10);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        box-shadow: var(--shadow);
        margin-bottom: 0.9rem;
    }
    .citation-meta {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        color: var(--muted);
        font-size: 0.85rem;
        margin-bottom: 0.55rem;
    }
    .answer-shell {
        background: linear-gradient(135deg, rgba(47, 124, 125, 0.12), rgba(255, 250, 243, 0.92));
        border: 1px solid rgba(47, 124, 125, 0.18);
        border-radius: 24px;
        padding: 1.15rem 1.2rem;
        box-shadow: var(--shadow);
    }
    .badge-row {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin: 0.9rem 0 0.2rem;
    }
    .badge-pill {
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        background: rgba(47, 124, 125, 0.1);
        border: 1px solid rgba(47, 124, 125, 0.12);
        color: var(--ink);
        font-size: 0.82rem;
    }
    .doc-shell {
        background: rgba(255, 250, 243, 0.88);
        border-radius: 24px;
        border: 1px solid rgba(29, 42, 57, 0.10);
        box-shadow: var(--shadow);
        padding: 1.1rem 1.2rem;
    }
</style>
"""


def configure_page(title: str, subtitle: str) -> None:
    st.set_page_config(page_title=title, layout="wide")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-kicker">GraphRAG Engine</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-copy">{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_title(label: str) -> None:
    st.markdown(f'<div class="section-title">{label}</div>', unsafe_allow_html=True)


def render_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="app-card">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_badges(items: Iterable[str]) -> None:
    rendered = "".join(f'<span class="badge-pill">{item}</span>' for item in items)
    st.markdown(f'<div class="badge-row">{rendered}</div>', unsafe_allow_html=True)


def render_warning_banner(messages: list[str]) -> None:
    for message in messages:
        st.markdown(f'<div class="callout-warn">{message}</div>', unsafe_allow_html=True)


def render_success_banner(message: str) -> None:
    st.markdown(f'<div class="callout-good">{message}</div>', unsafe_allow_html=True)


def render_doc_panel(markdown_text: str) -> None:
    st.markdown('<div class="doc-shell">', unsafe_allow_html=True)
    st.markdown(markdown_text)
    st.markdown("</div>", unsafe_allow_html=True)


def citation_card(citation, *, index: int) -> None:
    scores = citation.score_breakdown
    st.markdown(
        f"""
        <div class="citation-card">
            <div class="citation-meta">
                <strong>{index}. {citation.document_name}</strong>
                <span>{citation.article_ref or 'No article reference'}</span>
                <span>Pages {citation.page_start or '?'}-{citation.page_end or '?'}</span>
                <span>Fused score {scores.get('fused', 0.0):.4f}</span>
            </div>
            <div>{citation.snippet}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
