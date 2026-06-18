"""
ui_utils.py
-----------
Shared UI helpers for all Streamlit pages.
Import at the top of every page:
    from ui_utils import page_header, inject_css
"""
import html as _html

import streamlit as st
from pathlib import Path


def inject_css():
    """Re-inject the global CSS on sub-pages (Streamlit reloads CSS per page)."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    # Also import Google Fonts (inline so it works without the external CSS file)
    st.markdown(
        "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?"
        "family=IBM+Plex+Sans:wght@300;400;500;600"
        "&family=IBM+Plex+Mono:wght@400;500&display=swap'>",
        unsafe_allow_html=True,
    )


def prune_orphan_rois() -> int:
    """Drop in-memory ``roi_polygons`` entries whose slide ID is no longer in
    the configured study, so counts can't exceed the number of slides.

    Only the in-session dict is cleaned — the persistent ``roi_cache`` JSON
    files are left untouched, so a slide that is removed and re-added later
    still reloads its saved ROI. Returns the number of entries removed.
    """
    polygons = st.session_state.get("roi_polygons")
    if not polygons:
        return 0
    slide_ids = {s["slide_id"] for s in st.session_state.get("slides", [])}
    orphans = [sid for sid in polygons if sid not in slide_ids]
    for sid in orphans:
        del polygons[sid]
    return len(orphans)


def page_header(title: str, subtitle: str = ""):
    """Render the standard dark gradient page header."""
    safe_title = _html.escape(title)
    safe_sub = _html.escape(subtitle)
    sub_html = f"<p>{safe_sub}</p>" if subtitle else ""
    st.markdown(
        f'<div class="page-header"><h1>{safe_title}</h1>{sub_html}</div>',
        unsafe_allow_html=True,
    )
