"""
ui_utils.py
-----------
Shared UI helpers for all Streamlit pages.
Import at the top of every page:
    from ui_utils import page_header, inject_css
"""
import html as _html
import json

import streamlit as st
from pathlib import Path

# Repo root (this file lives at <repo>/app/ui_utils.py).
_ROOT = Path(__file__).parent.parent


def init_session_state() -> None:
    """Initialise the shared session-state defaults once per session.

    Single source of truth for every page — call it right after ``inject_css()``
    so deep-linking to any page (not just the home page) sets the same defaults.
    Only fills in keys that are missing, then restores any persisted pipeline
    settings (Leiden resolution, PCA components).
    """
    defaults = {
        "slides": [
            {"slide_id": f"AGED_{i}",  "condition": "AGED",  "run_dir": ""}
            for i in range(1, 5)
        ] + [
            {"slide_id": f"ADULT_{i}", "condition": "ADULT", "run_dir": ""}
            for i in range(1, 5)
        ],
        "base_panel_csv": str(_ROOT / "data" / "Xenium_mBrain_v1_1_metadata.csv"),
        "output_dir"    : str(Path.home() / "xenium_sample_pca_output"),
        "roi_cache_dir" : str(_ROOT / "roi_cache"),
        "panel_mode"    : "partial_union",
        "min_slides"    : 2,
        "roi_polygons"  : {},
        "leiden_resolution"            : 0.6,
        "n_pcs"                        : 50,
        "optimizer_results"            : None,
        "optimizer_best"               : None,
        "optimizer_best_row"           : None,
        "optimizer_cluster_assignments": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    _restore_persisted_settings()


def _restore_persisted_settings() -> None:
    """Restore the Leiden resolution and PCA component count the optimizer last
    applied, so a chosen value survives an app restart instead of reverting to
    the defaults. Runs once per session."""
    if st.session_state.get("_settings_restored"):
        return
    st.session_state["_settings_restored"] = True
    settings_path = (Path(st.session_state["output_dir"])
                     / "leiden_optimizer" / "pipeline_settings.json")
    if not settings_path.exists():
        return
    try:
        saved = json.loads(settings_path.read_text())
        if "leiden_resolution" in saved:
            st.session_state["leiden_resolution"] = float(saved["leiden_resolution"])
        if "n_pcs" in saved:
            st.session_state["n_pcs"] = int(saved["n_pcs"])
    except Exception:
        pass


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
