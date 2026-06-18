"""
ui_utils.py
-----------
Shared UI helpers for all Streamlit pages.
Import at the top of every page:
    from ui_utils import page_header, inject_css
"""
import html as _html
import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

import streamlit as st
from pathlib import Path

# Repo root (this file lives at <repo>/app/ui_utils.py).
_ROOT = Path(__file__).parent.parent


def applied_n_pcs(output_dir, default: int = 50) -> int:
    """The PCA-component count the Leiden Optimizer last *applied*, read from the
    persisted ``pipeline_settings.json``.

    Use this — not ``st.session_state['n_pcs']`` — anywhere outside the optimizer
    page. ``n_pcs`` is the optimizer number_input's widget key, and Streamlit
    drops widget-keyed session state on pages that don't render that widget, so
    the session value silently falls back to the default. The settings file is
    the stable source of truth.
    """
    import json
    p = Path(output_dir) / "leiden_optimizer" / "pipeline_settings.json"
    if p.exists():
        try:
            return int(json.loads(p.read_text()).get("n_pcs", default))
        except Exception:
            return default
    return default


# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = _ROOT / "logs"
LOG_FILE = LOG_DIR / "xenium_app.log"
_LOG_HANDLER_TAG = "_xenium_app_file"
# Third-party loggers that flood the file at DEBUG with little debugging value.
_NOISY_LOGGERS = ("matplotlib", "numba", "PIL", "fontTools", "h5py", "harmonypy")


def get_log_file() -> Path:
    """Path to the app's debug log file."""
    return LOG_FILE


def init_logging(level: int = logging.INFO) -> Path:
    """Attach a rotating file handler so the app and the ``xenium_spatial``
    package write debug output to ``<repo>/logs/xenium_app.log``.

    Idempotent: the handler is added once per process (tagged so repeated
    Streamlit reruns don't stack duplicates); subsequent calls just update the
    level. Returns the log file path.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    handler = next((h for h in root.handlers
                    if getattr(h, _LOG_HANDLER_TAG, False)), None)
    if handler is None:
        handler = RotatingFileHandler(
            LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        setattr(handler, _LOG_HANDLER_TAG, True)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(handler)
        logging.captureWarnings(True)  # route warnings.warn(...) into the log
        logging.getLogger("xenium_app").info("── logging initialised → %s", LOG_FILE)
    root.setLevel(level)
    handler.setLevel(level)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
    return LOG_FILE


def log_panel() -> None:
    """Render a compact debug-log panel: file location, level selector,
    download / clear buttons and a tail preview. Drop on any page."""
    log_file = get_log_file()
    with st.expander("🪵 Debug log"):
        st.caption(f"Log file: `{log_file}`")
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        cur = st.session_state.get("log_level", "INFO")
        new_level = st.selectbox(
            "Verbosity", levels, index=levels.index(cur) if cur in levels else 1,
            help="DEBUG captures the most detail. Applies to messages logged from now on.")
        if new_level != cur:
            st.session_state["log_level"] = new_level
            init_logging(logging.getLevelName(new_level))

        if log_file.exists() and log_file.stat().st_size:
            data = log_file.read_bytes()
            st.caption(f"{len(data):,} bytes · updated "
                       f"{datetime.fromtimestamp(log_file.stat().st_mtime):%Y-%m-%d %H:%M:%S}")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ Download log", data=data, file_name="xenium_app.log",
                                   mime="text/plain", use_container_width=True)
            with c2:
                if st.button("🗑 Clear log", use_container_width=True):
                    log_file.write_text("")
                    logging.getLogger("xenium_app").info("log cleared from UI")
                    st.rerun()
            tail = data.decode("utf-8", "replace").splitlines()[-40:]
            st.code("\n".join(tail) or "(empty)", language="log")
        else:
            st.info("Log is empty — run a step (Sample PCA / Leiden) to populate it.")


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
        "roi_last_slide": None,
        "leiden_resolution"            : 0.6,
        "n_pcs"                        : 50,
        "optimizer_results"            : None,
        "optimizer_best"               : None,
        "optimizer_best_row"           : None,
        "optimizer_cluster_assignments": None,
        "log_level"                    : "INFO",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    init_logging(logging.getLevelName(st.session_state["log_level"]))
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
            # Clamp to the optimizer widget's range — a stale/edited settings file
            # with n_pcs outside [2, 200] would otherwise crash the page when the
            # number_input is seeded from session_state.
            st.session_state["n_pcs"] = max(2, min(200, int(saved["n_pcs"])))
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


def _is_under(child, parent) -> bool:
    """True if ``child`` resolves to a location inside ``parent``."""
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except (ValueError, OSError, TypeError):
        return False


def paths_panel() -> None:
    """Show where the app is running from and where each configured path points,
    flagging any that live in a *different* checkout of this repo — the usual
    cause of "my edit didn't apply" / stale ROIs when several copies coexist.
    Offers a one-click reset of the repo-relative paths to this checkout.
    """
    repo = _ROOT
    # (label, session key, expected inside this repo?)
    specs = [
        ("Base panel CSV", "base_panel_csv", True),
        ("ROI cache",      "roi_cache_dir",  True),
        ("Output dir",     "output_dir",     False),  # intentionally under $HOME
    ]
    with st.expander("🗂 Paths & environment"):
        st.caption(f"App running from: `{repo}`")
        mismatched = []
        for label, key, want_inside in specs:
            raw = st.session_state.get(key)
            if not raw:  # None / "" from a malformed loaded config
                st.markdown(f"**{label}** ❌ not set")
                continue
            p = Path(str(raw))
            exists = p.exists()
            outside = want_inside and not _is_under(p, repo)
            if outside:
                mismatched.append(label)
            status = "✅" if exists else "❌ missing"
            warn = " · ⚠️ **outside this checkout**" if outside else ""
            st.markdown(f"**{label}** {status}{warn}  \n`{p}`")

        if mismatched:
            st.warning(
                f"{', '.join(mismatched)} point outside `{repo}` — likely a second "
                "copy of the project. Editing one checkout while the app reads "
                "another leads to stale ROIs and 'my fix didn't apply'. Reset the "
                "repo-relative paths to this checkout, or fix them in Study Setup."
            )
            if st.button("Reset paths to this checkout"):
                st.session_state["base_panel_csv"] = str(
                    repo / "data" / "Xenium_mBrain_v1_1_metadata.csv")
                st.session_state["roi_cache_dir"] = str(repo / "roi_cache")
                logging.getLogger("xenium_app").info(
                    "Reset base_panel_csv/roi_cache_dir to %s", repo)
                st.rerun()


def page_header(title: str, subtitle: str = ""):
    """Render the standard dark gradient page header."""
    safe_title = _html.escape(title)
    safe_sub = _html.escape(subtitle)
    sub_html = f"<p>{safe_sub}</p>" if subtitle else ""
    st.markdown(
        f'<div class="page-header"><h1>{safe_title}</h1>{sub_html}</div>',
        unsafe_allow_html=True,
    )
