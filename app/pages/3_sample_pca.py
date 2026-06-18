"""
pages/3_sample_pca.py
Sample PCA — pseudobulk PCA across the selected samples.

Loads the configured Xenium slides, applies the saved MBH ROIs, collapses
each slide into a pseudobulk profile, and runs PCA across the samples.
Shows how individual samples cluster and how the AGED / ADULT groups
separate, with a Nature-style figure that can be downloaded as PDF.
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header, init_session_state

st.set_page_config(page_title="Sample PCA · Xenium Sample PCA", page_icon="📊", layout="wide",
    initial_sidebar_state="expanded")

inject_css()
init_session_state()
# Make the xenium_spatial package importable when the app is run without an
# editable install (src layout: the package lives under <repo>/src).
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_dir(p: str) -> bool:
    return bool(p) and (Path(p) / "cell_feature_matrix" / "matrix.mtx.gz").exists()


def _roi_signature(slide_ids, roi_dir) -> tuple:
    """Signature that changes when any ROI file changes — invalidates the cache."""
    sig = []
    for sid in slide_ids:
        p = Path(roi_dir) / f"{sid.replace('/', '_').replace(' ', '_')}_roi.json"
        sig.append((sid, p.stat().st_mtime if p.exists() else 0.0))
    return tuple(sig)


@st.cache_resource(show_spinner=False)
def _load_combined(run_dirs, slide_ids, conditions, batches, base_csv,
                   roi_dir, use_roi, panel_mode, min_slides, roi_sig, output_dir=None):
    """Load + harmonise + ROI-filter + concatenate all slides (cached).

    ``roi_sig`` is the per-slide ROI-file signature (see ``_roi_signature``).
    It must NOT be underscore-prefixed: Streamlit skips hashing any argument
    whose name starts with ``_``, which would exclude it from the cache key
    and leave the cache stale after an ROI is edited. Keeping it hashed is
    what invalidates the cached load when a saved ROI changes.
    """
    from xenium_spatial.multislide_loader import SlideManifest, MultiSlideLoader
    from xenium_spatial.panel_registry import PanelRegistry
    from xenium_spatial.roi_selector import ROISelector

    manifest = SlideManifest()
    for sid, cond, d, b in zip(slide_ids, conditions, run_dirs, batches):
        manifest.add(slide_id=sid, condition=cond, run_dir=d, replicate_id=sid,
                     batch=b or sid)

    registry = PanelRegistry(base_csv)
    roi_selector = ROISelector(cache_dir=roi_dir) if use_roi else None

    loader = MultiSlideLoader(
        manifest=manifest, panel_registry=registry, roi_selector=roi_selector,
        panel_mode=panel_mode, min_slides=min_slides, apply_roi=use_roi,
        output_dir=output_dir,
    )
    return loader.load_all()


# ── Page ──────────────────────────────────────────────────────────────────────
page_header("📊 Sample PCA", "Pseudobulk PCA across the samples — group separation & outliers")

slides = st.session_state.get("slides", [])
valid_slides = [s for s in slides if _valid_dir(s.get("run_dir", ""))]
slide_ids = [s["slide_id"] for s in valid_slides]
n_roi = sum(1 for sid in slide_ids if sid in st.session_state["roi_polygons"]
            or (Path(st.session_state["roi_cache_dir"])
                / f"{sid.replace('/', '_').replace(' ', '_')}_roi.json").exists())

if len(valid_slides) < 2:
    st.warning("Need at least 2 valid slides. Configure them in **📁 Study Setup**.")
    st.stop()

n_cond = len({s["condition"] for s in valid_slides})
st.markdown(
    f"**{len(valid_slides)}** valid slides across **{n_cond}** group(s) · "
    f"**{n_roi}/{len(valid_slides)}** ROIs saved."
)

# ── Sample selection ────────────────────────────────────────────────────────
# Choose how many / which samples to include. The PCA runs on whatever is
# selected here (minimum 2), so you can compare just two samples or all of them.
selected_ids = st.multiselect(
    "Samples to include in the PCA",
    options=slide_ids,
    default=slide_ids,
    help="Pick the samples to analyse. At least 2 are required; the rest are "
         "ignored for this run.",
)
selected_slides = [s for s in valid_slides if s["slide_id"] in set(selected_ids)]

if len(selected_slides) < 2:
    st.warning("Select at least 2 samples to run the PCA.")
    st.stop()

n_cond_sel = len({s["condition"] for s in selected_slides})
st.caption(
    f"Selected **{len(selected_slides)}** sample(s) across **{n_cond_sel}** group(s)."
)

# ── Options ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    use_roi = st.toggle("Apply MBH ROIs", value=(n_roi > 0),
                        help="Restrict each slide to its saved ROI before pseudobulk.")
    if use_roi and n_roi < len(selected_slides):
        st.caption(f"⚠️ Only {n_roi}/{len(valid_slides)} ROIs saved — slides without one use the whole section.")
with c2:
    base_panel_only = st.toggle("Base panel only", value=True,
                                help="Restrict the PCA to the shared Xenium base panel "
                                     "(247 genes), dropping per-slide add-on genes so all "
                                     "samples are compared on the same gene set.")
with c3:
    n_top_genes = st.number_input("Top variable genes (0 = all)", min_value=0, max_value=5000,
                                  value=0, step=50,
                                  help="Restrict PCA to the N most variable genes. 0 uses all "
                                       "genes (recommended for targeted Xenium panels).")
with c4:
    scale_genes = st.toggle("Z-score genes", value=False,
                            help="Standardise each gene before PCA. Off by default "
                                 "(log1p already stabilises variance).")

out_root = Path(st.session_state["output_dir"]) / "sample_pca"


def _clear_outputs(out_root: Path) -> None:
    """Delete the previous run's figures and tables.

    The Results section below renders whatever figure files exist on disk.
    Without clearing them first, a failed run would leave the *previous*
    run's figures on display as if they were current. Clearing up front means
    the Results section always reflects the latest attempt: fresh on success,
    empty on failure (with the error shown).
    """
    figures = ["sample_pca_scatter", "sample_correlation_heatmap", "sample_pca_scree"]
    tables = ["sample_pca_coordinates.csv", "sample_pca_variance.csv",
              "pseudobulk_samples.h5ad"]
    for base in figures:
        for ext in ("png", "pdf", "svg"):
            (out_root / f"{base}.{ext}").unlink(missing_ok=True)
    for fname in tables:
        (out_root / fname).unlink(missing_ok=True)


run = st.button("▶ Run sample PCA", type="primary", use_container_width=True)

if run:
    _clear_outputs(out_root)
    try:
        with st.spinner("Loading slides, applying ROIs, and pseudobulking …"):
            run_dirs   = tuple(str(s["run_dir"]) for s in selected_slides)
            sids       = tuple(s["slide_id"] for s in selected_slides)
            conditions = tuple(s["condition"] for s in selected_slides)
            batches    = tuple((s.get("batch") or s["slide_id"]) for s in selected_slides)
            roi_sig    = _roi_signature(sids, st.session_state["roi_cache_dir"])

            adata = _load_combined(
                run_dirs, sids, conditions, batches,
                st.session_state["base_panel_csv"],
                st.session_state["roi_cache_dir"], use_roi,
                st.session_state["panel_mode"], int(st.session_state["min_slides"]),
                roi_sig, st.session_state["output_dir"],
            )

        with st.spinner("Running PCA and rendering figures …"):
            from xenium_spatial.sample_pca import (
                sample_level_pca_analysis,
                plot_sample_pca, plot_sample_correlation, plot_scree,
            )
            out_root.mkdir(parents=True, exist_ok=True)
            pb = sample_level_pca_analysis(
                adata, output_dir=out_root,
                sample_key="replicate", condition_key="condition",
                n_top_genes=int(n_top_genes), scale_genes=bool(scale_genes),
                base_panel_only=bool(base_panel_only),
                fmt="pdf",
            )
            # Also render PNGs for inline display.
            plot_sample_pca(pb, output_dir=out_root, fmt="png")
            plot_sample_correlation(pb, output_dir=out_root, fmt="png")
            plot_scree(pb, output_dir=out_root, fmt="png")

        st.session_state["pca_ran"] = True
        st.success(f"Done — {pb.n_obs} samples × {pb.n_vars} genes. Outputs in `{out_root}`.")
    except Exception as e:
        st.session_state["pca_ran"] = False
        logging.getLogger("xenium_app").exception("Sample PCA failed")
        st.error(f"Sample PCA failed: {e}")
        st.exception(e)

# ── Results ───────────────────────────────────────────────────────────────────
scatter_png = out_root / "sample_pca_scatter.png"
if scatter_png.exists():
    st.divider()
    st.subheader("Results")

    left, right = st.columns([3, 2])
    with left:
        st.image(str(scatter_png), caption="Sample-level PCA (pseudobulk)", use_container_width=True)
        _batches = [(s.get("batch") or s["slide_id"]) for s in selected_slides]
        if len(set(_batches)) > 1 and set(_batches) != {s["slide_id"] for s in selected_slides}:
            st.caption(
                "Marker **shape** = batch, **colour** = condition. If samples group "
                "by shape rather than colour, a technical batch — not the condition — "
                "is driving the separation."
            )
    with right:
        scree_png = out_root / "sample_pca_scree.png"
        if scree_png.exists():
            st.image(str(scree_png), caption="Variance explained", use_container_width=True)

    corr_png = out_root / "sample_correlation_heatmap.png"
    if corr_png.exists():
        st.image(str(corr_png), caption="Sample correlation (hierarchically ordered)",
                 use_container_width=False, width=480)

    # Coordinate table
    coord_csv = out_root / "sample_pca_coordinates.csv"
    if coord_csv.exists():
        st.markdown("**PC coordinates**")
        st.dataframe(pd.read_csv(coord_csv, index_col=0), use_container_width=True)

    # Downloads (Nature-style PDFs + tables)
    st.markdown("**Downloads**")
    dcols = st.columns(4)
    downloads = [
        ("sample_pca_scatter.pdf",         "⬇️ PCA scatter (PDF)",    "application/pdf"),
        ("sample_correlation_heatmap.pdf", "⬇️ Correlation (PDF)",    "application/pdf"),
        ("sample_pca_scree.pdf",           "⬇️ Scree (PDF)",          "application/pdf"),
        ("sample_pca_coordinates.csv",     "⬇️ Coordinates (CSV)",    "text/csv"),
    ]
    for col, (fname, label, mime) in zip(dcols, downloads):
        fpath = out_root / fname
        if fpath.exists():
            with col:
                st.download_button(label, data=fpath.read_bytes(),
                                   file_name=fname, mime=mime, use_container_width=True)
else:
    st.info("Configure slides and ROIs, then click **Run sample PCA**.")
