"""
pages/5_clusters.py
Cell clusters — UMAP + marker-based annotation.

Builds the final cell-level clustering at the applied Leiden resolution (UMAP +
Leiden on the same Harmony embedding the optimizer scored), then lets you
explore it as a UMAP, read per-cluster marker genes, and assign cell-type
labels. The clustered AnnData is persisted to ``<output_dir>/clustering/`` and
reused by the downstream quantification steps.
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header, init_session_state, applied_n_pcs

st.set_page_config(page_title="Clusters · Xenium Spatial Pipeline", page_icon="🔬", layout="wide",
    initial_sidebar_state="expanded")
inject_css()
init_session_state()

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import pipeline  # noqa: E402  (app/pipeline.py — shared cached embedding loader)
from xenium_spatial.cell_clustering import (  # noqa: E402
    build_clustered_adata, rank_markers, top_markers_by_cluster, cluster_composition,
    clustered_h5ad_path, load_annotations, save_annotations, annotations_path,
)

logger = logging.getLogger("xenium_app")


# ── Helpers ────────────────────────────────────────────────────────────────
def _valid_dir(p: str) -> bool:
    return bool(p) and (Path(p) / "cell_feature_matrix" / "matrix.mtx.gz").exists()


@st.cache_data(show_spinner=False)
def _markers_table(path: str, mtime: float, n_genes: int) -> pd.DataFrame:
    adata = pipeline.load_clustered(path, mtime)
    return rank_markers(adata, groupby="leiden", n_genes=n_genes)


# ── Page ───────────────────────────────────────────────────────────────────
page_header("🔬 Cell clusters", "UMAP + marker-based annotation of the MBH cells")

slides = st.session_state.get("slides", [])
valid_slides = [s for s in slides if _valid_dir(s.get("run_dir", ""))]
slide_ids = [s["slide_id"] for s in valid_slides]
if len(valid_slides) < 1:
    st.warning("Need at least one valid slide. Configure them in **📁 Study Setup**.")
    st.stop()

roi_cache_dir = st.session_state["roi_cache_dir"]
n_roi = sum(1 for sid in slide_ids if sid in st.session_state["roi_polygons"]
            or (Path(roi_cache_dir)
                / f"{sid.replace('/', '_').replace(' ', '_')}_roi.json").exists())

resolution = float(st.session_state.get("leiden_resolution", 0.6))
out_dir = st.session_state["output_dir"]
h5ad_path = clustered_h5ad_path(out_dir)

st.markdown(
    f"Builds clusters at the applied resolution **{resolution:.2f}** "
    "(set in the 🔎 Leiden Optimizer). Match the embedding to what you swept, then build."
)

# ── Embedding settings (match the optimizer's run) ──────────────────────────
with st.expander("⚙️ Embedding settings", expanded=not h5ad_path.exists()):
    c1, c2, c3 = st.columns(3)
    with c1:
        use_roi = st.toggle("Apply MBH ROIs", value=(n_roi > 0), key="cl_use_roi")
        base_panel_only = st.toggle("Base panel only", value=False, key="cl_base_only",
                                    help="Off keeps the per-slide add-on genes. When the "
                                         "custom panel is shared across all slides (no "
                                         "comparability problem), keeping it is recommended.")
    with c2:
        use_harmony = st.toggle("Harmony batch correction", value=(len(valid_slides) > 1),
                                key="cl_harmony",
                                help="Integrate on each slide's batch label (set in Study Setup).")
        n_neighbors = st.number_input("KNN neighbours", min_value=2, max_value=100, value=15,
                                      step=1, key="cl_knn")
    with c3:
        if "cl_n_pcs" not in st.session_state:
            st.session_state["cl_n_pcs"] = max(2, min(200, applied_n_pcs(out_dir)))
        n_pcs = st.number_input("PCA components", min_value=2, max_value=200, step=1,
                                key="cl_n_pcs",
                                help="Defaults to the value applied in the Leiden Optimizer. "
                                     "Use the elbow recommendation (~16 here) for a cleaner "
                                     "embedding than the 50-PC default.")
        scale_genes = st.toggle("Z-score genes before PCA", value=False, key="cl_scale")

    batch_key = "batch" if (use_harmony and len(valid_slides) > 1) else None
    build = st.button(f"🧬 Build clustering ({n_pcs} PCs · resolution {resolution:.2f})",
                      type="primary", use_container_width=True)

if build:
    try:
        with st.spinner("Loading slides, building the embedding, UMAP and clusters …"):
            run_dirs   = tuple(str(s["run_dir"]) for s in valid_slides)
            sids       = tuple(s["slide_id"] for s in valid_slides)
            conditions = tuple(s["condition"] for s in valid_slides)
            batches    = tuple((s.get("batch") or s["slide_id"]) for s in valid_slides)
            roi_sig    = pipeline.roi_signature(sids, roi_cache_dir)

            adata = pipeline.load_embedding(
                run_dirs, sids, conditions, batches,
                st.session_state["base_panel_csv"], roi_cache_dir, use_roi,
                st.session_state["panel_mode"], int(st.session_state["min_slides"]),
                roi_sig, bool(base_panel_only), int(n_pcs), int(n_neighbors),
                bool(scale_genes), batch_key, out_dir,
            )
            adata = build_clustered_adata(adata, resolution)
            adata.write_h5ad(h5ad_path)
        st.success(f"Built {adata.n_obs:,} cells · "
                   f"{adata.obs['leiden'].nunique()} clusters → saved to `{h5ad_path}`.")
        st.rerun()
    except Exception as e:
        logger.exception("Cluster build failed")
        st.error(f"Build failed: {e}")
        st.exception(e)

if not h5ad_path.exists():
    st.info("No clustering yet — set the embedding options above and click **Build clustering**.")
    st.stop()

# ── Load the persisted clustering ───────────────────────────────────────────
adata = pipeline.load_clustered(str(h5ad_path), h5ad_path.stat().st_mtime)
clusters = sorted(adata.obs["leiden"].astype(str).unique(), key=lambda x: int(x))
gene_list = list(adata.var_names)

m1, m2, m3 = st.columns(3)
m1.metric("Cells", f"{adata.n_obs:,}")
m2.metric("Clusters", f"{len(clusters)}")
m3.metric("Resolution", f"{resolution:.2f}")

# ── UMAP ────────────────────────────────────────────────────────────────────
st.subheader("UMAP")
if "X_umap" not in adata.obsm:
    st.error("No UMAP in the saved object — rebuild the clustering.")
    st.stop()

colour_by = st.radio("Colour by", ["Cluster", "Condition", "Batch", "Gene"],
                     horizontal=True, key="umap_colour")
sel_gene = None
if colour_by == "Gene":
    sel_gene = st.selectbox("Gene", gene_list, key="umap_gene")

# Subsample for a responsive scatter (display only).
MAX_PTS = 30_000
idx = np.arange(adata.n_obs)
if adata.n_obs > MAX_PTS:
    idx = np.random.default_rng(42).choice(idx, MAX_PTS, replace=False)

um = adata.obsm["X_umap"][idx]
df = pd.DataFrame({"UMAP1": um[:, 0], "UMAP2": um[:, 1]})
df["Cluster"] = adata.obs["leiden"].astype(str).values[idx]
if "condition" in adata.obs:
    df["Condition"] = adata.obs["condition"].astype(str).values[idx]
if "batch" in adata.obs:
    df["Batch"] = adata.obs["batch"].astype(str).values[idx]

import plotly.express as px  # noqa: E402

if colour_by == "Gene" and sel_gene is not None:
    layer = "lognorm" if "lognorm" in adata.layers else None
    expr = np.asarray(adata.obs_vector(sel_gene, layer=layer)).ravel()
    df[sel_gene] = expr[idx]
    fig = px.scatter(df, x="UMAP1", y="UMAP2", color=sel_gene,
                     color_continuous_scale="Viridis", render_mode="webgl",
                     title=f"{sel_gene} (log-normalised)")
else:
    key = colour_by if colour_by in df.columns else "Cluster"
    fig = px.scatter(df, x="UMAP1", y="UMAP2", color=key, render_mode="webgl",
                     category_orders={"Cluster": clusters})
fig.update_traces(marker=dict(size=3, opacity=0.7))
fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10),
                  legend=dict(itemsizing="constant"))
st.plotly_chart(fig, use_container_width=True)
st.caption("UMAP is for visualisation only — quantify on the cluster labels, not on UMAP distances. "
           f"Showing {len(df):,} of {adata.n_obs:,} cells.")

# ── Markers ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Marker genes")
st.caption("Wilcoxon rank-genes-groups per cluster (log-normalised expression).")
if st.button("Compute marker genes", key="compute_markers"):
    st.session_state["_markers_ready"] = True

markers = None
if st.session_state.get("_markers_ready"):
    try:
        with st.spinner("Ranking genes per cluster …"):
            markers = _markers_table(str(h5ad_path), h5ad_path.stat().st_mtime, 25)
    except Exception as e:
        logger.exception("Marker computation failed")
        st.error(f"Could not compute markers: {e}")

if markers is not None:
    tops = top_markers_by_cluster(markers, n=8)
    st.dataframe(markers, use_container_width=True, hide_index=True, height=320)
    st.download_button("⬇️ Markers (CSV)", data=markers.to_csv(index=False),
                       file_name="cluster_markers.csv", mime="text/csv")
else:
    tops = {}

# ── Annotation ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Annotate clusters")
comp = cluster_composition(adata)
existing = load_annotations(out_dir)
st.caption("Assign a cell-type label per cluster. Top marker genes (compute above) and the "
           "AGED/ADULT cell counts are shown to guide you. Saved to the study output.")

with st.form("annotation_form"):
    new_map = {}
    for cl in clusters:
        cols = st.columns([1, 3, 4])
        n_cells = int(comp.loc[cl, "n_cells"]) if cl in comp.index else 0
        cols[0].markdown(f"**{cl}**  \n<span style='font-size:11px;color:#888'>{n_cells:,} cells</span>",
                         unsafe_allow_html=True)
        hint = ", ".join(tops.get(cl, [])) if tops else "— compute markers for hints —"
        cols[1].caption(f"top: {hint}")
        new_map[cl] = cols[2].text_input(f"label_{cl}", value=existing.get(cl, ""),
                                         label_visibility="collapsed",
                                         placeholder="cell type (e.g. Tanycyte, Astrocyte)")
    submitted = st.form_submit_button("💾 Save annotations", type="primary")

if submitted:
    mapping = {cl: lbl.strip() for cl, lbl in new_map.items() if lbl.strip()}
    save_annotations(out_dir, mapping)
    # Bake labels into the h5ad so downstream steps see obs['cell_type'].
    try:
        import anndata as ad
        from xenium_spatial.cell_clustering import apply_annotations
        a = ad.read_h5ad(h5ad_path)
        apply_annotations(a, mapping)
        a.write_h5ad(h5ad_path)
    except Exception as e:
        logger.exception("Writing annotations to h5ad failed")
        st.warning(f"Saved labels, but could not update the h5ad: {e}")
    st.success(f"Saved {len(mapping)} label(s) to `{annotations_path(out_dir)}`.")
    st.rerun()

st.caption(f"Clustered object: `{h5ad_path}`")
