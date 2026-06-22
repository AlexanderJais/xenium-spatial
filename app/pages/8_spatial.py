"""
pages/8_spatial.py
Spatial maps + niches — the readouts that use the Xenium cell coordinates.

Reads the clustered AnnData and shows, per slide, where each cell type sits, plus
a neighbourhood-enrichment heatmap (which cell types are spatial neighbours more
or less than chance). The enrichment can be split by condition to look for niche
changes with age.
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header, init_session_state

st.set_page_config(page_title="Spatial · Xenium Spatial Pipeline", page_icon="🗺️", layout="wide",
    initial_sidebar_state="expanded")
inject_css()
init_session_state()

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import pipeline  # noqa: E402
from xenium_spatial.cell_clustering import clustered_h5ad_path  # noqa: E402

logger = logging.getLogger("xenium_app")


@st.cache_data(show_spinner=False)
def _enrichment(path, mtime, group_key, slide_subset, n_neighbors, n_perms):
    from xenium_spatial.spatial import neighborhood_enrichment
    adata = pipeline.load_clustered(path, mtime)
    if slide_subset:
        adata = adata[adata.obs["slide_id"].astype(str).isin(set(slide_subset))].copy()
    return neighborhood_enrichment(adata, group_key=group_key, n_neighbors=n_neighbors,
                                   n_perms=n_perms)


@st.cache_data(show_spinner=False)
def _spatialmap_pdf(path, mtime, group_key, slide, highlight):
    """Nature-style spatial cell-type map for one slide (PDF)."""
    from xenium_spatial import figure_export as fx
    a = pipeline.load_clustered(path, mtime)
    o = a.obs
    xy = np.asarray(a.obsm["spatial"])
    sl = o["slide_id"].astype(str).values if "slide_id" in o else np.array(["all"] * a.n_obs)
    m = sl == slide
    lab = o[group_key].astype(str).values[m]
    grps = sorted(set(lab), key=lambda x: (len(x), x))
    if highlight and highlight != "— all —":
        lab = np.where(lab == highlight, highlight, "other")
        order = [highlight, "other"]
        palette = {highlight: "#D55E00", "other": "#5A6470"}
    else:
        order = grps
        palette = None
    return fx.scatter_categorical(
        xy[m, 0], xy[m, 1], lab, order=order, palette=palette,
        xlabel="x (µm)", ylabel="y (µm)", title=f"{slide} — cell-type map",
        legend_title=group_key, point_size=1.5, equal_aspect=True,
        invert_y=True, dark_bg=True)


page_header("🗺️ Spatial maps & niches", "Where the cell types sit, and which ones co-localise")

out_dir = st.session_state["output_dir"]
h5ad_path = clustered_h5ad_path(out_dir)
if not h5ad_path.exists():
    st.warning("No clustering found. Build it first on the **🔬 Clusters** page.")
    st.stop()

mtime = h5ad_path.stat().st_mtime
adata = pipeline.load_clustered(str(h5ad_path), mtime)
obs = adata.obs
if "spatial" not in adata.obsm:
    st.error("The clustered object has no spatial coordinates — rebuild it on 🔬 Clusters.")
    st.stop()

has_celltype = "cell_type" in obs.columns
group_key = "cell_type"
if has_celltype:
    group_key = "cell_type" if st.toggle(
        "Group by annotated cell type (off = raw Leiden cluster)", value=True) else "leiden"
else:
    group_key = "leiden"
    st.caption("No annotations yet — using raw Leiden clusters. Annotate on 🔬 Clusters for names.")

groups = sorted(obs[group_key].astype(str).unique(), key=lambda x: (len(x), x))
slide_ids = sorted(obs["slide_id"].astype(str).unique()) if "slide_id" in obs else ["all"]

import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

# Fixed colour per cell type so the same type keeps its colour across slides
# (px otherwise assigns colours by order-of-appearance within each slide's data).
_palette = (px.colors.qualitative.Dark24 + px.colors.qualitative.Light24)
colour_map = {g: _palette[i % len(_palette)] for i, g in enumerate(groups)}

# ── Spatial cell-type map ────────────────────────────────────────────────────
st.subheader("Cell-type map")
mc1, mc2 = st.columns([1, 1])
with mc1:
    slide = st.selectbox("Slide", slide_ids)
with mc2:
    highlight = st.selectbox("Highlight one type (optional)", ["— all —"] + groups)

xy = np.asarray(adata.obsm["spatial"])
sl = obs["slide_id"].astype(str).values if "slide_id" in obs else np.array(["all"] * adata.n_obs)
m = sl == slide
dfm = pd.DataFrame({"x": xy[m, 0], "y": xy[m, 1], group_key: obs[group_key].astype(str).values[m]})

if highlight != "— all —":
    dfm["shown"] = np.where(dfm[group_key] == highlight, highlight, "other")
    fig_map = px.scatter(dfm, x="x", y="y", color="shown", render_mode="webgl",
                         color_discrete_map={highlight: "#D55E00", "other": "#D8DCE4"},
                         category_orders={"shown": [highlight, "other"]})
else:
    fig_map = px.scatter(dfm, x="x", y="y", color=group_key, render_mode="webgl",
                         color_discrete_map=colour_map,
                         category_orders={group_key: groups})
fig_map.update_traces(marker=dict(size=3, opacity=0.75))
fig_map.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(title="x (µm)", scaleanchor="y", showgrid=False),
                      yaxis=dict(title="y (µm)", autorange="reversed", showgrid=False),
                      plot_bgcolor="#111111", legend=dict(itemsizing="constant"))
st.plotly_chart(fig_map, use_container_width=True)
st.caption(f"{int(m.sum()):,} cells on slide **{slide}**. Y axis: 0 = dorsal, larger = ventral.")
try:
    _map_pdf = _spatialmap_pdf(str(h5ad_path), mtime, group_key, slide, highlight)
    st.download_button("⬇️ Cell-type map (PDF, publication)", data=_map_pdf,
                       file_name=f"spatial_map_{slide}.pdf", mime="application/pdf")
except Exception as e:  # noqa: BLE001
    logger.exception("Spatial map PDF export failed")
    st.caption(f"PDF export unavailable: {e}")

# ── Neighbourhood enrichment ─────────────────────────────────────────────────
st.divider()
st.subheader("Neighbourhood enrichment")
st.caption("Permutation z-score that two cell types are spatial neighbours more (red) or "
           "less (blue) than chance. Labels are shuffled within each slide, preserving "
           "the tissue structure.")

e1, e2, e3 = st.columns(3)
with e1:
    n_neighbors = st.number_input("k spatial neighbours", 3, 30, 6, 1)
with e2:
    n_perms = st.number_input("permutations", 20, 500, 100, 20)
with e3:
    split = st.toggle("Split by condition", value=False,
                      help="Compute enrichment separately within each condition's slides "
                           "to look for niche changes with age.")

def _heat(z: pd.DataFrame, title: str):
    if z.empty or not np.isfinite(z.values).any():
        st.warning(f"{title}: not enough cells per slide to compute enrichment "
                   "(slides with ≤ k cells are skipped).")
        return None
    vmax = float(np.nanmax(np.abs(z.values))) or 1.0
    fig = go.Figure(go.Heatmap(z=z.values, x=list(z.columns), y=list(z.index),
                               colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
                               colorbar=dict(title="z")))
    fig.update_layout(title=title, height=480, margin=dict(l=10, r=10, t=40, b=10),
                      yaxis=dict(autorange="reversed"))
    return fig


def _enrich_pdf(z: pd.DataFrame, title: str) -> bytes:
    from xenium_spatial import figure_export as fx
    return fx.heatmap(z.values, x_labels=list(z.columns), y_labels=list(z.index),
                      cmap="RdBu_r", center=0.0, cbar_label="z-score",
                      title=f"Neighbourhood enrichment — {title}", annotate=True)

if st.button("Compute neighbourhood enrichment", key="run_enrich"):
    st.session_state["_enrich_ready"] = True

if st.session_state.get("_enrich_ready"):
    try:
        with st.spinner("Building spatial graphs and permuting …"):
            conds = (sorted(obs["condition"].astype(str).unique())
                     if "condition" in obs else [])
            if split and len(conds) >= 2:
                cols = st.columns(len(conds))
                for col, c in zip(cols, conds):
                    c_slides = sorted(obs.loc[obs["condition"].astype(str) == c, "slide_id"]
                                      .astype(str).unique())
                    z = _enrichment(str(h5ad_path), mtime, group_key, tuple(c_slides),
                                    int(n_neighbors), int(n_perms))
                    fig = _heat(z, c)
                    if fig is not None:
                        col.plotly_chart(fig, use_container_width=True)
                        col.download_button(f"⬇️ {c} enrichment (PDF)", data=_enrich_pdf(z, c),
                                            file_name=f"enrichment_{c}.pdf",
                                            mime="application/pdf", key=f"enrich_pdf_{c}")
            else:
                if split:
                    st.info("Only one condition present — showing the combined enrichment.")
                z = _enrichment(str(h5ad_path), mtime, group_key, (), int(n_neighbors), int(n_perms))
                fig = _heat(z, "All slides")
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    d1, d2 = st.columns(2)
                    d1.download_button("⬇️ Enrichment z-scores (CSV)", data=z.to_csv(),
                                       file_name="neighbourhood_enrichment.csv", mime="text/csv")
                    d2.download_button("⬇️ Enrichment heatmap (PDF, publication)",
                                       data=_enrich_pdf(z, "all slides"),
                                       file_name="neighbourhood_enrichment.pdf",
                                       mime="application/pdf")
    except Exception as e:
        logger.exception("Neighbourhood enrichment failed")
        st.error(f"Enrichment failed: {e}")

st.info("Spatial niches are a strength of Xenium data. Aging niche shifts (e.g. microglia "
        "clustering near the ventricle, or tanycyte–neuron contacts changing) show up here — "
        "but the same n≈2 caveat applies: treat per-condition differences as discovery.")
