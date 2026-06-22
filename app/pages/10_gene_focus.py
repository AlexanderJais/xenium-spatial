"""
pages/10_gene_focus.py
Gene focus — quantitative analysis of one gene (default Galanin / Gal).

Reads the clustered AnnData and reports, for the selected gene: expression and
detection per cluster, per-cluster differential expression across conditions
(pseudobulk, per replicate), a per-slide spatial expression map, and a spatial
age-effect grid (AGED−ADULT difference per MBH sub-region).
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header, init_session_state

st.set_page_config(page_title="Gene focus · Xenium Spatial Pipeline", page_icon="🎯", layout="wide",
    initial_sidebar_state="expanded")
inject_css()
init_session_state()

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import pipeline  # noqa: E402
from xenium_spatial.cell_clustering import clustered_h5ad_path  # noqa: E402

logger = logging.getLogger("xenium_app")
_WONG = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442"]


@st.cache_data(show_spinner=False)
def _dge(path, mtime, gene, group_key):
    from xenium_spatial.gene_focus import gene_dge_across_clusters
    adata = pipeline.load_clustered(path, mtime)
    return gene_dge_across_clusters(adata, gene, group_key=group_key)


@st.cache_data(show_spinner=False)
def _grid(path, mtime, gene, n_bins):
    from xenium_spatial.gene_focus import gene_spatial_grid
    adata = pipeline.load_clustered(path, mtime)
    return gene_spatial_grid(adata, gene, n_bins=n_bins)


# ── Publication PDF builders (cached so they don't re-render every rerun) ─────
@st.cache_data(show_spinner=False)
def _violin_pdf(path, mtime, gene, group_key, split):
    from xenium_spatial import figure_export as fx
    from xenium_spatial.gene_focus import gene_by_cluster
    bc = gene_by_cluster(pipeline.load_clustered(path, mtime), gene, group_key=group_key)
    order = sorted(bc["group"].unique(), key=lambda x: (len(x), x))
    vdf = bc if len(bc) <= 40_000 else bc.sample(40_000, random_state=42)
    has_cond = "condition" in bc.columns and bc["condition"].nunique() >= 2
    conds = sorted(bc["condition"].unique()) if "condition" in bc.columns else []
    cc = {c: fx.WONG[i % len(fx.WONG)] for i, c in enumerate(conds)}
    if split and has_cond:
        return fx.violin_by_group(vdf, group_col="group", value_col="expr", order=order,
                                  xlabel=group_key, ylabel=f"{gene} (log-norm)",
                                  title=f"{gene} by cluster", split_col="condition",
                                  split_levels=conds, split_colour=cc)
    return fx.violin_by_group(vdf, group_col="group", value_col="expr", order=order,
                              xlabel=group_key, ylabel=f"{gene} (log-norm)",
                              title=f"{gene} by cluster")


@st.cache_data(show_spinner=False)
def _fc_pdf(path, mtime, gene, group_key):
    from xenium_spatial import figure_export as fx
    summary, _ = _dge(path, mtime, gene, group_key)
    bar = summary.dropna(subset=["log2fc"]).copy()
    if not len(bar):
        return None
    bar = bar.sort_values("log2fc")
    direction = next((d for d in summary["direction"] if d), "")
    return fx.hbar(bar["log2fc"].to_numpy(), bar["group"].tolist(),
                   xlabel=f"log2FC ({direction})", ylabel=group_key,
                   title=f"{gene} fold-change per cluster")


@st.cache_data(show_spinner=False)
def _genespatial_pdf(path, mtime, gene, slide):
    from xenium_spatial import figure_export as fx
    from xenium_spatial.gene_focus import gene_vector
    a = pipeline.load_clustered(path, mtime)
    if "spatial" not in a.obsm:
        return None
    o = a.obs
    xy = np.asarray(a.obsm["spatial"])
    sl = o["slide_id"].astype(str).values if "slide_id" in o else np.array(["all"] * a.n_obs)
    m = sl == slide
    expr = gene_vector(a, gene, "lognorm")[m]
    return fx.scatter_continuous(xy[m, 0], xy[m, 1], expr, xlabel="x (µm)", ylabel="y (µm)",
                                 title=f"{slide} — {gene}", cbar_label=f"{gene} (log-norm)",
                                 point_size=1.5, equal_aspect=True, invert_y=True, dark_bg=True)


@st.cache_data(show_spinner=False)
def _grid_diff_pdf(path, mtime, gene, n_bins, min_cells):
    from xenium_spatial import figure_export as fx
    g = _grid(path, mtime, gene, n_bins)
    if not g or g.get("diff") is None:
        return None
    both = np.minimum(np.nan_to_num(g["counts"][g["conds"][0]]),
                      np.nan_to_num(g["counts"][g["conds"][1]]))
    diff = g["diff"].copy()
    diff[both < min_cells] = np.nan
    if not np.isfinite(diff).any():
        return None
    vmax = float(np.nanmax(np.abs(diff)))
    return fx.heatmap(diff, x_labels=[], y_labels=[], cmap="RdBu_r", center=0.0,
                      vmin=-vmax, vmax=vmax, cbar_label="Δ (log-norm)",
                      title=f"{gene} difference ({g['direction']})",
                      xlabel="medial ↔ lateral", ylabel="dorsal → ventral")


page_header("🎯 Gene focus", "Quantitative analysis of one gene across clusters, conditions and space")

out_dir = st.session_state["output_dir"]
h5ad_path = clustered_h5ad_path(out_dir)
if not h5ad_path.exists():
    st.warning("No clustering found. Build it first on the **🔬 Clusters** page.")
    st.stop()

mtime = h5ad_path.stat().st_mtime
adata = pipeline.load_clustered(str(h5ad_path), mtime)
obs = adata.obs
genes = list(adata.var_names)
if not genes:
    st.error("The clustered object has no genes — rebuild it on 🔬 Clusters.")
    st.stop()

# Gene + grouping controls.
gc1, gc2 = st.columns([2, 2])
with gc1:
    default_ix = genes.index("Gal") if "Gal" in genes else 0
    gene = st.selectbox("Gene", genes, index=default_ix)
with gc2:
    has_celltype = "cell_type" in obs.columns
    group_key = "cell_type" if has_celltype else "leiden"
    if has_celltype:
        group_key = "cell_type" if st.toggle(
            "Group by annotated cell type (off = raw Leiden cluster)", value=True) else "leiden"

from xenium_spatial.gene_focus import gene_by_cluster  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

by_cell = gene_by_cluster(adata, gene, group_key=group_key)
order = sorted(by_cell["group"].unique(), key=lambda x: (len(x), x))
has_cond = "condition" in by_cell.columns and by_cell["condition"].nunique() >= 2
# Condition comparisons (DE + spatial grid) also need replicate labels.
can_dge = has_cond and "replicate" in obs.columns
conds = sorted(by_cell["condition"].unique()) if "condition" in by_cell.columns else []
cond_colour = {c: _WONG[i % len(_WONG)] for i, c in enumerate(conds)}

# ── A. Expression + detection per cluster ────────────────────────────────────
st.subheader(f"{gene} — expression by cluster")
det = (by_cell.groupby("group")["detected"].mean() * 100).reindex(order)
m1, m2 = st.columns(2)
m1.metric(f"{gene}+ cells", f"{int(by_cell['detected'].sum()):,} "
          f"({100 * by_cell['detected'].mean():.1f}%)")
m2.metric("Top cluster", f"{det.idxmax()} ({det.max():.0f}% +)" if len(det) else "—")

# Violin (subsample for responsiveness), split by condition if available.
vdf = by_cell
if len(vdf) > 40_000:
    vdf = vdf.sample(40_000, random_state=42)
split = st.toggle("Split violin by condition", value=has_cond, disabled=not has_cond)
if split and has_cond:
    figv = px.violin(vdf, x="group", y="expr", color="condition", points=False,
                     category_orders={"group": order}, color_discrete_map=cond_colour)
else:
    figv = px.violin(vdf, x="group", y="expr", points=False,
                     category_orders={"group": order})
figv.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                   xaxis_title=group_key, yaxis_title=f"{gene} (log-norm)")
st.plotly_chart(figv, use_container_width=True)
try:
    _vio = _violin_pdf(str(h5ad_path), mtime, gene, group_key, bool(split and has_cond))
    if _vio:
        st.download_button("⬇️ Expression violins (PDF, publication)", data=_vio,
                           file_name=f"{gene}_violins.pdf", mime="application/pdf")
except Exception as e:  # noqa: BLE001
    logger.exception("Gene violin PDF export failed")
    st.caption(f"PDF export unavailable: {e}")

figd = px.bar(x=det.index, y=det.values, labels={"x": group_key, "y": "% detected"})
figd.update_traces(marker_color="#1B4F8A")
figd.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10),
                   title=f"% of cells expressing {gene}")
st.plotly_chart(figd, use_container_width=True)

# ── B. Per-cluster differential expression (condition) ───────────────────────
st.divider()
st.subheader(f"{gene} — differential expression per cluster")
if not can_dge:
    st.info("Need ≥2 conditions and a `replicate` column for differential expression.")
else:
    summary, per_rep = _dge(str(h5ad_path), mtime, gene, group_key)
    direction = next((d for d in summary["direction"] if d), "")
    st.caption(f"Pseudobulk per replicate; log2FC is **{direction}** (positive = up in the "
               "second group). Each dot is one replicate.")

    # Per-replicate log2 CPM dots by cluster + condition.
    figp = go.Figure()
    for c in conds:
        s = per_rep[per_rep["condition"] == c]
        figp.add_trace(go.Scatter(x=s["group"], y=s["log2cpm"], mode="markers", name=c,
                                  marker=dict(color=cond_colour[c], size=9,
                                              line=dict(width=1, color="black"))))
    figp.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10),
                       xaxis=dict(categoryorder="array", categoryarray=order, title=group_key),
                       yaxis_title=f"{gene} log2 CPM (pseudobulk)",
                       legend=dict(orientation="h", y=1.02))
    st.plotly_chart(figp, use_container_width=True)

    # log2FC bar per cluster.
    bar = summary.dropna(subset=["log2fc"]).copy()
    if len(bar):
        bar = bar.sort_values("log2fc")
        figf = go.Figure(go.Bar(x=bar["log2fc"], y=bar["group"], orientation="h",
                                marker_color=np.where(bar["log2fc"] > 0, "#D55E00", "#0072B2")))
        figf.update_layout(height=max(240, 26 * len(bar)), margin=dict(l=10, r=10, t=20, b=10),
                           xaxis_title=f"log2FC ({direction})", yaxis_title=group_key,
                           title=f"{gene} fold-change per cluster")
        st.plotly_chart(figf, use_container_width=True)
        try:
            _fc = _fc_pdf(str(h5ad_path), mtime, gene, group_key)
            if _fc:
                st.download_button("⬇️ Fold-change per cluster (PDF, publication)", data=_fc,
                                   file_name=f"{gene}_log2fc_per_cluster.pdf",
                                   mime="application/pdf")
        except Exception as e:  # noqa: BLE001
            logger.exception("Gene log2FC PDF export failed")
            st.caption(f"PDF export unavailable: {e}")

    show = summary.copy()
    for col in show.columns:
        if col.endswith("_mean") or col == "log2fc":
            show[col] = show[col].round(3)
        if col in ("pval", "padj"):
            show[col] = show[col].round(4)
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Per-cluster DE (CSV)", data=summary.to_csv(index=False),
                       file_name=f"{gene}_dge_per_cluster.csv", mime="text/csv")

# ── C. Spatial expression map ────────────────────────────────────────────────
st.divider()
st.subheader(f"{gene} — spatial expression")
slide_ids = sorted(obs["slide_id"].astype(str).unique()) if "slide_id" in obs else ["all"]
slide = st.selectbox("Slide", slide_ids)
if "spatial" in adata.obsm:
    from xenium_spatial.gene_focus import gene_vector
    xy = np.asarray(adata.obsm["spatial"])
    sl = obs["slide_id"].astype(str).values if "slide_id" in obs else np.array(["all"] * adata.n_obs)
    m = sl == slide
    expr = gene_vector(adata, gene, "lognorm")[m]
    sdf = pd.DataFrame({"x": xy[m, 0], "y": xy[m, 1], gene: expr})
    sdf = sdf.sort_values(gene)  # plot high-expressing cells on top
    figs = px.scatter(sdf, x="x", y="y", color=gene, render_mode="webgl",
                      color_continuous_scale="Viridis")
    figs.update_traces(marker=dict(size=3))
    figs.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10),
                       xaxis=dict(title="x (µm)", scaleanchor="y", showgrid=False),
                       yaxis=dict(title="y (µm)", autorange="reversed", showgrid=False),
                       plot_bgcolor="#111111")
    st.plotly_chart(figs, use_container_width=True)
    st.caption(f"{int(m.sum()):,} cells on **{slide}** coloured by {gene} log-norm.")
    try:
        _smap = _genespatial_pdf(str(h5ad_path), mtime, gene, slide)
        if _smap:
            st.download_button("⬇️ Spatial expression (PDF, publication)", data=_smap,
                               file_name=f"{gene}_spatial_{slide}.pdf", mime="application/pdf")
    except Exception as e:  # noqa: BLE001
        logger.exception("Gene spatial PDF export failed")
        st.caption(f"PDF export unavailable: {e}")
else:
    st.info("No spatial coordinates in the clustered object.")

# ── D. Spatial age-effect grid ───────────────────────────────────────────────
st.divider()
st.subheader(f"{gene} — spatial age-effect grid")
if not can_dge:
    st.info("Need ≥2 conditions and a `replicate` column for the spatial age-effect grid.")
    st.stop()
st.caption("Each slide's coordinates are normalised to its ROI bounding box, tiled into a grid; "
           "the difference grid shows where the gene differs between conditions. Sparse bins "
           "(few cells) are masked. Discovery only — no per-bin test at n≈2.")
gc = st.columns(3)
with gc[0]:
    n_bins = st.number_input("Grid size (NxN)", 4, 16, 8, 1)
with gc[1]:
    min_cells = st.number_input("Min cells / bin", 1, 200, 10, 1)

if st.button("Compute spatial grid", key="run_grid"):
    st.session_state["_grid_ready"] = True
if st.session_state.get("_grid_ready"):
    try:
        with st.spinner("Binning …"):
            g = _grid(str(h5ad_path), mtime, gene, int(n_bins))
    except Exception as e:
        logger.exception("Spatial grid failed")
        st.error(f"Grid failed: {e}")
        g = None
    if g:
        if g["diff"] is not None:
            both = np.minimum(np.nan_to_num(g["counts"][g["conds"][0]]),
                              np.nan_to_num(g["counts"][g["conds"][1]]))
            diff = g["diff"].copy()
            diff[both < min_cells] = np.nan
            vmax = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1.0
            figg = go.Figure(go.Heatmap(z=diff, colorscale="RdBu_r", zmid=0,
                                        zmin=-vmax, zmax=vmax, colorbar=dict(title="Δ")))
            figg.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10),
                               title=f"{gene} difference  ({g['direction']})",
                               yaxis=dict(autorange="reversed", title="dorsal → ventral"),
                               xaxis_title="medial ↔ lateral")
            st.plotly_chart(figg, use_container_width=True)
            if not np.isfinite(diff).any():
                st.warning("Every bin is below the min-cell threshold — lower the grid size or threshold.")
            else:
                try:
                    _gd = _grid_diff_pdf(str(h5ad_path), mtime, gene, int(n_bins), int(min_cells))
                    if _gd:
                        st.download_button("⬇️ Age-effect grid (PDF, publication)", data=_gd,
                                           file_name=f"{gene}_ageeffect_grid.pdf",
                                           mime="application/pdf")
                except Exception as e:  # noqa: BLE001
                    logger.exception("Age-effect grid PDF export failed")
                    st.caption(f"PDF export unavailable: {e}")
        # Per-condition grids side by side, on a shared colour scale.
        cols = st.columns(len(g["conds"]))
        finite_max = [float(np.nanmax(v)) for v in g["grids"].values()
                      if np.isfinite(v).any()]
        gmax = max(finite_max) if finite_max else 1.0
        for col, c in zip(cols, g["conds"]):
            z = g["grids"][c].copy()
            z[np.nan_to_num(g["counts"][c]) < min_cells] = np.nan
            f = go.Figure(go.Heatmap(z=z, colorscale="Viridis", zmin=0, zmax=gmax,
                                     colorbar=dict(title=gene)))
            f.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), title=c,
                            yaxis=dict(autorange="reversed"))
            col.plotly_chart(f, use_container_width=True)

st.info("⚠️ n ≈ 2 per condition — treat condition differences (per-cluster DE and the spatial "
        "grid) as discovery; validate in an independent cohort. The grid's dorsal→ventral / "
        "medial↔lateral axes only line up across slides when each section is **straightened** "
        "and its MBH ROI framed consistently (set the per-slide rotation in the 🗺️ ROI Manager).")
