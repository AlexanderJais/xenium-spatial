"""
pages/7_dge.py
Pseudobulk differential expression — within a cell type, AGED vs ADULT.

Reads the clustered AnnData, pseudobulks the chosen cell type per replicate, and
tests each gene across the replicate-level values (never per cell). Shown as a
volcano + table, with a per-cell-type DE-count summary. Underpowered at n≈2 —
effect-size ranking, not significance.
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header, init_session_state

st.set_page_config(page_title="DGE · Xenium Sample PCA", page_icon="🧪", layout="wide",
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
def _dge(path, mtime, cell_type, group_key):
    import anndata as ad
    from xenium_spatial.pseudobulk_dge import dge_for_celltype
    adata = ad.read_h5ad(path)
    df, err = dge_for_celltype(adata, cell_type, group_key=group_key)
    return df, err


@st.cache_data(show_spinner=False)
def _summary(path, mtime, group_key, padj_thresh, lfc_thresh):
    import anndata as ad
    from xenium_spatial.pseudobulk_dge import dge_summary
    adata = ad.read_h5ad(path)
    return dge_summary(adata, group_key=group_key, padj_thresh=padj_thresh, lfc_thresh=lfc_thresh)


page_header("🧪 Pseudobulk DGE", "Within-cell-type differential expression across conditions")

out_dir = st.session_state["output_dir"]
h5ad_path = clustered_h5ad_path(out_dir)
if not h5ad_path.exists():
    st.warning("No clustering found. Build it first on the **🔬 Clusters** page.")
    st.stop()

mtime = h5ad_path.stat().st_mtime
adata = pipeline.load_clustered(str(h5ad_path), mtime)
obs = adata.obs

if "condition" not in obs.columns or obs["condition"].nunique() < 2:
    st.warning("Need at least two conditions to compare.")
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

# ── Thresholds + cell-type selection ────────────────────────────────────────
t1, t2, t3 = st.columns(3)
with t1:
    cell_type = st.selectbox(f"Cell type ({group_key})", groups)
with t2:
    padj_thresh = st.number_input("padj threshold", 0.0, 1.0, 0.10, 0.01)
with t3:
    lfc_thresh = st.number_input("|log2FC| threshold", 0.0, 5.0, 1.0, 0.5)

# ── Per-cell-type DE-count summary ──────────────────────────────────────────
with st.expander("DE-gene counts per cell type", expanded=False):
    with st.spinner("Scanning all cell types …"):
        summ = _summary(str(h5ad_path), mtime, group_key, float(padj_thresh), float(lfc_thresh))
    st.dataframe(summ, use_container_width=True, hide_index=True)
    st.caption(f"DE = padj < {padj_thresh:g} and |log2FC| > {lfc_thresh:g}. "
               "Cell types that can't be tested show a reason in `note`.")

# ── DGE for the selected cell type ──────────────────────────────────────────
st.subheader(f"{cell_type} — differential expression")
df, err = _dge(str(h5ad_path), mtime, cell_type, group_key)
if err:
    st.warning(f"Cannot run DGE for **{cell_type}**: {err}")
    st.stop()

direction = df["direction"].iloc[0] if "direction" in df.columns and len(df) else ""
n_sig = int(((df["padj"] < padj_thresh) & (df["log2fc"].abs() > lfc_thresh)).sum())
st.caption(f"{len(df):,} genes tested · {n_sig} pass padj < {padj_thresh:g} & "
           f"|log2FC| > {lfc_thresh:g} · fold change is **{direction}** (positive = up in "
           "the second group).")

import plotly.graph_objects as go  # noqa: E402

# Volcano.
plot = df.dropna(subset=["pval"]).copy()
plot["neglog10p"] = -np.log10(plot["pval"].clip(lower=1e-300))
plot["sig"] = ((plot["padj"] < padj_thresh) & (plot["log2fc"].abs() > lfc_thresh))
fig = go.Figure()
for is_sig, colour, name in [(False, "#B8C4D0", "ns"), (True, "#D55E00", "DE")]:
    s = plot[plot["sig"] == is_sig]
    fig.add_trace(go.Scatter(
        x=s["log2fc"], y=s["neglog10p"], mode="markers", name=name,
        marker=dict(size=6, color=colour, line=dict(width=0.3, color="black")),
        text=s["gene"], hovertemplate="%{text}<br>log2FC %{x:.2f}<br>-log10p %{y:.2f}<extra></extra>",
    ))
# Label the top genes by significance.
for _, r in plot.sort_values("pval").head(12).iterrows():
    fig.add_annotation(x=r["log2fc"], y=r["neglog10p"], text=r["gene"],
                       font=dict(size=9), showarrow=False, yshift=8)
fig.add_vline(x=lfc_thresh, line=dict(color="grey", width=0.5, dash="dash"))
fig.add_vline(x=-lfc_thresh, line=dict(color="grey", width=0.5, dash="dash"))
fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                  xaxis_title="log2 fold-change", yaxis_title="-log10 p",
                  legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig, use_container_width=True)

show = df.copy()
for c in show.columns:
    if c.endswith("_mean") or c in ("log2fc", "base_log2cpm"):
        show[c] = show[c].round(3)
    if c in ("pval", "padj"):
        show[c] = show[c].round(5)
st.dataframe(show, use_container_width=True, hide_index=True, height=360)
st.download_button("⬇️ DGE table (CSV)", data=df.to_csv(index=False),
                   file_name=f"dge_{cell_type}.csv", mime="text/csv")

st.info("⚠️ **n ≈ 2 per condition.** Pseudobulk + t-test is the right *level* (replicate, not "
        "cell), but it's underpowered here — rank by effect size, and confirm with a count "
        "model (DESeq2/edgeR) and an independent cohort before reporting hits.")
