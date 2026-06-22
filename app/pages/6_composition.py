"""
pages/6_composition.py
Cell-type composition — do cluster / cell-type proportions shift with condition?

Reads the clustered AnnData persisted by the 🔬 Clusters page and compares
per-replicate proportions of each cell type across conditions. Proportions are
computed per biological replicate (not per cell), and at the usual 2-vs-2 design
the focus is effect size; p-values are exploratory only.
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header, init_session_state

st.set_page_config(page_title="Composition · Xenium Spatial Pipeline", page_icon="📊", layout="wide",
    initial_sidebar_state="expanded")
inject_css()
init_session_state()

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import pipeline  # noqa: E402
from xenium_spatial.cell_clustering import clustered_h5ad_path  # noqa: E402
from xenium_spatial.composition import composition_long, composition_stats  # noqa: E402

logger = logging.getLogger("xenium_app")

_WONG = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442"]


page_header("📊 Cell-type composition", "Do cell-type proportions shift between conditions?")

out_dir = st.session_state["output_dir"]
h5ad_path = clustered_h5ad_path(out_dir)
if not h5ad_path.exists():
    st.warning("No clustering found. Build it first on the **🔬 Clusters** page.")
    st.stop()

adata = pipeline.load_clustered(str(h5ad_path), h5ad_path.stat().st_mtime)
obs = adata.obs

has_celltype = "cell_type" in obs.columns
group_key = "cell_type" if has_celltype else "leiden"
if has_celltype:
    group_key = "cell_type" if st.toggle(
        "Group by annotated cell type (off = raw Leiden cluster)", value=True) else "leiden"
else:
    st.caption("No annotations yet — grouping by raw Leiden cluster. "
               "Annotate on the 🔬 Clusters page for named cell types.")

if "condition" not in obs.columns or obs["condition"].nunique() < 2:
    st.warning("Need at least two conditions to compare composition.")
    st.stop()
if "replicate" not in obs.columns:
    st.error("The clustered object has no `replicate` column — rebuild it on 🔬 Clusters.")
    st.stop()

comp = composition_long(obs, group_key=group_key, sample_key="replicate",
                        condition_key="condition", batch_key="batch")
stats = composition_stats(comp, group_key=group_key, condition_key="condition")
conds = sorted(obs["condition"].astype(str).unique())
cond_colour = {c: _WONG[i % len(_WONG)] for i, c in enumerate(conds)}

import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

# ── Per-replicate stacked composition (overview / QC) ────────────────────────
st.subheader("Per-replicate composition")
st.caption("Each bar is one sample; segments are the cell-type fractions. "
           "A first check that replicates within a condition look alike.")
comp["percent"] = comp["proportion"] * 100
# Order samples by condition for a readable axis.
sample_order = (comp.drop_duplicates("replicate")
                    .sort_values(["condition", "replicate"])["replicate"].tolist())
group_order_full = sorted(comp[group_key].astype(str).unique(), key=lambda x: (len(x), x))
fig_stack = px.bar(comp, x="replicate", y="percent", color=group_key,
                   category_orders={"replicate": sample_order},
                   labels={"percent": "% of cells", "replicate": "sample"})
fig_stack.update_layout(height=420, barmode="stack", margin=dict(l=10, r=10, t=30, b=10),
                        legend_title=group_key)
st.plotly_chart(fig_stack, use_container_width=True)
try:
    from xenium_spatial import figure_export as fx
    _pal = {g: fx.WONG[i % len(fx.WONG)] for i, g in enumerate(group_order_full)}
    _stack_pdf = fx.stacked_bar(
        comp, sample_col="replicate", value_col="percent", group_col=group_key,
        sample_order=sample_order, group_order=group_order_full, palette=_pal,
        ylabel="% of cells", title="Per-replicate composition")
    st.download_button("⬇️ Stacked composition (PDF, publication)", data=_stack_pdf,
                       file_name="composition_stacked.pdf", mime="application/pdf")
except Exception as e:  # noqa: BLE001
    logger.exception("Stacked composition PDF export failed")
    st.caption(f"PDF export unavailable: {e}")

# ── Per-cell-type proportion by condition (honest n≈2 dots) ──────────────────
st.subheader("Proportion by condition")
st.caption("Each dot is one replicate; the bar is the condition mean. With ~2 replicates "
           "per group, read the dots and the effect size — not error bars.")
# Order cell types by absolute effect size from the stats table.
group_order = stats[group_key].astype(str).tolist()
means = (comp.groupby([group_key, "condition"], observed=True)["percent"]
             .mean().reset_index())

fig = go.Figure()
for c in conds:
    sub = means[means["condition"] == c]
    fig.add_bar(x=sub[group_key], y=sub["percent"], name=f"{c} (mean)",
                marker_color=cond_colour[c], opacity=0.45)
for c in conds:
    sub = comp[comp["condition"] == c]
    fig.add_trace(go.Scatter(
        x=sub[group_key], y=sub["percent"], mode="markers", name=f"{c} (replicates)",
        marker=dict(color=cond_colour[c], size=8, line=dict(width=1, color="black")),
    ))
fig.update_layout(height=460, barmode="group", margin=dict(l=10, r=10, t=30, b=10),
                  xaxis=dict(categoryorder="array", categoryarray=group_order,
                             title=group_key),
                  yaxis_title="% of cells", legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig, use_container_width=True)
try:
    from xenium_spatial import figure_export as fx
    _dots_pdf = fx.grouped_dots(
        comp, means, group_col=group_key, value_col="percent", cond_col="condition",
        group_order=group_order, conds=conds, cond_colour=cond_colour,
        ylabel="% of cells", title="Proportion by condition")
    st.download_button("⬇️ Proportion by condition (PDF, publication)", data=_dots_pdf,
                       file_name="composition_by_condition.pdf", mime="application/pdf")
except Exception as e:  # noqa: BLE001
    logger.exception("Proportion-by-condition PDF export failed")
    st.caption(f"PDF export unavailable: {e}")

# ── Stats table ──────────────────────────────────────────────────────────────
st.subheader("Effect sizes")
dir_txt = stats["direction"].iloc[0] if "direction" in stats.columns and len(stats) else ""
st.caption(f"Log2 fold-change of mean proportion ({dir_txt}); positive = higher in the "
           "second group. The t-test is **exploratory** — at n≈2 per condition it is "
           "underpowered and should rank candidates, not declare significance.")
show = stats.copy()
for col in show.columns:
    if col.endswith("_mean"):
        show[col] = (show[col] * 100).round(2)
    if col in ("log2fc",):
        show[col] = show[col].round(3)
    if col in ("t_pval", "t_padj"):
        show[col] = show[col].round(4)
st.dataframe(show, use_container_width=True, hide_index=True)
st.download_button("⬇️ Composition stats (CSV)", data=stats.to_csv(index=False),
                   file_name="composition_stats.csv", mime="text/csv")
st.download_button("⬇️ Per-replicate proportions (CSV)", data=comp.to_csv(index=False),
                   file_name="composition_per_replicate.csv", mime="text/csv")

st.info("⚠️ **n ≈ 2 per condition.** Treat this as discovery / hypothesis-generating: "
        "report effect sizes and per-replicate dots, and validate promising shifts in an "
        "independent cohort before claiming a proportion change.")
