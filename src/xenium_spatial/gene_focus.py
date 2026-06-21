"""
gene_focus.py
-------------
Quantitative single-gene analysis on the clustered AnnData: expression and
detection per cluster, per-cluster differential expression across conditions
(pseudobulk, per replicate), and a spatial age-effect grid.

All condition comparisons are done at the biological-replicate level. The
spatial grid normalises each slide's coordinates to its own bounding box so a
bin means roughly the same MBH sub-region across sections (an approximation,
not a registered alignment).

numpy / pandas / scipy only (no scanpy).
"""
import logging

import numpy as np
import pandas as pd

from .composition import _bh_adjust

logger = logging.getLogger(__name__)


def gene_vector(adata, gene, layer="lognorm") -> np.ndarray:
    """Per-cell values for one gene from a layer (falls back to X)."""
    use_layer = layer if (layer and layer in adata.layers) else None
    return np.asarray(adata.obs_vector(gene, layer=use_layer)).ravel()


def gene_by_cluster(adata, gene, group_key="cell_type",
                    condition_key="condition") -> pd.DataFrame:
    """Per-cell tidy frame: ``group``, log-norm ``expr``, ``detected`` (count>0),
    and ``condition`` — for violins and detection-rate bars."""
    df = pd.DataFrame({
        "group": adata.obs[group_key].astype(str).values,
        "expr": gene_vector(adata, gene, "lognorm"),
        "detected": gene_vector(adata, gene, "counts") > 0,
    })
    if condition_key in adata.obs.columns:
        df["condition"] = adata.obs[condition_key].astype(str).values
    return df


def _total_counts(adata) -> np.ndarray:
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    return np.asarray(X.sum(axis=1)).ravel()


def gene_dge_across_clusters(adata, gene, group_key="cell_type", sample_key="replicate",
                             condition_key="condition"):
    """Per-cluster pseudobulk DE for one gene.

    Sums the gene's counts and the total counts per (cluster, replicate), forms
    log2 CPM, then compares conditions per cluster (Welch t-test, BH-adjusted).
    Returns ``(summary, per_replicate)``: ``summary`` is one row per cluster
    (means, log2fc as condB-vs-condA, pval, padj, n_cells); ``per_replicate`` is
    the (cluster, replicate) log2 CPM table for the dot plot.
    """
    from scipy import stats

    obs = adata.obs
    df = pd.DataFrame({
        "group": obs[group_key].astype(str).values,
        "sample": obs[sample_key].astype(str).values,
        "condition": obs[condition_key].astype(str).values,
        "gene": gene_vector(adata, gene, "counts"),
        "total": _total_counts(adata),
    })
    pb = (df.groupby(["group", "sample"], observed=True)
            .agg(gene=("gene", "sum"), total=("total", "sum"),
                 condition=("condition", "first")).reset_index())
    pb["log2cpm"] = np.log2(pb["gene"] / np.clip(pb["total"], 1.0, None) * 1e6 + 1.0)

    conds = sorted(pb["condition"].dropna().unique())
    n_cells = df.groupby("group", observed=True).size()
    direction = f"{conds[1]} vs {conds[0]}" if len(conds) == 2 else ""
    rows = []
    for g, sub in pb.groupby("group", observed=True):
        rec = {"group": g, "n_cells": int(n_cells.get(g, 0))}
        per = []
        for c in conds:
            vals = sub.loc[sub["condition"] == c, "log2cpm"].values
            rec[f"{c}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            per.append(vals)
        rec["log2fc"], rec["pval"], rec["direction"] = np.nan, np.nan, direction
        if len(conds) == 2 and all(len(v) >= 2 for v in per):
            rec["log2fc"] = float(np.mean(per[1]) - np.mean(per[0]))
            with np.errstate(all="ignore"):
                try:
                    rec["pval"] = float(stats.ttest_ind(per[1], per[0], equal_var=False).pvalue)
                except Exception:
                    rec["pval"] = np.nan
        rows.append(rec)

    summary = pd.DataFrame(rows)
    summary["padj"] = _bh_adjust(summary["pval"].to_numpy())
    return summary, pb


def gene_spatial_grid(adata, gene, n_bins: int = 8, condition_key="condition",
                      sample_key="replicate", slide_key="slide_id") -> dict:
    """Spatial age-effect grid for one gene.

    Normalises each slide's coordinates to its own bounding box ([0,1]^2), tiles
    into ``n_bins`` x ``n_bins``, and per bin computes the mean log-norm
    expression **per condition** (averaging replicate means so one big sample
    can't dominate). Returns per-condition grids, per-condition cell-count grids
    (for masking sparse bins), and the difference grid (condB - condA).
    """
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] missing — rebuild the clustering.")
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    obs = adata.obs
    slides = (obs[slide_key].astype(str).values
              if slide_key in obs.columns else np.array(["all"] * adata.n_obs))

    nx = np.zeros(len(xy))
    ny = np.zeros(len(xy))
    for s in np.unique(slides):
        m = slides == s
        x, y = xy[m, 0], xy[m, 1]
        nx[m] = (x - x.min()) / ((x.max() - x.min()) or 1.0)
        ny[m] = (y - y.min()) / ((y.max() - y.min()) or 1.0)
    bx = np.clip((nx * n_bins).astype(int), 0, n_bins - 1)
    by = np.clip((ny * n_bins).astype(int), 0, n_bins - 1)

    df = pd.DataFrame({
        "bin": by * n_bins + bx,
        "expr": gene_vector(adata, gene, "lognorm"),
        "sample": obs[sample_key].astype(str).values,
        "condition": obs[condition_key].astype(str).values,
    })
    conds = sorted(df["condition"].dropna().unique())
    per_samp = (df.groupby(["condition", "sample", "bin"], observed=True)["expr"]
                  .mean().reset_index())
    cond_mean = per_samp.groupby(["condition", "bin"])["expr"].mean()
    cell_counts = df.groupby(["condition", "bin"]).size().astype(float)

    def _grid(series, cond):
        g = np.full((n_bins, n_bins), np.nan)
        if cond in series.index.get_level_values(0):
            for b, v in series.loc[cond].items():
                r, c = divmod(int(b), n_bins)
                g[r, c] = v
        return g

    grids = {c: _grid(cond_mean, c) for c in conds}
    counts = {c: _grid(cell_counts, c) for c in conds}
    diff, direction = None, ""
    if len(conds) == 2:
        diff = grids[conds[1]] - grids[conds[0]]
        direction = f"{conds[1]} − {conds[0]}"
    return {"conds": conds, "grids": grids, "counts": counts, "diff": diff,
            "direction": direction, "n_bins": n_bins}
