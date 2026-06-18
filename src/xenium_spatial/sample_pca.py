"""
sample_pca.py
-------------
Sample-level (pseudobulk) PCA for the AGED vs ADULT Xenium study.

This is the *first* exploratory step of the pipeline: before any
cell-level clustering or DGE, we ask a simpler question —

    "Do the samples separate by biological group, and are there
     any outlier slides we should worry about?"

To answer it we collapse every slide (biological replicate) into a
single pseudobulk expression profile, normalise for library size,
and run PCA across the 8 samples.  The resulting plots show:

  * how individual samples cluster in PC space,
  * how the AGED and ADULT groups separate,
  * a sample-by-sample correlation heatmap with hierarchical
    clustering (dendrogram),
  * a scree plot of variance explained per PC.

Pseudobulk PCA is the standard QC / sanity-check for replicated
spatial and bulk studies (cf. DESeq2's ``plotPCA``): it is robust at
n=4 per group because each point is one biological replicate, not one
cell.

Design notes
------------
* Operates on a single concatenated AnnData (output of
  :class:`xenium_spatial.multislide_loader.MultiSlideLoader`) that already has
  ROI filtering applied and raw counts in ``.layers['counts']``.
* Depends only on numpy / pandas / scipy / scikit-learn / matplotlib
  (no scanpy), so it can run in a minimal environment.
* Aggregation, normalisation, PCA and plotting are separate functions
  so they can be unit-tested and reused independently; the
  :func:`sample_level_pca_analysis` orchestrator wires them together.
"""

import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal figure style (kept local so this module has no seaborn / scanpy
# dependency — it only needs numpy/pandas/scipy/sklearn/matplotlib).
# Encodes the Nature-grade figure conventions used across the project
# (Arial-ish sans, thin spines, editable Type-42 PDF fonts).
# ---------------------------------------------------------------------------
_NATURE_RC = {
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
}
_WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73",
         "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
_CONDITION_COLOURS = {
    "Control": "#0072B2", "Treatment": "#D55E00",
    "ADULT": "#0072B2", "AGED": "#D55E00",
}


def _apply_style():
    import matplotlib as mpl
    mpl.rcParams.update(_NATURE_RC)


def _savefig(fig, path: Path, fmt: str = "pdf", dpi: int = 300) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = path.with_suffix(f".{fmt}")
    fig.savefig(out, format=fmt, dpi=dpi, transparent=False)
    logger.info("Saved: %s", out)
    return out


# ===========================================================================
# 1. Pseudobulk aggregation
# ===========================================================================

def pseudobulk_samples(
    adata: ad.AnnData,
    sample_key: str = "replicate",
    condition_key: str = "condition",
    batch_key: str = "batch",
    layer: str = "counts",
) -> ad.AnnData:
    """
    Collapse cells into one pseudobulk profile per sample.

    Counts are summed across all cells belonging to each sample
    (slide / biological replicate).  The result is a small AnnData of
    shape ``(n_samples, n_genes)`` with per-sample metadata preserved
    in ``.obs``.

    Parameters
    ----------
    adata:
        Concatenated cell-level AnnData.  Raw integer counts must be in
        ``adata.layers[layer]`` (default ``'counts'``); falls back to
        ``.X`` with a warning if the layer is absent.
    sample_key:
        ``obs`` column identifying the biological replicate / slide.
        One pseudobulk row is produced per unique value.
    condition_key:
        ``obs`` column with the biological group (e.g. AGED / ADULT).
        Carried through to the pseudobulk ``.obs`` for colouring.
    layer:
        Layer holding raw counts.

    Returns
    -------
    AnnData of pseudobulk samples (cells summed), with ``.obs`` columns
    ``[condition_key, 'n_cells', 'total_counts']`` and the raw summed
    counts stored in both ``.X`` and ``.layers['counts']``.
    """
    if sample_key not in adata.obs.columns:
        raise KeyError(
            f"sample_key='{sample_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    X = _get_counts(adata, layer)

    # Stable, readable sample order: group by condition then sample id.
    sample_ids = (
        adata.obs[[sample_key]]
        .assign(_cond=adata.obs[condition_key].astype(str)
                if condition_key in adata.obs.columns else "")
        .drop_duplicates(subset=[sample_key])
        .sort_values(["_cond", sample_key])[sample_key]
        .astype(str)
        .tolist()
    )

    obs_sample = adata.obs[sample_key].astype(str).values
    rows, meta_rows = [], []
    for sid in sample_ids:
        mask = obs_sample == sid
        n_cells = int(mask.sum())
        if n_cells == 0:
            continue
        summed = np.asarray(X[mask].sum(axis=0)).ravel()
        rows.append(summed)

        meta = {"sample": sid, "n_cells": n_cells, "total_counts": float(summed.sum())}
        if condition_key in adata.obs.columns:
            # All cells of one sample share the same condition; take the first.
            meta[condition_key] = str(adata.obs.loc[mask, condition_key].iloc[0])
        if batch_key in adata.obs.columns:
            # Likewise, all cells of one sample share the same batch label.
            meta[batch_key] = str(adata.obs.loc[mask, batch_key].iloc[0])
        meta_rows.append(meta)

    if not rows:
        raise ValueError("Pseudobulk produced no samples — check sample_key values.")

    counts = np.vstack(rows).astype(np.float32)
    obs = pd.DataFrame(meta_rows).set_index("sample")
    obs.index.name = "sample"

    pb = ad.AnnData(X=counts, obs=obs, var=adata.var.copy())
    pb.layers["counts"] = pb.X.copy()

    if condition_key in pb.obs.columns:
        pb.obs[condition_key] = pb.obs[condition_key].astype("category")
    if batch_key in pb.obs.columns:
        pb.obs[batch_key] = pb.obs[batch_key].astype("category")

    logger.info(
        "Pseudobulk: %d samples x %d genes (sample_key='%s'). Cells per sample: %s",
        pb.n_obs, pb.n_vars, sample_key,
        ", ".join(f"{s}={n}" for s, n in zip(pb.obs_names, pb.obs["n_cells"])),
    )
    return pb


# ===========================================================================
# 2. Normalisation
# ===========================================================================

def normalize_pseudobulk(
    pb: ad.AnnData,
    target_sum: float = 1e6,
    log: bool = True,
) -> ad.AnnData:
    """
    Library-size normalise pseudobulk counts, then optionally log1p.

    Different samples contain different numbers of cells and therefore
    very different total counts; without library-size normalisation PCA
    would simply rank samples by sequencing depth / cell number.  We
    scale each sample to a common total (counts-per-million by default)
    and apply ``log1p`` to stabilise variance.

    The normalised matrix is written to ``pb.X`` and
    ``pb.layers['lognorm']``; raw counts remain in ``pb.layers['counts']``.

    Returns the same (modified) AnnData.
    """
    counts = _as_dense(pb.layers["counts"]).astype(np.float64)
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0  # guard against empty samples
    normed = counts / lib * target_sum
    if log:
        normed = np.log1p(normed)

    pb.X = normed.astype(np.float32)
    pb.layers["lognorm"] = pb.X.copy()
    logger.info(
        "Pseudobulk normalised to target_sum=%.0f%s.",
        target_sum, " + log1p" if log else "",
    )
    return pb


# ===========================================================================
# 3. PCA across samples
# ===========================================================================

def run_sample_pca(
    pb: ad.AnnData,
    n_comps: Optional[int] = None,
    n_top_genes: int = 0,
    scale_genes: bool = False,
    random_state: int = 42,
) -> ad.AnnData:
    """
    PCA on the sample-by-gene matrix.

    Parameters
    ----------
    pb:
        Pseudobulk AnnData with normalised values in ``.X`` (run
        :func:`normalize_pseudobulk` first).
    n_comps:
        Number of principal components.  Defaults to
        ``min(n_samples - 1, n_genes)`` which is the maximum number of
        non-trivial components for a centred matrix.
    n_top_genes:
        If > 0, restrict PCA to the ``n_top_genes`` most variable genes
        (by variance across samples).  ``0`` (default) uses all genes —
        recommended for targeted Xenium panels (~300 genes), consistent
        with the rest of the pipeline.
    scale_genes:
        If True, z-score each gene (unit variance) before PCA so that
        highly expressed genes do not dominate.  Default False: ``log1p``
        already compresses the dynamic range and unscaled PCA is the
        convention for pseudobulk QC (cf. DESeq2 ``plotPCA``).
    random_state:
        Seed for the (deterministic) SVD solver.

    Returns
    -------
    The pseudobulk AnnData with:
        ``.obsm['X_pca']``                (n_samples, n_comps)
        ``.uns['pca']['variance_ratio']`` explained-variance fractions
        ``.uns['pca']['genes_used']``     genes entering the PCA
        ``.varm['PCs']``                  gene loadings (top-gene subset
                                          rows only if ``n_top_genes>0``)
    """
    from sklearn.decomposition import PCA

    X = _as_dense(pb.X).astype(np.float64)
    genes = np.asarray(pb.var_names)

    if 0 < n_top_genes < X.shape[1]:
        var = X.var(axis=0)
        keep = np.argsort(var)[::-1][:n_top_genes]
        keep.sort()
        X = X[:, keep]
        genes = genes[keep]
        logger.info("PCA restricted to %d most variable genes.", n_top_genes)

    if scale_genes:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd

    max_comps = max(1, min(X.shape[0] - 1, X.shape[1]))
    if n_comps is None:
        n_comps = max_comps
    else:
        n_comps = min(n_comps, max_comps)

    pca = PCA(n_components=n_comps, svd_solver="full", random_state=random_state)
    scores = pca.fit_transform(X)

    pb.obsm["X_pca"] = scores.astype(np.float32)
    pb.uns["pca"] = {
        "variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "variance": pca.explained_variance_.astype(np.float32),
        "n_comps": int(n_comps),
        "genes_used": genes.tolist(),
        "scaled": bool(scale_genes),
    }
    # Loadings: (n_genes_used, n_comps)
    pb.uns["pca"]["loadings"] = pca.components_.T.astype(np.float32)

    logger.info(
        "Sample PCA: %d components | PC1=%.1f%%, PC2=%.1f%% variance (%d genes).",
        n_comps,
        pca.explained_variance_ratio_[0] * 100,
        pca.explained_variance_ratio_[1] * 100 if n_comps > 1 else 0.0,
        len(genes),
    )
    return pb


# ===========================================================================
# 4. Plots
# ===========================================================================

def plot_sample_pca(
    pb: ad.AnnData,
    condition_key: str = "condition",
    batch_key: str = "batch",
    output_dir: Path | str = ".",
    pc_x: int = 1,
    pc_y: int = 2,
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Scatter of samples in PC space, coloured by group and labelled by id.

    Each point is one individual sample (pseudobulk replicate); axis
    labels report the fraction of variance explained by each PC. When a
    ``batch_key`` column carries more than one batch, batch is encoded as the
    marker *shape* (condition stays colour) so you can see at a glance whether
    samples separate by technical batch rather than by condition.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _apply_style()
    CONDITION_COLOURS, WONG = _CONDITION_COLOURS, _WONG

    scores = pb.obsm["X_pca"]
    vr = pb.uns["pca"]["variance_ratio"]
    n_pc = scores.shape[1]
    ix, iy = pc_x - 1, pc_y - 1

    # With only two samples PCA yields a single non-trivial component, so the
    # requested second axis (PC2) does not exist. Fall back to a flat y=0 axis
    # so the samples can still be displayed along PC1.
    xs = scores[:, ix]
    ys = scores[:, iy] if iy < n_pc else np.zeros(scores.shape[0])
    y_label = (
        f"PC{pc_y} ({vr[iy] * 100:.1f}%)" if iy < n_pc
        else f"PC{pc_y} (n/a — only {n_pc} component)"
    )

    conds = (
        pb.obs[condition_key].astype(str).values
        if condition_key in pb.obs.columns
        else np.array([""] * pb.n_obs)
    )
    uniq = sorted(set(conds))
    colour = _condition_colours(uniq, CONDITION_COLOURS, WONG)

    # Optional batch encoding via marker shape (only when >1 batch is present
    # and it isn't just a copy of the sample id, which carries no information).
    batches = (
        pb.obs[batch_key].astype(str).values
        if batch_key in pb.obs.columns
        else np.array([""] * pb.n_obs)
    )
    uniq_batch = sorted(set(b for b in batches if b))
    # Batch is informative only when it *groups* samples — i.e. at least one
    # batch is shared by ≥2 samples (fewer distinct batches than samples). When
    # every sample has its own batch (the default, batch == slide_id), the
    # marker shapes carry no signal, so fall back to the plain colour plot.
    show_batch = 1 < len(uniq_batch) < pb.n_obs
    _MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
    batch_marker = {b: _MARKERS[i % len(_MARKERS)] for i, b in enumerate(uniq_batch)}

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    if show_batch:
        for cond in uniq:
            for b in uniq_batch:
                m = (conds == cond) & (batches == b)
                if m.any():
                    ax.scatter(xs[m], ys[m], s=55, c=colour[cond],
                               marker=batch_marker[b], edgecolors="black",
                               linewidths=0.5, zorder=3)
    else:
        for cond in uniq:
            m = conds == cond
            ax.scatter(xs[m], ys[m], s=55, c=colour[cond], edgecolors="black",
                       linewidths=0.5, zorder=3)

    for i, sid in enumerate(pb.obs_names):
        ax.annotate(
            sid, (xs[i], ys[i]),
            xytext=(3, 3), textcoords="offset points",
            fontsize=5.5, zorder=4,
        )

    ax.set_xlabel(f"PC{pc_x} ({vr[ix] * 100:.1f}%)")
    ax.set_ylabel(y_label)
    ax.set_title("Sample-level PCA (pseudobulk)")
    ax.axhline(0, color="grey", lw=0.4, ls="--", zorder=1)
    ax.axvline(0, color="grey", lw=0.4, ls="--", zorder=1)

    # Condition legend (colour); plus a batch legend (marker shape) when shown.
    if len(uniq) > 1:
        cond_handles = [Line2D([0], [0], marker="o", ls="", markerfacecolor=colour[c],
                               markeredgecolor="black", markersize=7, label=c)
                        for c in uniq]
        leg1 = ax.legend(handles=cond_handles, title=condition_key, frameon=False,
                         loc="best", fontsize=5.5)
        ax.add_artist(leg1)
    if show_batch:
        batch_handles = [Line2D([0], [0], marker=batch_marker[b], ls="",
                                markerfacecolor="lightgrey", markeredgecolor="black",
                                markersize=7, label=b) for b in uniq_batch]
        ax.legend(handles=batch_handles, title=batch_key, frameon=False,
                  loc="lower right", fontsize=5.5)

    fig.tight_layout()
    return _savefig(fig, Path(output_dir) / "sample_pca_scatter", fmt=fmt, dpi=dpi)


def plot_sample_correlation(
    pb: ad.AnnData,
    condition_key: str = "condition",
    output_dir: Path | str = ".",
    method: str = "pearson",
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """
    Sample-by-sample correlation heatmap with hierarchical clustering.

    Samples are ordered by hierarchical (average-linkage) clustering of
    ``1 - correlation`` distances and annotated with a coloured group
    bar so that group cohesion / outliers are visible at a glance.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    _apply_style()
    CONDITION_COLOURS, WONG = _CONDITION_COLOURS, _WONG

    X = _as_dense(pb.X)
    corr = np.corrcoef(X) if method == "pearson" else _spearman_corr(X)
    corr = np.clip(corr, -1.0, 1.0)
    n = corr.shape[0]

    # Hierarchical ordering on correlation distance.
    if n > 2:
        dist = 1.0 - corr
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0  # enforce symmetry for squareform
        Z = linkage(squareform(dist, checks=False), method="average")
        order = leaves_list(Z)
    else:
        order = np.arange(n)

    corr_o = corr[np.ix_(order, order)]
    labels = np.asarray(pb.obs_names)[order]

    conds = (
        pb.obs[condition_key].astype(str).values[order]
        if condition_key in pb.obs.columns
        else np.array([""] * n)
    )
    uniq = sorted(set(conds))
    colour = _condition_colours(uniq, CONDITION_COLOURS, WONG)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(corr_o, cmap="viridis", aspect="equal", vmin=corr_o.min(), vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
    ax.set_yticklabels(labels, fontsize=5.5)

    # Group colour bar along the top.
    for i, cond in enumerate(conds):
        ax.add_patch(plt.Rectangle(
            (i - 0.5, -1.2), 1.0, 0.6, facecolor=colour[cond],
            edgecolor="none", clip_on=False,
        ))
    ax.set_ylim(n - 0.5, -1.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{method.capitalize()} r", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    handles = [plt.Line2D([0], [0], marker="s", ls="", markerfacecolor=colour[c],
                          markeredgecolor="none", markersize=6, label=c) for c in uniq]
    if len(uniq) > 1:
        ax.legend(handles=handles, title=condition_key, frameon=False,
                  loc="upper left", bbox_to_anchor=(1.25, 1.0), fontsize=5.5)
    ax.set_title("Sample correlation (hierarchically ordered)")
    fig.tight_layout()
    return _savefig(fig, Path(output_dir) / "sample_correlation_heatmap", fmt=fmt, dpi=dpi)


def plot_scree(
    pb: ad.AnnData,
    output_dir: Path | str = ".",
    fmt: str = "pdf",
    dpi: int = 300,
) -> Path:
    """Bar plot of variance explained per PC, with cumulative line."""
    import matplotlib.pyplot as plt

    _apply_style()

    vr = np.asarray(pb.uns["pca"]["variance_ratio"]) * 100
    pcs = np.arange(1, len(vr) + 1)

    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.bar(pcs, vr, color="#0072B2", width=0.7)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_xticks(pcs)

    ax2 = ax.twinx()
    ax2.plot(pcs, np.cumsum(vr), "-o", color="#D55E00", ms=3, lw=0.8)
    ax2.set_ylabel("Cumulative (%)", color="#D55E00")
    ax2.tick_params(axis="y", labelcolor="#D55E00")
    ax2.set_ylim(0, 105)
    ax2.spines["right"].set_visible(True)

    ax.set_title("PCA scree plot")
    fig.tight_layout()
    return _savefig(fig, Path(output_dir) / "sample_pca_scree", fmt=fmt, dpi=dpi)


# ===========================================================================
# 5. Orchestrator
# ===========================================================================

def sample_level_pca_analysis(
    adata: ad.AnnData,
    output_dir: Path | str,
    sample_key: str = "replicate",
    condition_key: str = "condition",
    batch_key: str = "batch",
    n_top_genes: int = 0,
    scale_genes: bool = False,
    base_panel_only: bool = True,
    fmt: str = "pdf",
    dpi: int = 300,
    random_state: int = 42,
) -> ad.AnnData:
    """
    Run the full sample-level PCA workflow and save figures + tables.

    Steps: (optionally restrict to base panel) -> pseudobulk ->
    library-size normalise -> PCA -> scatter, correlation heatmap, scree
    plot, and a CSV of PC coordinates.

    Parameters
    ----------
    base_panel_only:
        If True (default), restrict the analysis to the shared Xenium base
        panel genes (``var['panel_type'] == 'base'``) before pseudobulking,
        dropping every add-on / custom gene.  This keeps the PCA comparable
        across slides that carry different add-on panels.  Set False to use
        whichever gene set the loader produced (base + custom).

    Returns the pseudobulk AnnData (with PCA in ``.obsm['X_pca']``) so
    the caller can do further inspection.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Sample-level PCA  |  sample_key='%s', group='%s'", sample_key, condition_key)
    logger.info("=" * 60)

    if base_panel_only:
        adata = _restrict_to_base_panel(adata)

    pb = pseudobulk_samples(adata, sample_key=sample_key, condition_key=condition_key,
                            batch_key=batch_key)
    pb = normalize_pseudobulk(pb)
    pb = run_sample_pca(
        pb, n_top_genes=n_top_genes, scale_genes=scale_genes, random_state=random_state,
    )

    plot_sample_pca(pb, condition_key=condition_key, batch_key=batch_key,
                    output_dir=output_dir, fmt=fmt, dpi=dpi)
    plot_sample_correlation(pb, condition_key=condition_key, output_dir=output_dir, fmt=fmt, dpi=dpi)
    plot_scree(pb, output_dir=output_dir, fmt=fmt, dpi=dpi)

    # Export PC coordinates + metadata for downstream inspection.
    vr = pb.uns["pca"]["variance_ratio"]
    pc_cols = [f"PC{i+1}" for i in range(pb.obsm["X_pca"].shape[1])]
    coord_df = pd.DataFrame(pb.obsm["X_pca"], index=pb.obs_names, columns=pc_cols)
    meta_cols = [c for c in (condition_key, batch_key, "n_cells", "total_counts")
                 if c in pb.obs.columns]
    coord_df = pb.obs[meta_cols].join(coord_df)
    coord_path = output_dir / "sample_pca_coordinates.csv"
    coord_df.to_csv(coord_path)
    logger.info("Sample PCA coordinates saved to %s", coord_path)

    var_df = pd.DataFrame({
        "PC": pc_cols,
        "variance_ratio": np.asarray(vr),
        "cumulative": np.cumsum(np.asarray(vr)),
    })
    var_df.to_csv(output_dir / "sample_pca_variance.csv", index=False)

    pb.write_h5ad(output_dir / "pseudobulk_samples.h5ad")
    logger.info("Sample-level PCA complete. Outputs in %s/", output_dir)
    return pb


# ===========================================================================
# Internal helpers
# ===========================================================================

def _restrict_to_base_panel(adata: ad.AnnData) -> ad.AnnData:
    """
    Subset an AnnData to the shared Xenium base panel genes.

    Relies on the ``var['panel_type']`` column written by
    :class:`xenium_spatial.panel_registry.PanelRegistry` during harmonisation, where
    base-panel genes are tagged ``'base'`` and add-on genes ``'custom'`` /
    ``'custom_shared'`` / ``'custom_unique'``.  Add-on genes are dropped so
    the PCA only uses the 247 genes common to every slide.

    If the column is absent (e.g. an AnnData not produced by the loader)
    the input is returned unchanged with a warning.
    """
    if "panel_type" not in adata.var.columns:
        logger.warning(
            "base_panel_only requested but var['panel_type'] is missing; "
            "using all %d genes. Load slides via MultiSlideLoader to enable "
            "base-panel restriction.", adata.n_vars,
        )
        return adata

    panel = adata.var["panel_type"].astype(str)
    is_base = (panel == "base").values
    n_base = int(is_base.sum())
    if n_base == 0:
        logger.warning(
            "base_panel_only requested but no genes are tagged 'base'; "
            "using all %d genes.", adata.n_vars,
        )
        return adata

    n_dropped = adata.n_vars - n_base
    logger.info(
        "Restricting to base panel: %d base genes kept, %d add-on genes dropped.",
        n_base, n_dropped,
    )
    return adata[:, is_base].copy()


def _get_counts(adata: ad.AnnData, layer: str) -> np.ndarray | sp.spmatrix:
    """Return the raw count matrix from a layer, falling back to .X."""
    if layer in adata.layers:
        return adata.layers[layer]
    logger.warning(
        "Layer '%s' not found; using adata.X for pseudobulk. If .X is "
        "log-normalised this will distort the aggregation.", layer,
    )
    return adata.X


def _as_dense(X) -> np.ndarray:
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def _condition_colours(uniq, named: dict, fallback: list) -> dict:
    """Map group labels to colours, preferring the pipeline's named palette."""
    out = {}
    for i, c in enumerate(uniq):
        out[c] = named.get(c, fallback[i % len(fallback)])
    return out


def _spearman_corr(X: np.ndarray) -> np.ndarray:
    """Spearman correlation = Pearson correlation of per-gene ranks."""
    from scipy.stats import rankdata
    ranks = np.vstack([rankdata(row) for row in X])
    return np.corrcoef(ranks)
