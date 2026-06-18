"""
leiden_optimizer.py
-------------------
Automated Leiden-resolution optimisation for the refactored Xenium pipeline.

The pseudobulk Sample-PCA step answers "do the samples separate by group?".
This module answers the *next* question — "at what resolution should the
cells be clustered?" — by sweeping a grid of Leiden resolutions and scoring
each one with five complementary cluster-quality metrics:

  * **Silhouette score**        — cluster separation in PCA space (higher = better)
  * **Calinski-Harabasz index** — between/within-cluster variance (higher = better)
  * **Davies-Bouldin index**    — avg similarity to nearest cluster (lower = better)
  * **Spatial coherence**       — fraction of spatial neighbours in the same
                                  cluster (higher = better; needs obsm['spatial'])
  * **Modularity**              — community structure on the KNN graph (higher = better)

A weighted combined score recommends the resolution that best balances
cluster quality and granularity.

Design notes
------------
The refactored pipeline loads cell-level data via
:class:`xenium_spatial.multislide_loader.MultiSlideLoader` but only ever pseudobulks it,
so the single-cell substrate the sweep needs (a PCA embedding and a KNN
neighbour graph) does not exist yet.  :func:`preprocess_for_clustering`
builds that substrate on the fly — normalise -> log1p -> PCA ->
``sc.pp.neighbors`` — keeping ``obsm['spatial']`` intact so spatial coherence
can be scored.  :func:`optimize_leiden_resolution` then runs the sweep.

Unlike :mod:`xenium_spatial.sample_pca`, this module depends on **scanpy** + **igraph** +
**leidenalg** (Leiden clustering and modularity are not in the minimal stack).
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


# ===========================================================================
# Cell-level preprocessing (build the PCA embedding + KNN graph for the sweep)
# ===========================================================================

def preprocess_for_clustering(
    adata: ad.AnnData,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    target_sum: float = 1e4,
    scale_genes: bool = False,
    batch_key: Optional[str] = None,
    counts_layer: str = "counts",
    random_state: int = 42,
) -> ad.AnnData:
    """
    Build the single-cell embedding + neighbour graph the sweep requires.

    The :class:`~xenium_spatial.multislide_loader.MultiSlideLoader` output carries only
    raw counts (in ``.X`` / ``layers['counts']``) and ``obsm['spatial']``.
    The Leiden sweep needs ``obsm['X_pca']`` and ``obsp['connectivities']``,
    so this runs the standard cell-level recipe:

        normalise_total -> log1p -> (optional z-score) -> PCA
          -> (optional Harmony) -> neighbours

    Raw counts are preserved in ``layers['counts']`` and ``obsm['spatial']``
    is carried through untouched.

    Parameters
    ----------
    adata
        Concatenated cell-level AnnData with raw counts.
    n_pcs
        Number of principal components.  Capped at ``min(n_obs, n_vars) - 1``.
    n_neighbors
        Neighbours for ``sc.pp.neighbors`` (the KNN graph scored by the sweep).
    target_sum
        Library-size normalisation target (counts-per-10k by default).
    scale_genes
        Z-score each gene before PCA.  Off by default — log1p already
        stabilises variance for the targeted Xenium panel.
    batch_key
        If given, run Harmony batch correction on the PCA embedding using this
        ``.obs`` column (e.g. ``'slide_id'``) and build the neighbour graph on
        the corrected ``X_pca_harmony`` instead of ``X_pca``.  Use this when
        pooling many slides so clusters reflect cell type rather than which
        slide a cell came from.  ``None`` (default) skips integration.
    counts_layer
        Layer holding the raw counts to normalise from.  Falls back to ``.X``.
    random_state
        Seed for PCA / Harmony / neighbour-graph reproducibility.

    Returns
    -------
    A new AnnData (the input is not modified) with ``obsm['X_pca']`` (and
    ``obsm['X_pca_harmony']`` when ``batch_key`` is set),
    ``obsp['connectivities']`` and the preserved ``obsm['spatial']``.
    """
    import scanpy as sc

    adata = adata.copy()

    # Start from raw counts so the recipe is well-defined regardless of what
    # an upstream step may have left in .X.
    if counts_layer in adata.layers:
        adata.X = adata.layers[counts_layer].copy()
    else:
        logger.warning(
            "Layer '%s' not found; normalising from .X. If .X is already "
            "log-normalised this will distort the embedding.", counts_layer,
        )
        adata.layers[counts_layer] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()

    if scale_genes:
        sc.pp.scale(adata, max_value=10)

    # PCA — cap components at the matrix rank so small ROIs / panels are safe.
    max_pcs = max(1, min(adata.n_obs, adata.n_vars) - 1)
    n_pcs = min(n_pcs, max_pcs)
    sc.pp.pca(adata, n_comps=n_pcs, random_state=random_state)

    # Record the elbow-plot recommendation for how many PCs to retain, so the
    # caller (and the Streamlit UI) can compare the requested ``n_pcs`` against
    # a data-driven estimate. This only annotates uns; it does not change the
    # embedding that was just built.
    try:
        elbow = compute_elbow_n_pcs(adata.uns["pca"]["variance"])
        adata.uns["pca_elbow"] = elbow
        logger.info(
            "Elbow metric: ~%d PCs capture the signal "
            "(cumulative>90%% & <5%% cutoff at PC%s; flattening cutoff at PC%s). "
            "Using n_pcs=%d for the neighbour graph.",
            elbow["n_pcs"], elbow["co1"], elbow["co2"], n_pcs,
        )
    except Exception as e:  # pragma: no cover - diagnostics only
        logger.debug("Elbow PC estimate skipped: %s", e)

    # Optional Harmony batch correction. The neighbour graph (and therefore the
    # clustering + every metric the sweep scores) is then built on the corrected
    # embedding so it reflects biology rather than batch.
    use_rep = "X_pca"
    if batch_key is not None:
        if batch_key not in adata.obs.columns:
            logger.warning(
                "batch_key='%s' not in adata.obs; skipping Harmony.", batch_key,
            )
        else:
            adata = run_harmony(adata, batch_key=batch_key, random_state=random_state)
            use_rep = "X_pca_harmony"

    # KNN graph — this is the graph the sweep evaluates for modularity and
    # the connectivities the Leiden algorithm clusters on.
    n_neighbors = min(n_neighbors, max(2, adata.n_obs - 1))
    sc.pp.neighbors(
        adata, n_neighbors=n_neighbors, n_pcs=n_pcs,
        use_rep=use_rep, random_state=random_state,
    )

    logger.info(
        "Preprocessed for clustering: %d cells x %d genes | PCA=%d comps, "
        "KNN k=%d on '%s'%s | spatial=%s",
        adata.n_obs, adata.n_vars, n_pcs, n_neighbors, use_rep,
        " (z-scored)" if scale_genes else "",
        "yes" if "spatial" in adata.obsm else "no",
    )
    return adata


# ===========================================================================
# PC selection (elbow plot)
# ===========================================================================

def compute_elbow_n_pcs(
    variance,
    cum_threshold: float = 90.0,
    pct_threshold: float = 5.0,
    change_threshold: float = 0.1,
) -> dict:
    """
    Recommend how many principal components to keep from a PCA variance curve.

    Implements the two-criterion elbow heuristic from the HBC scRNA-seq
    training (https://hbctraining.github.io/scRNA-seq/lessons/elbow_plot_metric.html),
    which picks the number of PCs quantitatively instead of reading the elbow
    off the plot by eye.  Both criteria work on the per-PC *standard deviation*
    expressed as a percentage of the total (``stdev / sum(stdev) * 100``), the
    same scale Seurat's ``ElbowPlot`` uses:

      * **co1** — the first PC at which the *cumulative* percentage exceeds
        ``cum_threshold`` (90%) while that PC individually contributes less
        than ``pct_threshold`` (5%).  Past this point extra PCs add little.
      * **co2** — the last PC where the drop in percentage to the next PC is
        still greater than ``change_threshold`` (0.1%); beyond it the scree
        curve has flattened into its tail.

    The recommendation is ``min(co1, co2)`` — the more conservative cutoff.

    Parameters
    ----------
    variance
        Per-PC variance (eigenvalues), ordered from PC1 downwards — e.g.
        ``adata.uns['pca']['variance']`` from scanpy.  Standard deviations are
        derived internally as ``sqrt(variance)``.
    cum_threshold, pct_threshold, change_threshold
        The 90% / 5% / 0.1% cutoffs above; exposed so the heuristic can be
        tuned for unusual panels.

    Returns
    -------
    dict with keys:
        ``n_pcs``      recommended number of PCs (int, 1-based count),
        ``co1``        the cumulative/individual cutoff PC (int or None),
        ``co2``        the flattening cutoff PC (int or None),
        ``pct``        per-PC percentage of total stdev (list),
        ``cumulative`` cumulative percentage (list).
    """
    var = np.asarray(variance, dtype=np.float64).ravel()
    if var.size == 0:
        raise ValueError("variance is empty; run PCA before estimating PCs.")

    stdev = np.sqrt(np.clip(var, 0.0, None))
    total = stdev.sum()
    if total <= 0:
        raise ValueError("All PCs have zero variance; cannot estimate PCs.")

    pct = stdev / total * 100.0
    cumu = np.cumsum(pct)
    n = pct.size

    # co1: first PC past 90% cumulative whose own contribution is < 5%.
    co1_hits = np.where((cumu > cum_threshold) & (pct < pct_threshold))[0]
    co1 = int(co1_hits[0] + 1) if co1_hits.size else n

    # co2: last PC whose drop to the next PC still exceeds 0.1%.
    if n > 1:
        drops = pct[:-1] - pct[1:]
        co2_hits = np.where(drops > change_threshold)[0]
        co2 = int(co2_hits[-1] + 2) if co2_hits.size else 1
    else:
        co2 = 1

    n_pcs = int(min(co1, co2))
    n_pcs = max(1, min(n_pcs, n))

    return {
        "n_pcs": n_pcs,
        "co1": co1,
        "co2": co2,
        "pct": pct.tolist(),
        "cumulative": cumu.tolist(),
    }


def plot_pca_elbow(
    adata: ad.AnnData,
    output_dir=None,
    fmt: str = "pdf",
    dpi: int = 300,
    elbow: Optional[dict] = None,
):
    """
    Elbow / scree plot of per-PC variance with the recommended cutoff marked.

    Plots the percentage of standard deviation explained by each PC (the
    Seurat-style elbow) and draws a vertical line at the number of PCs
    recommended by :func:`compute_elbow_n_pcs`.  Returns the saved path when
    ``output_dir`` is given, otherwise the matplotlib ``Figure`` so a caller
    (e.g. the Streamlit page) can display it directly.

    ``adata`` must already carry a PCA (``adata.uns['pca']['variance']``), as
    produced by :func:`preprocess_for_clustering`.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if "pca" not in adata.uns or "variance" not in adata.uns["pca"]:
        raise KeyError(
            "adata.uns['pca']['variance'] missing. "
            "Run preprocess_for_clustering() (or sc.pp.pca) first."
        )

    if elbow is None:
        elbow = adata.uns.get("pca_elbow") or compute_elbow_n_pcs(
            adata.uns["pca"]["variance"]
        )

    pct = np.asarray(elbow["pct"])
    pcs = np.arange(1, len(pct) + 1)
    n_rec = int(elbow["n_pcs"])

    mpl.rcParams.update({
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(pcs, pct, "-o", color="#1B4F8A", ms=3, lw=0.9, zorder=3)
    ax.axvline(n_rec, color="#D55E00", ls="--", lw=0.9, zorder=2,
               label=f"recommended: {n_rec} PCs")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Std. dev. explained (%)")
    ax.set_title("PCA elbow plot")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    if output_dir is None:
        return fig

    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"pca_elbow.{fmt}"
    fig.savefig(path, format=fmt, dpi=dpi)
    plt.close(fig)
    logger.info("Saved PCA elbow plot: %s", path)
    return path


# ===========================================================================
# Harmony batch correction
# ===========================================================================

def run_harmony(
    adata: ad.AnnData,
    batch_key: str = "slide_id",
    max_iter: int = 30,
    random_state: int = 42,
) -> ad.AnnData:
    """
    Correct batch effects with Harmony on the PCA embedding.

    Requires ``harmonypy`` to be installed.  Stores the corrected embedding in
    ``.obsm['X_pca_harmony']``.

    Parameters
    ----------
    adata
        AnnData with ``.obsm['X_pca']`` already computed.
    batch_key
        Column in ``.obs`` distinguishing technical batches.  Use ``'slide_id'``
        for a multi-slide study — using ``'condition'`` here would remove the
        biological effect you are trying to detect.
    max_iter
        Maximum number of Harmony iterations.
    random_state
        Seed for reproducibility.
    """
    n_batches = adata.obs[batch_key].nunique()
    if n_batches < 2:
        logger.warning(
            "Harmony skipped: only %d unique value(s) for batch_key='%s'. "
            "Copying X_pca to X_pca_harmony unchanged.",
            n_batches, batch_key,
        )
        adata.obsm["X_pca_harmony"] = adata.obsm["X_pca"].copy()
        return adata

    # Warn when each batch is a separate biological replicate: Harmony will then
    # correct between-replicate variation, partially removing biological signal
    # (Korsunsky et al. 2019, Tran et al. 2020).
    if "condition" in adata.obs.columns:
        batches_per_cond = adata.obs.groupby("condition", observed=True)[batch_key].nunique()
        if batches_per_cond.sum() == n_batches:
            logger.warning(
                "Harmony: batch_key='%s' has %d unique values across %d conditions "
                "(%s). Each batch is a separate biological replicate — Harmony will "
                "correct between-replicate variation, risking partial removal of "
                "true biological signal. Verify clusters still separate conditions.",
                batch_key, n_batches, adata.obs["condition"].nunique(),
                ", ".join(f"{c}: {n}" for c, n in batches_per_cond.items()),
            )

    # Fail fast if harmonypy is absent, before touching any data.
    try:
        import harmonypy as _hm_check  # noqa: F401
    except ImportError:
        try:
            import scanpy.external as _sce_check  # noqa: F401
        except (ImportError, AttributeError):
            raise ImportError(
                "harmonypy is required for batch correction. "
                "Install with: pip install harmonypy"
            )

    logger.info("Running Harmony integration on key '%s' (%d batches) …", batch_key, n_batches)

    # The scanpy harmony wrapper has a known shape-mismatch bug in some
    # scanpy/harmonypy combinations (stores the transposed embedding). Probe it
    # on a tiny subsample first; if it misbehaves, fall through to direct harmonypy.
    _wrapper_ok = False
    try:
        import scanpy.external as sce
        _n_probe = min(20, adata.n_obs)
        _probe = adata[:_n_probe].copy()
        sce.pp.harmony_integrate(
            _probe, key=batch_key, basis="X_pca", adjusted_basis="X_pca_harmony",
            max_iter_harmony=1, random_state=random_state, verbose=False,
        )
        if ("X_pca_harmony" in _probe.obsm
                and _probe.obsm["X_pca_harmony"].shape[0] == _n_probe):
            _wrapper_ok = True
        del _probe
    except Exception:
        pass

    if _wrapper_ok:
        sce.pp.harmony_integrate(
            adata, key=batch_key, basis="X_pca", adjusted_basis="X_pca_harmony",
            max_iter_harmony=max_iter, random_state=random_state,
        )
        logger.info("Harmony complete via scanpy wrapper. Shape: %s",
                    adata.obsm["X_pca_harmony"].shape)
    else:
        logger.debug("scanpy harmony wrapper unavailable/shape mismatch; using harmonypy directly.")
        try:
            import harmonypy as hm
        except ImportError:
            raise ImportError("harmonypy is required. Install with: pip install harmonypy")

        import logging as _logging
        _logging.getLogger("harmonypy").setLevel(_logging.WARNING)

        pca_mat = adata.obsm["X_pca"].copy()
        meta = adata.obs[[batch_key]].copy()
        ho = hm.run_harmony(
            pca_mat, meta, batch_key,
            max_iter_harmony=max_iter, random_state=random_state, verbose=False,
        )
        n_cells, n_pcs = pca_mat.shape
        Z = None
        for attr in ["Z_corr", "result", "embedding"]:
            candidate = getattr(ho, attr, None)
            if candidate is not None:
                Z = np.array(candidate)
                break
        if Z is None:
            raise AttributeError(
                "harmonypy result has no recognised embedding attribute. "
                f"Available attrs: {[a for a in dir(ho) if not a.startswith('_')]}"
            )
        if Z.ndim == 1:
            raise ValueError(
                f"harmonypy returned a 1D array of shape {Z.shape}. "
                "Please upgrade harmonypy: pip install --upgrade harmonypy"
            )
        if Z.shape == (n_pcs, n_cells):
            Z = Z.T   # old API: (n_pcs, n_cells) -> (n_cells, n_pcs)
        elif Z.shape != (n_cells, n_pcs):
            raise ValueError(
                f"harmonypy returned unexpected shape {Z.shape}; "
                f"expected ({n_cells}, {n_pcs}) or ({n_pcs}, {n_cells})."
            )
        adata.obsm["X_pca_harmony"] = Z.astype(np.float32)
        logger.info("Harmony complete via direct harmonypy. Shape: %s",
                    adata.obsm["X_pca_harmony"].shape)
    return adata


# ===========================================================================
# Resolution sweep + multi-metric scoring
# ===========================================================================

def optimize_leiden_resolution(
    adata: ad.AnnData,
    resolutions: Optional[list[float]] = None,
    random_state: int = 42,
    use_rep: Optional[str] = None,
    n_sample: int = 50_000,
    callback=None,
) -> dict:
    """
    Sweep Leiden resolutions and score each with multiple cluster quality
    metrics to recommend the best resolution.

    Metrics computed at each resolution:

    * **Silhouette score** — measures how similar each cell is to its own
      cluster versus the nearest neighbouring cluster in PCA/latent space.
      Range [-1, 1]; higher = better separated clusters.
    * **Calinski-Harabasz index** (variance ratio criterion) — ratio of
      between-cluster to within-cluster dispersion.  Higher = more compact
      and well-separated clusters.
    * **Davies-Bouldin index** — average similarity between each cluster and
      its most similar one.  Lower = better.
    * **Modularity** — community structure quality on the KNN graph.
    * **Spatial coherence** — fraction of each cell's spatial neighbours
      that belong to the same cluster (requires ``adata.obsm['spatial']``).
      Higher = more spatially contiguous clusters.

    Additionally, cluster assignments at every resolution are stored so that
    a **clustree** plot can be generated downstream.

    Parameters
    ----------
    adata
        Pre-processed AnnData with a neighbour graph already computed
        (i.e. ``adata.obsp['connectivities']`` exists).
    resolutions
        List of resolution values to test.  Defaults to a fine grid
        from 0.1 to 2.0.
    random_state
        Random seed for reproducibility.
    use_rep
        Representation in ``adata.obsm`` for silhouette / CH / DB scores
        (e.g. ``'X_pca'``, ``'X_pca_harmony'``).  Auto-detected if *None*.
    n_sample
        Max cells to subsample for silhouette score (expensive at O(n²)).
        Calinski-Harabasz and Davies-Bouldin are also evaluated on the
        subsample for consistency.  Set to 0 to use all cells.
    callback
        Optional ``callback(step, total, resolution, metrics_dict)``
        called after each resolution is evaluated — useful for progress
        bars in the Streamlit UI.

    Returns
    -------
    dict with keys:
        ``"results"``   – :class:`pandas.DataFrame` with columns
            ``resolution``, ``n_clusters``, ``silhouette``,
            ``calinski_harabasz``, ``davies_bouldin``,
            ``spatial_coherence``, ``modularity``, ``combined_score``.
        ``"best_resolution"`` – float, the resolution with the highest
            combined score.
        ``"best_row"``  – dict of the best row.
        ``"cluster_assignments"`` – :class:`pandas.DataFrame` with one
            column per resolution (``leiden_0.10``, ``leiden_0.20``, …)
            and one row per cell.  Used for clustree visualisation.
    """
    import scanpy as sc
    from sklearn.metrics import (
        silhouette_score as _silhouette_score,
        calinski_harabasz_score as _ch_score,
        davies_bouldin_score as _db_score,
    )

    if "connectivities" not in adata.obsp:
        raise ValueError(
            "adata.obsp['connectivities'] is missing. Run "
            "preprocess_for_clustering() (or sc.pp.neighbors) first."
        )

    if resolutions is None:
        resolutions = [round(r, 2) for r in np.arange(0.1, 2.05, 0.1)]

    # Auto-detect embedding for cluster quality metrics
    if use_rep is None:
        for candidate in ("X_pca_harmony", "X_pca"):
            if candidate in adata.obsm:
                use_rep = candidate
                break
        if use_rep is None:
            raise ValueError(
                "No PCA embedding found in adata.obsm. "
                "Run PCA (and optionally Harmony) before resolution optimisation."
            )

    # Detect spatial coordinates for spatial coherence
    _has_spatial = "spatial" in adata.obsm
    _spatial_k = 15  # neighbours for spatial coherence
    if _has_spatial:
        from scipy.spatial import cKDTree
        xy = adata.obsm["spatial"].astype(np.float64)
        _spatial_k = min(_spatial_k, adata.n_obs - 1)
        _, _spatial_nbr_idx = cKDTree(xy).query(xy, k=_spatial_k + 1)
        _spatial_nbr_idx = _spatial_nbr_idx[:, 1:]  # exclude self
        logger.info(
            "Spatial coherence enabled: k=%d spatial neighbours", _spatial_k,
        )
    else:
        logger.info(
            "No spatial coordinates found (obsm['spatial']); "
            "spatial coherence will be reported as NaN."
        )

    logger.info(
        "Leiden resolution sweep: %d resolutions (%.2f – %.2f), "
        "metrics on '%s', %s cells",
        len(resolutions), min(resolutions), max(resolutions),
        use_rep, f"subsampled to {n_sample}" if 0 < n_sample < adata.n_obs else "all",
    )

    # Subsample indices once for consistent metric evaluation
    if 0 < n_sample < adata.n_obs:
        rng = np.random.RandomState(random_state)
        sample_idx = rng.choice(adata.n_obs, size=n_sample, replace=False)
    else:
        sample_idx = np.arange(adata.n_obs)

    embedding = adata.obsm[use_rep][sample_idx]

    # Retrieve the adjacency / connectivity graph for modularity
    try:
        import igraph as ig
        import scipy.sparse as sp
        adj = adata.obsp["connectivities"]
        # Convert sparse matrix to igraph via edge list (memory-efficient)
        if sp.issparse(adj):
            coo = adj.tocoo()
            edges = list(zip(coo.row.tolist(), coo.col.tolist()))
            weights = coo.data.tolist()
            g = ig.Graph(n=adj.shape[0], edges=edges, directed=False)
            g.es["weight"] = weights
            # Remove self-loops and duplicate edges from symmetry
            g.simplify(combine_edges="first")
        else:
            g = ig.Graph.Weighted_Adjacency(adj.tolist(), mode="undirected")
        _has_igraph = True
    except (ImportError, KeyError):
        _has_igraph = False

    rows: list[dict] = []
    cluster_cols: dict[str, np.ndarray] = {}  # for clustree
    # The sweep mutates `adata` in place: each iteration writes Leiden labels to
    # this temp obs column (removed after the loop) and scanpy stamps
    # adata.uns['leiden']. Callers that pass a shared/cached object (e.g. the
    # Streamlit page's @st.cache_resource embedding) get it back clean, but
    # should not rely on it being untouched mid-sweep.
    tmp_key = f"_leiden_opt_{random_state}"

    for step_i, res in enumerate(resolutions):
        # Run Leiden at this resolution
        try:
            sc.tl.leiden(
                adata, resolution=res, key_added=tmp_key,
                random_state=random_state, flavor="igraph",
                n_iterations=2, directed=False,
            )
        except TypeError:
            sc.tl.leiden(
                adata, resolution=res, key_added=tmp_key,
                random_state=random_state,
            )

        labels = adata.obs[tmp_key].astype("category")
        n_clusters = labels.nunique()
        labels_int = labels.cat.codes.values  # integer codes for sklearn

        # Store cluster assignments for clustree
        col_name = f"leiden_{res:.2f}"
        cluster_cols[col_name] = labels.values.copy()

        # --- Silhouette score (on subsample) ---
        labels_sub = labels_int[sample_idx]
        # The cluster-quality metrics need >=2 distinct labels *in the
        # subsample*. n_clusters counts clusters on the full data, so an extreme
        # imbalance can leave the subsample single-label even when n_clusters>=2;
        # sklearn then raises. Guard on the subsample and degrade to NaN rather
        # than aborting the whole sweep.
        n_sub_labels = len(np.unique(labels_sub))
        if n_clusters < 2 or n_sub_labels < 2:
            sil = -1.0
            ch = 0.0
            db = float("nan")
        else:
            try:
                sil = float(_silhouette_score(
                    embedding, labels_sub, metric="euclidean", sample_size=None,
                ))
                # --- Calinski-Harabasz index (on subsample) ---
                ch = float(_ch_score(embedding, labels_sub))
                # --- Davies-Bouldin index (on subsample) ---
                db = float(_db_score(embedding, labels_sub))
            except ValueError as e:
                logger.warning(
                    "Metric computation failed at res=%.2f (%s); recording NaN.",
                    res, e,
                )
                sil, ch, db = -1.0, 0.0, float("nan")

        # --- Spatial coherence ---
        if _has_spatial and n_clusters >= 2:
            # For each cell, fraction of spatial neighbours in the same cluster
            labels_arr = labels_int
            nbr_labels = labels_arr[_spatial_nbr_idx]  # (n_cells, k)
            same_cluster = (nbr_labels == labels_arr[:, None]).mean(axis=1)
            spatial_coh = float(same_cluster.mean())
        else:
            spatial_coh = float("nan")

        # --- Modularity (on full graph) ---
        if _has_igraph and n_clusters >= 2:
            membership = [int(x) for x in labels.values]
            mod = float(g.modularity(membership, weights="weight"))
        else:
            mod = 0.0

        row = {
            "resolution": res,
            "n_clusters": n_clusters,
            "silhouette": round(sil, 4),
            "calinski_harabasz": round(ch, 2),
            "davies_bouldin": round(db, 4) if not np.isnan(db) else float("nan"),
            "spatial_coherence": round(spatial_coh, 4) if not np.isnan(spatial_coh) else float("nan"),
            "modularity": round(mod, 4),
        }
        rows.append(row)
        logger.info(
            "  res=%.2f  clusters=%d  sil=%.4f  CH=%.1f  DB=%.4f  "
            "spatial_coh=%.4f  mod=%.4f",
            res, n_clusters, sil, ch,
            db if not np.isnan(db) else 0.0,
            spatial_coh if not np.isnan(spatial_coh) else 0.0,
            mod,
        )

        if callback is not None:
            callback(step_i + 1, len(resolutions), res, row)

    # Clean up temporary obs column
    if tmp_key in adata.obs.columns:
        del adata.obs[tmp_key]

    df = pd.DataFrame(rows)

    # --- Combined score ---
    # Normalise each metric to [0, 1] then take weighted average.
    # Silhouette [-1, 1]: higher is better
    # Calinski-Harabasz [0, ∞): higher is better
    # Davies-Bouldin [0, ∞): LOWER is better → invert
    # Spatial coherence [0, 1]: higher is better (may be NaN)
    # Modularity [−0.5, 1]: higher is better

    def _norm_col(s: pd.Series, invert: bool = False) -> pd.Series:
        """Min-max normalise to [0, 1]; if invert, flip so lower raw = higher norm.

        NaN values are filled with 0.5 (neutral) after normalisation so they
        don't dominate or crash the combined score.
        """
        s = s.copy()
        s_min, s_max = s.dropna().min(), s.dropna().max()
        rng = s_max - s_min if s_max > s_min else 1.0
        normed = (s - s_min) / rng
        normed = normed.fillna(0.5)  # neutral score for NaN entries
        return (1.0 - normed) if invert else normed

    sil_norm = _norm_col(df["silhouette"])
    ch_norm = _norm_col(df["calinski_harabasz"])
    db_norm = _norm_col(df["davies_bouldin"], invert=True)
    mod_norm = _norm_col(df["modularity"])

    # Use the spatial-aware weighting whenever spatial coordinates exist and at
    # least one resolution produced a score. Gating on ``.all()`` instead would
    # let a single degenerate resolution (n_clusters<2 -> NaN coherence)
    # silently flip the whole sweep to the non-spatial weights; the NaN rows are
    # already neutralised by _norm_col's fillna(0.5).
    has_spatial_scores = _has_spatial and df["spatial_coherence"].notna().any()
    if has_spatial_scores:
        sc_norm = _norm_col(df["spatial_coherence"])
        # Weights: silhouette 30%, CH 15%, DB 15%, spatial coherence 20%, modularity 20%
        df["combined_score"] = (
            0.30 * sil_norm
            + 0.15 * ch_norm
            + 0.15 * db_norm
            + 0.20 * sc_norm
            + 0.20 * mod_norm
        ).round(4)
    else:
        # No spatial data — fall back to non-spatial weights
        # Weights: silhouette 35%, CH 15%, DB 15%, modularity 35%
        df["combined_score"] = (
            0.35 * sil_norm
            + 0.15 * ch_norm
            + 0.15 * db_norm
            + 0.35 * mod_norm
        ).round(4)

    best_idx = int(df["combined_score"].idxmax())
    best_row = df.iloc[best_idx]
    best_res = float(best_row["resolution"])

    logger.info(
        "Optimal Leiden resolution: %.2f  (clusters=%d, sil=%.4f, "
        "CH=%.1f, DB=%.4f, spatial_coh=%.4f, mod=%.4f, combined=%.4f)",
        best_res, int(best_row["n_clusters"]),
        best_row["silhouette"], best_row["calinski_harabasz"],
        best_row["davies_bouldin"], best_row.get("spatial_coherence", 0.0),
        best_row["modularity"], best_row["combined_score"],
    )

    # Build clustree assignment DataFrame
    cluster_df = pd.DataFrame(cluster_cols, index=adata.obs_names)

    return {
        "results": df,
        "best_resolution": best_res,
        "best_row": df.iloc[best_idx].to_dict(),
        "cluster_assignments": cluster_df,
    }
