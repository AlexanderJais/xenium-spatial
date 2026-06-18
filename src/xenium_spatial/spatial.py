"""
spatial.py
----------
Spatial readouts that use the Xenium cell coordinates (``obsm['spatial']``),
which the clustering and DGE steps ignore:

  * a tidy frame for per-slide cell-type maps, and
  * neighbourhood enrichment — for each pair of cell types, are they spatial
    neighbours more (or less) often than expected by chance? (the Palla et al.
    2022 / squidpy ``nhood_enrichment`` permutation z-score, reimplemented with
    scikit-learn so squidpy is not a dependency).

Permutations are done **within each slide** so the real spatial structure is
preserved and only the labels are shuffled.

numpy / pandas / scikit-learn only.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def spatial_frame(adata, group_key: str = "cell_type", slide_key: str = "slide_id",
                  condition_key: str = "condition") -> pd.DataFrame:
    """Tidy frame (x, y, group, slide, condition) for per-slide spatial scatters."""
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] missing — rebuild the clustering.")
    xy = np.asarray(adata.obsm["spatial"])
    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})
    df[group_key] = adata.obs[group_key].astype(str).values
    if slide_key in adata.obs:
        df["slide"] = adata.obs[slide_key].astype(str).values
    if condition_key in adata.obs:
        df["condition"] = adata.obs[condition_key].astype(str).values
    return df


def neighborhood_enrichment(adata, group_key: str = "cell_type",
                            slide_key: str = "slide_id", n_neighbors: int = 6,
                            n_perms: int = 100, seed: int = 0) -> pd.DataFrame:
    """Neighbourhood-enrichment z-scores between cell types.

    Builds a spatial kNN graph per slide, counts observed neighbour pairs per
    (type, type), then compares against ``n_perms`` within-slide label
    permutations. Returns a symmetric ``types × types`` DataFrame of z-scores:
    positive = the two types are neighbours more than chance (co-localised),
    negative = they avoid each other.
    """
    from sklearn.neighbors import NearestNeighbors

    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] missing — rebuild the clustering.")

    rng = np.random.default_rng(seed)
    coords_all = np.asarray(adata.obsm["spatial"])
    labels_all = adata.obs[group_key].astype(str).values
    cats = sorted(pd.unique(labels_all))
    cat_idx = {c: i for i, c in enumerate(cats)}
    K = len(cats)
    if K < 2:
        return pd.DataFrame(np.zeros((K, K)), index=cats, columns=cats)

    slides = (adata.obs[slide_key].astype(str).values
              if slide_key in adata.obs else np.array(["all"] * adata.n_obs))

    # Per-slide neighbour edges + integer labels (built once, reused per perm).
    per_slide = []
    for s in pd.unique(slides):
        m = slides == s
        coords = coords_all[m]
        n = coords.shape[0]
        if n <= n_neighbors:
            continue
        k = min(n_neighbors, n - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        _, idx = nn.kneighbors(coords)
        src = np.repeat(np.arange(n), k)
        dst = idx[:, 1:].ravel()            # drop self-neighbour
        lab = np.array([cat_idx[c] for c in labels_all[m]])
        per_slide.append((src, dst, lab))

    if not per_slide:
        return pd.DataFrame(np.full((K, K), np.nan), index=cats, columns=cats)

    def _counts(label_arrays):
        M = np.zeros((K, K), dtype=float)
        for (src, dst, _), lab in zip(per_slide, label_arrays):
            flat = lab[src] * K + lab[dst]
            M += np.bincount(flat, minlength=K * K).reshape(K, K)
        return M + M.T  # symmetrise (undirected co-occurrence)

    observed = _counts([lab for _, _, lab in per_slide])
    perms = np.empty((n_perms, K, K))
    for p in range(n_perms):
        shuffled = [rng.permutation(lab) for _, _, lab in per_slide]
        perms[p] = _counts(shuffled)

    mean = perms.mean(axis=0)
    std = perms.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (observed - mean) / std
    z[~np.isfinite(z)] = 0.0
    return pd.DataFrame(z, index=cats, columns=cats)
