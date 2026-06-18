"""
pseudobulk_dge.py
-----------------
Differential expression between conditions, within a cell type, done the
statistically honest way for replicated single-cell / spatial data:

  cells of a cell type  ->  sum counts per biological replicate (pseudobulk)
  ->  CPM + log2  ->  per-gene test across the replicate-level values.

This avoids pseudoreplication (treating cells as replicates). It reuses the
Sample-PCA ``pseudobulk_samples`` aggregator. With the usual 2-vs-2 design the
per-gene Welch t-test is underpowered — rank by effect size and validate; a
proper count model (DESeq2/edgeR) is the gold standard for publication, but is
intentionally out of scope for this Python-only tool.

numpy / pandas / scipy only (no scanpy at import).
"""
import logging
import collections

import numpy as np
import pandas as pd

from .composition import _bh_adjust

logger = logging.getLogger(__name__)


def dge_for_celltype(adata, cell_type, group_key="cell_type", sample_key="replicate",
                     condition_key="condition", min_cpm: float = 1.0,
                     min_samples: int | None = None):
    """Pseudobulk DE for one ``cell_type``.

    Returns ``(DataFrame, error)``. On success ``error`` is None and the frame
    has: ``gene``, ``<condA>_mean``, ``<condB>_mean`` (log2 CPM), ``log2fc``
    (condB vs condA, i.e. second vs first alphabetically), ``pval``, ``padj``,
    ``base_log2cpm``. On failure the frame is None and ``error`` explains why.
    """
    from .sample_pca import pseudobulk_samples

    mask = (adata.obs[group_key].astype(str) == str(cell_type)).values
    if mask.sum() == 0:
        return None, "no cells of this type"
    sub = adata[mask].copy()

    pb = pseudobulk_samples(sub, sample_key=sample_key, condition_key=condition_key)
    counts = np.asarray(pb.X, dtype=float)            # samples x genes
    meta = pb.obs[condition_key].astype(str).values if condition_key in pb.obs else None
    genes = np.asarray(pb.var_names)

    if meta is None:
        return None, "no condition labels"
    conds = sorted(pd.unique(meta))
    if len(conds) != 2:
        return None, f"need exactly 2 conditions present (have {list(conds)})"
    n_per = collections.Counter(meta)
    if min(n_per[conds[0]], n_per[conds[1]]) < 2:
        return None, (f"need ≥2 replicates per condition with cells of this type "
                      f"(have {dict(n_per)})")

    # CPM + log2 per pseudobulk sample.
    lib = counts.sum(axis=1, keepdims=True)
    cpm = counts / np.clip(lib, 1.0, None) * 1e6
    logcpm = np.log2(cpm + 1.0)

    # Expression filter on CPM, not raw counts: pseudobulk libraries vary by
    # orders of magnitude (cells per replicate differ), so a raw-count mean is
    # dominated by the largest-library sample and would admit genes detectable
    # in only one sample. Require detection (CPM ≥ min_cpm) in enough samples
    # that a single sample can't carry a gene.
    n_samples = counts.shape[0]
    if min_samples is None:
        min_samples = max(2, n_samples // 2)
    keep = (cpm >= min_cpm).sum(axis=0) >= min_samples
    if keep.sum() == 0:
        return None, "no genes pass the expression filter (CPM detection)"

    a_idx = meta == conds[0]
    b_idx = meta == conds[1]
    from scipy import stats

    rows = []
    for j in np.where(keep)[0]:
        a = logcpm[a_idx, j]
        b = logcpm[b_idx, j]
        with np.errstate(all="ignore"):
            try:
                p = float(stats.ttest_ind(b, a, equal_var=False).pvalue)
            except Exception:
                p = np.nan
        rows.append({
            "gene"            : str(genes[j]),
            f"{conds[0]}_mean": float(a.mean()),
            f"{conds[1]}_mean": float(b.mean()),
            "base_log2cpm"    : float(logcpm[:, j].mean()),
            "log2fc"          : float(b.mean() - a.mean()),
            "pval"            : p,
        })

    df = pd.DataFrame(rows)
    df["padj"] = _bh_adjust(df["pval"].to_numpy())
    df["direction"] = f"{conds[1]} vs {conds[0]}"
    return df.sort_values("pval", na_position="last").reset_index(drop=True), None


def all_dge(adata, group_key="cell_type", sample_key="replicate",
            condition_key="condition"):
    """Run DGE for every cell type once (threshold-independent), so the caller
    can cache this and count DE genes at any threshold cheaply.

    Returns ``(results, notes)``: ``results`` is a tidy DataFrame of every tested
    gene across all testable cell types (with a ``cell_type`` column), and
    ``notes`` is one row per cell type with ``n_cells`` and a ``note`` (empty if
    tested, otherwise the reason it was skipped).
    """
    groups = adata.obs[group_key].astype(str)
    frames, notes = [], []
    for ct in sorted(groups.unique(), key=lambda x: (len(x), x)):
        n_cells = int((groups == ct).sum())
        df, err = dge_for_celltype(adata, ct, group_key, sample_key, condition_key)
        if err:
            notes.append({group_key: ct, "n_cells": n_cells, "note": err})
            continue
        df = df.copy()
        df.insert(0, group_key, ct)
        frames.append(df)
        notes.append({group_key: ct, "n_cells": n_cells, "note": ""})
    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return results, pd.DataFrame(notes)
