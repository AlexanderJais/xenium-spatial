"""
cell_clustering.py
------------------
Cell-level clustering products built on top of the optimizer's embedding:
a UMAP, final Leiden labels at a chosen resolution, marker genes per cluster,
and a cluster -> cell-type annotation layer.

The heavy single-cell dependency (scanpy) is imported lazily inside each
function so importing this module stays cheap for pages that only read the
persisted artifacts.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

logger = logging.getLogger(__name__)


# ── Artifact paths ─────────────────────────────────────────────────────────
def clustering_dir(output_dir) -> Path:
    d = Path(output_dir) / "clustering"
    d.mkdir(parents=True, exist_ok=True)
    return d


def clustered_h5ad_path(output_dir) -> Path:
    return clustering_dir(output_dir) / "clustered.h5ad"


def annotations_path(output_dir) -> Path:
    return clustering_dir(output_dir) / "annotations.json"


# ── Build ──────────────────────────────────────────────────────────────────
def build_clustered_adata(adata: ad.AnnData, resolution: float,
                          random_state: int = 42) -> ad.AnnData:
    """Add a UMAP and final Leiden labels (at ``resolution``) to a preprocessed
    AnnData that already carries a neighbour graph (from
    ``preprocess_for_clustering``).

    The input is copied, so a cached embedding passed in is not mutated.
    Returns the copy with ``obsm['X_umap']`` and a categorical ``obs['leiden']``.
    """
    import scanpy as sc

    adata = adata.copy()
    sc.tl.umap(adata, random_state=random_state)
    # Mirror the optimizer's Leiden call (flavor/fallback) so labels match the
    # sweep's clustering at the same resolution.
    try:
        sc.tl.leiden(adata, resolution=resolution, key_added="leiden",
                     random_state=random_state, flavor="igraph",
                     n_iterations=2, directed=False)
    except TypeError:
        sc.tl.leiden(adata, resolution=resolution, key_added="leiden",
                     random_state=random_state)
    adata.obs["leiden"] = adata.obs["leiden"].astype("category")
    logger.info("Clustered: %d cells, %d Leiden clusters at res=%.2f; UMAP built.",
                adata.n_obs, adata.obs["leiden"].nunique(), resolution)
    return adata


# ── Markers ────────────────────────────────────────────────────────────────
def rank_markers(adata: ad.AnnData, groupby: str = "leiden",
                 n_genes: int = 25, layer: str = "lognorm") -> pd.DataFrame:
    """Wilcoxon rank-genes-groups per cluster on log-normalised expression.

    Returns a tidy DataFrame: ``cluster, rank, gene, log2fc, score, pval_adj``.
    """
    import scanpy as sc

    use_layer = layer if layer in adata.layers else None
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon",
                            layer=use_layer, use_raw=False)
    res = adata.uns["rank_genes_groups"]
    groups = list(res["names"].dtype.names)
    rows = []
    for g in groups:
        k = min(n_genes, len(res["names"][g]))
        for i in range(k):
            rows.append({
                "cluster" : g,
                "rank"    : i + 1,
                "gene"    : str(res["names"][g][i]),
                "log2fc"  : float(res["logfoldchanges"][g][i]),
                "score"   : float(res["scores"][g][i]),
                "pval_adj": float(res["pvals_adj"][g][i]),
            })
    return pd.DataFrame(rows)


def top_markers_by_cluster(markers: pd.DataFrame, n: int = 8) -> dict:
    """{cluster -> ['GeneA', 'GeneB', ...]} of the top ``n`` markers, for
    annotation hints."""
    out: dict[str, list[str]] = {}
    for cl, sub in markers.groupby("cluster"):
        out[str(cl)] = sub.sort_values("rank")["gene"].head(n).tolist()
    return out


# ── Composition (used by the annotation table now; composition page later) ──
def cluster_composition(adata: ad.AnnData, cluster_key: str = "leiden",
                        condition_key: str = "condition") -> pd.DataFrame:
    """Per-cluster cell counts split by condition (wide), plus an ``n_cells``
    total column, ordered by cluster id."""
    obs = adata.obs
    if condition_key in obs.columns:
        comp = (obs.groupby([cluster_key, condition_key], observed=True).size()
                   .unstack(fill_value=0))
    else:
        comp = obs.groupby([cluster_key], observed=True).size().to_frame("count")
    comp["n_cells"] = comp.sum(axis=1)
    # Sort clusters numerically when possible.
    try:
        comp = comp.reindex(sorted(comp.index, key=lambda x: int(x)))
    except (TypeError, ValueError):
        pass
    return comp


# ── Annotation ─────────────────────────────────────────────────────────────
def load_annotations(output_dir) -> dict:
    p = annotations_path(output_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_annotations(output_dir, mapping: dict) -> Path:
    p = annotations_path(output_dir)
    p.write_text(json.dumps(mapping, indent=2))
    return p


def apply_annotations(adata: ad.AnnData, mapping: dict,
                      cluster_key: str = "leiden") -> ad.AnnData:
    """Add ``obs['cell_type']`` from a {cluster_id -> label} mapping; unlabelled
    clusters keep their cluster id."""
    labels = adata.obs[cluster_key].astype(str)
    adata.obs["cell_type"] = labels.map(lambda c: mapping.get(c) or c).astype("category")
    return adata
