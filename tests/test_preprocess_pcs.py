"""
Regression tests for ``preprocess_for_clustering``'s PCA / elbow handling.

The elbow recommendation logged during the sweep used to be computed from a
PCA truncated to the *requested* ``n_pcs``. That truncated the variance curve
so ``compute_elbow_n_pcs`` could never recommend more components than were
asked for — a self-reinforcing trap that collapsed to a degenerate handful of
PCs once a small ``n_pcs`` was chosen. These tests pin the fixed behaviour:
the recommendation is data-driven (independent of ``n_pcs``), while the
embedding still carries exactly the requested number of components.
"""

import numpy as np
import pytest


def _synthetic_counts_adata(n_cells=300, n_genes=60, n_factors=8, seed=0):
    """A low-rank-plus-noise count matrix: a handful of informative PCs
    followed by a noise floor, so the elbow sits well above 2."""
    anndata = pytest.importorskip("anndata")
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n_factors, n_genes))
    scores = rng.normal(size=(n_cells, n_factors)) * np.linspace(6.0, 1.5, n_factors)
    signal = scores @ loadings
    rate = np.exp((signal - signal.mean()) / (signal.std() + 1e-9))
    counts = rng.poisson(rate * 2.0 + 0.5).astype("float32")
    return anndata.AnnData(counts)


def test_elbow_recommendation_independent_of_requested_n_pcs():
    pytest.importorskip("scanpy")
    from xenium_spatial.leiden_optimizer import preprocess_for_clustering

    low = preprocess_for_clustering(_synthetic_counts_adata(), n_pcs=2, n_neighbors=10)
    high = preprocess_for_clustering(_synthetic_counts_adata(), n_pcs=30, n_neighbors=10)

    rec_low = low.uns["pca_elbow"]["n_pcs"]
    rec_high = high.uns["pca_elbow"]["n_pcs"]

    # The recommendation no longer collapses to the requested n_pcs ...
    assert rec_low > 2
    # ... and is the same regardless of how many PCs were requested.
    assert rec_low == rec_high


def test_embedding_keeps_requested_n_pcs():
    pytest.importorskip("scanpy")
    from xenium_spatial.leiden_optimizer import preprocess_for_clustering

    for n in (2, 5, 30):
        out = preprocess_for_clustering(
            _synthetic_counts_adata(), n_pcs=n, n_neighbors=10)
        assert out.obsm["X_pca"].shape[1] == n
