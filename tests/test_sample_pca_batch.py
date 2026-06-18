"""
Tests that the sample-level pseudobulk carries the per-slide ``batch`` label
through to ``pb.obs``, so the Sample-PCA scatter can encode batch as marker
shape and reveal batch-vs-condition confounding.
"""

import pytest


def _toy_adata():
    anndata = pytest.importorskip("anndata")
    import numpy as np
    import pandas as pd

    obs = pd.DataFrame(
        {
            "replicate": ["S1", "S1", "S2", "S2"],
            "condition": ["AGED", "AGED", "ADULT", "ADULT"],
            "batch": ["run_A", "run_A", "run_B", "run_B"],
        },
        index=["c0", "c1", "c2", "c3"],
    )
    X = np.arange(8, dtype="float32").reshape(4, 2)
    a = anndata.AnnData(X=X, obs=obs)
    a.layers["counts"] = a.X.copy()
    return a


def test_pseudobulk_carries_batch():
    from xenium_spatial.sample_pca import pseudobulk_samples

    pb = pseudobulk_samples(_toy_adata())
    assert "batch" in pb.obs.columns
    assert pb.obs.loc["S1", "batch"] == "run_A"
    assert pb.obs.loc["S2", "batch"] == "run_B"


def test_pseudobulk_without_batch_column_is_fine():
    """If the input lacks a batch column, pseudobulk still works (no batch obs)."""
    a = _toy_adata()
    del a.obs["batch"]
    from xenium_spatial.sample_pca import pseudobulk_samples

    pb = pseudobulk_samples(a)
    assert "batch" not in pb.obs.columns
    assert "condition" in pb.obs.columns
