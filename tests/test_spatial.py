"""
Test neighbourhood enrichment: two spatially segregated cell types should be
self-enriched (positive z on the diagonal) and mutually depleted (negative z
off-diagonal), versus within-slide label permutation.
"""
import numpy as np
import pytest


def _adata_two_blobs():
    anndata = pytest.importorskip("anndata")
    pytest.importorskip("sklearn")
    import pandas as pd
    rng = np.random.default_rng(0)
    a = rng.normal(loc=(0, 0), scale=1.0, size=(60, 2))
    b = rng.normal(loc=(100, 100), scale=1.0, size=(60, 2))
    coords = np.vstack([a, b])
    labels = ["A"] * 60 + ["B"] * 60
    ad = anndata.AnnData(
        X=np.zeros((120, 1), dtype="float32"),
        obs=pd.DataFrame({"cell_type": labels, "slide_id": ["S1"] * 120}),
        var=pd.DataFrame(index=["g0"]),
    )
    ad.obsm["spatial"] = coords
    return ad


def test_segregated_types_self_enrich():
    from xenium_spatial.spatial import neighborhood_enrichment
    z = neighborhood_enrichment(_adata_two_blobs(), group_key="cell_type",
                                n_neighbors=6, n_perms=50, seed=0)
    assert z.loc["A", "A"] > 0
    assert z.loc["B", "B"] > 0
    assert z.loc["A", "B"] < 0   # the two blobs avoid each other
