"""
Tests for single-gene quantification: per-cluster pseudobulk DE direction and
the normalised spatial-grid difference.
"""
import numpy as np
import pytest


def test_per_cluster_dge_direction():
    anndata = pytest.importorskip("anndata")
    pytest.importorskip("scipy")
    import pandas as pd
    # Cluster A, 2 AGED + 2 ADULT samples, 6 cells each. Gal high in AGED.
    rows, counts = [], []
    for sid, cond in [("S1", "AGED"), ("S2", "AGED"), ("S3", "ADULT"), ("S4", "ADULT")]:
        for _ in range(6):
            gal = 50 if cond == "AGED" else 5
            counts.append([gal, 20, 20])           # Gal, g1, g2
            rows.append({"replicate": sid, "condition": cond, "cell_type": "A"})
    X = np.array(counts, dtype="float32")
    a = anndata.AnnData(X=X, obs=pd.DataFrame(rows),
                        var=pd.DataFrame(index=["Gal", "g1", "g2"]))
    a.layers["counts"] = a.X.copy()
    a.layers["lognorm"] = np.log1p(a.X)

    from xenium_spatial.gene_focus import gene_dge_across_clusters
    summary, per_rep = gene_dge_across_clusters(a, "Gal")
    row = summary[summary["group"] == "A"].iloc[0]
    assert row["direction"] == "AGED vs ADULT"
    assert row["log2fc"] > 0                 # Gal up in AGED
    assert set(per_rep["sample"]) == {"S1", "S2", "S3", "S4"}


def test_spatial_grid_uniform_difference():
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    # Two slides, one per condition, expression uniform within condition
    # (AGED=2, ADULT=1) -> every populated bin's difference is exactly 1.
    gx, gy = np.meshgrid(np.linspace(0, 100, 6), np.linspace(0, 100, 6))
    coords_one = np.column_stack([gx.ravel(), gy.ravel()])
    coords = np.vstack([coords_one, coords_one])
    n = coords_one.shape[0]
    obs = pd.DataFrame({
        "slide_id": ["A"] * n + ["B"] * n,
        "condition": ["AGED"] * n + ["ADULT"] * n,
        "replicate": ["A"] * n + ["B"] * n,
    })
    expr = np.array([2.0] * n + [1.0] * n, dtype="float32")
    a = anndata.AnnData(X=expr.reshape(-1, 1), obs=obs,
                        var=pd.DataFrame(index=["Gal"]))
    a.layers["lognorm"] = a.X.copy()
    a.obsm["spatial"] = coords

    from xenium_spatial.gene_focus import gene_spatial_grid
    g = gene_spatial_grid(a, "Gal", n_bins=4)
    assert g["conds"] == ["ADULT", "AGED"]
    assert g["diff"].shape == (4, 4)
    finite = g["diff"][np.isfinite(g["diff"])]
    assert finite.size > 0
    assert np.allclose(finite, 1.0)          # AGED(2) - ADULT(1)
