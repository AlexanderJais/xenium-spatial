"""
Tests for within-cell-type pseudobulk DGE: a gene seeded as up in AGED comes
out with a positive log2FC (AGED vs ADULT) and ranks above a flat gene, and
under-replicated cell types are rejected rather than silently tested per cell.
"""
import numpy as np
import pytest


def _adata():
    anndata = pytest.importorskip("anndata")
    rng = np.random.default_rng(0)
    # 4 samples x 6 cells, all one cell type "A". Genes: DE_up (high in AGED),
    # FLAT (same), NOISE.
    samples = [("S1", "AGED"), ("S2", "AGED"), ("S3", "ADULT"), ("S4", "ADULT")]
    obs_rows, counts = [], []
    for sid, cond in samples:
        for _ in range(6):
            de = rng.poisson(80 if cond == "AGED" else 8)
            flat = rng.poisson(40)
            noise = rng.poisson(5)
            counts.append([de, flat, noise])
            obs_rows.append({"replicate": sid, "condition": cond, "cell_type": "A"})
    import pandas as pd
    X = np.array(counts, dtype="float32")
    a = anndata.AnnData(X=X, obs=pd.DataFrame(obs_rows),
                        var=pd.DataFrame(index=["DE_up", "FLAT", "NOISE"]))
    a.layers["counts"] = a.X.copy()
    return a


def test_de_gene_has_positive_log2fc_and_ranks_first():
    pytest.importorskip("scipy")
    from xenium_spatial.pseudobulk_dge import dge_for_celltype
    df, err = dge_for_celltype(_adata(), "A")
    assert err is None
    assert df["direction"].iloc[0] == "AGED vs ADULT"
    de = df[df["gene"] == "DE_up"].iloc[0]
    assert de["log2fc"] > 1.0  # strongly up in AGED
    # The seeded gene should be the top hit by p-value.
    assert df.iloc[0]["gene"] == "DE_up"


def test_under_replicated_celltype_is_rejected():
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    # One AGED sample only -> cannot test, must return an error not a crash.
    X = np.array([[10, 5, 1]] * 4, dtype="float32")
    obs = pd.DataFrame({"replicate": ["S1", "S1", "S3", "S4"],
                        "condition": ["AGED", "AGED", "ADULT", "ADULT"],
                        "cell_type": ["A"] * 4})
    a = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["g0", "g1", "g2"]))
    a.layers["counts"] = a.X.copy()
    from xenium_spatial.pseudobulk_dge import dge_for_celltype
    df, err = dge_for_celltype(a, "A")
    assert df is None and err is not None
