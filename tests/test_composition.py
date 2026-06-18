"""
Tests for cell-type composition math: per-replicate proportions (with explicit
zeros for absent types) and the AGED-vs-ADULT effect-size direction.
"""
import numpy as np
import pandas as pd
import pytest


def _obs():
    # 4 samples (2 AGED, 2 ADULT); type A enriched in AGED, type C only in S1.
    rows = []
    spec = {
        ("S1", "AGED"):  {"A": 6, "B": 3, "C": 1},
        ("S2", "AGED"):  {"A": 5, "B": 5},
        ("S3", "ADULT"): {"A": 2, "B": 8},
        ("S4", "ADULT"): {"A": 3, "B": 7},
    }
    for (sample, cond), types in spec.items():
        for t, n in types.items():
            rows += [{"replicate": sample, "condition": cond, "batch": "r1",
                      "cell_type": t}] * n
    return pd.DataFrame(rows)


def test_proportions_sum_to_one_per_sample():
    from xenium_spatial.composition import composition_long
    comp = composition_long(_obs(), group_key="cell_type")
    per_sample = comp.groupby("replicate")["proportion"].sum()
    assert np.allclose(per_sample.values, 1.0)


def test_absent_type_is_zero_not_missing():
    from xenium_spatial.composition import composition_long
    comp = composition_long(_obs(), group_key="cell_type")
    # Type C only exists in S1; S2/S3/S4 must have explicit 0-count rows.
    c_rows = comp[comp["cell_type"] == "C"]
    assert len(c_rows) == 4
    assert (c_rows[c_rows["replicate"] != "S1"]["count"] == 0).all()


def test_effect_size_direction_aged_vs_adult():
    pytest.importorskip("scipy")
    from xenium_spatial.composition import composition_long, composition_stats
    comp = composition_long(_obs(), group_key="cell_type")
    stats = composition_stats(comp, group_key="cell_type")
    row = stats[stats["cell_type"] == "A"].iloc[0]
    # A is enriched in AGED -> positive log2fc (AGED vs ADULT).
    assert row["direction"] == "AGED vs ADULT"
    assert row["AGED_mean"] > row["ADULT_mean"]
    assert row["log2fc"] > 0
