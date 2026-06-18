"""
Tests for the elbow-plot PC-selection heuristic (``compute_elbow_n_pcs``).

The reference behaviour is the two-criterion metric from the HBC scRNA-seq
training (https://hbctraining.github.io/scRNA-seq/lessons/elbow_plot_metric.html):

    co1 = first PC where cumulative % > 90 and that PC's own % < 5
    co2 = last PC where the drop in % to the next PC still exceeds 0.1
    n_pcs = min(co1, co2)

computed on the per-PC standard deviation expressed as a % of the total.
"""

import numpy as np
import pytest

from xenium_spatial.leiden_optimizer import compute_elbow_n_pcs


def test_steep_then_flat_curve():
    """A clear elbow: the recommendation sits where the curve flattens."""
    var = np.array([50, 30, 12, 5, 2, 1, 0.5, 0.4, 0.35, 0.33, 0.32, 0.31, 0.30])
    out = compute_elbow_n_pcs(var)
    assert out["n_pcs"] == min(out["co1"], out["co2"])
    assert 1 <= out["n_pcs"] <= var.size
    # The flat tail (PCs ~9+) should not be recommended.
    assert out["n_pcs"] < 11


def test_recommendation_is_min_of_cutoffs():
    rng = np.random.default_rng(0)
    stdev = np.concatenate([np.linspace(7, 2, 12), np.linspace(1.9, 0.3, 38)])
    out = compute_elbow_n_pcs(stdev ** 2)
    assert out["n_pcs"] == min(out["co1"], out["co2"])


def test_percentages_use_stdev_and_sum_to_100():
    var = np.array([9.0, 4.0, 1.0])  # stdev = 3, 2, 1 -> total 6
    out = compute_elbow_n_pcs(var)
    assert pytest.approx(sum(out["pct"]), abs=1e-6) == 100.0
    # First PC: 3/6 = 50%
    assert pytest.approx(out["pct"][0], abs=1e-6) == 50.0
    assert pytest.approx(out["cumulative"][-1], abs=1e-6) == 100.0


def test_single_component():
    out = compute_elbow_n_pcs([10.0])
    assert out["n_pcs"] == 1
    assert out["co1"] == 1
    assert out["co2"] == 1


def test_empty_raises():
    with pytest.raises(ValueError):
        compute_elbow_n_pcs([])


def test_all_zero_variance_raises():
    with pytest.raises(ValueError):
        compute_elbow_n_pcs([0.0, 0.0, 0.0])


def test_thresholds_are_tunable():
    var = np.array([40, 30, 20, 6, 4])
    strict = compute_elbow_n_pcs(var, change_threshold=10.0)
    loose = compute_elbow_n_pcs(var, change_threshold=0.001)
    # A larger change_threshold makes co2 trigger earlier (fewer PCs kept).
    assert strict["co2"] <= loose["co2"]
