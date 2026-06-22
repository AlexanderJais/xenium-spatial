"""
Tests for the section-straightening transform: rotation is rigid (distances
preserved), round-trips, the PCA auto-suggest recovers a known tilt, and an
oblique rectangle becomes axis-aligned after applying the suggested angle.
"""
import math

import pytest


def test_identity_is_noop():
    np = pytest.importorskip("numpy")
    from xenium_spatial.transform import apply_transform

    xy = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, -1.0]])
    out = apply_transform(xy, rotation_deg=0.0)
    assert np.allclose(out, xy)


def test_rotation_is_rigid_and_round_trips():
    np = pytest.importorskip("numpy")
    from xenium_spatial.transform import apply_transform

    rng = np.random.default_rng(0)
    xy = rng.normal(size=(50, 2)) * 100 + 500
    pivot = xy.mean(axis=0)

    rot = apply_transform(xy, 37.0, pivot)
    # Pairwise distances preserved (rigid).
    d0 = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    d1 = np.linalg.norm(rot[:, None] - rot[None], axis=-1)
    assert np.allclose(d0, d1, atol=1e-6)
    # Inverse rotation about the same pivot returns the original.
    back = apply_transform(rot, -37.0, pivot)
    assert np.allclose(back, xy, atol=1e-6)


def test_pca_suggestion_recovers_known_tilt():
    np = pytest.importorskip("numpy")
    from xenium_spatial.transform import apply_transform, principal_axis_angle

    # A horizontal bar (long axis along x), then tilt it by +20°.
    x = np.linspace(-100, 100, 400)
    bar = np.column_stack([x, np.zeros_like(x)])
    tilted = apply_transform(bar, 20.0, pivot=(0.0, 0.0))

    sugg = principal_axis_angle(tilted)
    # Suggestion should undo the tilt (long axis back to horizontal).
    assert math.isclose(sugg, -20.0, abs_tol=1.0)
    fixed = apply_transform(tilted, sugg, pivot=(0.0, 0.0))
    assert np.std(fixed[:, 1]) < 1.0  # flat along y again


def test_angle_between_makes_line_vertical():
    pytest.importorskip("numpy")
    from xenium_spatial.transform import angle_between, apply_transform
    import numpy as np

    p0, p1 = (0.0, 0.0), (10.0, 10.0)  # 45° line
    rot = angle_between(p0, p1, target="vertical")
    line = np.array([p0, p1])
    out = apply_transform(line, rot, pivot=(0.0, 0.0))
    assert abs(out[1, 0] - out[0, 0]) < 1e-6  # same x → vertical


def test_transform_from_roi_defaults_to_identity():
    from xenium_spatial.transform import transform_from_roi, is_identity

    assert is_identity(transform_from_roi({}))
    assert is_identity(transform_from_roi({"transform": {"rotation_deg": 0.0}}))
    tf = transform_from_roi({"transform": {"rotation_deg": -7.5, "pivot": [1.0, 2.0]}})
    assert not is_identity(tf)
    assert tf["rotation_deg"] == -7.5 and tf["pivot"] == [1.0, 2.0]
