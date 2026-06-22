"""
transform.py
------------
Per-slide **section-straightening** transform for Xenium coordinates.

Some sections are mounted at a slight angle, so the brain's dorsal–ventral axis
is not aligned with the image y-axis. That breaks the axis-aligned bounding-box
ROI (it pulls in extra tissue), the "y = dorsal→ventral" reading of the spatial
maps, and the axis-aligned spatial age-effect grid.

The fix is a per-slide rigid rotation about the tissue centroid, recorded
alongside the ROI and applied to ``obsm['spatial']`` at load time so the whole
pipeline works in one canonical orientation. Rotation is rigid, so it does **not**
change clustering, markers, composition, pseudobulk DGE, or neighbourhood
enrichment (all rotation-invariant) — it only straightens the ROI framing, the
maps, and the grid.

This module is the single source of truth for the maths, shared by the ROI
Manager (which previews the straightened tissue and frames the box in the
canonical frame) and :mod:`roi_selector` (which applies the same transform when
filtering), so the preview and the filter can never disagree.

Transform record (stored inside the ROI JSON, identity when absent)::

    "transform": {"rotation_deg": -7.5, "pivot": [cx, cy], "method": "manual"}

``rotation_deg`` is a counter-clockwise rotation in the raw data frame; ``pivot``
is the centre of rotation (the tissue centroid) so coordinate ranges stay put.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

# Default (identity) transform — used whenever a slide has no saved transform.
IDENTITY = {"rotation_deg": 0.0, "pivot": None, "method": "identity"}


def rotation_matrix(angle_deg: float) -> np.ndarray:
    """2×2 matrix rotating a point counter-clockwise by ``angle_deg``."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], dtype=float)


def tissue_centroid(xy) -> np.ndarray:
    """Centroid (mean x, y) of a set of cell coordinates — the rotation pivot."""
    return np.asarray(xy, dtype=float).mean(axis=0)


def apply_transform(
    xy, rotation_deg: float = 0.0, pivot: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Rotate ``xy`` (N×2) by ``rotation_deg`` about ``pivot``.

    ``pivot`` defaults to the centroid of ``xy``. With ``rotation_deg == 0`` the
    coordinates are returned unchanged (a float copy), so identity transforms are
    a no-op and existing ROIs keep selecting exactly the same cells.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.size == 0:
        return xy.copy()
    pv = tissue_centroid(xy) if pivot is None else np.asarray(pivot, dtype=float)
    if not rotation_deg:
        return xy.astype(float, copy=True)
    centred = xy - pv
    rotated = centred @ rotation_matrix(rotation_deg).T
    return rotated + pv


def principal_axis_angle(xy) -> float:
    """Suggest a straightening angle from the tissue's principal axis (PCA).

    Returns the rotation (degrees, in ``[-90, 90]``) that brings the tissue's
    long axis horizontal — a sensible starting guess for a coronal section,
    which the user then nudges. Returns 0.0 for degenerate inputs.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.shape[0] < 3:
        return 0.0
    centred = xy - xy.mean(axis=0)
    cov = np.cov(centred.T)
    if not np.all(np.isfinite(cov)):
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    ang = math.degrees(math.atan2(float(major[1]), float(major[0])))
    sugg = -ang  # rotate the long axis back to horizontal
    while sugg > 90.0:
        sugg -= 180.0
    while sugg < -90.0:
        sugg += 180.0
    return round(sugg, 1)


def angle_between(p0: Sequence[float], p1: Sequence[float],
                  target: str = "vertical") -> float:
    """Rotation (degrees) that makes the line ``p0→p1`` vertical or horizontal.

    Helper for a landmark-based workflow (e.g. aligning the third-ventricle
    midline). ``target='vertical'`` makes the line run dorsal–ventral.
    """
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    line = math.degrees(math.atan2(dy, dx))
    want = 90.0 if target == "vertical" else 0.0
    rot = want - line
    while rot > 90.0:
        rot -= 180.0
    while rot < -90.0:
        rot += 180.0
    return round(rot, 1)


def transform_from_roi(roi: dict) -> dict:
    """Read a transform record from an ROI dict, falling back to identity."""
    tf = (roi or {}).get("transform")
    if not isinstance(tf, dict):
        return dict(IDENTITY)
    return {
        "rotation_deg": float(tf.get("rotation_deg", 0.0) or 0.0),
        "pivot": tf.get("pivot"),
        "method": tf.get("method", "manual"),
    }


def is_identity(tf: dict) -> bool:
    """True when a transform record leaves coordinates unchanged."""
    return not tf or not float(tf.get("rotation_deg", 0.0) or 0.0)
