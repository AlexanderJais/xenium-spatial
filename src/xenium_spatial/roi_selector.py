"""
roi_selector.py
---------------
Region-of-Interest (ROI) persistence and application for Xenium spatial data.

ROIs are framed interactively in the Streamlit ROI Manager (Plotly + sliders)
and written to ``roi_cache/<slide_id>_roi.json``.  This module reads those
saved ROIs back and applies them to AnnData objects during loading.

ROI JSON format
---------------
    {
      "slide_id": "AGED_1",
      "roi_name": "MBH",
      "vertices": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
      "n_cells_selected": 1234,
      ...
    }

The polygon is defined by ``vertices`` (in µm); a rectangle is just a
4-vertex polygon.

Usage
-----
    from xenium_spatial.roi_selector import ROISelector

    selector = ROISelector(cache_dir="roi_cache")
    adata_mbh = selector.apply_roi(adata_slide1, slide_id="AGED_1")

    # Batch (apply all saved ROIs):
    filtered = selector.apply_all(adatas, slide_ids)
"""

import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np

logger = logging.getLogger(__name__)


class ROISelector:
    """
    Reads saved ROI polygons and applies them to Xenium AnnData objects.

    Parameters
    ----------
    cache_dir:
        Directory where per-slide ROI JSON files live
        (``<slide_id>_roi.json``).  Created if it does not exist.
    """

    def __init__(self, cache_dir: Path | str = "roi_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ROISelector: cache directory = %s", self.cache_dir.absolute())

    # ------------------------------------------------------------------
    # Apply saved ROI
    # ------------------------------------------------------------------

    def apply_roi(
        self,
        adata: ad.AnnData,
        slide_id: str,
        invert: bool = False,
    ) -> ad.AnnData:
        """
        Subset ``adata`` to cells inside the saved ROI polygon for ``slide_id``.

        Parameters
        ----------
        adata:
            AnnData with ``.obsm['spatial']`` (cell centroids in µm).
        slide_id:
            Identifier matching the saved ROI file.
        invert:
            If True, keep cells OUTSIDE the ROI instead.

        Returns
        -------
        Filtered AnnData (a copy; the original is not modified).
        """
        roi_path = self._roi_path(slide_id)
        if not roi_path.exists():
            raise FileNotFoundError(
                f"No ROI found for slide '{slide_id}' at {roi_path}. "
                "Frame it in the ROI Manager first."
            )
        roi = self._load_roi(roi_path)
        vertices = np.array(roi["vertices"], dtype=np.float64).reshape(-1, 2)

        if "spatial" not in adata.obsm:
            raise ValueError("adata.obsm['spatial'] required.")

        xy = adata.obsm["spatial"].astype(np.float64)
        inside = _points_in_polygon(xy, vertices)
        mask = ~inside if invert else inside

        result = adata[mask].copy()
        result.obs["roi_name"] = roi.get("roi_name", "ROI")
        logger.info(
            "ROI '%s' applied to '%s': %d / %d cells selected (%.1f%%)",
            roi.get("roi_name"), slide_id,
            mask.sum(), adata.n_obs, 100 * mask.sum() / max(adata.n_obs, 1),
        )
        return result

    def has_roi(self, slide_id: str) -> bool:
        """Return True if a saved ROI exists for this slide_id."""
        return self._roi_path(slide_id).exists()

    def apply_all(
        self,
        adatas: list[ad.AnnData],
        slide_ids: list[str],
    ) -> list[ad.AnnData]:
        """
        Apply saved ROIs to every slide in the list.
        Slides without a saved ROI are returned unchanged (with a warning).
        """
        results = []
        for adata, sid in zip(adatas, slide_ids):
            if self.has_roi(sid):
                results.append(self.apply_roi(adata, sid))
            else:
                logger.warning(
                    "No ROI found for '%s'; returning full slide. "
                    "Frame it in the ROI Manager to enable filtering.", sid
                )
                results.append(adata.copy())
        return results

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------

    def _roi_path(self, slide_id: str) -> Path:
        safe_id = slide_id.replace("/", "_").replace(" ", "_")
        return self.cache_dir / f"{safe_id}_roi.json"

    def _load_roi(self, path: Path) -> dict:
        with open(path) as fh:
            return json.load(fh)


# ===========================================================================
# Geometry helper
# ===========================================================================

def _points_in_polygon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Ray-casting test: boolean mask of points inside a polygon.

    Parameters
    ----------
    points:   (N, 2) array of (x, y) coordinates
    vertices: (M, 2) array of polygon vertices (need not be closed)

    Returns
    -------
    Boolean array of shape (N,).
    """
    from matplotlib.path import Path as MplPath
    return MplPath(vertices).contains_points(points)
