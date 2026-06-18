"""
pipeline.py
-----------
Shared, Streamlit-cached loaders for the cell-level pages (Leiden Optimizer and
the cluster/UMAP page). Centralised here so both pages build the *same*
embedding from the same inputs without duplicating the loader.

scanpy / the xenium_spatial package are imported lazily inside the function, at
which point each page has already put ``<repo>/src`` on sys.path.
"""
from pathlib import Path

import streamlit as st


def roi_signature(slide_ids, roi_dir) -> tuple:
    """Per-slide ROI-file signature (slide_id, mtime). Changes when any saved
    ROI changes, so it can key the embedding cache and invalidate it on edit."""
    sig = []
    for sid in slide_ids:
        p = Path(roi_dir) / f"{sid.replace('/', '_').replace(' ', '_')}_roi.json"
        sig.append((sid, p.stat().st_mtime if p.exists() else 0.0))
    return tuple(sig)


@st.cache_resource(show_spinner=False)
def load_embedding(run_dirs, slide_ids, conditions, batches, base_csv, roi_dir,
                   use_roi, panel_mode, min_slides, roi_sig,
                   base_panel_only, n_pcs, n_neighbors, scale_genes,
                   batch_key, output_dir=None):
    """Load + harmonise + ROI-filter + concatenate the slides, then build the
    cell-level PCA embedding and KNN graph (optionally Harmony-corrected).

    ``roi_sig`` must NOT be underscore-prefixed: Streamlit skips hashing args
    whose name starts with ``_``, which would drop it from the cache key.
    """
    from xenium_spatial.multislide_loader import SlideManifest, MultiSlideLoader
    from xenium_spatial.panel_registry import PanelRegistry
    from xenium_spatial.roi_selector import ROISelector
    from xenium_spatial.sample_pca import _restrict_to_base_panel
    from xenium_spatial.leiden_optimizer import preprocess_for_clustering

    manifest = SlideManifest()
    for sid, cond, d, b in zip(slide_ids, conditions, run_dirs, batches):
        manifest.add(slide_id=sid, condition=cond, run_dir=d, replicate_id=sid,
                     batch=b or sid)

    registry = PanelRegistry(base_csv)
    roi_selector = ROISelector(cache_dir=roi_dir) if use_roi else None
    loader = MultiSlideLoader(
        manifest=manifest, panel_registry=registry, roi_selector=roi_selector,
        panel_mode=panel_mode, min_slides=min_slides, apply_roi=use_roi,
        output_dir=output_dir,
    )
    adata = loader.load_all()
    if base_panel_only:
        adata = _restrict_to_base_panel(adata)
    return preprocess_for_clustering(
        adata, n_pcs=n_pcs, n_neighbors=n_neighbors, scale_genes=scale_genes,
        batch_key=batch_key,
    )
