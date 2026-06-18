"""
xenium_spatial
==============
Tools for the first exploratory steps of a 10x Genomics Xenium spatial study:
load and harmonise multi-slide runs, frame ROIs, run sample-level (pseudobulk)
PCA, and optimise Leiden clustering resolution at the single-cell level.

The convenience names below (``SlideManifest``, ``MultiSlideLoader`` …) are
re-exported lazily: importing :mod:`xenium_spatial` itself is cheap, and the
heavier dependencies (anndata, and — for clustering — scanpy / igraph /
leidenalg / harmonypy) are only pulled in when you actually touch the relevant
object. This is why a lightweight helper such as
``xenium_spatial.leiden_optimizer.compute_elbow_n_pcs`` can be imported with
just numpy installed.
"""

import importlib

__version__ = "0.1.0"

# Public name -> module that defines it. Imported on first access (PEP 562).
_LAZY_API = {
    "SlideManifest": "xenium_spatial.multislide_loader",
    "MultiSlideLoader": "xenium_spatial.multislide_loader",
    "PanelRegistry": "xenium_spatial.panel_registry",
    "ROISelector": "xenium_spatial.roi_selector",
    "sample_level_pca_analysis": "xenium_spatial.sample_pca",
}

__all__ = [*_LAZY_API, "__version__"]


def __getattr__(name: str):
    if name in _LAZY_API:
        module = importlib.import_module(_LAZY_API[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_LAZY_API])
