"""
run_sample_pca.py
-----------------
First analysis step for the AGED vs ADULT Xenium study:
load the slides, apply the per-slide MBH ROIs, and run a
sample-level (pseudobulk) PCA.

This answers the question that has to come *before* any cell-level
clustering or DGE: do the samples separate by group (AGED vs
ADULT), and are there outlier slides we should be aware of?

Usage
-----
    # Default: load all configured slides, apply saved ROIs, run sample PCA
    # on the shared base panel only
    python run_sample_pca.py

    # Run on just two samples
    python run_sample_pca.py --samples AGED_1 ADULT_1

    # Ignore ROIs and use whole sections
    python run_sample_pca.py --no-roi

    # Include the add-on (custom) genes as well as the base panel
    python run_sample_pca.py --all-genes

    # Restrict PCA to the 200 most variable genes and z-score them
    python run_sample_pca.py --n-top-genes 200 --scale-genes

Outputs (in figures_output_sample_pca/)
    sample_pca_scatter.<fmt>          PC1 vs PC2, coloured by group
    sample_correlation_heatmap.<fmt>  hierarchically-clustered sample r
    sample_pca_scree.<fmt>            variance explained per PC
    sample_pca_coordinates.csv        PC coordinates + metadata
    sample_pca_variance.csv           variance ratios
    pseudobulk_samples.h5ad           pseudobulk AnnData
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Make the xenium_spatial package importable without an editable install
# (src layout: the package lives under <repo>/src; this script is in <repo>/scripts).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from xenium_spatial.multislide_loader import SlideManifest, MultiSlideLoader
from xenium_spatial.panel_registry import PanelRegistry
from xenium_spatial.roi_selector import ROISelector
from xenium_spatial.sample_pca import sample_level_pca_analysis

logger = logging.getLogger("SamplePCA")


# ===========================================================================
# Study configuration (edit these paths, or use the web app's Study Setup)
# ===========================================================================

ROOT_DATA  = _REPO_ROOT / "data"
OUTPUT_DIR = _REPO_ROOT / "figures_output_sample_pca"
ROI_CACHE  = _REPO_ROOT / "roi_cache"
BASE_PANEL = _REPO_ROOT / "data" / "Xenium_mBrain_v1_1_metadata.csv"

SLIDES = [
    {"slide_id": "AGED_1",  "condition": "AGED",  "run_dir": ROOT_DATA / "AGED_1"},
    {"slide_id": "AGED_2",  "condition": "AGED",  "run_dir": ROOT_DATA / "AGED_2"},
    {"slide_id": "AGED_3",  "condition": "AGED",  "run_dir": ROOT_DATA / "AGED_3"},
    {"slide_id": "AGED_4",  "condition": "AGED",  "run_dir": ROOT_DATA / "AGED_4"},
    {"slide_id": "ADULT_1", "condition": "ADULT", "run_dir": ROOT_DATA / "ADULT_1"},
    {"slide_id": "ADULT_2", "condition": "ADULT", "run_dir": ROOT_DATA / "ADULT_2"},
    {"slide_id": "ADULT_3", "condition": "ADULT", "run_dir": ROOT_DATA / "ADULT_3"},
    {"slide_id": "ADULT_4", "condition": "ADULT", "run_dir": ROOT_DATA / "ADULT_4"},
]


def main(
    use_roi: bool = True,
    panel_mode: str = "partial_union",
    min_slides: int = 2,
    n_top_genes: int = 0,
    scale_genes: bool = False,
    base_panel_only: bool = True,
    samples: Optional[list[str]] = None,
    fmt: str = "pdf",
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Select which slides to analyse. `samples` lets you run on a subset
    #    (e.g. 2 of the configured slides); default is every slide in SLIDES.
    if samples:
        known = {s["slide_id"] for s in SLIDES}
        unknown = [s for s in samples if s not in known]
        if unknown:
            raise SystemExit(
                f"Unknown sample id(s): {unknown}. Available: {sorted(known)}"
            )
        selected = [s for s in SLIDES if s["slide_id"] in set(samples)]
    else:
        selected = list(SLIDES)

    if len(selected) < 2:
        raise SystemExit(
            f"Sample PCA needs at least 2 samples; got {len(selected)}. "
            "Select more with --samples."
        )

    logger.info(
        "Running sample PCA on %d sample(s): %s",
        len(selected), ", ".join(s["slide_id"] for s in selected),
    )

    # 2. Manifest of the selected slides.
    manifest = SlideManifest()
    for s in selected:
        manifest.add(
            slide_id=s["slide_id"], condition=s["condition"],
            run_dir=s["run_dir"], replicate_id=s["slide_id"],
        )

    # 3. Panel registry + optional ROI selector.
    registry = PanelRegistry(BASE_PANEL)
    roi_selector = ROISelector(cache_dir=ROI_CACHE) if use_roi else None
    if use_roi and not any(roi_selector.has_roi(s["slide_id"]) for s in selected):
        logger.warning(
            "ROI requested but no saved ROIs found in %s/. Frame ROIs first "
            "in the web app's ROI Manager, or pass --no-roi to use whole sections.",
            ROI_CACHE,
        )

    # 4. Load + harmonise + ROI-filter + concatenate.
    loader = MultiSlideLoader(
        manifest=manifest,
        panel_registry=registry,
        roi_selector=roi_selector,
        panel_mode=panel_mode,
        min_slides=min_slides,
        apply_roi=use_roi,
        output_dir=OUTPUT_DIR,
    )
    adata = loader.load_all()

    # 5. Sample-level PCA.
    sample_level_pca_analysis(
        adata,
        output_dir=OUTPUT_DIR,
        sample_key="replicate",
        condition_key="condition",
        n_top_genes=n_top_genes,
        scale_genes=scale_genes,
        base_panel_only=base_panel_only,
        fmt=fmt,
    )

    logger.info("Done. See %s/ for figures and tables.", OUTPUT_DIR)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sample-level PCA for the AGED vs ADULT study.")
    p.add_argument("--no-roi", action="store_true", help="Use whole sections (skip ROI filtering).")
    p.add_argument("--panel-mode", default="partial_union",
                   choices=["intersection", "partial_union", "union"])
    p.add_argument("--min-slides", type=int, default=2,
                   help="Min slides a custom gene must appear in (partial_union).")
    p.add_argument("--samples", nargs="+", metavar="SLIDE_ID",
                   help="Slide IDs to include (>=2). Default: all configured slides.")
    p.add_argument("--all-genes", action="store_true",
                   help="Include add-on (custom) genes too. Default: base panel only.")
    p.add_argument("--n-top-genes", type=int, default=0,
                   help="Restrict PCA to N most variable genes (0 = all genes).")
    p.add_argument("--scale-genes", action="store_true",
                   help="Z-score genes before PCA.")
    p.add_argument("--fmt", default="pdf", choices=["pdf", "svg", "png"])
    args = p.parse_args()

    main(
        use_roi=not args.no_roi,
        panel_mode=args.panel_mode,
        min_slides=args.min_slides,
        n_top_genes=args.n_top_genes,
        scale_genes=args.scale_genes,
        base_panel_only=not args.all_genes,
        samples=args.samples,
        fmt=args.fmt,
    )
