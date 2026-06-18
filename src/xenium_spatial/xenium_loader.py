"""
xenium_loader.py
----------------
Loads a single 10x Genomics Xenium output directory into AnnData.

Panel structure
---------------
Every Xenium run produces ONE count matrix (matrix.mtx.gz) that contains
ALL genes for that run -- base panel genes AND custom panel genes together.
There is no separate file for base vs custom genes.

For this AGED/ADULT study the structure per slide is:

    ~247 base genes  (Xenium_mBrain_v1_1, IDENTICAL across all slides)
  +  ~50 custom genes (DIFFERS between slides, partial overlap)
  = ~297 total genes  stored together in that slide's matrix.mtx.gz

This loader reads the entire matrix without any filtering or splitting.
The classification of genes into "base" vs "custom" happens afterwards
in PanelRegistry, which compares loaded gene names against the base CSV.

Expected directory layout:
    <run_dir>/
        cell_feature_matrix/
            barcodes.tsv.gz   one barcode per line
            features.tsv.gz   one gene per line: gene_id, gene_name, type
            matrix.mtx.gz     sparse counts (genes x cells in MTX format)
        cells.parquet         cell metadata: centroid_x/y, area, ...
        transcripts.parquet   optional per-molecule positions
        cell_boundaries.parquet  optional polygon boundaries
        experiment.xenium     JSON run metadata
"""

import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io

logger = logging.getLogger(__name__)


def load_xenium_run(
    run_dir: Path | str,
    condition_label: str,
    slide_id: str | None = None,
    load_transcripts: bool = False,
) -> ad.AnnData:
    """
    Load a single Xenium run directory into AnnData.

    Reads the full count matrix exactly as written by the Xenium instrument.
    The matrix contains all genes on that slide (base + custom combined).
    No gene filtering is applied here.

    Parameters
    ----------
    run_dir:
        Xenium output directory (must contain cell_feature_matrix/).
    condition_label:
        Stored in adata.obs['condition'] (e.g. "AGED" or "ADULT").
    slide_id:
        Identifier stored in adata.obs['slide_id']. Defaults to dir name.
    load_transcripts:
        Load transcripts.parquet into adata.uns['transcripts'].

    Returns
    -------
    AnnData:
        .X                 raw counts (cells x all_genes), CSR float32
        .obs               cell metadata from cells.parquet
        .var               gene metadata: gene_id, feature_type
        .obs['condition']   condition_label
        .obs['slide_id']    slide identifier
        .obsm['spatial']   (N,2) float32 centroid coordinates in um
        .layers['counts']  identical copy of .X (survives normalisation)
        .uns['slide_info'] n_cells, n_genes, slide_id, condition, run_dir
    """
    run_dir  = Path(run_dir)
    slide_id = slide_id or run_dir.name
    _check_dir(run_dir)

    logger.info("Loading slide '%s' [%s]  from: %s", slide_id, condition_label, run_dir)

    # ------------------------------------------------------------------
    # 1. Count matrix -- RNA targets only (Gene Expression rows)
    # ------------------------------------------------------------------
    # The Xenium matrix.mtx.gz contains ALL feature types in row order:
    #   - Gene Expression   : the RNA targets (predesigned + custom)
    #   - Blank Codeword    : quality-control codewords (not real genes)
    #   - Negative Control Codeword / Negative Control Probe : controls
    #
    # We load the full matrix first, then slice to Gene Expression rows
    # only.  This keeps blank/negative controls OUT of the AnnData so
    # they never contaminate normalisation, HVG selection, or DGE.
    # ------------------------------------------------------------------
    mtx_dir = run_dir / "cell_feature_matrix"
    if not mtx_dir.exists():
        raise FileNotFoundError(
            f"'cell_feature_matrix/' not found in {run_dir}.\n"
            "Check this is a valid Xenium output directory."
        )

    # MTX convention: rows=genes, cols=cells -> transpose to cells x genes
    X_full = scipy.io.mmread(mtx_dir / "matrix.mtx.gz").T.tocsr().astype(np.float32)

    barcodes = _read_tsv_gz(mtx_dir / "barcodes.tsv.gz", header=None)[0].values
    features = _read_tsv_gz(
        mtx_dir / "features.tsv.gz",
        header=None,
        names=["gene_id", "gene_name", "feature_type"],
    )

    if X_full.shape != (len(barcodes), len(features)):
        raise ValueError(
            f"Slide '{slide_id}': matrix shape {X_full.shape} != "
            f"{len(barcodes)} barcodes x {len(features)} features. "
            "The MTX files may be mismatched or corrupted."
        )

    # Log feature type breakdown
    type_counts   = features["feature_type"].value_counts()
    n_rna         = int(type_counts.get("Gene Expression", 0))
    # NOTE: 247 is the Xenium mBrain v1.1 base panel size; this is informational
    # only and is derived from PanelRegistry.annotate() later.  We log the full
    # RNA count here and let PanelRegistry fill in base/custom split afterwards.
    non_rna_parts = [
        f"{int(v)} {k}"
        for k, v in type_counts.items()
        if k != "Gene Expression"
    ]
    non_rna_str = (", ".join(non_rna_parts) + " excluded") if non_rna_parts else "no controls present"
    logger.info(
        "  -> %d RNA targets (base/custom split populated by PanelRegistry)  |  %s",
        n_rna, non_rna_str,
    )

    # Slice to Gene Expression features only
    rna_mask = (features["feature_type"] == "Gene Expression").values
    features  = features[rna_mask].reset_index(drop=True)
    X         = X_full[:, rna_mask]

    # ------------------------------------------------------------------
    # 2. Cell metadata
    # ------------------------------------------------------------------
    # Support both run formats:
    #   New (xenium_cell_segmentation_stains_v1): cells.parquet
    #   Old (imported segmentation, no stain):    cells.csv.gz or cells.csv
    cells_df = None
    for candidate, loader in [
        (run_dir / "cells.parquet",  lambda p: pd.read_parquet(p)),
        (run_dir / "cells.csv.gz",   lambda p: pd.read_csv(p, compression="gzip")),
        (run_dir / "cells.csv",      lambda p: pd.read_csv(p)),
    ]:
        if candidate.exists():
            cells_df = loader(candidate)
            cells_df.index = cells_df.index.astype(str)
            cells_df = _standardise_centroid_cols(cells_df)
            logger.info("  -> cell metadata from %s", candidate.name)
            break

    if cells_df is None:
        logger.warning(
            "Slide '%s': no cells.parquet or cells.csv[.gz] found; "
            "obs will be minimal and spatial plots unavailable.", slide_id
        )
        cells_df = pd.DataFrame(index=barcodes)
        obs = cells_df
    else:
        obs = _align_cells_to_barcodes(cells_df, barcodes, slide_id)
    obs.index.name = "cell_id"

    # ------------------------------------------------------------------
    # 3. Assemble AnnData
    # ------------------------------------------------------------------
    var = features.set_index("gene_name")
    var.index.name = "gene_name"

    # Warn on duplicate gene symbols before make_unique suffixes them
    # ("Foo" -> "Foo-1"). A suffixed copy no longer matches the base panel by
    # name, so PanelRegistry would silently classify it as a custom gene.
    # Surfacing it lets the user notice rather than chase a phantom gene.
    dup_names = var.index[var.index.duplicated()].unique().tolist()
    if dup_names:
        logger.warning(
            "Slide '%s': %d duplicate gene symbol(s) in features.tsv.gz "
            "(%s%s); make_unique will suffix the copies, which PanelRegistry "
            "then treats as custom genes.",
            slide_id, len(dup_names),
            ", ".join(map(str, dup_names[:10])), " ..." if len(dup_names) > 10 else "",
        )

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs_names = barcodes.tolist()
    adata.var_names_make_unique()

    # Raw counts layer -- preserved through all normalisation steps
    adata.layers["counts"] = adata.X.copy()

    # ------------------------------------------------------------------
    # 4. Spatial coordinates
    # ------------------------------------------------------------------
    if "centroid_x" in adata.obs.columns and "centroid_y" in adata.obs.columns:
        adata.obsm["spatial"] = (
            adata.obs[["centroid_x", "centroid_y"]].values.astype(np.float32)
        )
    else:
        logger.warning(
            "Slide '%s': centroid columns missing -- spatial plots unavailable.", slide_id
        )

    # ------------------------------------------------------------------
    # 5. Metadata columns
    # ------------------------------------------------------------------
    adata.obs["condition"] = pd.Categorical([condition_label] * adata.n_obs)
    adata.obs["slide_id"]  = pd.Categorical([slide_id]        * adata.n_obs)
    adata.obs["run_dir"]   = str(run_dir)

    # ------------------------------------------------------------------
    # 6. uns: slide-level bookkeeping
    # ------------------------------------------------------------------
    xenium_meta = {}
    if (run_dir / "experiment.xenium").exists():
        with open(run_dir / "experiment.xenium") as fh:
            xenium_meta = json.load(fh)

    adata.uns["slide_info"] = {
        "slide_id"        : slide_id,
        "condition"       : condition_label,
        "run_dir"         : str(run_dir),
        "n_cells"         : adata.n_obs,
        "n_genes_total"   : adata.n_vars,
        # base/custom split populated later by PanelRegistry.annotate()
        "n_genes_base"    : None,
        "n_genes_custom"  : None,
        "xenium_metadata" : xenium_meta,
    }

    # ------------------------------------------------------------------
    # 7. Optional per-transcript table
    # ------------------------------------------------------------------
    if load_transcripts:
        tx_path = run_dir / "transcripts.parquet"
        if tx_path.exists():
            logger.info("  Loading transcripts.parquet ...")
            adata.uns["transcripts"] = pd.read_parquet(tx_path)
        else:
            logger.warning("Slide '%s': transcripts.parquet not found.", slide_id)

    logger.info(
        "  -> %d cells x %d genes (Gene Expression only)",
        adata.n_obs, adata.n_vars,
    )
    return adata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _align_cells_to_barcodes(
    cells_df: pd.DataFrame,
    barcodes: np.ndarray,
    slide_id: str,
) -> pd.DataFrame:
    """
    Align cells.parquet rows to the barcode order from barcodes.tsv.gz.

    Xenium barcodes.tsv.gz contains one cell ID per line (e.g. "1", "2", ...).
    cells.parquet may have:
      - A 'cell_id' column with matching IDs
      - A RangeIndex (0-based) when barcodes are 1-based -> off-by-one
      - An index that already matches barcodes as strings
      - The same number of rows as barcodes -> positional alignment

    We try each strategy in order and log which one worked.
    """
    n_bc = len(barcodes)
    barcodes_str = [str(b) for b in barcodes]

    # Strategy 1: cells_df has a 'cell_id' column -> use it as index.
    # Work on a copy so a failed match here does not leave the column promoted
    # to the index when Strategy 2 falls back to the original frame's index.
    for col in ["cell_id", "Cell ID", "cellid", "CellID"]:
        if col in cells_df.columns:
            by_id = cells_df.set_index(col)
            by_id.index = by_id.index.astype(str)
            matched = by_id.reindex(barcodes_str)
            n_matched = matched.notna().any(axis=1).sum() if not matched.empty else 0
            if n_matched > 0:
                logger.info(
                    "  -> cells aligned via column '%s': %d/%d matched",
                    col, n_matched, n_bc,
                )
                return matched
            break  # found the column but it didn't match -- try other strategies

    # Strategy 2: direct string index match
    cells_df.index = cells_df.index.astype(str)
    matched = cells_df.reindex(barcodes_str)
    n_matched = matched.notna().any(axis=1).sum() if not matched.empty else 0
    if n_matched > 0:
        logger.info(
            "  -> cells aligned via string index: %d/%d matched", n_matched, n_bc
        )
        return matched

    # Strategy 3: barcodes are 1-based integers, cells_df has 0-based RangeIndex
    # Try re-indexing with barcodes interpreted as 0-based (subtract 1)
    try:
        bc_int = [int(b) for b in barcodes]
        bc_0based = [i - 1 for i in bc_int]
        if all(0 <= i < len(cells_df) for i in bc_0based):
            result = cells_df.iloc[bc_0based].copy()
            result.index = pd.Index(barcodes_str)
            logger.info(
                "  -> cells aligned via 1-based->0-based index conversion: %d cells",
                len(result),
            )
            return result
    except (ValueError, TypeError):
        pass

    # Strategy 4: same row count — last-resort positional alignment.
    # WARNING: this assumes cells_df rows are in the same order as barcodes.tsv.gz.
    # If they differ (e.g. cells.parquet sorted spatially vs barcodes sorted by
    # detection order), spatial coordinates will be silently mis-assigned.
    # Strategies 1-3 cover all standard Xenium output formats; reaching here
    # means the data format is unexpected.
    if len(cells_df) == n_bc:
        result = cells_df.copy()
        result.index = pd.Index(barcodes_str)
        logger.warning(
            "Slide '%s': falling back to POSITIONAL alignment "
            "(%d rows == %d barcodes) because no cell_id column or index match "
            "was found. This is only correct if cells.parquet rows are in the "
            "same order as barcodes.tsv.gz — verify your data if spatial "
            "coordinates look wrong.",
            slide_id, len(cells_df), n_bc,
        )
        return result

    # Nothing worked: log what we have and return empty obs
    logger.warning(
        "Slide '%s': could not align cells_df (index sample: %s) "
        "to barcodes (sample: %s). "
        "Spatial coordinates will be unavailable.",
        slide_id,
        list(cells_df.index[:5]),
        barcodes_str[:5],
    )
    return pd.DataFrame(index=pd.Index(barcodes_str))


def _check_dir(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Xenium run directory not found: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {p}")


def _read_tsv_gz(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="gzip", **kwargs)


def _standardise_centroid_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise centroid column names across Xenium software versions."""
    renames = {}
    for col in df.columns:
        lc = col.lower()
        if lc in {"x_centroid", "x", "centroid_x_um", "cell_centroid_x", "x_um"}:
            renames[col] = "centroid_x"
        elif lc in {"y_centroid", "y", "centroid_y_um", "cell_centroid_y", "y_um"}:
            renames[col] = "centroid_y"
    return df.rename(columns=renames)
