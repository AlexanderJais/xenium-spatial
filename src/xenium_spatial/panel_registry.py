"""
panel_registry.py
-----------------
Manages the gene panels for the AGED/ADULT Xenium study.

Panel structure per slide
--------------------------
Each Xenium run produces ONE count matrix containing ALL genes measured on
that slide.  The genes come in two groups:

    Base panel  (~247 genes)
        The Xenium_mBrain_v1_1 panel.
        IDENTICAL across ALL 8 slides.
        Defined in Xenium_mBrain_v1_1_metadata.csv.

    Custom panel (~50 genes per slide)
        Additional genes added on top of the base panel.
        DIFFERS between slides -- there is partial overlap between runs
        but no two slides necessarily have the exact same custom set.

Total per slide: ~297 genes stored together in one matrix.mtx.gz.
The base/custom distinction is NOT stored by Xenium -- this class reads
all gene names from each slide and classifies them by comparing against
the base panel CSV.

Three harmonisation modes
--------------------------
intersection
    Keep only the 247 base genes guaranteed in every slide.
    No custom genes. Zero zero-inflation. Safest for DGE.

partial_union  [RECOMMENDED for this study]
    Base genes + custom genes present in at least `min_slides` slides.
    e.g. min_slides=2 keeps custom genes shared by 2 or more of 8 slides.
    Slides missing a kept custom gene get zero-filled columns, and those
    columns are flagged in adata.var['zero_filled'] so downstream DGE
    can mask or down-weight them.
    Balances coverage vs zero-inflation.

union
    Base genes + every custom gene seen in any slide.
    Maximises coverage; maximises zero-inflation for rare custom genes.
    Useful for exploratory work only.

Custom gene tracking
--------------------
After harmonisation every adata.var gains:
    panel_type          "base" | "custom_shared" | "custom_unique"
    n_slides_present    int  (how many slides carried this gene)
    slides_present      str  (comma-separated slide IDs)
    zero_filled         bool (True = this slide got a zero column for this gene)
    cell_type_annotation, ensembl_id, n_probesets  (base genes only)
"""

import logging
from pathlib import Path
from typing import Literal, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

_CSV_GENE_COL   = "Genes"
_CSV_ENSEMBL    = "Ensembl_ID"
_CSV_PROBESETS  = "Num_Probesets"
_CSV_ANNOTATION = "Annotation"


class PanelRegistry:
    """
    Gene panel manager for Xenium multi-slide experiments.

    Parameters
    ----------
    base_panel_csv:
        Path to Xenium_mBrain_v1_1_metadata.csv.
    """

    def __init__(self, base_panel_csv: Path | str):
        self._csv_path = Path(base_panel_csv)
        self._meta: pd.DataFrame = self._load_metadata()
        logger.info(
            "PanelRegistry: loaded %d base genes from %s",
            len(self._meta), self._csv_path.name,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base_genes(self) -> list[str]:
        return self._meta[_CSV_GENE_COL].tolist()

    @property
    def base_gene_set(self) -> set[str]:
        return set(self.base_genes)

    @property
    def metadata(self) -> pd.DataFrame:
        return self._meta.set_index(_CSV_GENE_COL)

    def cell_type_for_gene(self, gene: str) -> Optional[str]:
        row = self._meta[self._meta[_CSV_GENE_COL] == gene]
        return None if row.empty else row[_CSV_ANNOTATION].iloc[0]

    # ------------------------------------------------------------------
    # Slide validation
    # ------------------------------------------------------------------

    def validate_slides(
        self,
        adatas: list[ad.AnnData],
        slide_ids: list[str],
        raise_on_missing_base: bool = False,
    ) -> pd.DataFrame:
        """
        Validate that each slide has the expected panel structure and
        return a per-slide summary DataFrame.

        For each slide checks:
          1. Which of the 247 base genes are present (should be all).
          2. How many custom genes are present (~50).
          3. Total gene count (~297).

        Parameters
        ----------
        adatas:
            List of per-slide AnnData objects as returned by load_xenium_run().
            Each contains all genes for that slide (base + custom combined).
        slide_ids:
            Identifiers matching adatas.
        raise_on_missing_base:
            If True, raise ValueError when any slide is missing base genes.
            If False (default), log a warning and continue.

        Returns
        -------
        DataFrame with columns:
            slide_id, condition, n_total, n_base, n_custom,
            base_complete, n_base_missing, missing_base_genes,
            custom_genes
        """
        rows = []
        any_problem = False

        for adata, sid in zip(adatas, slide_ids):
            gene_set   = set(adata.var_names)
            n_base     = len(gene_set & self.base_gene_set)
            n_custom   = len(gene_set - self.base_gene_set)
            missing    = sorted(self.base_gene_set - gene_set)
            custom_lst = sorted(gene_set - self.base_gene_set)
            complete   = len(missing) == 0

            condition = (
                adata.obs["condition"].iloc[0]
                if "condition" in adata.obs.columns
                else "unknown"
            )

            rows.append({
                "slide_id"          : sid,
                "condition"         : condition,
                "n_total"           : adata.n_vars,
                "n_base"            : n_base,
                "n_custom"          : n_custom,
                "base_complete"     : complete,
                "n_base_missing"    : len(missing),
                "missing_base_genes": missing,
                "custom_genes"      : custom_lst,
            })

            if not complete:
                any_problem = True
                logger.warning(
                    "Slide '%s': %d base genes missing from matrix: %s",
                    sid, len(missing),
                    ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""),
                )

        df = pd.DataFrame(rows)

        if any_problem and raise_on_missing_base:
            bad = df[~df["base_complete"]]["slide_id"].tolist()
            raise ValueError(
                f"Missing base panel genes in slides: {bad}. "
                "Check that the correct Xenium run directories are loaded."
            )

        return df

    # ------------------------------------------------------------------
    # Per-slide classification
    # ------------------------------------------------------------------

    def classify_genes(self, var_names) -> pd.Series:
        """Classify var_names as 'base' or 'custom'."""
        idx = pd.Index(var_names)
        return pd.Series(
            np.where(idx.isin(self.base_gene_set), "base", "custom"),
            index=idx, name="panel_type", dtype="category",
        )

    # ------------------------------------------------------------------
    # Overlap analysis (call before harmonisation to inspect your panels)
    # ------------------------------------------------------------------

    def custom_overlap_matrix(
        self,
        adatas: list[ad.AnnData],
        slide_ids: list[str],
    ) -> pd.DataFrame:
        """
        Binary presence matrix: custom genes (rows) x slides (columns).

        Returns a boolean DataFrame sorted by descending slide count.
        Use this to decide the min_slides threshold for partial_union.
        """
        all_custom: set[str] = set()
        for a in adatas:
            all_custom |= set(a.var_names) - self.base_gene_set

        presence = {
            gene: {sid: gene in set(a.var_names)
                   for a, sid in zip(adatas, slide_ids)}
            for gene in sorted(all_custom)
        }
        df = pd.DataFrame(presence).T          # genes x slides
        n_slides_col = df.sum(axis=1)
        df = df.loc[n_slides_col.sort_values(ascending=False).index]
        return df

    def custom_gene_counts(
        self,
        adatas: list[ad.AnnData],
        slide_ids: list[str],
    ) -> pd.DataFrame:
        """
        Per-custom-gene summary with slide counts and overlap category.

        Returns
        -------
        DataFrame columns: gene, n_slides, slides_present, category
        category is one of: 'shared_all', 'shared_partial', 'unique'
        """
        matrix = self.custom_overlap_matrix(adatas, slide_ids)
        n      = len(slide_ids)

        counts     = matrix.sum(axis=1).rename("n_slides")
        slides_str = matrix.apply(
            lambda row: ",".join(s for s in slide_ids if row[s]), axis=1
        ).rename("slides_present")

        def _cat(k):
            if k == n:   return "shared_all"
            elif k > 1:  return "shared_partial"
            else:        return "unique"

        df = pd.concat([counts, slides_str], axis=1)
        df["category"] = df["n_slides"].map(_cat).astype("category")
        df.index.name  = "gene"
        return df.reset_index().sort_values("n_slides", ascending=False).reset_index(drop=True)

    def consensus_panel(self, gene_sets_by_slide: dict) -> dict:
        """Consensus gene panel across slides: genes present in EVERY slide.

        Strict intersection — base and custom genes alike must be measured in
        all slides to enter the consensus, so nothing is ever zero-filled and
        every consensus gene is genuinely comparable across samples. Computed
        from gene *names* so the Consensus-Panel page (which reads each slide's
        ``features.tsv.gz``) and :meth:`harmonise` (which reads ``var_names``)
        agree on exactly the same set.

        Parameters
        ----------
        gene_sets_by_slide:
            Mapping ``slide_id -> set(gene_name)`` of each slide's
            Gene-Expression targets.

        Returns
        -------
        dict: ``slides``, ``base`` (base genes in all, panel order),
        ``addon`` (custom genes in all, sorted), ``consensus`` (base+addon),
        ``excluded_addon`` / ``excluded_base`` (present in some but not all),
        ``presence`` (``{gene: {slide: bool}}`` for addon + excluded_addon),
        and ``per_slide`` (total / base / custom / custom_in_consensus counts).
        """
        slides = list(gene_sets_by_slide)
        empty = {"slides": [], "base": [], "addon": [], "consensus": [],
                 "excluded_addon": [], "excluded_base": [], "presence": {},
                 "per_slide": {}}
        if not slides:
            return empty
        sets = [set(gene_sets_by_slide[s]) for s in slides]
        universal = set.intersection(*sets)
        union_all = set.union(*sets)
        base_set = self.base_gene_set

        base  = [g for g in self.base_genes if g in universal]
        addon = sorted(universal - base_set)
        excluded_addon = sorted((union_all - base_set) - universal)
        excluded_base  = [g for g in self.base_genes
                          if g in union_all and g not in universal]
        presence = {g: {s: (g in gs) for s, gs in zip(slides, sets)}
                    for g in (addon + excluded_addon)}
        per_slide = {}
        for s, gs in zip(slides, sets):
            custom = gs - base_set
            per_slide[s] = {"total": len(gs), "base": len(gs & base_set),
                            "custom": len(custom),
                            "custom_in_consensus": len(custom & universal)}
        return {"slides": slides, "base": base, "addon": addon,
                "consensus": base + addon, "excluded_addon": excluded_addon,
                "excluded_base": excluded_base, "presence": presence,
                "per_slide": per_slide}

    def recommend_min_slides(
        self,
        adatas: list[ad.AnnData],
        slide_ids: list[str],
        target_custom_genes: int = 30,
    ) -> int:
        """
        Suggest a min_slides threshold that retains approximately
        target_custom_genes custom genes after partial_union filtering.

        Prints a threshold table and returns the recommended value.
        """
        df = self.custom_gene_counts(adatas, slide_ids)
        n  = len(slide_ids)

        logger.info("Custom gene retention by min_slides threshold:")
        logger.info("  min_slides  |  custom genes kept  |  zero-filled columns / slide (avg)")
        # Find the MINIMUM threshold that still retains >= target_custom_genes.
        # Iterate from strictest (n) to most lenient (1) and keep updating
        # 'chosen' — the last update (lowest t that meets target) wins.
        chosen = 1
        for t in range(n, 0, -1):
            kept   = (df["n_slides"] >= t).sum()
            avg_zf = (
                df[df["n_slides"] >= t].apply(lambda row: n - row["n_slides"], axis=1).sum()
                / max(n, 1)
            )
            marker = " <-- recommended" if kept >= target_custom_genes else ""
            logger.info("      %d         |        %3d            |  %.1f%s", t, kept, avg_zf, marker)
            if kept >= target_custom_genes:
                chosen = t   # keep the lowest t that satisfies the target
        return chosen

    # ------------------------------------------------------------------
    # AnnData var annotation
    # ------------------------------------------------------------------

    def annotate(
        self,
        adata: ad.AnnData,
        slide_id: Optional[str] = None,
        overlap_df: Optional[pd.DataFrame] = None,
    ) -> ad.AnnData:
        """
        Add panel metadata columns to adata.var in-place.

        Parameters
        ----------
        adata:       AnnData to annotate.
        slide_id:    Slide identifier — used to flag zero_filled columns.
        overlap_df:  Output of custom_gene_counts(). Enables rich panel_type
                     and n_slides_present / slides_present / zero_filled columns.
        """
        meta = self.metadata

        # Basic classification
        adata.var["panel_type"] = self.classify_genes(adata.var_names)

        if overlap_df is not None:
            g2cat = overlap_df.set_index("gene")["category"].to_dict()
            g2n   = overlap_df.set_index("gene")["n_slides"].to_dict()
            g2sl  = overlap_df.set_index("gene")["slides_present"].to_dict()

            rich_type   = []
            zero_filled = []
            n_slides_v  = []
            slides_v    = []

            for g in adata.var_names:
                if g in self.base_gene_set:
                    rich_type.append("base")
                    zero_filled.append(False)
                    n_slides_v.append(np.nan)
                    slides_v.append("")
                else:
                    cat = g2cat.get(g, "unique")
                    rich_type.append(
                        "custom_shared" if "shared" in str(cat) else "custom_unique"
                    )
                    n_slides_v.append(int(g2n.get(g, 1)))
                    sl_str = g2sl.get(g, "")
                    slides_v.append(sl_str)
                    # zero_filled: this slide is NOT in the gene's slides_present
                    if slide_id and sl_str:
                        zero_filled.append(slide_id not in sl_str.split(","))
                    else:
                        zero_filled.append(False)

            adata.var["panel_type"]       = pd.Categorical(rich_type)
            adata.var["n_slides_present"] = n_slides_v
            adata.var["slides_present"]   = slides_v
            adata.var["zero_filled"]      = zero_filled

        # Base panel metadata (NaN for custom genes)
        adata.var["cell_type_annotation"] = adata.var_names.map(
            meta[_CSV_ANNOTATION].to_dict()
        )
        adata.var["ensembl_id"]  = adata.var_names.map(meta[_CSV_ENSEMBL].to_dict())
        adata.var["n_probesets"] = adata.var_names.map(meta[_CSV_PROBESETS].to_dict())
        return adata

    # ------------------------------------------------------------------
    # Multi-slide harmonisation
    # ------------------------------------------------------------------

    def harmonise(
        self,
        adatas: list[ad.AnnData],
        slide_ids: list[str],
        mode: Literal["consensus", "intersection", "partial_union", "union"] = "consensus",
        min_slides: int = 2,
        fill_value: float = 0.0,
    ) -> list[ad.AnnData]:
        """
        Harmonise gene panels across all slides to a common gene set.

        Parameters
        ----------
        adatas:
            Per-slide AnnData objects (raw counts).
        slide_ids:
            Identifiers matching adatas (same order).
        mode:
            'intersection'  : 247 base genes only.
            'partial_union' : base + custom genes in >= min_slides slides.
            'union'         : base + every custom gene.
        min_slides:
            Minimum number of slides a custom gene must appear in to be
            retained (partial_union only). Default 2.

        Returns
        -------
        List of harmonised AnnData objects. Each has extended .var columns.
        """
        assert len(adatas) == len(slide_ids)
        n = len(slide_ids)

        # Pre-compute overlap for annotation and filtering
        overlap_df = self.custom_gene_counts(adatas, slide_ids)

        if mode == "consensus":
            # Strict intersection across all slides (base + custom alike). Every
            # gene is present in every slide, so nothing is zero-filled.
            universal = set(adatas[0].var_names)
            for a in adatas[1:]:
                universal &= set(a.var_names)
            base_in_all  = [g for g in self.base_genes if g in universal]
            addon_in_all = sorted(universal - self.base_gene_set)
            gene_order   = base_in_all + addon_in_all
            n_excl_base  = sum(
                1 for g in self.base_genes
                if any(g in set(a.var_names) for a in adatas) and g not in universal
            )
            logger.info(
                "Harmonise [consensus]: strict intersection across %d slides\n"
                "  Base genes (in all)  : %d\n"
                "  Add-on genes (in all): %d\n"
                "  Base excluded (missing in >=1 slide): %d\n"
                "  Total consensus set  : %d  (no zero-filling)",
                n, len(base_in_all), len(addon_in_all), n_excl_base, len(gene_order),
            )

        elif mode == "intersection":
            common     = self.base_gene_set.copy()
            for a in adatas:
                common &= set(a.var_names)
            gene_order = [g for g in self.base_genes if g in common]
            logger.info(
                "Harmonise [intersection]: %d base genes kept, 0 custom genes.",
                len(gene_order),
            )

        elif mode == "partial_union":
            kept_custom = (
                overlap_df[overlap_df["n_slides"] >= min_slides]["gene"].tolist()
            )
            base_in_any = [
                g for g in self.base_genes
                if any(g in set(a.var_names) for a in adatas)
            ]
            gene_order = base_in_any + kept_custom

            n_all     = (overlap_df["category"] == "shared_all").sum()
            n_dropped = (overlap_df["n_slides"] < min_slides).sum()
            logger.info(
                "Harmonise [partial_union, min_slides=%d/%d]:\n"
                "  Base genes         : %d\n"
                "  Custom kept        : %d  "
                "(shared_all=%d, shared_partial=%d)\n"
                "  Custom dropped     : %d  "
                "(present in < %d slides)\n"
                "  Total gene set     : %d",
                min_slides, n,
                len(base_in_any),
                len(kept_custom), n_all, len(kept_custom) - n_all,
                n_dropped, min_slides,
                len(gene_order),
            )

        elif mode == "union":
            base_in_any = [
                g for g in self.base_genes
                if any(g in set(a.var_names) for a in adatas)
            ]
            all_custom  = overlap_df["gene"].tolist()
            gene_order  = base_in_any + all_custom
            logger.info(
                "Harmonise [union]: %d base + %d custom = %d total genes.",
                len(base_in_any), len(all_custom), len(gene_order),
            )

        else:
            raise ValueError(
                "mode must be 'consensus', 'intersection', 'partial_union', or "
                f"'union'. Got '{mode}'."
            )

        # Zero-fill missing genes and subset to gene_order for every slide
        results = []
        for adata, sid in zip(adatas, slide_ids):
            harmonised = self._zero_fill_and_subset(adata, gene_order, fill_value)
            self.annotate(harmonised, slide_id=sid, overlap_df=overlap_df)
            results.append(harmonised)

        # Summary log
        self._log_summary(results, slide_ids)
        return results

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def report(
        self,
        adatas: list[ad.AnnData],
        slide_ids: list[str],
    ) -> pd.DataFrame:
        """
        Per-slide panel composition table.

        Columns: slide_id, n_base, n_custom_total,
                 n_custom_shared_all, n_custom_shared_partial,
                 n_custom_unique, custom_unique_genes
        """
        overlap_df = self.custom_gene_counts(adatas, slide_ids)
        shared_all     = set(overlap_df[overlap_df["category"] == "shared_all"]["gene"])
        shared_partial = set(overlap_df[overlap_df["category"] == "shared_partial"]["gene"])
        unique_genes   = set(overlap_df[overlap_df["category"] == "unique"]["gene"])

        rows = []
        for adata, sid in zip(adatas, slide_ids):
            gene_set = set(adata.var_names)
            custom   = gene_set - self.base_gene_set
            rows.append({
                "slide_id"               : sid,
                "n_base"                 : len(gene_set & self.base_gene_set),
                "n_custom_total"         : len(custom),
                "n_custom_shared_all"    : len(custom & shared_all),
                "n_custom_shared_partial": len(custom & shared_partial),
                "n_custom_unique"        : len(custom & unique_genes),
                "custom_unique_genes"    : sorted(custom & unique_genes),
            })
        return pd.DataFrame(rows)

    def print_overlap_summary(
        self,
        adatas: list[ad.AnnData],
        slide_ids: list[str],
    ):
        """Print a human-readable overlap table."""
        df = self.custom_gene_counts(adatas, slide_ids)
        n  = len(slide_ids)
        shared_all     = df[df["n_slides"] == n]
        shared_partial = df[(df["n_slides"] > 1) & (df["n_slides"] < n)]
        unique         = df[df["n_slides"] == 1]

        logger.info("─" * 55)
        logger.info("Custom gene overlap summary (%d slides, ~%d custom genes):",
                    n, len(df))
        logger.info("  Shared across ALL %d slides : %3d genes", n, len(shared_all))
        if not shared_all.empty:
            logger.info("    %s",
                ", ".join(shared_all["gene"].tolist()[:20])
                + (" …" if len(shared_all) > 20 else ""))
        logger.info("  Shared across 2-%d slides   : %3d genes", n - 1, len(shared_partial))
        logger.info("  Unique to exactly 1 slide  : %3d genes", len(unique))
        for _, row in unique.iterrows():
            logger.info("    %-20s  in slide: %s", row["gene"], row["slides_present"])
        logger.info("─" * 55)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zero_fill_and_subset(
        self,
        adata: ad.AnnData,
        gene_order: list[str],
        fill_value: float = 0.0,
    ) -> ad.AnnData:
        """
        Return a new AnnData with exactly the genes in gene_order.
        Genes absent from this slide get sparse zero-filled columns.
        Genes present in adata but not in gene_order are dropped.
        """
        present = set(adata.var_names)
        missing = [g for g in gene_order if g not in present]

        if missing:
            # Use genuinely sparse zero matrix (not dense then converted)
            zero_block = sp.csr_matrix((adata.n_obs, len(missing)), dtype=np.float32)
            var_miss   = pd.DataFrame(index=pd.Index(missing, name=adata.var.index.name or "gene_name"))
            adata_miss = ad.AnnData(
                X=zero_block,
                obs=adata.obs.copy(),
                var=var_miss,
            )
            # Add zero-filled layers to adata_miss so they survive the concat
            for lyr in adata.layers:
                adata_miss.layers[lyr] = zero_block.copy()
            combined = ad.concat([adata, adata_miss], axis=1, merge="first")
        else:
            combined = adata

        keep = [g for g in gene_order if g in set(combined.var_names)]
        result = combined[:, keep].copy()

        # Restore counts layer if it was dropped (safety net)
        if "counts" not in result.layers:
            result.layers["counts"] = result.X.copy()
        return result

    def _load_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self._csv_path)
        expected = [_CSV_GENE_COL, _CSV_ENSEMBL, _CSV_PROBESETS, _CSV_ANNOTATION]
        for col in expected:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in {self._csv_path}.\n"
                    f"Columns present: {list(df.columns)}"
                )
        return df[expected].drop_duplicates(subset=[_CSV_GENE_COL]).reset_index(drop=True)

    def _log_summary(self, results, slide_ids):
        for a, sid in zip(results, slide_ids):
            pt = a.var["panel_type"] if "panel_type" in a.var.columns else pd.Series(dtype=str)
            n_zf = int(a.var["zero_filled"].sum()) if "zero_filled" in a.var.columns else 0
            n_base = int((pt == "base").sum())
            n_cs   = int((pt == "custom_shared").sum())
            n_cu   = int((pt == "custom_unique").sum())
            logger.info(
                "  %-12s  total=%d  base=%d  custom_shared=%d  "
                "custom_unique=%d  zero_filled=%d",
                sid, a.n_vars, n_base, n_cs, n_cu, n_zf,
            )
