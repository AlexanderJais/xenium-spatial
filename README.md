# Xenium Spatial Pipeline

**An end-to-end exploratory pipeline for a 10x Genomics Xenium spatial study — from sample-level QC through cell-level clustering to cell-type, spatial, and single-gene quantification (default: AGED vs ADULT mouse mediobasal hypothalamus).**

A streamlined tool for the first exploratory steps of a [10x Genomics Xenium](https://www.10xgenomics.com/platforms/xenium) spatial study: load the slides, frame the mediobasal hypothalamus (MBH) region on each, collapse every slide into a pseudobulk profile, and run PCA across the samples to see **how the samples cluster and how the AGED and ADULT groups separate**. When you are ready to move from samples to cells, an optional **Leiden Optimizer** sweeps clustering resolutions on the single cells and recommends the one that best balances cluster quality and granularity — so the resolution you take into cell-level analysis is chosen by metrics, not by eye.

From there a cell-level **quantification arm** carries the chosen clustering through UMAP + marker-based **annotation**, cell-type **composition** shifts, within-cell-type **differential expression**, **spatial** maps and neighbourhood niches, and a single-**gene focus** view (default Galanin) — every condition comparison done at the biological-replicate level. See [cell-level quantification](#cell-level-quantification).

Built around a multi-replicate, two-condition study using the `Xenium_mBrain_v1_1` base panel (~247 genes) plus per-slide custom panels (~50 genes each, partially overlapping). The default example is 4 AGED + 4 ADULT brain sections, but **the number of slides and the condition labels are not fixed** — add or remove slides in **Study Setup** (or the `SLIDES` list / a manifest CSV for the CLI), use whatever group names your study needs, and select any subset of slides to analyse at each step (the Sample PCA needs ≥ 2 samples).

Runs entirely on your machine. No data leaves your computer.

---

## Table of contents

- [Two ways to run](#two-ways-to-run)
- [Installation](#installation)
- [Web interface](#web-interface)
- [Command line](#command-line)
- [How the PCA works](#how-the-pca-works)
- [How the Leiden Optimizer works](#how-the-leiden-optimizer-works)
- [Cell-level quantification](#cell-level-quantification)
- [Batch correction](#batch-correction)
- [Panel structure](#panel-structure)
- [Consensus panel](#consensus-panel)
- [Outputs](#outputs)
- [Configuration file format](#configuration-file-format)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Two ways to run

| Method | Command | Best for |
|--------|---------|----------|
| **A. Web interface** (recommended) | `streamlit run app/app.py` | Interactive ROI framing, inline Nature-style figures |
| **B. Command line** | `python scripts/run_sample_pca.py` | Scripted/headless runs once ROIs are saved |

Both paths share the same loader, ROI cache (`roi_cache/`), and PCA module (`src/xenium_spatial/sample_pca.py`). See the [Quick Start guide](QUICKSTART_MAC.md) for step-by-step instructions.

---

## Installation

The project is a `src`-layout Python package (`xenium_spatial`). Install it
editable so the package is importable from anywhere:

```bash
cd /path/to/xenium-spatial-analysis
pip install -e .              # core: sample-PCA workflow + scripts/run_sample_pca.py
pip install -e ".[app]"       # + the Streamlit web interface
pip install -e ".[clustering]"  # + the single-cell Leiden Optimizer stack
pip install -e ".[dev]"       # everything, plus pytest
```

The `requirements.txt` (`pip install -r requirements.txt`) remains as a
single-file superset (same lower-bound version constraints as the extras) if
you prefer not to use extras.

The core workflow (Study Setup → ROI Manager → Sample PCA, and the
`scripts/run_sample_pca.py` CLI) uses an intentionally small dependency set:
NumPy/pandas/SciPy/scikit-learn/Matplotlib + AnnData for data handling,
Streamlit/Plotly for the UI, PyArrow for `cells.parquet`. No DESeq2 is required.

The **cell-level steps** — the Leiden Optimizer **and** the quantification arm (steps 4–9) — need the single-cell stack: `scanpy`, `igraph`, `leidenalg`, and `harmonypy` (the `clustering` extra). These are imported lazily, so the Sample-PCA workflow (steps 1–3) runs fine even if they are not installed; you only need them once you move from samples to cells.

> **macOS Apple Silicon:** `./install_mac.sh` creates a native ARM64 conda environment and installs everything for you.

---

## Web interface

```bash
streamlit run app/app.py
```

Or double-click `start_app.command` in Finder. Your browser opens at http://localhost:8501.

The app is a numbered, sequential workflow — each page is also reachable directly from the sidebar:

| Step | Page | Purpose |
|------|------|---------|
| 1 | **📁 Study Setup** | Enter the path to each Xenium output directory — add or remove slides as needed (any number, any condition labels; the default is 4 AGED + 4 ADULT). A green tick confirms each is valid and shows its gene/cell counts. An optional **Batch** column records a technical batch (e.g. sequencing run / capture date) for Harmony — see [batch correction](#batch-correction). Save/load the full config as JSON. |
| 2 | **🧬 Consensus Panel** | Builds the **consensus gene panel** — the strict intersection of every sample's panel (base genes in all + add-on genes in all) — from each slide's `features.tsv.gz` metadata. Shows the shared add-on genes, any partially-shared add-ons that are **excluded** (with a sample × gene presence matrix), and per-sample contributions, then locks the consensus in as the gene set the whole pipeline runs on. Because it's a strict intersection, **no gene is ever zero-filled**, which removes the main way a differing add-on panel could fake a DE hit. See [consensus panel](#consensus-panel). |
| 3 | **🗺️ ROI Manager** | Interactive Plotly scatter per slide. If a section is mounted slightly rotated, **straighten it first** with the *Rotate (°)* slider (live preview; ✨ *Auto-suggest* seeds the angle from the tissue's principal axis) so the brain sits upright before you frame the box. Then **drag a box** on the (straightened) tissue to frame the MBH bounding rectangle (or fine-tune with the four edge sliders); the cell count and box dimensions update live. ◀/▶ buttons and an auto-advance toggle step through slides; an "All slides" table flags cell-count outliers. Manual coordinate entry is available as a fallback. ROIs — and the per-slide rotation — are saved to `roi_cache/` and reused automatically. |
| 4 | **📊 Sample PCA** | Loads the slides, applies the saved ROIs, pseudobulks each sample, and runs PCA across them. Shows the Nature-style PCA scatter (coloured by group; if a **batch** is set, batch is the marker *shape* so batch-vs-condition confounding is visible), a hierarchically-clustered sample correlation heatmap, and a scree plot — inline, with PDF/CSV downloads. |
| 5 | **🔎 Leiden Optimizer** | Loads and ROI-filters the same slides, builds a single-cell PCA + KNN graph (with optional **Harmony** batch correction on each slide's **batch** label, on by default for multi-slide runs), then sweeps a grid of Leiden resolutions. The **📐 How many PCs? — elbow plot** tool (under *Preprocessing*) estimates how many principal components to keep before you sweep. Each resolution is scored on silhouette, Calinski-Harabasz, Davies-Bouldin, spatial coherence, and modularity; a weighted combined score recommends the best. A clustree shows how clusters split/merge, and one click applies the chosen resolution to the pipeline settings (persisted to the output dir and the study-config JSON). Needs `scanpy`, `igraph`, `leidenalg`, and `harmonypy`. |
| 6 | **🔬 Clusters** | Builds the final clustering at the applied resolution (UMAP + Leiden on the same Harmony embedding), shown as an interactive UMAP coloured by cluster / condition / batch / gene. Computes per-cluster **marker genes** (Wilcoxon) and provides a cluster→**cell-type annotation** form. The clustered object is persisted to `<output_dir>/clustering/clustered.h5ad` (labels to `annotations.json`) for the downstream quantification steps. Needs the `clustering` extra. |
| 7 | **📊 Composition** | Reads the clustered object and compares **per-replicate** cell-type proportions across conditions: a stacked per-sample overview, a per-cell-type dot plot (one dot per replicate, bar = condition mean), and an effect-size table (log2 fold-change + an *exploratory* t-test). Proportions are computed per biological replicate — never per cell — and the UI is explicit that at n≈2 this is discovery, not significance. |
| 8 | **🧪 Pseudobulk DGE** | Within a chosen cell type, pseudobulks the cells per replicate (reusing the Sample-PCA aggregator), then tests each gene across the replicate-level CPM values — a **volcano** + table + per-cell-type DE-count summary. Replicate-level (not per-cell) to avoid pseudoreplication; at n≈2 it's effect-size ranking, with a count model (DESeq2/edgeR) recommended for publication. |
| 9 | **🗺️ Spatial maps & niches** | Uses the Xenium cell coordinates the other steps ignore: per-slide **cell-type maps** (optionally highlighting one type) and a **neighbourhood-enrichment** heatmap (permutation z-score for which cell types are spatial neighbours more/less than chance, shuffled within slide). The enrichment can be split by condition to surface aging niche changes. |
| 10 | **🎯 Gene focus** | Quantitative analysis of one gene (default **Galanin / Gal**): expression + detection rate per cluster, **per-cluster differential expression** across conditions (pseudobulk per replicate, log2FC forest + table), a per-slide spatial expression map, and a **spatial age-effect grid** (AGED−ADULT difference per MBH sub-region, on slide-normalised coordinates). Replicate-level throughout; discovery only at n≈2. |

The landing page also carries a progress summary (slides configured, ROIs saved, Sample PCA status, Leiden resolution, PCA components) and two diagnostics in expanders:

- **🗂 Paths & environment** — shows where the app is running from and where each configured path (base panel CSV, ROI cache, output dir) points, flags any that resolve to a *different* checkout of the project (a common cause of stale ROIs when several copies coexist), and offers a one-click reset of the repo-relative paths to the running checkout.
- **🪵 Debug log** — the app and the `xenium_spatial` package write to a rotating log at `logs/xenium_app.log`. The panel has a verbosity selector (DEBUG…ERROR) and produces a single **copy-paste block** combining the environment + key package versions with the recent log lines (length selectable) — copy it (the code-block copy icon) and paste it back when reporting an issue, so the report carries both the activity and the validation context. Full-log download / clear are also there.

---

## Command line

Once paths and ROIs are set (the runner reads the same `roi_cache/`):

```bash
python scripts/run_sample_pca.py                 # load all configured slides, base panel only, run PCA
python scripts/run_sample_pca.py --samples AGED_1 ADULT_1   # run on a subset (>=2 samples)
python scripts/run_sample_pca.py --all-genes     # include per-slide add-on genes, not just the base panel
python scripts/run_sample_pca.py --no-roi        # use whole sections (skip ROI filtering)
python scripts/run_sample_pca.py --n-top-genes 200 --scale-genes   # restrict to top-variable genes, z-scored
python scripts/run_sample_pca.py --fmt png       # PNG instead of PDF figures
```

Slide paths are configured at the top of `scripts/run_sample_pca.py` (the `SLIDES` list), mirroring the web app's Study Setup. By default the PCA is restricted to the shared base panel and uses every configured slide; `--samples` selects a subset (minimum 2) and `--all-genes` opts back into the add-on genes. The web app's Sample PCA page exposes the same controls (a sample multiselect and a "Base panel only" toggle).

---

## How the PCA works

The analysis lives in `src/xenium_spatial/sample_pca.py` and runs in these steps:

0. **Restrict to the base panel** (default) — drop per-slide add-on genes so every sample is compared on the shared `Xenium_mBrain_v1_1` panel (~247 genes). This matters because samples can carry different add-on panels; pass `--all-genes` (or untick "Base panel only") to keep them.
1. **Pseudobulk** (`pseudobulk_samples`) — sum raw counts across all cells of each slide, giving one expression profile per biological replicate (one point per sample).
2. **Normalise** (`normalize_pseudobulk`) — library-size normalise each sample to counts-per-million, then `log1p`. Without this, PCA would just rank samples by cell number / sequencing depth.
3. **PCA** (`run_sample_pca`) — PCA across samples via scikit-learn. Uses all (base panel) genes by default (recommended for targeted Xenium panels); optionally restricts to the top-variable genes and/or z-scores genes.
4. **Plot** — a PC1/PC2 scatter coloured by group with sample labels, a sample-by-sample correlation heatmap ordered by hierarchical clustering, and a scree plot. (With only two samples PCA yields a single component, so the scatter spreads the samples along PC1.)

Pseudobulk PCA is the standard QC / sanity-check for replicated studies (cf. DESeq2's `plotPCA`): each point is one biological replicate, so it is robust at n=4 per group, and it makes outlier slides immediately visible.

---

## How the Leiden Optimizer works

The optimizer lives in `src/xenium_spatial/leiden_optimizer.py` and is driven by the **🔎 Leiden Optimizer** page. Where the Sample PCA collapses each slide to one point, the optimizer works at the **single-cell** level to answer the next question: *at what resolution should the cells be clustered?*

It runs in three stages:

1. **Load + embed** (`preprocess_for_clustering`) — the same slides and ROIs as the Sample PCA are loaded and concatenated, then a single-cell embedding is built on the fly: `normalize_total` → `log1p` → PCA → `sc.pp.neighbors` (the KNN graph). `obsm['spatial']` is carried through so spatial metrics stay available. The refactored pipeline only pseudobulks the cells, so this substrate (a PCA embedding + neighbour graph) does not otherwise exist — the optimizer builds it itself.
   - **How many PCs?** — rather than guessing the number of principal components for the KNN graph, the page's **📐 How many PCs? — elbow plot** tool estimates it from the data. It uses the two-criterion [HBC elbow heuristic](https://hbctraining.github.io/scRNA-seq/lessons/elbow_plot_metric.html) (`compute_elbow_n_pcs`): the recommended cutoff is the more conservative of (a) the first PC past 90% cumulative variation that itself adds < 5%, and (b) the last PC whose drop in variation to the next is still > 0.1%. Every embedding also records this recommendation in `adata.uns['pca_elbow']`, and `plot_pca_elbow` saves the scree/elbow figure.
2. **Optional Harmony integration** (`run_harmony`) — when several slides are pooled, plain PCA tends to separate cells by *which slide* they came from rather than by cell type. Harmony corrects the embedding (batch = each slide's **`batch`** label, see [batch correction](#batch-correction)) before the neighbour graph is built, so clustering and every metric below are computed on the batch-corrected space. On by default for multi-slide runs.
3. **Resolution sweep** (`optimize_leiden_resolution`) — Leiden clustering is run across a grid of resolutions, and each is scored with five complementary cluster-quality metrics:

| Metric | Direction | What it captures |
|--------|-----------|------------------|
| Silhouette | higher = better | Cluster separation in PCA / Harmony space |
| Calinski-Harabasz | higher = better | Between- vs within-cluster variance ratio |
| Davies-Bouldin | lower = better | Average similarity to the most-similar cluster |
| Spatial coherence | higher = better | Fraction of each cell's spatial neighbours in the same cluster |
| Modularity | higher = better | Community structure quality on the KNN graph |

Each metric is min-max normalised to [0, 1] and combined into a single weighted score. With spatial coordinates the weights are **silhouette 30% · Calinski-Harabasz 15% · Davies-Bouldin 15% · spatial coherence 20% · modularity 20%**; without them, silhouette and modularity each take 35%. The resolution with the highest combined score is recommended.

The page shows the per-metric curves, a **clustree** (Sankey diagram of how clusters split and merge across resolutions), and a one-click **Apply** that writes the chosen resolution to the pipeline settings. Silhouette is O(n²), so metrics are computed on a subsample (50k cells by default).

---

## Cell-level quantification

Once you **apply** a resolution in the Leiden Optimizer, the cell-level steps build and reuse a single artifact — `<output_dir>/clustering/clustered.h5ad` (UMAP, Leiden labels, cell-type annotation, condition / replicate / batch, and the spatial coordinates) — so every downstream view shares the same cells and labels.

| Page | Question | Method |
|------|----------|--------|
| **🔬 Clusters** | What are the cell types? | UMAP + Leiden at the applied resolution on the Harmony embedding; Wilcoxon marker genes per cluster; a cluster → cell-type annotation form (saved to `annotations.json`, baked into `obs['cell_type']`). |
| **📊 Composition** | Do cell-type *proportions* shift with condition? | Per-replicate proportions, AGED vs ADULT; log2 fold-change + an exploratory t-test. |
| **🧪 Pseudobulk DGE** | Which genes change *within* a cell type? | Pseudobulk per replicate within the cell type; Welch test on CPM; volcano + per-cell-type DE counts. |
| **🗺️ Spatial maps & niches** | Where do cell types sit, and what co-localises? | Per-slide cell-type maps; neighbourhood-enrichment z-score from within-slide label permutation. |
| **🎯 Gene focus** | How does *one gene* behave? | Expression + detection per cluster, per-cluster DE, a spatial expression map, and an AGED−ADULT spatial age-effect grid. |

**Statistical stance.** Every condition comparison (composition, DGE, gene focus) is computed at the **biological-replicate** level — proportions or pseudobulk summed per sample, never per cell — to avoid pseudoreplication (treating thousands of cells as independent observations, which manufactures significance). With the usual 2-vs-2 design the per-feature tests are underpowered, so the UI is explicit that these are **effect-size / discovery** analyses: rank by effect size, read the per-replicate dots, and validate hits in an independent cohort (and with a count model such as DESeq2/edgeR for DGE) before reporting them.

### Section straightening

If a section is mounted at an angle, the **ROI Manager** lets you record a per-slide rotation (slider + ✨ principal-axis *Auto-suggest*) that brings the brain to a canonical upright orientation. The rotation is saved inside the ROI file (`roi_cache/<slide>_roi.json`, as a `transform` block; absent = identity, so existing ROIs are unchanged) and applied to `obsm['spatial']` at load time by `roi_selector` — using the *same* `transform.py` maths the Manager previews, so the framed box and the filtered cells can never disagree. The raw coordinates are kept in `obsm['spatial_raw']`.

Because rotation is a **rigid** transform, it does not change clustering, marker genes, composition, pseudobulk DGE, or neighbourhood enrichment (all rotation-invariant). What it fixes is (1) the axis-aligned **bounding-box ROI** — an upright section lets a tight box capture just the MBH instead of dragging in oblique tissue, which in turn cleans up *every* downstream analysis via the cell set that enters it; (2) the **orientation of the spatial maps**; and (3) the **spatial age-effect grid**, whose dorsal→ventral (y) and medial↔lateral (x) axes only line up across slides once each section is straightened. After changing a rotation, re-save the ROI and re-run Sample PCA / rebuild the clustering (the ROI-file change invalidates the load caches automatically).

---

## Batch correction

Harmony (Leiden Optimizer) and the Sample-PCA scatter both use a per-slide **`batch`** label, set in **Study Setup** (blank = the slide's own `slide_id`). This carries through the loader to `adata.obs['batch']`.

**Why it matters.** In a 4 + 4 replicate design, each slide *is* a biological replicate of its condition. If the batch is left at the default `slide_id`, Harmony integrates *across* the conditions and can attenuate genuine AGED-vs-ADULT signal — the very effect you are studying. To remove technical variation **without** erasing the condition difference, set a `batch` that is **shared across conditions** (e.g. the sequencing run / capture date, so each batch contains both AGED and ADULT slides). When batches are *crossed* with condition like this, Harmony corrects the run effect while preserving biology.

The Leiden Optimizer checks this automatically: it warns when every batch maps to a single condition (confounded) and confirms when batches are crossed. On the Sample-PCA scatter, batch is drawn as the **marker shape** (condition stays the colour) — if samples group by shape rather than colour, a technical batch is driving the separation. (Sample PCA itself does not *correct* for batch — with one pseudobulk point per sample that would be inappropriate; the marker shape is a diagnostic.)

A manifest CSV for the CLI may carry the batch as an optional 5th column (`slide_id, condition, run_dir, replicate_id, batch`).

---

## Panel structure

Every Xenium run produces one count matrix containing all genes for that slide:

| Group | Count | Description |
|-------|-------|-------------|
| **Base panel** | ~247 | `Xenium_mBrain_v1_1` — identical across all slides |
| **Custom panel** | ~50 | Additional genes — differs between slides, partial overlap |
| **Total** | ~297 | Stored together in `matrix.mtx.gz` |

`PanelRegistry` classifies each gene by comparing names against the base panel CSV and harmonises the slides to a common gene set before concatenation.

### Harmonisation modes

| Mode | Custom genes kept | Recommended when |
|------|-------------------|------------------|
| **`consensus`** | Present in **all** slides (and base in all) | **Default — strict intersection, never zero-fills** |
| `intersection` | None (base only) | You only need the 247 base panel genes |
| `partial_union` | Present in ≥ `min_slides` slides | More custom genes, accepts zero-filling |
| `union` | All custom genes | Exploratory analysis only |

In `partial_union` / `union` modes, slides missing a retained custom gene receive a zero-filled column. Because a gene can be zero-filled in one slide yet measured in another, the concatenated AnnData records this per slide in `adata.varm['zero_filled_by_slide']` (genes × slides), with study-level summaries in `adata.var['zero_filled_any']` and `adata.var['n_slides_zero_filled']`.

### Consensus panel

The default `consensus` mode is the **strict intersection** of every configured sample's panel — base genes present in all samples, plus add-on genes present in all samples. Because every consensus gene is genuinely measured in every sample, **nothing is ever zero-filled**, which structurally removes the main way a differing add-on panel could fake a differential-expression hit (a gene simply absent from a panel otherwise reads as "not expressed" there).

The **🧬 Consensus Panel** page (step 2) builds this from each slide's `features.tsv.gz` and shows exactly what is kept and what is excluded:

- the **shared add-on genes** that enter the consensus;
- any **partially-shared add-on genes** that are dropped, with a sample × gene presence matrix so you can see which samples carried each one;
- any **base gene** missing from a sample (excluded under strict intersection — normally none);
- per-sample contributions, and a `consensus_panel.json` / `.csv` export.

Clicking **Use this consensus panel** records `consensus_panel.json` in the output directory and keeps `consensus` active for Sample PCA, clustering and all downstream quantification. The `partial_union` / `union` / `intersection` modes remain available via the study-config JSON for users who deliberately want the wider (zero-filled) gene set.

---

## Outputs

All files are written to `<output_dir>/sample_pca/` (web app) or `figures_output_sample_pca/` (CLI):

| File | Description |
|------|-------------|
| `sample_pca_scatter.pdf` | PC1 vs PC2, coloured by group, samples labelled (Nature-style) |
| `sample_correlation_heatmap.pdf` | Sample-by-sample correlation, hierarchically ordered |
| `sample_pca_scree.pdf` | Variance explained per PC + cumulative line |
| `sample_pca_coordinates.csv` | PC coordinates + condition / batch / n_cells / total_counts per sample |
| `sample_pca_variance.csv` | Variance ratio and cumulative variance per PC |
| `pseudobulk_samples.h5ad` | Pseudobulk AnnData (counts, lognorm, `obsm['X_pca']`) |

The publication figures (Sample-PCA scatter / correlation heatmap / scree, and the PCA elbow) follow **Nature Publishing Group** conventions via a shared style (`src/xenium_spatial/figure_style.py`): a sans-serif typeface (Arial/Helvetica), thin spines, editable PDF (Type-42 fonts), and colour-blind-safe [Wong (2011)](https://doi.org/10.1038/nmeth.1618) group colours, at a font scale sized for single-column print. (The interactive quantification charts in steps 5–9 are Plotly, for on-screen exploration.)

The Leiden Optimizer writes to `<output_dir>/leiden_optimizer/`:

| File | Description |
|------|-------------|
| `leiden_resolution_sweep.csv` | Per-resolution metrics (n_clusters, silhouette, CH, DB, spatial coherence, modularity, combined score) |
| `pipeline_settings.json` | The applied `leiden_resolution`, restored on app start and merged into the study config |

The recommended resolution is also stored in session state and saved with the study configuration JSON from **Study Setup**, so it travels with the rest of your settings.

The cell-level quantification persists to `<output_dir>/clustering/`:

| File | Description |
|------|-------------|
| `clustered.h5ad` | The final clustering — UMAP, Leiden labels, `cell_type` annotation, condition / replicate / batch, spatial coords; the shared input to steps 6–9 |
| `annotations.json` | The cluster → cell-type label mapping |

Each quantification page also offers its own CSV downloads (cluster marker genes, composition stats and per-replicate proportions, per-cell-type DGE tables, neighbourhood-enrichment z-scores, and per-cluster gene fold-changes) **and a publication-ready PDF for every on-screen chart** — the interactive Plotly view is for exploration, while a one-click *“(PDF, publication)”* button beside it exports the matching **Nature-style, Type-42-editable** vector figure (shared style with the Sample-PCA / elbow figures). Covered figures: the UMAP and top-marker heatmap (Clusters); the stacked-composition and proportion-by-condition plots (Composition); the volcano (DGE); the cell-type map and neighbourhood-enrichment heatmap (Spatial); and the expression violins, per-cluster fold-change bar, spatial expression map and age-effect grid (Gene focus).

A `panel_validation.csv` (per-slide base/custom gene breakdown) is written to the **output directory** alongside these results. The app's diagnostic log goes to `logs/xenium_app.log` (see the **🪵 Debug log** panel).

---

## Configuration file format

Study Setup can save/load a JSON configuration so you never re-enter paths:

```json
{
  "slides": [
    { "run_dir": "/path/to/AGED_1_output", "slide_id": "AGED_1", "condition": "AGED", "batch": "run_0701" }
  ],
  "output_dir": "/path/to/results",
  "base_panel_csv": "data/Xenium_mBrain_v1_1_metadata.csv",
  "roi_cache_dir": "roi_cache",
  "leiden_resolution": 0.6,
  "n_pcs": 50
}
```

Only `slides` is required; the rest fall back to sensible defaults. Each slide's `batch` is optional (blank/absent → the slide's `slide_id`; see [batch correction](#batch-correction)). `leiden_resolution` and `n_pcs` are filled in by the Leiden Optimizer when you apply a recommendation (defaults `0.6` / `50` until then).

---

## Project structure

```
xenium-spatial-analysis/
├── pyproject.toml               Packaging metadata + optional-dependency extras
├── requirements.txt             Pinned dependency superset (alternative to extras)
├── LICENSE                      MIT
├── start_app.command            Double-click to launch the web interface
├── install_mac.sh               macOS installer (Apple Silicon)
│
├── src/
│   └── xenium_spatial/          Core analysis library (the installable package)
│       ├── __init__.py          Lazy public API (SlideManifest, PanelRegistry, …)
│       ├── xenium_loader.py     Load a Xenium run directory into AnnData
│       ├── multislide_loader.py Multi-slide manifest, validation, concat
│       ├── panel_registry.py    Gene classification and panel harmonisation
│       ├── roi_selector.py      ROI persistence + apply (reads roi_cache/, straightens coords)
│       ├── transform.py         Per-slide section-straightening rotation (shared maths)
│       ├── sample_pca.py        Pseudobulk, normalise, PCA, and figures
│       ├── leiden_optimizer.py  Cell-level embedding, elbow PC selection, Leiden sweep
│       ├── cell_clustering.py   UMAP + Leiden labels, marker genes, annotation
│       ├── composition.py       Per-replicate cell-type proportions + stats
│       ├── pseudobulk_dge.py    Within-cell-type pseudobulk differential expression
│       ├── spatial.py           Cell-type maps + neighbourhood-enrichment
│       ├── gene_focus.py        Single-gene quantification (expression, DE, spatial grid)
│       ├── figure_style.py      Shared Nature-grade matplotlib style (rcParams)
│       └── figure_export.py     Nature-style PDF renderers for the quantification charts
│
├── app/                         Web interface (Streamlit)
│   ├── app.py                   Landing page: progress + paths/log diagnostics
│   ├── ui_utils.py              Shared helpers: session init, logging, paths/log panels, CSS
│   ├── pipeline.py              Shared cached loaders (embedding + clustered.h5ad)
│   ├── styles.css               Custom Streamlit styles
│   ├── .streamlit/config.toml   Theme and server settings
│   └── pages/
│       ├── 1_study_setup.py     Slide folders (incl. batch) + JSON save/load
│       ├── 2_consensus_panel.py Consensus gene set (strict intersection) the pipeline runs on
│       ├── 3_roi_manager.py     Interactive ROI framing (draw-box / sliders / straighten)
│       ├── 4_sample_pca.py      Pseudobulk PCA + Nature-style figures
│       ├── 5_leiden_optimizer.py  Elbow plot, resolution sweep, scoring, clustree
│       ├── 6_clusters.py        UMAP + marker genes + cell-type annotation
│       ├── 7_composition.py     Per-replicate cell-type composition shifts
│       ├── 8_dge.py             Within-cell-type pseudobulk DGE + volcano
│       ├── 9_spatial.py         Cell-type maps + neighbourhood niches
│       └── 10_gene_focus.py     Single-gene quantification (default Gal)
│
├── scripts/
│   └── run_sample_pca.py        CLI entry point for the sample PCA
│
├── tests/                       pytest suite (elbow, manifest, composition,
│                                DGE, spatial, gene-focus, …)
│
├── data/
│   └── Xenium_mBrain_v1_1_metadata.csv   Base panel gene list + annotations
│
└── .github/workflows/ci.yml     Lint/test on push + PR
```

---

## Requirements

See [`requirements.txt`](requirements.txt). Key packages:

| Package | Min version | Purpose |
|---------|-------------|---------|
| streamlit | 1.35 | Web interface |
| plotly | 5.20 | Interactive ROI scatter |
| numpy / pandas / scipy | — | Core numerics |
| scikit-learn | 1.3 | PCA |
| matplotlib | 3.8 | Nature-style figures |
| anndata | 0.10 | Annotated data matrices |
| pyarrow | 14.0 | Parquet support (`cells.parquet`) |

Cell-level steps only (steps 4–9 — lazily imported, not needed for Sample PCA):

| Package | Min version | Purpose |
|---------|-------------|---------|
| scanpy | 1.10 | Normalisation, PCA, neighbour graph, Leiden, UMAP, marker genes |
| igraph | 0.11 | Graph backend + modularity |
| leidenalg | 0.10 | Leiden community detection |
| harmonypy | 0.0.9 | Cross-slide batch integration (Harmony) |

scikit-learn (a core dependency) also powers the spatial neighbourhood-enrichment graph.

---

## Troubleshooting

**ROI sliders not responding**
Refresh the page (Cmd+R / Ctrl+R). If it persists, use the manual coordinate entry panel to type x,y pairs directly.

**`cell_feature_matrix/` not found**
The selected path must be the Xenium run output directory itself, not a parent folder. It must directly contain `cell_feature_matrix/` (with `matrix.mtx.gz`, `barcodes.tsv.gz`, `features.tsv.gz`) and `cells.parquet`.

**PCA separates samples by cell number, not biology**
This usually means library-size normalisation was bypassed. The built-in workflow always CPM-normalises before PCA; if you are calling the functions directly, run `normalize_pseudobulk` before `run_sample_pca`.

**Custom genes not appearing after harmonisation**
Lower `min_slides`, or switch `panel_mode` to `union`.

**Leiden clusters track the slide instead of cell type**
This is batch effect across slides. Enable **Harmony batch correction** on the Leiden Optimizer page (on by default for multi-slide runs) so clustering happens on the integrated embedding. Set a `batch` shared across conditions first — see [batch correction](#batch-correction) — so you don't remove the condition signal along with the batch effect.

**`No module named 'scanpy'` on the Leiden Optimizer / cell-level pages**
The cell-level steps (4–9) need the single-cell stack. Install it with `pip install -e ".[clustering]"` (or `pip install scanpy igraph leidenalg harmonypy`); steps 1–3 (Study Setup → ROI Manager → Sample PCA) do not require it.

**"No clustering found" on Clusters / Composition / DGE / Spatial / Gene focus**
Those pages read `clustering/clustered.h5ad`. Build it first: on the **🔬 Clusters** page set the embedding (match what you swept) and click *Build clustering*. The other quantification pages then pick it up automatically (they cache on the file's modification time, so re-building or re-annotating refreshes them).

**ROI selects 0 cells**
The MBH sits in the ventral 50–80% of a coronal section (larger y, since y increases toward ventral). Re-frame by dragging a box there; the live cell count confirms when the region is populated.

---

## Citation

If you use this tool, please cite the underlying methods:

| Tool | Reference |
|------|-----------|
| scikit-learn (PCA) | Pedregosa et al., *JMLR* 2011 |
| AnnData | Virshup et al., *JOSS* 2024 |
| Colour palette | Wong, *Nature Methods* 2011 |
| Xenium | 10x Genomics Xenium In Situ platform |
