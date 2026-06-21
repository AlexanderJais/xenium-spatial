# Xenium Spatial Pipeline — macOS Quick Start (Apple Silicon)

> Tested on MacBook Pro M1/M2/M3/M4, macOS Ventura/Sonoma/Sequoia.
> For full documentation see [README.md](README.md).

---

## Prerequisites

You need **nothing** pre-installed except macOS. The installer handles everything.

---

## 1. Install (one-time)

Open **Terminal**, navigate to the project folder, and run:

```bash
cd /path/to/xenium-spatial-analysis
chmod +x install_mac.sh
./install_mac.sh
```

This installs Miniforge3 (ARM64 conda) if needed, creates a Python 3.11 environment, and installs the dependencies:

| Component | Details |
|-----------|---------|
| Python 3.11 | Native Apple Silicon via conda-forge |
| Core stack | numpy, pandas, scipy, scikit-learn, matplotlib |
| Data | anndata, pyarrow (`cells.parquet`) |
| Web interface | streamlit, plotly |
| Cell-level steps (4–9) | scanpy, igraph, leidenalg, harmonypy |

The cell-level (single-cell) stack is installed last and is non-fatal: if it fails, the Sample-PCA workflow (steps 1–3) still works and you can add it later with `conda install -n xenium_sample_pca -c conda-forge scanpy python-igraph leidenalg harmonypy`.

Alternatively, in any environment: `pip install -r requirements.txt`.

---

## 2. Launch

**Option A — Web app (recommended):**

Double-click `start_app.command` in Finder, or from Terminal:
```bash
conda activate xenium_sample_pca
streamlit run app/app.py
```
Your browser opens at http://localhost:8501.

**Option B — Command line (headless, after ROIs are saved):**
```bash
conda activate xenium_sample_pca
python scripts/run_sample_pca.py            # apply saved ROIs, run PCA
python scripts/run_sample_pca.py --no-roi   # use whole sections
```

---

## 3. Step 1 — Study Setup

Each Xenium run directory must contain:

```
<run_dir>/
    cell_feature_matrix/
        barcodes.tsv.gz
        features.tsv.gz
        matrix.mtx.gz
    cells.parquet
    experiment.xenium
```

Go to **📁 Study Setup** and paste the full path to each run directory. Use **➕ Add slide** / the 🗑 button to match your sample count (the page starts with the 4 + 4 AGED/ADULT template), and edit the condition labels if your groups differ. A green tick confirms validity; the page shows the cell and gene counts per slide once validated.

**Tip:** On macOS, right-click a folder in Finder → Get Info → copy the path from *Where*.

Click **Save configuration to JSON** to store all paths so you never re-enter them — **Load** restores them next session. See the [README](README.md#configuration-file-format) for the schema.

---

## 4. Step 2 — ROI Manager

Define the mediobasal hypothalamus (MBH) boundary on each slide.

1. Select a slide from the dropdown (or step through with the ◀/▶ buttons).
2. **Drag a box** on the tissue to frame the MBH bounding rectangle, or fine-tune with the four edge sliders (left/right x, top/bottom y).
3. The scatter, the live cell count, and the box dimensions update as you adjust. The MBH sits in the ventral 50–80% of a coronal section (larger y = ventral).
4. Click **Save ROI** when the rectangle covers the MBH (turn on *auto-advance* to jump to the next unsaved slide).

**Precise coordinates:** use the *Paste coordinates* panel — one `x, y` pair per line in micrometres:
```
3200, 4100
3800, 4100
3800, 4700
3200, 4700
```

**Copy ROIs:** if sections are at similar coordinates, save once and copy to other slides via *Copy to other slides*. Saved ROIs live in `roi_cache/` and are reused automatically on every run.

---

## 5. Step 3 — Sample PCA

Go to **📊 Sample PCA** and click **Run sample PCA**. The app loads the slides, applies the saved ROIs, pseudobulks each sample, and runs PCA across them.

Options:
- **Samples to include** — pick which samples go into the PCA (minimum 2); the rest are ignored for that run.
- **Base panel only** — on by default; restricts the PCA to the shared base panel so samples with different add-on panels stay comparable. Untick to include add-on genes.
- **Apply MBH ROIs** — on by default once ROIs exist; turn off to use whole sections.
- **Top variable genes** — 0 uses all genes (recommended for the targeted panel).
- **Z-score genes** — off by default (`log1p` already stabilises variance).

You get three figures inline:
- **PCA scatter** — PC1 vs PC2, samples coloured by group (AGED/ADULT) and individually labelled.
- **Correlation heatmap** — sample-by-sample correlation, hierarchically ordered (spot outliers).
- **Scree plot** — variance explained per PC.

### Output files

Written to `<output_dir>/sample_pca/`:

| File | Description |
|------|-------------|
| `sample_pca_scatter.pdf` | PC1 vs PC2 coloured by group |
| `sample_correlation_heatmap.pdf` | Hierarchically-ordered sample correlation |
| `sample_pca_scree.pdf` | Variance explained per PC |
| `sample_pca_coordinates.csv` | PC coordinates + metadata per sample |
| `sample_pca_variance.csv` | Variance ratios |
| `pseudobulk_samples.h5ad` | Pseudobulk AnnData |

---

## 6. Step 4 — Leiden Optimizer (optional)

Once the Sample PCA looks sensible, go to **🔎 Leiden Optimizer** to choose a cell-level clustering resolution by metrics instead of by eye. It reuses the same slides and ROIs, builds a single-cell PCA + KNN graph, then sweeps Leiden resolutions and scores each on silhouette, Calinski-Harabasz, Davies-Bouldin, spatial coherence, and modularity.

Set the options, then click **Run resolution sweep**:
- **Apply MBH ROIs / Base panel only** — same meaning as on the Sample PCA page.
- **PCA components / KNN neighbours** — the embedding and graph the sweep runs on (defaults 50 / 15 are fine for the targeted panel).
- **Harmony batch correction** — on by default for multi-slide runs. Integrates slides on each slide's **batch** label (set in Study Setup) so clusters reflect cell type, not which slide a cell came from. *In a 4 + 4 replicate design, leaving the batch at the default `slide_id` corrects across conditions and can dampen real AGED-vs-ADULT differences. Set a `batch` shared across conditions (e.g. the sequencing run / date) so each batch contains both conditions — the page warns when batches are confounded and confirms when they're crossed.*
- **Min / Max / Step resolution** — the grid to sweep (default 0.1 → 2.0 by 0.1).
- **Max cells for metric computation** — silhouette is O(n²); 50k is a good default, and the page warns above that.

You get the per-metric curves, a **clustree** (how clusters split/merge across resolutions), and a recommended resolution. Click **Apply recommended resolution** (or pick another from the sweep) to store it in the pipeline settings — it appears in the sidebar, is saved to `<output_dir>/leiden_optimizer/pipeline_settings.json`, and travels with the Study Setup config JSON.

### Output files

Written to `<output_dir>/leiden_optimizer/`:

| File | Description |
|------|-------------|
| `leiden_resolution_sweep.csv` | Per-resolution metrics and combined score |
| `pipeline_settings.json` | The applied `leiden_resolution` (restored on app start) |

---

## 7. Steps 5–9 — cell-level quantification (optional)

After you **apply** a resolution, the remaining pages quantify the cells. Start on **🔬 Clusters** and click *Build clustering* — it computes the UMAP + Leiden labels and writes `<output_dir>/clustering/clustered.h5ad`, which every later page reads.

| Page | What you get |
|------|--------------|
| **🔬 Clusters** | UMAP (colour by cluster / condition / batch / gene), per-cluster **marker genes**, and a cluster → **cell-type** annotation form. |
| **📊 Composition** | Per-replicate cell-type **proportions**, AGED vs ADULT (dot plot + effect-size table). |
| **🧪 Pseudobulk DGE** | **Differential expression** within a cell type across conditions (volcano + table). |
| **🗺️ Spatial maps & niches** | Cell-type **maps** per slide and a neighbourhood-**enrichment** heatmap. |
| **🎯 Gene focus** | Everything for one **gene** (default Galanin): per-cluster expression + DE, spatial map, and an AGED−ADULT spatial grid. |

All condition comparisons are per biological replicate; at n ≈ 2 they're for **discovery** (effect sizes), not significance. These pages need the `clustering` extra installed.

---

## Troubleshooting

**ROI sliders not responding**
Refresh the page (Cmd+R). If it persists, use the *Paste coordinates* panel.

**`cell_feature_matrix/` not found**
The path must be the Xenium run directory itself (containing `cell_feature_matrix/` and `cells.parquet`), not a parent folder.

**ROI selects 0 cells**
The MBH is ventral (larger y). Drag the box lower on the section; the live cell count confirms when the region is populated.

**PCA separates samples by cell number, not biology**
The built-in workflow always CPM-normalises before PCA. If you call the functions directly, run `normalize_pseudobulk` before `run_sample_pca`.

**App is slow to load a slide scatter**
The ROI Manager loads `cells.parquet` on demand and subsamples large slides for display. A few seconds for very large sections is normal.

**Leiden clusters track the slide, not cell type**
Turn on **Harmony batch correction** on the Leiden Optimizer page (default on for multi-slide runs) so clustering runs on the batch-integrated embedding.

**`No module named 'scanpy'` on the Leiden Optimizer page**
The optimizer stack did not install. Run `conda install -n xenium_sample_pca -c conda-forge scanpy python-igraph leidenalg harmonypy`. The other three pages work without it.

For more, see the [README](README.md#troubleshooting).
