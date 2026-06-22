"""
pages/5_leiden_optimizer.py
Leiden Resolution Optimizer — automated resolution sweep with silhouette +
modularity scoring, run directly on the study's slides.

Unlike the Sample-PCA step (which pseudobulks the slides), the optimizer works
at the single-cell level: it loads the configured slides, applies the saved
MBH ROIs, builds a cell-level PCA embedding + KNN graph on the fly, then sweeps
a grid of Leiden resolutions and scores each with five complementary metrics.
One click applies the recommended resolution to the pipeline settings.
"""

import sys
import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header, init_session_state

st.set_page_config(
    page_title="Leiden Optimizer · Xenium Spatial Pipeline",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
init_session_state()

_ROOT = Path(__file__).parent.parent.parent
# Make the xenium_spatial package importable without an editable install
# (src layout: the package lives under <repo>/src).
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_dir(p: str) -> bool:
    return bool(p) and (Path(p) / "cell_feature_matrix" / "matrix.mtx.gz").exists()


def _roi_signature(slide_ids, roi_dir) -> tuple:
    """Signature that changes when any ROI file changes — invalidates the cache."""
    sig = []
    for sid in slide_ids:
        p = Path(roi_dir) / f"{sid.replace('/', '_').replace(' ', '_')}_roi.json"
        sig.append((sid, p.stat().st_mtime if p.exists() else 0.0))
    return tuple(sig)


@st.cache_resource(show_spinner=False)
def _load_and_preprocess(run_dirs, slide_ids, conditions, batches, base_csv, roi_dir,
                         use_roi, panel_mode, min_slides, roi_sig,
                         base_panel_only, n_pcs, n_neighbors, scale_genes,
                         batch_key, output_dir=None):
    """Load + harmonise + ROI-filter + concatenate slides, then build the
    cell-level PCA embedding and KNN graph the Leiden sweep needs (cached).

    ``roi_sig`` is the per-slide ROI-file signature (see ``_roi_signature``).
    It must NOT be underscore-prefixed: Streamlit skips hashing any argument
    whose name starts with ``_``, which would exclude it from the cache key
    and leave the cache stale after an ROI is edited.
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


@st.cache_data(show_spinner=False)
def _estimate_pca_elbow(run_dirs, slide_ids, conditions, base_csv, roi_dir,
                        use_roi, panel_mode, min_slides, roi_sig,
                        base_panel_only, scale_genes, max_pcs, output_dir=None):
    """Load + embed the slides with a generous PCA, then return the elbow-plot
    data and the recommended number of PCs.

    Kept separate from ``_load_and_preprocess`` (and from Harmony / the KNN
    graph) so the elbow estimate is driven purely by the data, not by the
    ``n_pcs`` the user happens to have selected for the sweep. Cached on the
    same loading inputs so it is cheap to re-open.
    """
    from xenium_spatial.multislide_loader import SlideManifest, MultiSlideLoader
    from xenium_spatial.panel_registry import PanelRegistry
    from xenium_spatial.roi_selector import ROISelector
    from xenium_spatial.sample_pca import _restrict_to_base_panel
    from xenium_spatial.leiden_optimizer import preprocess_for_clustering

    manifest = SlideManifest()
    for sid, cond, d in zip(slide_ids, conditions, run_dirs):
        manifest.add(slide_id=sid, condition=cond, run_dir=d, replicate_id=sid)

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

    # n_neighbors is irrelevant to the PCA variance curve; keep it small/cheap.
    adata = preprocess_for_clustering(
        adata, n_pcs=max_pcs, n_neighbors=2, scale_genes=scale_genes,
        batch_key=None,
    )
    return dict(adata.uns["pca_elbow"])


def _persist_resolution(res: float, output_dir: Path) -> Path:
    """Write the chosen Leiden resolution to a settings JSON in the output dir.

    Persists the value beyond the Streamlit session so a future clustering step
    (or a re-launched app via Study Setup's config load) can pick it up.
    """
    out = Path(output_dir) / "leiden_optimizer"
    out.mkdir(parents=True, exist_ok=True)
    settings_path = out / "pipeline_settings.json"
    settings = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except Exception:
            settings = {}
    settings["leiden_resolution"] = float(res)
    settings["n_pcs"] = int(st.session_state.get("n_pcs", 50))
    settings_path.write_text(json.dumps(settings, indent=2))
    return settings_path


def _build_clustree(cluster_df: pd.DataFrame, best_res: float) -> None:
    """Build and display a clustree Sankey diagram.

    For each consecutive pair of resolutions, compute the overlap matrix:
    for every cluster at resolution r_{i+1}, count how many cells came from
    each cluster at resolution r_i.  Render as a Plotly Sankey diagram.
    """
    import plotly.graph_objects as go

    all_cols = sorted(cluster_df.columns, key=lambda c: float(c.split("_")[1]))
    if len(all_cols) < 2:
        st.info("Need at least 2 resolutions for a clustree plot.")
        return

    # Limit to avoid overly dense diagrams — subsample if > 12 levels
    if len(all_cols) > 12:
        step = max(1, len(all_cols) // 12)
        cols = all_cols[::step]
        if cols[-1] != all_cols[-1]:
            cols.append(all_cols[-1])
    else:
        cols = all_cols

    node_labels, node_x, node_y, node_colors = [], [], [], []
    node_map = {}  # (col_name, cluster_id) -> node_index
    n_levels = len(cols)

    for level_i, col in enumerate(cols):
        res_val = float(col.split("_")[1])
        clusters = sorted(cluster_df[col].unique(), key=lambda x: int(x))
        n_cl = len(clusters)
        for ci, cl in enumerate(clusters):
            node_map[(col, cl)] = len(node_labels)
            node_labels.append(f"r{res_val:.1f} c{cl}")
            node_x.append((level_i + 0.5) / (n_levels + 0.5))
            node_y.append((ci + 0.5) / max(n_cl + 0.5, 1))
            if abs(res_val - best_res) < 0.005:
                node_colors.append("rgba(231, 76, 60, 0.85)")
            else:
                node_colors.append("rgba(52, 152, 219, 0.65)")

    sources, targets, values, edge_colors = [], [], [], []
    for i in range(len(cols) - 1):
        col_lo, col_hi = cols[i], cols[i + 1]
        lo = cluster_df[col_lo].values
        hi = cluster_df[col_hi].values
        clusters_lo = sorted(set(lo), key=lambda x: int(x))
        clusters_hi = sorted(set(hi), key=lambda x: int(x))
        for cl_hi in clusters_hi:
            mask_hi = hi == cl_hi
            total_hi = mask_hi.sum()
            if total_hi == 0:
                continue
            for cl_lo in clusters_lo:
                overlap = ((lo == cl_lo) & mask_hi).sum()
                if overlap == 0:
                    continue
                src = node_map.get((col_lo, cl_lo))
                tgt = node_map.get((col_hi, cl_hi))
                if src is not None and tgt is not None:
                    sources.append(src)
                    targets.append(tgt)
                    values.append(int(overlap))
                    frac = overlap / total_hi
                    edge_colors.append(
                        f"rgba(100, 100, 100, {max(0.08, min(0.6, frac))})"
                    )

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            label=node_labels, x=node_x, y=node_y, color=node_colors,
            pad=4, thickness=14, line=dict(color="black", width=0.3),
        ),
        link=dict(source=sources, target=targets, value=values, color=edge_colors),
    )])
    # Height scales with the *widest* level (most clusters in any single
    # resolution), since levels are laid out left-to-right. Using the total
    # node count across all levels produced an absurdly tall, tangled figure.
    max_nodes_per_level = max((cluster_df[c].nunique() for c in cols), default=1)
    fig.update_layout(
        title_text="Clustree: Cluster Lineage Across Resolutions",
        title_x=0.5, font_size=10,
        height=int(min(max(450, 40 * max_nodes_per_level), 1600)),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Page ──────────────────────────────────────────────────────────────────────
page_header(
    "🔎 Leiden Resolution Optimizer",
    "Automated resolution sweep with silhouette + modularity scoring",
)

st.markdown(
    "This runs Leiden clustering at multiple resolutions on your slides "
    "(loaded and ROI-filtered exactly like the Sample-PCA step, then "
    "embedded with PCA + a KNN graph), and scores each resolution with five "
    "complementary metrics:\n\n"
    "- **Silhouette score** — cluster separation in PCA space (higher = better)\n"
    "- **Calinski-Harabasz index** — between- vs within-cluster variance (higher = better)\n"
    "- **Davies-Bouldin index** — average similarity to most-similar cluster (lower = better)\n"
    "- **Spatial coherence** — fraction of spatial neighbours in the same cluster (higher = better)\n"
    "- **Modularity** — community structure quality on the KNN graph (higher = better)\n\n"
    "A weighted combined score identifies the resolution that best balances "
    "cluster quality and granularity. A **clustree** plot shows how clusters "
    "split and merge across resolutions. One click applies the recommendation "
    "to the pipeline settings.\n\n"
    "Before sweeping, the **📐 How many PCs? — elbow plot** tool (under "
    "*Preprocessing*, below) estimates how many principal components to keep "
    "for the embedding, so the *PCA components* setting is chosen from the data "
    "rather than guessed."
)
st.divider()

# ── Slide selection ───────────────────────────────────────────────────────────
slides = st.session_state.get("slides", [])
valid_slides = [s for s in slides if _valid_dir(s.get("run_dir", ""))]
slide_ids = [s["slide_id"] for s in valid_slides]
n_roi = sum(1 for sid in slide_ids if sid in st.session_state["roi_polygons"]
            or (Path(st.session_state["roi_cache_dir"])
                / f"{sid.replace('/', '_').replace(' ', '_')}_roi.json").exists())

if len(valid_slides) < 1:
    st.warning("No valid slides configured. Set them up in **📁 Study Setup**.")
    st.stop()

n_cond = len({s["condition"] for s in valid_slides})
st.markdown(
    f"**{len(valid_slides)}** valid slides across **{n_cond}** group(s) · "
    f"**{n_roi}/{len(valid_slides)}** ROIs saved."
)

selected_ids = st.multiselect(
    "Samples to include in the clustering",
    options=slide_ids,
    default=slide_ids,
    help="The cells from these slides are pooled, embedded, and clustered "
         "together. At least one slide is required.",
)
selected_slides = [s for s in valid_slides if s["slide_id"] in set(selected_ids)]
if len(selected_slides) < 1:
    st.warning("Select at least one slide to run the optimizer.")
    st.stop()

# ── Preprocessing options ─────────────────────────────────────────────────────
st.subheader("Preprocessing")
p1, p2, p3, p4 = st.columns(4)
with p1:
    use_roi = st.toggle("Apply MBH ROIs", value=(n_roi > 0),
                        help="Restrict each slide to its saved ROI before clustering.")
    if use_roi and n_roi < len(selected_slides):
        st.caption(f"⚠️ Only {n_roi}/{len(valid_slides)} ROIs saved — slides without one use the whole section.")
with p2:
    base_panel_only = st.toggle("Base panel only", value=True,
                                help="Cluster on the shared Xenium base panel (247 genes), "
                                     "dropping per-slide add-on genes so the embedding is "
                                     "comparable across slides.")
with p3:
    # Clamp a restored/config-loaded value into the widget's range before it is
    # instantiated; seeding the key with an out-of-range value would raise.
    st.session_state["n_pcs"] = max(2, min(200, int(st.session_state.get("n_pcs", 50))))
    n_pcs = st.number_input("PCA components", min_value=2, max_value=200, step=5, key="n_pcs",
                            help="Principal components used for the embedding and KNN graph. "
                                 "Not sure how many? Use the elbow-plot tool just below to "
                                 "estimate it from the data.")
with p4:
    n_neighbors = st.number_input("KNN neighbours", min_value=2, max_value=100, value=15, step=1,
                                  help="Neighbours for the graph that Leiden clusters and "
                                       "modularity is scored on.")
o1, o2 = st.columns(2)
with o1:
    scale_genes = st.toggle("Z-score genes before PCA", value=False,
                            help="Standardise each gene before PCA. Off by default "
                                 "(log1p already stabilises variance).")
with o2:
    n_selected = len(selected_slides)
    use_harmony = st.toggle(
        "Harmony batch correction", value=(n_selected > 1),
        disabled=(n_selected < 2),
        help="Integrate slides with Harmony on the PCA embedding before building "
             "the graph, using each slide's **batch** label (set it in Study "
             "Setup; defaults to slide_id). Use a batch shared across conditions "
             "(e.g. sequencing run/date) to remove technical variation without "
             "erasing the condition difference. Requires harmonypy.",
    )
    if n_selected < 2:
        st.caption("Only one slide selected — nothing to integrate.")
    elif use_harmony:
        st.caption("Clusters will be computed on the Harmony-corrected embedding (batch = `batch`).")

# ── How many PCs? (elbow plot) ────────────────────────────────────────────────
with st.expander("📐 How many PCs? — elbow plot", expanded=True):
    st.caption(
        "Estimate how many principal components actually carry signal, so the "
        "**PCA components** above isn't just a guess. Uses the two-criterion "
        "elbow heuristic from the "
        "[HBC scRNA-seq training](https://hbctraining.github.io/scRNA-seq/lessons/elbow_plot_metric.html): "
        "the recommended cutoff is the more conservative of (a) the first PC past "
        "90% cumulative variation that itself adds <5%, and (b) the last PC whose "
        "drop in variation to the next is still >0.1%."
    )
    if st.button("Estimate from data", key="estimate_pcs"):
        try:
            with st.spinner("Loading slides and computing PCA variance …"):
                run_dirs   = tuple(str(s["run_dir"]) for s in selected_slides)
                sids       = tuple(s["slide_id"] for s in selected_slides)
                conditions = tuple(s["condition"] for s in selected_slides)
                roi_sig    = _roi_signature(sids, st.session_state["roi_cache_dir"])
                elbow = _estimate_pca_elbow(
                    run_dirs, sids, conditions,
                    st.session_state["base_panel_csv"],
                    st.session_state["roi_cache_dir"], use_roi,
                    st.session_state["panel_mode"], int(st.session_state["min_slides"]),
                    roi_sig, bool(base_panel_only), bool(scale_genes), 50,
                    st.session_state["output_dir"],
                )
            st.session_state["pca_elbow"] = elbow
        except Exception as e:
            logging.getLogger("xenium_app").exception("PC estimation failed")
            st.error(f"Could not estimate PCs: {e}")

    elbow = st.session_state.get("pca_elbow")
    if elbow:
        import plotly.graph_objects as go

        pct = elbow["pct"]
        pcs = list(range(1, len(pct) + 1))
        n_rec = int(elbow["n_pcs"])
        st.success(
            f"**Recommended: {n_rec} PCs**  "
            f"(90%/5% cutoff at PC{elbow['co1']}, flattening cutoff at PC{elbow['co2']}). "
            f"Set **PCA components** above to {n_rec}."
        )
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=pcs, y=pct, mode="lines+markers", name="Std. dev. (%)",
            line=dict(color="#1B4F8A", width=2), marker=dict(size=5),
        ))
        fig_elbow.add_vline(
            x=n_rec, line_dash="dash", line_color="#D55E00",
            annotation_text=f"recommended: {n_rec}", annotation_position="top right",
        )
        fig_elbow.update_layout(
            xaxis_title="Principal component", yaxis_title="Std. dev. explained (%)",
            template="plotly_white", height=320, margin=dict(t=20, b=40, l=50, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

batch_key = "batch" if (use_harmony and len(selected_slides) > 1) else None

if batch_key is not None:
    # Map each batch label to the conditions it spans. If every batch sits inside
    # a single condition (e.g. the default where batch == slide_id), Harmony pulls
    # the conditions together and can erase the very difference under study. If
    # batches are *crossed* with condition (a batch contains multiple conditions,
    # e.g. one sequencing run with both AGED and ADULT), correction removes
    # technical variation while preserving the condition signal.
    from collections import defaultdict
    _batch_conds = defaultdict(set)
    for s in selected_slides:
        _batch_conds[s.get("batch") or s["slide_id"]].add(s["condition"])
    _n_conds = len({s["condition"] for s in selected_slides})
    _crossed = any(len(cs) > 1 for cs in _batch_conds.values())
    if _n_conds >= 2 and not _crossed:
        st.warning(
            "⚠️ Every batch label maps to a single condition, so Harmony will "
            "correct **across** conditions and may remove genuine between-condition "
            "signal. In **Study Setup**, set a `batch` that is shared across "
            "conditions (e.g. the sequencing run / capture date) so each batch "
            "contains both conditions."
        )
    elif _n_conds >= 2 and _crossed:
        st.success(
            "✓ Batches are crossed with condition — Harmony will remove technical "
            "batch variation while preserving the condition difference."
        )

st.divider()

# ── Sweep configuration ───────────────────────────────────────────────────────
st.subheader("Sweep configuration")
c1, c2, c3 = st.columns(3)
with c1:
    res_min = st.number_input("Min resolution", 0.05, 5.0, 0.1, 0.05, format="%.2f")
with c2:
    res_max = st.number_input("Max resolution", 0.1, 5.0, 2.0, 0.1, format="%.2f")
with c3:
    res_step = st.number_input("Step size", 0.05, 1.0, 0.1, 0.05, format="%.2f")

if res_min >= res_max:
    st.error("Min resolution must be less than max resolution.")
    st.stop()

n_steps = int(round((res_max - res_min) / res_step)) + 1
resolutions = [round(res_min + i * res_step, 2) for i in range(n_steps)]
resolutions = [r for r in resolutions if r <= res_max + 1e-9]
st.caption(f"Will test **{len(resolutions)}** resolutions: {resolutions[0]} – {resolutions[-1]}")

c4, c5 = st.columns(2)
with c4:
    n_sample = st.number_input(
        "Max cells for metric computation",
        1000, 100_000, 50_000, 5000,
        help="Silhouette score is O(n^2) in time and memory. Subsampling speeds "
             "up the sweep with minimal impact on ranking. CH and DB also use "
             "the subsample.",
    )
    if n_sample > 50_000:
        st.caption("⚠️ Silhouette is O(n²) — values above ~50k can be slow and memory-heavy.")
with c5:
    st.info(
        "**Scoring weights (with spatial data):**\n"
        "- 30% silhouette\n- 15% Calinski-Harabasz\n- 15% Davies-Bouldin (inverted)\n"
        "- 20% spatial coherence\n- 20% modularity\n\n"
        "Without spatial coordinates, silhouette and modularity each get 35%."
    )

st.divider()

# ── Run sweep ─────────────────────────────────────────────────────────────────
run_clicked = st.button("▶ Run resolution sweep", type="primary", use_container_width=True)

if run_clicked:
    from xenium_spatial.leiden_optimizer import optimize_leiden_resolution

    try:
        with st.spinner("Loading slides, applying ROIs, and building the embedding …"):
            run_dirs   = tuple(str(s["run_dir"]) for s in selected_slides)
            sids       = tuple(s["slide_id"] for s in selected_slides)
            conditions = tuple(s["condition"] for s in selected_slides)
            batches    = tuple((s.get("batch") or s["slide_id"]) for s in selected_slides)
            roi_sig    = _roi_signature(sids, st.session_state["roi_cache_dir"])

            adata = _load_and_preprocess(
                run_dirs, sids, conditions, batches,
                st.session_state["base_panel_csv"],
                st.session_state["roi_cache_dir"], use_roi,
                st.session_state["panel_mode"], int(st.session_state["min_slides"]),
                roi_sig, bool(base_panel_only), int(n_pcs), int(n_neighbors),
                bool(scale_genes), batch_key, st.session_state["output_dir"],
            )

        has_spatial = "spatial" in adata.obsm
        spatial_msg = ("spatial coherence enabled" if has_spatial
                       else "no spatial coords — spatial coherence disabled")
        embed_msg = ("Harmony-integrated embedding" if "X_pca_harmony" in adata.obsm
                     else "PCA embedding (no batch correction)")
        st.info(
            f"Embedded {adata.n_obs:,} cells × {adata.n_vars:,} genes on the "
            f"{embed_msg}  ({spatial_msg}).  Sweeping {len(resolutions)} resolutions …"
        )

        progress_bar = st.progress(0, text="Starting sweep …")

        def _progress_callback(step, total, res, metrics):
            progress_bar.progress(
                step / total,
                text=f"Resolution {res:.2f} — {metrics['n_clusters']} clusters, "
                     f"silhouette {metrics['silhouette']:.4f}  ({step}/{total})",
            )

        result = optimize_leiden_resolution(
            adata, resolutions=resolutions, random_state=42,
            n_sample=int(n_sample), callback=_progress_callback,
        )
        progress_bar.progress(1.0, text="Sweep complete!")

        st.session_state["optimizer_results"] = result["results"]
        st.session_state["optimizer_best"] = result["best_resolution"]
        st.session_state["optimizer_best_row"] = result["best_row"]
        st.session_state["optimizer_cluster_assignments"] = result["cluster_assignments"]

        # Persist the sweep table for the record.
        out_dir = Path(st.session_state["output_dir"]) / "leiden_optimizer"
        out_dir.mkdir(parents=True, exist_ok=True)
        result["results"].to_csv(out_dir / "leiden_resolution_sweep.csv", index=False)

        st.rerun()
    except Exception as e:
        logging.getLogger("xenium_app").exception("Leiden optimisation failed")
        st.error(f"Leiden optimisation failed: {e}")
        st.exception(e)


# ── Display results ───────────────────────────────────────────────────────────
df = st.session_state.get("optimizer_results")
best_res = st.session_state.get("optimizer_best")
best_row = st.session_state.get("optimizer_best_row")
cluster_assignments = st.session_state.get("optimizer_cluster_assignments")

if df is not None and best_res is not None:
    st.subheader("Results")

    # Match the optimizer's spatial-weighting decision: spatial coherence is in
    # play if any resolution produced a (non-NaN) score. Using .all() here would
    # disagree with the combined-score weighting whenever a degenerate
    # single-cluster resolution leaves one NaN.
    has_spatial = df["spatial_coherence"].notna().any()
    if has_spatial:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m6 = None

    with m1:
        st.metric("Optimal resolution", f"{best_res:.2f}")
    with m2:
        st.metric("Clusters", int(best_row["n_clusters"]))
    with m3:
        st.metric("Silhouette", f"{best_row['silhouette']:.4f}")
    with m4:
        st.metric("Calinski-Harabasz", f"{best_row['calinski_harabasz']:.1f}")
    with m5:
        _db = best_row["davies_bouldin"]
        st.metric("Davies-Bouldin", f"{_db:.4f}" if _db == _db else "N/A")
    if m6 is not None:
        with m6:
            _sc = best_row["spatial_coherence"]
            st.metric("Spatial coherence", f"{_sc:.4f}" if _sc == _sc else "N/A")

    st.divider()

    # ── Metric plots ─────────────────────────────────────────────────────────
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_metric_rows = 3 if has_spatial else 2
    subplot_titles = (
        "Combined Score", "Number of Clusters",
        "Silhouette Score (higher = better)",
        ("Spatial Coherence (higher = better)" if has_spatial
         else "Modularity (higher = better)"),
        "Calinski-Harabasz Index (higher = better)",
        "Davies-Bouldin Index (lower = better)",
    )
    fig = make_subplots(
        rows=n_metric_rows, cols=2, subplot_titles=subplot_titles,
        vertical_spacing=0.10, horizontal_spacing=0.08,
    )
    _best_color = "#E74C3C"

    fig.add_trace(go.Scatter(
        x=df["resolution"], y=df["combined_score"], mode="lines+markers",
        name="Combined", line=dict(color="#1B4F8A", width=2.5), marker=dict(size=7),
    ), row=1, col=1)
    fig.add_vline(x=best_res, line_dash="dash", line_color=_best_color,
                  annotation_text=f"Best: {best_res:.2f}",
                  annotation_position="top right", row=1, col=1)

    fig.add_trace(go.Bar(
        x=df["resolution"], y=df["n_clusters"], name="Clusters",
        marker_color=[_best_color if abs(r - best_res) < 1e-6 else "#90C8F0"
                      for r in df["resolution"]],
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=df["resolution"], y=df["silhouette"], mode="lines+markers",
        name="Silhouette", line=dict(color="#0A7E6E", width=2), marker=dict(size=6),
    ), row=2, col=1)
    fig.add_vline(x=best_res, line_dash="dash", line_color=_best_color, row=2, col=1)

    if has_spatial:
        fig.add_trace(go.Scatter(
            x=df["resolution"], y=df["spatial_coherence"], mode="lines+markers",
            name="Spatial Coherence", line=dict(color="#8E44AD", width=2), marker=dict(size=6),
        ), row=2, col=2)
    else:
        fig.add_trace(go.Scatter(
            x=df["resolution"], y=df["modularity"], mode="lines+markers",
            name="Modularity", line=dict(color="#CE9178", width=2), marker=dict(size=6),
        ), row=2, col=2)
    fig.add_vline(x=best_res, line_dash="dash", line_color=_best_color, row=2, col=2)

    fig.add_trace(go.Scatter(
        x=df["resolution"], y=df["calinski_harabasz"], mode="lines+markers",
        name="Calinski-Harabasz", line=dict(color="#D4AC0D", width=2), marker=dict(size=6),
    ), row=n_metric_rows, col=1)
    fig.add_vline(x=best_res, line_dash="dash", line_color=_best_color, row=n_metric_rows, col=1)

    fig.add_trace(go.Scatter(
        x=df["resolution"], y=df["davies_bouldin"], mode="lines+markers",
        name="Davies-Bouldin", line=dict(color="#E67E22", width=2), marker=dict(size=6),
    ), row=n_metric_rows, col=2)
    fig.add_vline(x=best_res, line_dash="dash", line_color=_best_color, row=n_metric_rows, col=2)

    fig.update_layout(
        height=300 * n_metric_rows, showlegend=False, template="plotly_white",
        title_text="Leiden Resolution Sweep — Cluster Quality Metrics", title_x=0.5,
    )
    for i in range(1, n_metric_rows + 1):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Resolution", row=i, col=j)
    fig.update_yaxes(title_text="Combined Score", row=1, col=1)
    fig.update_yaxes(title_text="Clusters", row=1, col=2)
    fig.update_yaxes(title_text="Silhouette", row=2, col=1)
    fig.update_yaxes(title_text="Spatial Coherence" if has_spatial else "Modularity",
                     row=2, col=2)
    fig.update_yaxes(title_text="Calinski-Harabasz", row=n_metric_rows, col=1)
    fig.update_yaxes(title_text="Davies-Bouldin", row=n_metric_rows, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # ── Clustree ─────────────────────────────────────────────────────────────
    if cluster_assignments is not None and len(cluster_assignments.columns) >= 2:
        st.divider()
        st.subheader("Clustree — cluster lineage across resolutions")
        st.caption(
            "Each node is a cluster at a given resolution. Edges show what "
            "fraction of cells in a higher-resolution cluster came from each "
            "lower-resolution cluster. Edge width encodes cell proportion."
        )
        _build_clustree(cluster_assignments, best_res)

    if has_spatial:
        with st.expander("Modularity across resolutions"):
            fig_mod = go.Figure()
            fig_mod.add_trace(go.Scatter(
                x=df["resolution"], y=df["modularity"], mode="lines+markers",
                name="Modularity", line=dict(color="#CE9178", width=2), marker=dict(size=6),
            ))
            fig_mod.add_vline(x=best_res, line_dash="dash", line_color=_best_color)
            fig_mod.update_layout(height=300, template="plotly_white",
                                  xaxis_title="Resolution", yaxis_title="Modularity")
            st.plotly_chart(fig_mod, use_container_width=True)

    with st.expander("Full results table", expanded=False):
        styler = (
            df.style
            .highlight_max(subset=["combined_score"], color="#D4EDDA")
            .highlight_max(subset=["silhouette"], color="#D4EDDA")
            .highlight_max(subset=["calinski_harabasz"], color="#D4EDDA")
        )
        if df["davies_bouldin"].notna().all():
            styler = styler.highlight_min(subset=["davies_bouldin"], color="#D4EDDA")
        st.dataframe(styler, use_container_width=True, hide_index=True)

        sweep_csv = df.to_csv(index=False)
        st.download_button("⬇️ Sweep results (CSV)", data=sweep_csv,
                           file_name="leiden_resolution_sweep.csv", mime="text/csv")

    st.divider()

    # ── Apply ────────────────────────────────────────────────────────────────
    st.subheader("Apply optimal resolution")
    col_current, col_apply = st.columns(2)
    with col_current:
        _db_str = (f"{best_row['davies_bouldin']:.4f}"
                   if best_row["davies_bouldin"] == best_row["davies_bouldin"] else "N/A")
        st.info(
            f"**Current setting:** {st.session_state['leiden_resolution']:.2f}\n\n"
            f"**Recommended:** {best_res:.2f} "
            f"({int(best_row['n_clusters'])} clusters, "
            f"silhouette {best_row['silhouette']:.4f}, "
            f"CH {best_row['calinski_harabasz']:.1f}, DB {_db_str})"
        )
    with col_apply:
        if st.button(f"✅ Apply recommended resolution ({best_res:.2f})",
                     type="primary", use_container_width=True, key="apply_best_resolution"):
            st.session_state["leiden_resolution"] = float(best_res)
            try:
                path = _persist_resolution(best_res, Path(st.session_state["output_dir"]))
                st.success(
                    f"Leiden resolution set to **{best_res:.2f}** and saved to "
                    f"`{path}`. It's now part of the pipeline settings (visible in "
                    "the sidebar and saved with the study config in **Study Setup**)."
                )
                st.balloons()
            except OSError as e:
                st.warning(
                    f"Leiden resolution set to **{best_res:.2f}** for this session, "
                    f"but it could not be saved to the output directory ({e}). "
                    "Check the output path in **Study Setup**."
                )

        manual_pick = st.selectbox(
            "Or pick a different resolution from the sweep",
            options=df["resolution"].tolist(),
            index=df["resolution"].tolist().index(best_res),
            format_func=lambda r: (
                f"{r:.2f}  —  {int(df.loc[df['resolution']==r, 'n_clusters'].iloc[0])} clusters, "
                f"sil {df.loc[df['resolution']==r, 'silhouette'].iloc[0]:.4f}, "
                f"CH {df.loc[df['resolution']==r, 'calinski_harabasz'].iloc[0]:.1f}"
            ),
        )
        if st.button("Apply selected resolution", key="apply_manual_resolution"):
            st.session_state["leiden_resolution"] = float(manual_pick)
            try:
                path = _persist_resolution(manual_pick, Path(st.session_state["output_dir"]))
                st.success(f"Leiden resolution set to **{manual_pick:.2f}** and saved to `{path}`.")
            except OSError as e:
                st.warning(
                    f"Leiden resolution set to **{manual_pick:.2f}** for this session, "
                    f"but it could not be saved to the output directory ({e})."
                )
else:
    st.info(
        "Configure the preprocessing and sweep parameters above, then click "
        "**Run resolution sweep** to find the optimal Leiden resolution."
    )
