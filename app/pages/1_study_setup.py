"""
pages/1_study_setup.py
Study Setup page — configure the slide folders (add or remove as needed).
"""

import json
import uuid

import pandas as pd
import streamlit as st
from pathlib import Path

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header, prune_orphan_rois

st.set_page_config(page_title="Study Setup · Xenium Sample PCA", page_icon="📁", layout="wide",
    initial_sidebar_state="expanded")


inject_css()
# ── Shared defaults (duplicated here so pages work standalone) ───────────────
if "slides" not in st.session_state:
    st.session_state["slides"] = [
        {"slide_id": f"AGED_{i}",  "condition": "AGED",  "run_dir": ""} for i in range(1,5)
    ] + [
        {"slide_id": f"ADULT_{i}", "condition": "ADULT", "run_dir": ""} for i in range(1,5)
    ]
for k, v in {
    "base_panel_csv": str(Path(__file__).parent.parent.parent / "data" / "Xenium_mBrain_v1_1_metadata.csv"),
    "output_dir"    : str(Path.home() / "xenium_sample_pca_output"),
    "roi_cache_dir" : str(Path(__file__).parent.parent.parent / "roi_cache"),
    "leiden_resolution": 0.6,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _ensure_keys(slides: list) -> list:
    """Give every slide a stable unique 'key' so widgets survive add/remove.

    Streamlit widgets are identified by their key. Keying rows on the list
    index would mean deleting a row shifts every later row's key and makes
    their stored values jump around. A per-slide uuid keeps each row's
    widgets glued to that row regardless of insertions or deletions.
    """
    for s in slides:
        if not s.get("key"):
            s["key"] = uuid.uuid4().hex[:8]
    return slides


st.session_state["slides"] = _ensure_keys(st.session_state["slides"])

# ── Helpers ──────────────────────────────────────────────────────────────────
# Wong 2011 colour-blind-safe palette; blue first, then vermillion
_WONG_POOL = ["#0072B2", "#D55E00", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]

def _condition_colours() -> dict:
    """Build condition → colour from the slides currently in session state."""
    conds = sorted({s["condition"] for s in st.session_state.get("slides", []) if s["condition"]})
    return {c: _WONG_POOL[i % len(_WONG_POOL)] for i, c in enumerate(conds)}

# Keep a module-level alias updated on each render
CONDITION_COLOURS = _condition_colours()

def _xenium_dir_status(path_str: str) -> tuple[bool, str]:
    """Return (valid, message) for a Xenium run directory.

    Supports both old-format runs (no stain, imported segmentation, cells.csv.gz
    instead of cells.parquet) and new-format runs (xenium_cell_segmentation_stains_v1
    with cells.parquet).  The matrix files are the only strict requirement.
    """
    if not path_str.strip():
        return False, "No path entered"
    p = Path(path_str)
    if not p.exists():
        return False, f"Directory not found: {p}"
    if not p.is_dir():
        return False, "Path is not a directory"
    mtx = p / "cell_feature_matrix"
    if not mtx.exists():
        return False, "Missing cell_feature_matrix/"
    for f in ["matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz"]:
        if not (mtx / f).exists():
            return False, f"Missing {f}"
    # cells.parquet is present in newer runs (xenium_cell_segmentation_stains_v1).
    # Older runs (imported segmentation) may have cells.csv.gz or similar instead.
    # We warn but do not block -- xenium_loader handles the fallback.
    has_cells_parquet = (p / "cells.parquet").exists()
    has_cells_csv     = (p / "cells.csv.gz").exists() or (p / "cells.csv").exists()
    if not has_cells_parquet and not has_cells_csv:
        return True, "Valid (no cells.parquet — spatial coords may be unavailable)"
    return True, "Valid Xenium run directory"

# ── Page ─────────────────────────────────────────────────────────────────────
page_header("📁 Study Setup", "Configure the Xenium run directories for your slides")
st.markdown(
    "Enter the path to each Xenium output directory. "
    "Each folder must contain `cell_feature_matrix/` (with `matrix.mtx.gz`, "
    "`barcodes.tsv.gz`, `features.tsv.gz`) and `cells.parquet`."
)
st.info(
    "💡 **Tip:** On macOS, right-click a folder in Finder → "
    "**Get Info** → copy the path from *Where*, "
    "or drag the folder into this browser window's address bar to get its path."
)
st.divider()

# ── Slide table ──────────────────────────────────────────────────────────────
st.subheader("Slide folders")
st.caption(
    "Add a row per Xenium run. Conditions are free text — use whatever group "
    "labels your study needs (the defaults are AGED / ADULT)."
)

# Header row
h_cond, h_id, h_path, h_status, h_del = st.columns([1.3, 1.5, 4.4, 1.3, 0.6])
h_cond.markdown("**Condition**")
h_id.markdown("**Slide ID**")
h_path.markdown("**Run directory**")
h_status.markdown("**Status**")

slides = st.session_state["slides"]
delete_key = None  # set if a row's delete button is pressed; processed after loop

for i, slide in enumerate(slides):
    row_key = slide["key"]

    col_cond, col_id, col_path, col_status, col_del = st.columns([1.3, 1.5, 4.4, 1.3, 0.6])

    with col_cond:
        new_cond = st.text_input(
            "Condition",
            value=slide["condition"],
            key=f"cond_{row_key}",
            label_visibility="collapsed",
        )
        slides[i]["condition"] = new_cond
        st.markdown(
            f'<div style="height:4px;border-radius:2px;background:{CONDITION_COLOURS.get(new_cond, "#888")};margin-top:-6px"></div>',
            unsafe_allow_html=True,
        )

    with col_id:
        new_id = st.text_input(
            "Slide ID",
            value=slide["slide_id"],
            key=f"slide_id_{row_key}",
            label_visibility="collapsed",
        )
        slides[i]["slide_id"] = new_id

    with col_path:
        new_path = st.text_input(
            "Run directory",
            value=slide["run_dir"],
            placeholder="/path/to/xenium_run",
            key=f"run_dir_{row_key}",
            label_visibility="collapsed",
        )
        slides[i]["run_dir"] = new_path

    with col_status:
        if new_path.strip():
            ok, msg = _xenium_dir_status(new_path)
            if ok:
                st.success("✓", icon=None)
            else:
                st.error(msg[:40])
        else:
            st.caption("—")

    with col_del:
        if st.button("🗑", key=f"del_slide_{row_key}", help="Remove this slide"):
            delete_key = row_key

    # Show gene count if valid
    if slide["run_dir"].strip():
        ok, _ = _xenium_dir_status(slide["run_dir"])
        if ok:
            try:
                feat_path = (
                    Path(slide["run_dir"]) / "cell_feature_matrix" / "features.tsv.gz"
                )
                # features.tsv.gz columns: gene_id, gene_name, feature_type
                # feature_type values: "Gene Expression", "Blank Codeword",
                #   "Negative Control Codeword", "Negative Control Probe"
                feats = pd.read_csv(
                    feat_path, sep="\t", header=None,
                    names=["gene_id", "gene_name", "feature_type"],
                    compression="gzip",
                )
                type_counts = feats["feature_type"].value_counts()
                n_rna       = int(type_counts.get("Gene Expression", 0))
                n_blank     = int(type_counts.get("Blank Codeword", 0))
                n_neg_cw    = int(type_counts.get("Negative Control Codeword", 0))
                n_neg_probe = int(type_counts.get("Negative Control Probe", 0))
                # Classify the slide's RNA genes by NAME against the base panel
                # (matching PanelRegistry), rather than subtracting a fixed base
                # count — which misreports whenever a slide is missing base genes
                # or carries extra ones. Fall back to the count-difference only
                # if the base panel CSV cannot be read.
                rna_genes = set(
                    feats.loc[feats["feature_type"] == "Gene Expression", "gene_name"]
                )
                base_genes: set = set()
                try:
                    _base_csv = Path(st.session_state.get("base_panel_csv", ""))
                    if _base_csv.exists():
                        _base_df = pd.read_csv(_base_csv)
                        _gene_col = "Genes" if "Genes" in _base_df.columns else _base_df.columns[0]
                        base_genes = set(_base_df[_gene_col].astype(str))
                except Exception:
                    base_genes = set()
                if base_genes:
                    n_predesigned = len(rna_genes & base_genes)
                    n_custom      = len(rna_genes - base_genes)
                else:
                    n_predesigned = min(n_rna, 247)
                    n_custom      = max(0, n_rna - 247)

                bc_path = (
                    Path(slide["run_dir"]) / "cell_feature_matrix" / "barcodes.tsv.gz"
                )
                n_cells = len(pd.read_csv(bc_path, header=None, compression="gzip"))

                control_parts = []
                if n_blank > 0:
                    control_parts.append(f"{n_blank} blank codewords")
                if n_neg_cw > 0:
                    control_parts.append(f"{n_neg_cw} neg. control codewords")
                if n_neg_probe > 0:
                    control_parts.append(f"{n_neg_probe} neg. control probes")
                control_str = (
                    f" + {', '.join(control_parts)}" if control_parts else ""
                )

                st.caption(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {n_cells:,} cells · "
                    f"{n_rna} RNA targets "
                    f"({n_predesigned} predesigned + {n_custom} custom)"
                    f"{control_str}",
                    unsafe_allow_html=True,
                )
            except Exception as _e:
                st.caption(f"Could not read gene counts: {_e}")

# Apply a deletion requested above, then rerun so the table redraws.
if delete_key is not None:
    if len(slides) <= 1:
        st.warning("At least one slide row is required.")
    else:
        st.session_state["slides"] = [s for s in slides if s["key"] != delete_key]
        # Drop the deleted row's widget state so its values can't leak elsewhere.
        for prefix in ("cond_", "slide_id_", "run_dir_", "del_slide_"):
            st.session_state.pop(f"{prefix}{delete_key}", None)
        prune_orphan_rois()  # drop the removed slide's ROI from the in-memory dict
        st.rerun()

st.session_state["slides"] = slides

# ── Add slide ─────────────────────────────────────────────────────────────────
if st.button("➕ Add slide", use_container_width=False):
    default_cond = slides[-1]["condition"] if slides else "AGED"
    slides.append({
        "slide_id" : f"Sample_{len(slides) + 1}",
        "condition": default_cond,
        "run_dir"  : "",
        "key"      : uuid.uuid4().hex[:8],
    })
    st.session_state["slides"] = slides
    st.rerun()

# ── Summary banner ────────────────────────────────────────────────────────────
st.divider()
n_total = len(slides)
valid_flags = [_xenium_dir_status(s["run_dir"])[0] for s in slides]
n_ok = sum(valid_flags)

by_cond: dict[str, int] = {}
for s, ok in zip(slides, valid_flags):
    if ok:
        by_cond[s["condition"]] = by_cond.get(s["condition"], 0) + 1
cond_breakdown = ", ".join(f"{n} {c}" for c, n in sorted(by_cond.items())) or "—"

if n_total > 0 and n_ok == n_total:
    st.success(f"✅ All {n_total} slides configured ({cond_breakdown})")
elif n_ok > 0:
    st.warning(f"⚠️ {n_ok}/{n_total} slides configured ({cond_breakdown}) — "
               f"{n_total - n_ok} still need paths")
else:
    st.error("No valid slide directories entered yet")

# ── File paths ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("File paths")

col_a, col_b = st.columns(2)

with col_a:
    csv = st.text_input(
        "Base panel CSV (Xenium_mBrain_v1_1_metadata.csv)",
        value=st.session_state["base_panel_csv"],
        help="The 10x Genomics metadata CSV that defines the 247 base panel genes.",
    )
    st.session_state["base_panel_csv"] = csv
    if Path(csv).exists():
        try:
            n = len(pd.read_csv(csv))
            st.caption(f"✓ Found — {n} genes")
        except Exception as e:
            st.error(str(e))
    else:
        st.error("File not found")

    roi_dir = st.text_input(
        "ROI cache directory",
        value=st.session_state["roi_cache_dir"],
        help="Polygon ROIs for each slide are stored here as JSON files.",
    )
    st.session_state["roi_cache_dir"] = roi_dir
    roi_path = Path(roi_dir)
    if roi_path.exists():
        n_saved = len(list(roi_path.glob("*_roi.json")))
        st.caption(f"{n_saved} ROI file(s) saved")
    else:
        st.caption("Directory will be created when the pipeline runs.")

with col_b:
    out = st.text_input(
        "Output directory",
        value=st.session_state["output_dir"],
        help="All figures and results files are written here.",
    )
    st.session_state["output_dir"] = out
    st.caption(f"Will be created if it does not exist: {out}")

# ── Save / load config ────────────────────────────────────────────────────────
st.divider()
st.subheader("Save / load configuration")

col_save, col_load = st.columns(2)

with col_save:
    if st.button("💾 Save configuration to JSON", use_container_width=True):
        cfg = {
            "slides"        : st.session_state["slides"],
            "base_panel_csv": st.session_state["base_panel_csv"],
            "output_dir"    : st.session_state["output_dir"],
            "roi_cache_dir" : st.session_state["roi_cache_dir"],
            "leiden_resolution": st.session_state.get("leiden_resolution", 0.6),
        }
        cfg_str = json.dumps(cfg, indent=2)
        st.download_button(
            "⬇️ Download config.json",
            data=cfg_str,
            file_name="xenium_pipeline_config.json",
            mime="application/json",
            use_container_width=True,
        )

with col_load:
    uploaded = st.file_uploader(
        "📂 Load configuration from JSON", type="json"
    )
    if uploaded:
        try:
            cfg = json.load(uploaded)
            if "slides" in cfg:
                st.session_state["slides"] = _ensure_keys(cfg["slides"])
                prune_orphan_rois()  # drop ROIs for slides not in the new config
            for k in ["base_panel_csv", "output_dir", "roi_cache_dir", "leiden_resolution"]:
                if k in cfg:
                    st.session_state[k] = cfg[k]
            st.success("Configuration loaded — refresh the page to see updated paths.")
        except Exception as e:
            st.error(f"Could not load config: {e}")
