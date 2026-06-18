"""
pages/2_roi_manager.py
ROI Manager — frame the MBH region for each slide.

Approach
--------
Plotly selection events (lasso / box / on_select) are unreliable across
Streamlit/Plotly versions, so we use FOUR SLIDERS — one per edge of the
bounding rectangle (x_min, x_max, y_min, y_max).  Slider ranges come from
each slide's actual tissue bounds, so every value is a valid coordinate.

Moving a slider updates the rectangle overlay and the live cell count.

ROIs are saved to ``roi_cache/<slide>_roi.json`` (the same format the
loader reads via ``src.roi_selector.ROISelector``) and reused automatically.
A "Paste coordinates" fallback handles non-rectangular regions.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from ui_utils import inject_css, page_header

st.set_page_config(page_title="ROI Manager · Xenium Sample PCA", page_icon="🗺️", layout="wide",
    initial_sidebar_state="expanded")

inject_css()
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "slides"        : [],
    "roi_polygons"  : {},
    "roi_cache_dir" : str(Path(__file__).parent.parent.parent / "roi_cache"),
    "roi_last_slide": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_cells(run_dir: str) -> tuple:
    """Load centroid_x, centroid_y from the run directory. Returns (df, error_str).

    Cached on ``run_dir``: the parquet/CSV files don't change during a session,
    so this avoids re-reading every slide from disk on every slider movement.
    """
    p = Path(run_dir)
    for cell_path, loader in [
        (p / "cells.parquet", lambda f: pd.read_parquet(f)),
        (p / "cells.csv.gz",  lambda f: pd.read_csv(f, compression="gzip")),
        (p / "cells.csv",     lambda f: pd.read_csv(f)),
    ]:
        if not cell_path.exists():
            continue
        try:
            df = loader(cell_path)
            renames = {}
            for col in df.columns:
                lc = col.lower()
                if lc in {"x_centroid","centroid_x_um","cell_centroid_x","x_um","x","centroid_x"}:
                    renames[col] = "centroid_x"
                elif lc in {"y_centroid","centroid_y_um","cell_centroid_y","y_um","y","centroid_y"}:
                    renames[col] = "centroid_y"
            df = df.rename(columns=renames)
            if "centroid_x" in df.columns and "centroid_y" in df.columns:
                return df[["centroid_x","centroid_y"]].dropna().reset_index(drop=True), ""
            return None, f"No centroid columns in {cell_path.name}. Found: {list(df.columns)}"
        except Exception as e:
            return None, f"Error reading {cell_path.name}: {e}"
    return None, f"No cell file found in {run_dir}"


def _roi_path(slide_id: str) -> Path:
    cache = Path(st.session_state["roi_cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"{slide_id.replace('/','_').replace(' ','_')}_roi.json"


def _save_roi(slide_id: str, vertices: list, n_cells: int):
    roi = {
        "slide_id"        : slide_id,
        "roi_name"        : "MBH",
        "vertices"        : [[float(x), float(y)] for x, y in vertices],
        "n_cells_selected": n_cells,
        "created_at"      : datetime.now().isoformat(),
        "method"          : "slider_rectangle",
    }
    _roi_path(slide_id).write_text(json.dumps(roi, indent=2))
    st.session_state["roi_polygons"][slide_id] = roi["vertices"]


def _delete_roi(slide_id: str):
    st.session_state["roi_polygons"].pop(slide_id, None)
    p = _roi_path(slide_id)
    if p.exists():
        p.unlink()


def _load_roi(slide_id: str) -> list | None:
    p = _roi_path(slide_id)
    if p.exists():
        try:
            return json.loads(p.read_text()).get("vertices", [])
        except Exception:
            return None
    return None


def _load_all_saved_rois():
    for s in st.session_state.get("slides", []):
        sid = s["slide_id"]
        if sid not in st.session_state["roi_polygons"]:
            v = _load_roi(sid)
            if v:
                st.session_state["roi_polygons"][sid] = v


def _count_in_rect(cells_df: pd.DataFrame, x0, x1, y0, y1) -> int:
    mask = (
        (cells_df["centroid_x"] >= x0) & (cells_df["centroid_x"] <= x1) &
        (cells_df["centroid_y"] >= y0) & (cells_df["centroid_y"] <= y1)
    )
    return int(mask.sum())


def _rect_to_verts(x0, x1, y0, y1) -> list:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _count_in_polygon(cells_df, verts) -> int:
    if cells_df is None or not verts or len(verts) < 3:
        return 0
    from matplotlib.path import Path as MplPath
    xy   = cells_df[["centroid_x","centroid_y"]].values
    mask = MplPath(verts).contains_points(xy)
    return int(mask.sum())


@st.cache_data(show_spinner=False)
def _count_in_polygon_cached(run_dir: str, verts_key: tuple) -> int | None:
    """Load (cached) and count cells inside a polygon for the summary table.

    Cached on ``(run_dir, verts_key)`` so the per-slide counts aren't recomputed
    on every slider movement. Returns ``None`` if cells can't be read.
    """
    df, _ = _load_cells(run_dir)
    if df is None:
        return None
    return _count_in_polygon(df, [list(v) for v in verts_key])


def _export_all() -> str:
    export = {}
    for sid, verts in st.session_state["roi_polygons"].items():
        p = _roi_path(sid)
        n_cells = 0
        if p.exists():
            try:
                n_cells = json.loads(p.read_text()).get("n_cells_selected", 0)
            except Exception:
                pass
        export[sid] = {"vertices": verts, "roi_name": "MBH",
                       "n_cells_selected": n_cells,
                       "exported_at": datetime.now().isoformat()}
    return json.dumps(export, indent=2)


def _import_rois(json_str: str) -> tuple[int, str]:
    try:
        data = json.loads(json_str)
        n = 0
        for sid, entry in data.items():
            verts = entry.get("vertices") or entry
            if not isinstance(verts, list) or len(verts) < 3:
                continue
            import math
            valid = True
            for v in verts:
                if (not isinstance(v, (list, tuple)) or len(v) != 2
                        or not all(isinstance(c, (int, float)) and math.isfinite(c) for c in v)):
                    valid = False
                    break
            if not valid:
                continue
            _save_roi(sid, verts, entry.get("n_cells_selected") or 0)
            n += 1
        return n, ""
    except Exception as e:
        return 0, str(e)


# ── Init ──────────────────────────────────────────────────────────────────────
_load_all_saved_rois()

# ── Page ──────────────────────────────────────────────────────────────────────
page_header("🗺️ ROI Manager", "Define the mediobasal hypothalamus boundary for each slide")
st.markdown(
    "Use the sliders to frame the **mediobasal hypothalamus (MBH)** on each section. "
    "The scatter and cell count update live."
)

slides = st.session_state.get("slides", [])
if not slides:
    st.warning("No slides configured. Go to **📁 Study Setup** first.")
    st.stop()

slide_ids = [s["slide_id"] for s in slides]
n_saved   = sum(1 for sid in slide_ids if sid in st.session_state["roi_polygons"])

if n_saved == len(slide_ids):
    st.success(f"✅ All {n_saved} ROIs saved and ready")
else:
    missing = [s for s in slide_ids if s not in st.session_state["roi_polygons"]]
    st.warning(f"**{n_saved}/{len(slide_ids)} ROIs saved** — still needed: {', '.join(missing)}")

# ── Export / import ───────────────────────────────────────────────────────────
with st.expander("💾 Export / Import all ROIs"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Export** — save all ROIs to a JSON file to reuse next session")
        if n_saved > 0:
            st.download_button(
                f"⬇️ Download roi_coordinates.json ({n_saved} ROIs)",
                data=_export_all(), file_name="roi_coordinates.json",
                mime="application/json", use_container_width=True,
            )
        else:
            st.info("No ROIs saved yet.")
    with c2:
        st.markdown("**Import** — restore previously exported ROIs instantly")
        uploaded = st.file_uploader("Upload roi_coordinates.json", type="json",
                                     key="roi_import")
        if uploaded:
            n_imp, err = _import_rois(uploaded.read().decode())
            if err:
                st.error(f"Import failed: {err}")
            else:
                st.success(f"Imported {n_imp} ROIs!")
                st.rerun()

if n_saved > 0:
    if st.button("🗑 Delete ALL ROIs and start over",
                 help="Use this if your ROIs are selecting 0 cells or wrong tissue."):
        for sid in list(st.session_state["roi_polygons"].keys()):
            _delete_roi(sid)
        st.success("All ROIs deleted.")
        st.rerun()

st.divider()

# ── Slide selector ────────────────────────────────────────────────────────────
selected_id = st.selectbox(
    "Select slide",
    options=slide_ids,
    format_func=lambda sid: f"{'✅' if sid in st.session_state['roi_polygons'] else '⬜'} {sid}",
)
selected_slide = next((s for s in slides if s["slide_id"] == selected_id), None)

if st.session_state["roi_last_slide"] != selected_id:
    for k in [f"sl_x0_{selected_id}", f"sl_x1_{selected_id}",
              f"sl_y0_{selected_id}", f"sl_y1_{selected_id}"]:
        st.session_state.pop(k, None)
    st.session_state["roi_last_slide"] = selected_id

cells_df, load_err = None, ""
if selected_slide and selected_slide.get("run_dir"):
    with st.spinner(f"Loading {selected_id} …"):
        cells_df, load_err = _load_cells(selected_slide["run_dir"])

# ── Layout ────────────────────────────────────────────────────────────────────
chart_col, ctrl_col = st.columns([3, 1])

# Defaults for variables consumed by the chart column.
x0 = x1 = y0 = y1 = 0
n_preview = 0

with ctrl_col:
    cond = selected_slide["condition"] if selected_slide else "—"
    st.markdown(f"**{selected_id}** — `{cond}`")

    if cells_df is not None:
        st.metric("Total cells", f"{len(cells_df):,}")

        tx0 = float(cells_df["centroid_x"].min())
        tx1 = float(cells_df["centroid_x"].max())
        ty0 = float(cells_df["centroid_y"].min())
        ty1 = float(cells_df["centroid_y"].max())
        tw  = tx1 - tx0
        th  = ty1 - ty0

        # Default ROI: horizontal centre, ventral 55-80% (where MBH sits).
        def_x0 = round(tx0 + tw * 0.35)
        def_x1 = round(tx0 + tw * 0.65)
        def_y0 = round(ty0 + th * 0.55)
        def_y1 = round(ty0 + th * 0.80)

        st.divider()
        st.markdown("**Rectangle ROI**")
        st.caption("Slide the edges to frame the MBH. The scatter and cell count update instantly.")

        step = max(1.0, round(min(tw, th) / 100))

        # Apply any values queued by "Load into sliders" before the widgets are
        # instantiated (writing widget keys after instantiation is forbidden).
        pending = st.session_state.pop(f"sl_pending_{selected_id}", None)
        if pending:
            st.session_state[f"sl_x0_{selected_id}"] = pending["x0"]
            st.session_state[f"sl_x1_{selected_id}"] = pending["x1"]
            st.session_state[f"sl_y0_{selected_id}"] = pending["y0"]
            st.session_state[f"sl_y1_{selected_id}"] = pending["y1"]

        x0 = st.slider("Left edge (x min)", min_value=int(tx0), max_value=int(tx1),
                       value=int(st.session_state.get(f"sl_x0_{selected_id}", def_x0)),
                       step=int(step), key=f"sl_x0_{selected_id}")
        x1 = st.slider("Right edge (x max)", min_value=int(tx0), max_value=int(tx1),
                       value=int(st.session_state.get(f"sl_x1_{selected_id}", def_x1)),
                       step=int(step), key=f"sl_x1_{selected_id}")
        y0 = st.slider("Top edge (y min — dorsal)", min_value=int(ty0), max_value=int(ty1),
                       value=int(st.session_state.get(f"sl_y0_{selected_id}", def_y0)),
                       step=int(step), key=f"sl_y0_{selected_id}")
        y1 = st.slider("Bottom edge (y max — ventral)", min_value=int(ty0), max_value=int(ty1),
                       value=int(st.session_state.get(f"sl_y1_{selected_id}", def_y1)),
                       step=int(step), key=f"sl_y1_{selected_id}")

        if x0 >= x1:
            x1 = min(x0 + int(step), int(tx1))
        if y0 >= y1:
            y1 = min(y0 + int(step), int(ty1))

        n_preview = _count_in_rect(cells_df, x0, x1, y0, y1)
        pct = 100 * n_preview / max(len(cells_df), 1)
        if n_preview == 0:
            st.error("0 cells in this region — adjust the sliders.")
        elif pct > 60:
            st.warning(f"{n_preview:,} cells ({pct:.1f}%) — region may be too large")
        else:
            st.success(f"**{n_preview:,} cells** ({pct:.1f}%)")

        st.divider()

        saved_verts = st.session_state["roi_polygons"].get(selected_id)
        can_save = n_preview > 0 and x0 < x1 and y0 < y1
        if st.button("✅ Save ROI", type="primary", use_container_width=True,
                     disabled=not can_save, key=f"save_{selected_id}"):
            _save_roi(selected_id, _rect_to_verts(x0, x1, y0, y1), n_preview)
            st.session_state[f"n_cells_{selected_id}"] = n_preview
            st.session_state["roi_just_saved"] = selected_id
            st.rerun()

        if st.session_state.get("roi_just_saved") == selected_id:
            st.success(f"Saved! {st.session_state.get(f'n_cells_{selected_id}', 0):,} cells in MBH ROI")
            del st.session_state["roi_just_saved"]

        if saved_verts:
            n_saved_c = _count_in_polygon(cells_df, saved_verts)
            pct_s = 100 * n_saved_c / max(len(cells_df), 1)
            st.divider()
            if n_saved_c == 0:
                st.error("⚠️ Saved ROI has 0 cells — please save a new one above.")
            else:
                st.info(f"Saved: **{n_saved_c:,}** cells ({pct_s:.1f}%)")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🗑 Delete", use_container_width=True, key=f"del_{selected_id}"):
                    _delete_roi(selected_id)
                    st.rerun()
            with c2:
                if st.button("📐 Load into sliders", use_container_width=True,
                             key=f"load_{selected_id}",
                             help="Restore saved ROI values into the sliders for editing"):
                    sv = np.array(saved_verts)
                    xs, ys = sv[:, 0], sv[:, 1]
                    is_rect = (len(saved_verts) == 4
                               and len(set(xs.round(1))) == 2
                               and len(set(ys.round(1))) == 2)
                    if not is_rect:
                        st.warning(
                            f"This ROI has {len(saved_verts)} vertices and is not a rectangle. "
                            "Loading it into the sliders will use its bounding box, which is "
                            "larger and will include extra cells."
                        )
                    else:
                        # Stash the values and apply them on the next run, before
                        # the slider widgets are instantiated. Writing the widget
                        # keys here (after the sliders exist) raises a
                        # StreamlitAPIException.
                        st.session_state[f"sl_pending_{selected_id}"] = {
                            "x0": int(xs.min()), "x1": int(xs.max()),
                            "y0": int(ys.min()), "y1": int(ys.max()),
                        }
                        st.rerun()

        if saved_verts and _count_in_polygon(cells_df, saved_verts) > 0 and len(slides) > 1:
            st.divider()
            with st.expander("📋 Copy to other slides"):
                st.caption("Sections at the same stereotaxic level have similar MBH coordinates. "
                           "Copy and verify the count.")
                targets = [s["slide_id"] for s in slides if s["slide_id"] != selected_id]
                sel_targets = st.multiselect("Copy to", targets, key=f"copy_targets_{selected_id}")
                if st.button("Copy", key=f"do_copy_{selected_id}") and sel_targets:
                    for t in sel_targets:
                        t_slide = next((s for s in slides if s["slide_id"] == t), None)
                        t_df, _ = _load_cells(t_slide["run_dir"]) if t_slide else (None, "")
                        t_n = _count_in_polygon(t_df, saved_verts) if t_df is not None else 0
                        _save_roi(t, saved_verts, t_n)
                    st.success(f"Copied to: {', '.join(sel_targets)}")
                    st.rerun()

        with st.expander("📋 Paste coordinates (advanced)"):
            st.caption("One x,y pair per line in µm:")
            paste = st.text_area("Vertices", height=90,
                                  placeholder="3200, 4100\n4800, 4100\n4800, 5600\n3200, 5600",
                                  key=f"paste_{selected_id}")
            if st.button("Save pasted ROI", key=f"load_paste_{selected_id}"):
                try:
                    lines = [l.strip() for l in paste.strip().splitlines() if l.strip()]
                    verts = [[float(v.strip()) for v in l.replace(";", ",").split(",")
                              if v.strip()][:2] for l in lines]
                    if any(len(v) != 2 for v in verts):
                        raise ValueError("Each line must contain exactly 2 values (x, y).")
                    if len(verts) >= 3:
                        n_p = _count_in_polygon(cells_df, verts)
                        if n_p == 0:
                            st.error("0 cells in pasted region — check coordinates match the tissue.")
                        else:
                            _save_roi(selected_id, verts, n_p)
                            st.success(f"Saved {len(verts)}-vertex ROI: {n_p:,} cells")
                            st.rerun()
                    else:
                        st.error("Need at least 3 vertices")
                except Exception as e:
                    st.error(f"Parse error: {e}")

    elif load_err:
        st.error(f"Could not load cells:\n{load_err}")
    else:
        st.info("Set the run directory in **📁 Study Setup** first.")

# ── Chart ─────────────────────────────────────────────────────────────────────
with chart_col:
    if cells_df is None:
        if load_err:
            st.error(f"Cannot display scatter: {load_err}")
        else:
            st.info("Configure the run directory in **📁 Study Setup** to see the tissue.")
    else:
        MAX_DISPLAY = 100_000
        df_plot = (cells_df if len(cells_df) <= MAX_DISPLAY
                   else cells_df.sample(MAX_DISPLAY, random_state=42))

        saved_verts = st.session_state["roi_polygons"].get(selected_id)

        inside_mask = (
            (df_plot["centroid_x"] >= x0) & (df_plot["centroid_x"] <= x1) &
            (df_plot["centroid_y"] >= y0) & (df_plot["centroid_y"] <= y1)
        )

        fig = go.Figure()
        out_mask = ~inside_mask
        if out_mask.any():
            fig.add_trace(go.Scatter(
                x=df_plot.loc[out_mask, "centroid_x"], y=df_plot.loc[out_mask, "centroid_y"],
                mode="markers", marker=dict(size=1.5, color="#2A5298", opacity=0.12),
                name="Outside", hoverinfo="skip", showlegend=False,
            ))
        if inside_mask.any():
            fig.add_trace(go.Scatter(
                x=df_plot.loc[inside_mask, "centroid_x"], y=df_plot.loc[inside_mask, "centroid_y"],
                mode="markers", marker=dict(size=2.5, color="#F5A623", opacity=0.7),
                name=f"In ROI ({n_preview:,})",
                hovertemplate="x: %{x:.0f} µm<br>y: %{y:.0f} µm<extra></extra>",
            ))

        # Current slider rectangle
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      line=dict(color="#F5A623", width=2.5),
                      fillcolor="rgba(245,166,35,0.05)")

        # Saved ROI polygon (green)
        if saved_verts and len(saved_verts) >= 3:
            sv = np.array(saved_verts)
            n_sv = _count_in_polygon(cells_df, saved_verts)
            fig.add_trace(go.Scatter(
                x=list(sv[:,0]) + [sv[0,0]], y=list(sv[:,1]) + [sv[0,1]],
                mode="lines", line=dict(color="#009E73", width=3),
                fill="toself", fillcolor="rgba(0,158,115,0.08)",
                name=f"Saved ROI ({n_sv:,} cells)", hoverinfo="skip",
            ))

        fig.update_layout(
            height=600, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="x (µm)", scaleanchor="y", showgrid=False, zeroline=False),
            yaxis=dict(title="y (µm)", autorange="reversed", showgrid=False, zeroline=False),
            plot_bgcolor="#111111", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                        font=dict(size=11), bgcolor="rgba(255,255,255,0.85)"),
            dragmode="zoom",
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"scrollZoom": True, "displaylogo": False})

        st.caption(
            f"Tissue bounds: x = {cells_df['centroid_x'].min():.0f}–{cells_df['centroid_x'].max():.0f} µm, "
            f"y = {cells_df['centroid_y'].min():.0f}–{cells_df['centroid_y'].max():.0f} µm  |  "
            f"Orange rectangle = current selection · green = saved ROI  |  "
            f"Y axis: 0 = dorsal, larger = ventral"
        )

# ── Summary table ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("All slides")
rows = []
for s in slides:
    sid   = s["slide_id"]
    verts = st.session_state["roi_polygons"].get(sid)
    n_inside = None
    if verts and s.get("run_dir"):
        n_inside = _count_in_polygon_cached(
            s["run_dir"], tuple(tuple(v) for v in verts))
    roi_str = (
        "⚠️ 0 cells — invalid" if verts and n_inside == 0
        else (f"✅ {n_inside:,} cells" if n_inside else ("✅ saved" if verts else "⬜ missing"))
    )
    rows.append({
        "Slide": sid, "Condition": s["condition"], "ROI": roi_str,
        # n_inside is None when unknown (no ROI / cells unreadable); show the
        # number otherwise, including a genuine 0 (don't collapse it to "—").
        "Cells in ROI": f"{n_inside:,}" if n_inside is not None else "—",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
