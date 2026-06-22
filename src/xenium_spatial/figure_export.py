"""
figure_export.py
----------------
Publication-grade (Nature-style) matplotlib renderers for the cell-level
quantification pages (clusters, composition, DGE, spatial, gene focus).

The Streamlit pages render *interactive* Plotly charts on screen for
exploration; these functions render the matching **static, editable PDF**
that goes into a figure. Each builder returns the figure as raw PDF
``bytes`` so a page can offer it straight from a ``st.download_button``
(no temp files), while sharing the same Nature-grade style
(:mod:`figure_style`) as the Sample-PCA / elbow figures.

Design:
* matplotlib-only (no scanpy/plotly) so the PDF is vector + editable
  (Type-42 fonts) in Illustrator.
* Each function takes already-aggregated arrays / DataFrames computed by
  the page (which are cheap and cached) and is otherwise pure — easy to
  unit-test and to cache on the page side.
* Single-column Nature width is ~88 mm (3.46 in); figures default to a
  size in that neighbourhood and can be scaled by the caller.
"""

from __future__ import annotations

import io
from typing import Optional, Sequence

import numpy as np

from .figure_style import apply_nature_style, ANNOT_FONTSIZE

# Colour-blind-safe palettes (Wong). The first is the general categorical
# cycle; the condition map keeps AGED/ADULT consistent with the Plotly pages.
WONG = ["#0072B2", "#D55E00", "#009E73", "#E69F00",
        "#56B4E9", "#CC79A7", "#F0E442", "#000000"]
# Longer distinct cycle (Wong first, then a Tableau-/Glasbey-style spread) so
# many-cluster categorical figures don't repeat a colour after only 8 groups.
CATEGORICAL = WONG + [
    "#984EA3", "#A65628", "#999999", "#66C2A5", "#FC8D62", "#8DA0CB",
    "#E78AC3", "#A6D854", "#FFD92F", "#1B9E77", "#7570B3", "#E7298A",
    "#B3B3B3", "#386CB0", "#BF5B17", "#666666", "#1F78B4", "#33A02C",
    "#FB9A99", "#6A3D9A", "#B15928", "#FDBF6F", "#CAB2D6", "#FFFF99",
]
CONDITION_COLOURS = {"ADULT": "#0072B2", "AGED": "#D55E00",
                     "Control": "#0072B2", "Treatment": "#D55E00"}

# Single-column Nature figure width in inches (~88 mm).
COL_W = 3.46


def _new_fig(w: float = COL_W, h: float = 3.0):
    """Apply the Nature style and return a fresh ``(fig, ax)``."""
    apply_nature_style()
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=(w, h))


def _to_pdf(fig) -> bytes:
    """Serialise a figure to PDF bytes and close it."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _palette_for(order: Sequence[str], palette: Optional[dict] = None) -> dict:
    if palette:
        return palette
    return {g: CATEGORICAL[i % len(CATEGORICAL)] for i, g in enumerate(order)}


# ---------------------------------------------------------------------------
# Scatter (UMAP, spatial maps)
# ---------------------------------------------------------------------------
def scatter_categorical(
    x, y, labels, *, order: Sequence[str], palette: Optional[dict] = None,
    xlabel: str = "", ylabel: str = "", title: str = "",
    point_size: float = 2.0, legend_title: Optional[str] = None,
    equal_aspect: bool = False, invert_y: bool = False, dark_bg: bool = False,
    w: float = COL_W + 1.0, h: float = 3.2,
) -> bytes:
    """Categorical scatter (e.g. UMAP / spatial map coloured by cluster).

    Points are rasterised (the PDF stays small with tens of thousands of
    cells) while axes, text and the legend stay vector/editable.
    """
    x = np.asarray(x); y = np.asarray(y)
    labels = np.asarray(labels).astype(str)
    palette = _palette_for(order, palette)
    fig, ax = _new_fig(w, h)
    if dark_bg:
        ax.set_facecolor("#111111")
    for g in order:
        m = labels == g
        if not m.any():
            continue
        ax.scatter(x[m], y[m], s=point_size, linewidths=0,
                   c=palette.get(g, "#888888"), label=str(g), rasterized=True)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    if invert_y:
        ax.invert_yaxis()
    leg = ax.legend(title=legend_title, loc="center left", bbox_to_anchor=(1.01, 0.5),
                    markerscale=max(2.0, 6.0 / max(point_size, 1e-6)),
                    handletextpad=0.3, labelspacing=0.25, borderaxespad=0.0)
    if leg and leg.get_title():
        leg.get_title().set_fontsize(ANNOT_FONTSIZE + 0.5)
    return _to_pdf(fig)


def scatter_continuous(
    x, y, values, *, xlabel: str = "", ylabel: str = "", title: str = "",
    cbar_label: str = "", cmap: str = "viridis", point_size: float = 2.0,
    equal_aspect: bool = False, invert_y: bool = False,
    sort_ascending: bool = True, dark_bg: bool = False,
    w: float = COL_W + 0.6, h: float = 3.2,
) -> bytes:
    """Continuous scatter (e.g. UMAP / spatial map coloured by gene)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    v = np.asarray(values, dtype=float)
    if sort_ascending:  # draw high-expressing cells on top
        o = np.argsort(v)
        x, y, v = x[o], y[o], v[o]
    fig, ax = _new_fig(w, h)
    if dark_bg:
        ax.set_facecolor("#111111")
    sc = ax.scatter(x, y, c=v, s=point_size, linewidths=0, cmap=cmap, rasterized=True)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    if invert_y:
        ax.invert_yaxis()
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label)
    cb.outline.set_linewidth(0.4)
    return _to_pdf(fig)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
def stacked_bar(
    frame, *, sample_col: str, value_col: str, group_col: str,
    sample_order: Sequence[str], group_order: Sequence[str],
    palette: Optional[dict] = None, ylabel: str = "% of cells", title: str = "",
    w: float = COL_W + 0.8, h: float = 3.2,
) -> bytes:
    """Per-sample stacked composition bar (segments = cell types)."""
    import pandas as pd  # noqa: F401  (frame is a DataFrame)
    palette = _palette_for(group_order, palette)
    fig, ax = _new_fig(w, h)
    bottoms = np.zeros(len(sample_order))
    pos = np.arange(len(sample_order))
    pivot = (frame.pivot_table(index=sample_col, columns=group_col, values=value_col,
                               aggfunc="sum", observed=True)
                  .reindex(index=sample_order))
    for g in group_order:
        vals = pivot[g].to_numpy() if g in pivot.columns else np.zeros(len(sample_order))
        vals = np.nan_to_num(vals)
        ax.bar(pos, vals, bottom=bottoms, width=0.8,
               color=palette.get(g, "#888888"), label=str(g),
               edgecolor="white", linewidth=0.3)
        bottoms += vals
    ax.set_xticks(pos)
    ax.set_xticklabels(sample_order, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), handletextpad=0.3,
              labelspacing=0.25, borderaxespad=0.0, title=group_col)
    return _to_pdf(fig)


def grouped_dots(
    points_df, means_df, *, group_col: str, value_col: str, cond_col: str,
    group_order: Sequence[str], conds: Sequence[str], cond_colour: dict,
    ylabel: str = "% of cells", title: str = "",
    w: float = COL_W + 1.0, h: float = 3.2,
) -> bytes:
    """Per-condition mean bars with one dot per replicate (honest n≈2)."""
    fig, ax = _new_fig(w, h)
    n_cond = max(len(conds), 1)
    bar_w = 0.8 / n_cond
    idx = {g: i for i, g in enumerate(group_order)}
    base = np.arange(len(group_order))
    for j, c in enumerate(conds):
        off = (j - (n_cond - 1) / 2) * bar_w
        sub = means_df[means_df[cond_col] == c]
        gx = np.array([idx[g] for g in sub[group_col] if g in idx])
        gy = np.array([v for g, v in zip(sub[group_col], sub[value_col]) if g in idx])
        ax.bar(base[gx.astype(int)] + off if len(gx) else [], gy if len(gx) else [],
               width=bar_w, color=cond_colour.get(c, "#888888"), alpha=0.45,
               label=f"{c} (mean)")
        pts = points_df[points_df[cond_col] == c]
        px = np.array([idx[g] + off for g in pts[group_col] if g in idx], dtype=float)
        py = np.array([v for g, v in zip(pts[group_col], pts[value_col]) if g in idx],
                      dtype=float)
        if len(px):
            jit = (np.random.default_rng(0).random(len(px)) - 0.5) * bar_w * 0.5
            ax.scatter(px + jit, py, s=12, color=cond_colour.get(c, "#888888"),
                       edgecolors="black", linewidths=0.4, zorder=3,
                       label=f"{c} (replicates)")
    ax.set_xticks(base)
    ax.set_xticklabels(list(group_order), rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), handletextpad=0.3,
              labelspacing=0.25, borderaxespad=0.0)
    return _to_pdf(fig)


# ---------------------------------------------------------------------------
# DGE volcano
# ---------------------------------------------------------------------------
def volcano(
    df, *, lfc_thresh: float, padj_thresh: float, direction: str = "",
    top_n: int = 12, title: str = "", gene_col: str = "gene",
    lfc_col: str = "log2fc", p_col: str = "pval", padj_col: str = "padj",
    w: float = COL_W + 0.4, h: float = 3.4,
) -> bytes:
    """Volcano: -log10 p vs log2FC, DE genes highlighted + top genes labelled."""
    d = df.dropna(subset=[p_col]).copy()
    lfc = d[lfc_col].to_numpy(dtype=float)
    nlp = -np.log10(np.clip(d[p_col].to_numpy(dtype=float), 1e-300, None))
    sig = ((d[padj_col].to_numpy(dtype=float) < padj_thresh)
           & (np.abs(lfc) > lfc_thresh))
    fig, ax = _new_fig(w, h)
    ax.scatter(lfc[~sig], nlp[~sig], s=7, c="#B8C4D0", linewidths=0,
               rasterized=True, label="ns")
    ax.scatter(lfc[sig], nlp[sig], s=9, c="#D55E00",
               edgecolors="black", linewidths=0.2, label="DE")
    for thr in (lfc_thresh, -lfc_thresh):
        ax.axvline(thr, color="grey", lw=0.5, ls="--")
    # Label the most significant genes.
    d2 = d.assign(_nlp=nlp).sort_values(p_col).head(top_n)
    for _, r in d2.iterrows():
        ax.annotate(str(r[gene_col]), (r[lfc_col], r["_nlp"]),
                    fontsize=ANNOT_FONTSIZE, xytext=(0, 3),
                    textcoords="offset points", ha="center")
    xl = f"log2 fold-change ({direction})" if direction else "log2 fold-change"
    ax.set_xlabel(xl); ax.set_ylabel(r"$-\log_{10}$ p")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    return _to_pdf(fig)


# ---------------------------------------------------------------------------
# Heatmaps (neighbourhood enrichment, spatial grids)
# ---------------------------------------------------------------------------
def heatmap(
    matrix, *, x_labels: Sequence[str], y_labels: Sequence[str],
    cmap: str = "RdBu_r", center: Optional[float] = 0.0,
    vmin: Optional[float] = None, vmax: Optional[float] = None,
    cbar_label: str = "", title: str = "", invert_y: bool = True,
    xlabel: str = "", ylabel: str = "", annotate: bool = False,
    w: float = COL_W + 0.6, h: float = 3.4,
) -> bytes:
    """Generic heatmap (diverging by default, centred at ``center``)."""
    M = np.asarray(matrix, dtype=float)
    if center is not None and (vmin is None or vmax is None):
        a = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
        vmin, vmax = center - a, center + a
    fig, ax = _new_fig(w, h)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   origin="upper" if invert_y else "lower")
    if len(x_labels):
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(list(x_labels), rotation=45, ha="right")
    if len(y_labels):
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(list(y_labels))
    if annotate and M.size <= 400:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                            fontsize=ANNOT_FONTSIZE - 0.5, color="black")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label)
    cb.outline.set_linewidth(0.4)
    return _to_pdf(fig)


# ---------------------------------------------------------------------------
# Gene focus: violins + bars
# ---------------------------------------------------------------------------
def violin_by_group(
    df, *, group_col: str, value_col: str, order: Sequence[str],
    xlabel: str = "", ylabel: str = "", title: str = "",
    split_col: Optional[str] = None, split_levels: Optional[Sequence[str]] = None,
    split_colour: Optional[dict] = None,
    w: float = COL_W + 1.2, h: float = 3.0,
) -> bytes:
    """Per-group violins of a continuous value, optionally split by condition."""
    fig, ax = _new_fig(w, h)
    base = np.arange(len(order))

    def _violin(positions, data, colour):
        data = [np.asarray(d, dtype=float) for d in data]
        keep = [(p, d) for p, d in zip(positions, data) if len(d) > 1]
        if not keep:
            return
        ps, ds = zip(*keep)
        parts = ax.violinplot(list(ds), positions=list(ps), widths=0.8,
                              showmeans=False, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(colour); b.set_edgecolor("none"); b.set_alpha(0.75)

    if split_col and split_levels:
        n = len(split_levels)
        bw = 0.8 / n
        for j, lvl in enumerate(split_levels):
            off = (j - (n - 1) / 2) * bw
            sub = df[df[split_col] == lvl]
            data = [sub.loc[sub[group_col] == g, value_col].to_numpy() for g in order]
            colour = (split_colour or {}).get(lvl, WONG[j % len(WONG)])
            _violin(base + off, data, colour)
            ax.scatter([], [], color=colour, label=str(lvl))  # legend proxy
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                  handletextpad=0.3, borderaxespad=0.0, title=split_col)
    else:
        data = [df.loc[df[group_col] == g, value_col].to_numpy() for g in order]
        _violin(base, data, WONG[0])
    ax.set_xticks(base)
    ax.set_xticklabels(list(order), rotation=45, ha="right")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return _to_pdf(fig)


def vbar(
    x, y, *, xlabel: str = "", ylabel: str = "", title: str = "",
    colour: str = "#1B4F8A", w: float = COL_W + 0.6, h: float = 2.8,
) -> bytes:
    """Simple vertical bar chart (e.g. % of cells expressing a gene)."""
    x = list(x); y = np.asarray(y, dtype=float)
    pos = np.arange(len(x))
    fig, ax = _new_fig(w, h)
    ax.bar(pos, y, width=0.8, color=colour)
    ax.set_xticks(pos)
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return _to_pdf(fig)


def hbar(
    values, labels, *, xlabel: str = "", ylabel: str = "", title: str = "",
    pos_colour: str = "#D55E00", neg_colour: str = "#0072B2",
    w: float = COL_W + 0.4, h: float = 3.0,
) -> bytes:
    """Horizontal diverging bar (e.g. per-cluster log2FC)."""
    values = np.asarray(values, dtype=float)
    labels = list(labels)
    pos = np.arange(len(labels))
    colours = np.where(values > 0, pos_colour, neg_colour)
    fig, ax = _new_fig(w, max(h, 0.26 * len(labels) + 0.8))
    ax.barh(pos, values, color=list(colours))
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return _to_pdf(fig)
