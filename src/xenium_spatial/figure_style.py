"""
figure_style.py
---------------
Shared Nature-grade matplotlib style for the publication figures (Sample-PCA
scatter, correlation heatmap, scree plot, PCA elbow).

Follows the Nature figure guidelines: a sans-serif typeface, thin spines,
colour-blind-safe palettes (applied by the callers), and editable Type-42
fonts in the PDF. Font sizes are set a step larger than the journal minimum
for on-page readability at the final single-column width (~88 mm / 3.5 in).
"""

# rcParams applied to every publication figure.
NATURE_RC = {
    "font.family"      : "sans-serif",
    "font.sans-serif"  : ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "font.size"        : 8,     # base (was 7)
    "axes.titlesize"   : 9,     # panel titles (was 8)
    "axes.labelsize"   : 8,     # axis labels (was 7)
    "xtick.labelsize"  : 7,     # tick labels (was 6)
    "ytick.labelsize"  : 7,
    "legend.fontsize"  : 7,     # (was 6)
    "legend.frameon"   : False,
    "axes.linewidth"   : 0.6,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size" : 2.5,
    "ytick.major.size" : 2.5,
    "xtick.direction"  : "out",
    "ytick.direction"  : "out",
    "lines.linewidth"  : 1.0,
    "savefig.bbox"     : "tight",
    "pdf.fonttype"     : 42,    # editable text in Illustrator
    "ps.fonttype"      : 42,
}

# Small in-plot text (e.g. per-point sample labels): one step below tick labels.
ANNOT_FONTSIZE = 6.5


def apply_nature_style() -> None:
    """Apply the Nature-grade rcParams; call once inside each plotting function."""
    import matplotlib as mpl
    mpl.rcParams.update(NATURE_RC)
