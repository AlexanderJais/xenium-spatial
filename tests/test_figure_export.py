"""
Tests for the publication PDF renderers (figure_export). These guard that each
builder returns a non-empty, valid PDF byte string for representative inputs —
matplotlib/numpy/pandas are imported lazily, so the suite skips cleanly in a
minimal environment.
"""
import pytest


def _is_pdf(b: bytes) -> bool:
    return isinstance(b, (bytes, bytearray)) and len(b) > 100 and b[:5] == b"%PDF-"


def test_scatter_and_heatmap_pdf():
    pytest.importorskip("matplotlib")
    np = pytest.importorskip("numpy")
    from xenium_spatial import figure_export as fx

    rng = np.random.default_rng(0)
    x, y = rng.normal(size=200), rng.normal(size=200)
    labels = np.where(x > 0, "A", "B")
    cat = fx.scatter_categorical(x, y, labels, order=["A", "B"],
                                 xlabel="UMAP1", ylabel="UMAP2", legend_title="grp")
    assert _is_pdf(cat)

    cont = fx.scatter_continuous(x, y, rng.random(200), cbar_label="expr")
    assert _is_pdf(cont)

    z = rng.normal(size=(4, 5))
    hm = fx.heatmap(z, x_labels=list("abcde"), y_labels=list("wxyz"),
                    cbar_label="z", annotate=True)
    assert _is_pdf(hm)


def test_volcano_and_bars_pdf():
    pytest.importorskip("matplotlib")
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from xenium_spatial import figure_export as fx

    df = pd.DataFrame({
        "gene": [f"g{i}" for i in range(20)],
        "log2fc": np.linspace(-3, 3, 20),
        "pval": np.linspace(1e-6, 0.9, 20),
        "padj": np.linspace(1e-5, 0.95, 20),
    })
    assert _is_pdf(fx.volcano(df, lfc_thresh=1.0, padj_thresh=0.1, direction="AGED/ADULT"))
    assert _is_pdf(fx.vbar(list("abcd"), [1, 2, 3, 4], ylabel="%"))
    assert _is_pdf(fx.hbar([-2.0, 1.0, 0.5], ["a", "b", "c"], xlabel="log2FC"))


def test_composition_and_violin_pdf():
    pytest.importorskip("matplotlib")
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from xenium_spatial import figure_export as fx

    comp = pd.DataFrame({
        "replicate": ["S1", "S1", "S2", "S2", "S3", "S3", "S4", "S4"],
        "condition": ["AGED", "AGED", "AGED", "AGED", "ADULT", "ADULT", "ADULT", "ADULT"],
        "cell_type": ["A", "B"] * 4,
        "percent": [60, 40, 55, 45, 30, 70, 35, 65],
    })
    samp_order = ["S1", "S2", "S3", "S4"]
    grp_order = ["A", "B"]
    assert _is_pdf(fx.stacked_bar(comp, sample_col="replicate", value_col="percent",
                                  group_col="cell_type", sample_order=samp_order,
                                  group_order=grp_order))
    means = (comp.groupby(["cell_type", "condition"])["percent"].mean().reset_index())
    cc = {"AGED": "#D55E00", "ADULT": "#0072B2"}
    assert _is_pdf(fx.grouped_dots(comp, means, group_col="cell_type", value_col="percent",
                                   cond_col="condition", group_order=grp_order,
                                   conds=["ADULT", "AGED"], cond_colour=cc))

    vio = pd.DataFrame({"group": np.repeat(list("AB"), 50),
                        "expr": np.r_[np.random.default_rng(1).normal(1, 0.5, 50),
                                      np.random.default_rng(2).normal(2, 0.5, 50)],
                        "condition": np.tile(["AGED", "ADULT"], 50)})
    assert _is_pdf(fx.violin_by_group(vio, group_col="group", value_col="expr",
                                      order=["A", "B"], ylabel="expr"))
    assert _is_pdf(fx.violin_by_group(vio, group_col="group", value_col="expr",
                                      order=["A", "B"], ylabel="expr",
                                      split_col="condition",
                                      split_levels=["AGED", "ADULT"], split_colour=cc))
