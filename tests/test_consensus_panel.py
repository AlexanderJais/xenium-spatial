"""
Tests for the consensus gene panel: the strict intersection of every sample's
panel (base + add-on), used to guarantee no gene is ever zero-filled.
"""
import pytest


def _registry(tmp_path):
    pd = pytest.importorskip("pandas")
    # Minimal base panel CSV (PanelRegistry reads the first column as gene names).
    csv = tmp_path / "base.csv"
    pd.DataFrame({"Genes": ["A", "B", "C"]}).to_csv(csv, index=False)
    from xenium_spatial.panel_registry import PanelRegistry
    return PanelRegistry(csv)


def test_all_shared_keeps_everything(tmp_path):
    reg = _registry(tmp_path)
    gs = {"S1": {"A", "B", "C", "Gal", "Galr1"},
          "S2": {"A", "B", "C", "Gal", "Galr1"}}
    con = reg.consensus_panel(gs)
    assert con["base"] == ["A", "B", "C"]
    assert con["addon"] == ["Gal", "Galr1"]
    assert con["consensus"] == ["A", "B", "C", "Gal", "Galr1"]
    assert con["excluded_addon"] == [] and con["excluded_base"] == []


def test_partial_addon_is_excluded(tmp_path):
    reg = _registry(tmp_path)
    # Gal in both; Galr1 only in S1; Avp only in S2 → both excluded.
    gs = {"S1": {"A", "B", "C", "Gal", "Galr1"},
          "S2": {"A", "B", "C", "Gal", "Avp"}}
    con = reg.consensus_panel(gs)
    assert con["addon"] == ["Gal"]
    assert con["excluded_addon"] == ["Avp", "Galr1"]
    # Presence matrix reflects which slide carries each excluded gene.
    assert con["presence"]["Galr1"] == {"S1": True, "S2": False}
    assert con["presence"]["Avp"] == {"S1": False, "S2": True}
    assert con["per_slide"]["S1"]["custom_in_consensus"] == 1  # only Gal


def test_missing_base_gene_is_dropped_strictly(tmp_path):
    reg = _registry(tmp_path)
    gs = {"S1": {"A", "B", "C"}, "S2": {"A", "B"}}  # C missing from S2
    con = reg.consensus_panel(gs)
    assert con["base"] == ["A", "B"]
    assert con["excluded_base"] == ["C"]
    assert "C" not in con["consensus"]


def test_harmonise_consensus_mode_zero_fills_nothing(tmp_path):
    ad = pytest.importorskip("anndata")
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    reg = _registry(tmp_path)

    def _adata(genes):
        return ad.AnnData(X=np.ones((4, len(genes)), dtype="float32"),
                          var=pd.DataFrame(index=list(genes)))

    a1 = _adata(["A", "B", "C", "Gal", "Galr1"])
    a2 = _adata(["A", "B", "C", "Gal", "Avp"])
    out = reg.harmonise([a1, a2], ["S1", "S2"], mode="consensus")
    # Consensus = A,B,C,Gal (Galr1/Avp partial → excluded); identical per slide.
    for a in out:
        assert list(a.var_names) == ["A", "B", "C", "Gal"]
        if "zero_filled" in a.var:
            assert not bool(a.var["zero_filled"].any())
