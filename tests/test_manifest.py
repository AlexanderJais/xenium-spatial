"""
Tests for ``SlideManifest.from_csv`` — in particular the header auto-detection,
which must not depend on whether the run_dir paths resolve on this machine.

Regression: a headerless manifest with relative paths (data living elsewhere)
used to lose its first slide, because the detector probed the filesystem and,
finding the path absent, mistook the first data row for a header.
"""

import pandas as pd

from xenium_spatial.multislide_loader import SlideManifest


def _write_csv(path, rows, header=None):
    df = pd.DataFrame(rows)
    df.to_csv(path, header=header if header is not None else False, index=False)
    return path


def test_headerless_relative_paths_keeps_all_slides(tmp_path):
    """No header + relative paths that don't exist here -> every row kept."""
    csv = _write_csv(
        tmp_path / "manifest.csv",
        [
            ["AGED_1", "AGED", "relative/path/aged1"],
            ["ADULT_1", "ADULT", "relative/path/adult1"],
        ],
    )
    m = SlideManifest.from_csv(csv)
    assert len(m) == 2
    assert m.slide_ids == ["AGED_1", "ADULT_1"]
    assert m.conditions == ["AGED", "ADULT"]


def test_header_row_is_detected_and_dropped(tmp_path):
    """A real header row is recognised and not turned into a slide."""
    csv = tmp_path / "manifest.csv"
    pd.DataFrame(
        [["AGED_1", "AGED", "/data/aged1"]],
        columns=["slide_id", "condition", "run_dir"],
    ).to_csv(csv, index=False)

    m = SlideManifest.from_csv(csv)
    assert len(m) == 1
    assert m.slide_ids == ["AGED_1"]
    # The header tokens must not leak in as a slide.
    assert "slide_id" not in m.slide_ids


def test_optional_replicate_id_column(tmp_path):
    """A 4th column is read as replicate_id; otherwise it defaults to slide_id."""
    csv = _write_csv(
        tmp_path / "manifest.csv",
        [
            ["AGED_1", "AGED", "/data/aged1", "rep_a"],
            ["ADULT_1", "ADULT", "/data/adult1", "rep_b"],
        ],
    )
    m = SlideManifest.from_csv(csv)
    reps = [s["replicate_id"] for s in m]
    assert reps == ["rep_a", "rep_b"]


def test_headerless_absolute_paths(tmp_path):
    """Absolute paths with no header are all retained (the original happy path)."""
    csv = _write_csv(
        tmp_path / "manifest.csv",
        [
            ["AGED_1", "AGED", "/data/aged1"],
            ["ADULT_1", "ADULT", "/data/adult1"],
        ],
    )
    m = SlideManifest.from_csv(csv)
    assert m.slide_ids == ["AGED_1", "ADULT_1"]
