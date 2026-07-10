import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.inventory import inventory_experiment, validate_cell


def test_validate_cell_missing_files(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    cell.mkdir(parents=True)
    (cell / "Composite_stack.tif").write_bytes(b"x")
    result = validate_cell(str(cell), experiment_set="20220304_L1")
    assert result.valid is False
    assert "nuclei_mask.tif" in result.missing_files
    assert "background_mask.tif" in result.missing_files
    assert result.cell_id == "20220304_L1/10_2"
    assert result.stage == "L1"


def test_validate_cell_complete(tmp_path):
    cell = tmp_path / "20220304_L2" / "5_3"
    cell.mkdir(parents=True)
    for name in ("Composite_stack.tif", "nuclei_mask.tif", "background_mask.tif"):
        (cell / name).write_bytes(b"x")
    result = validate_cell(str(cell), experiment_set="20220304_L2")
    assert result.valid is True
    assert result.missing_files == []
    assert result.stage == "L2"


def test_inventory_includes_valid_cells(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    cell.mkdir(parents=True)
    for name in ("Composite_stack.tif", "nuclei_mask.tif", "background_mask.tif"):
        (cell / name).write_bytes(b"x")
    bad = tmp_path / "20220304_L1" / "bad_cell"
    bad.mkdir(parents=True)
    (bad / "Composite_stack.tif").write_bytes(b"x")

    manifest = inventory_experiment(str(tmp_path))
    assert len(manifest.cells_included) == 1
    assert manifest.cells_included[0].stage == "L1"
    assert len(manifest.cells_excluded) == 1
    assert "nuclei_mask.tif" in manifest.cells_excluded[0].missing_files
