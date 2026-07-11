import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from skimage import io

from pipeline.measure import measure_intensity_batch, measure_intensity_cell, measure_shapes
from pipeline.types import CellRecord, Manifest


def _sphere_mask(shape=(8, 24, 24), radius=7):
    mask = np.zeros(shape, dtype=np.uint8)
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 255
    return mask


def _write_intensity_cell(cell_dir, *, bg_level=20.0, dilute_level=100.0, dense_level=400.0):
    """Synthetic ZYX C=3 stack + masks with known intensity levels on ch2."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    z, y, x = 6, 32, 32
    raw = np.zeros((z, y, x, 3), dtype=np.float32)

    nucleus = np.zeros((z, y, x), dtype=np.uint8)
    nucleus[:, 4:28, 4:28] = 255

    background = np.zeros((z, y, x), dtype=np.uint8)
    background[:, 0:3, 0:3] = 255

    gc = np.zeros((z, y, x), dtype=np.uint8)
    gc[:, 12:20, 12:20] = 255
    hole_filled = gc.copy()

    # ch2: background / nucleoplasm / GC regions
    raw[:, :, :, 2] = dilute_level
    raw[:, 0:3, 0:3, 2] = bg_level
    raw[:, 12:20, 12:20, 2] = dense_level
    # Keep ch0/ch1 nonzero inside GC so CV path has finite means
    raw[gc > 0, 0] = 50.0
    raw[gc > 0, 1] = 60.0

    io.imsave(str(cell_dir / "Composite_stack.tif"), raw, check_contrast=False)
    io.imsave(str(cell_dir / "nuclei_mask.tif"), nucleus, check_contrast=False)
    io.imsave(str(cell_dir / "background_mask.tif"), background, check_contrast=False)
    io.imsave(str(cell_dir / "gc.tif"), gc, check_contrast=False)
    io.imsave(str(cell_dir / "hole_filled.tif"), hole_filled, check_contrast=False)


def test_measure_shapes_concat(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    cell.mkdir(parents=True)
    mask = _sphere_mask()
    io.imsave(str(cell / "gc.tif"), mask, check_contrast=False)
    # batch_measure_shape also needs the folder walk; inventory not required
    df = measure_shapes(str(tmp_path), mask_name="gc.tif")
    assert not df.empty
    assert "volume" in df.columns
    assert "stage" in df.columns
    assert df["stage"].iloc[0] == "L1"


def test_measure_intensity_cell_applies_bg_subtraction(tmp_path):
    """C_* must use bg-subtracted intensities (regression: 2D bg_sub was a no-op)."""
    cell = tmp_path / "20220304_L1" / "10_2"
    bg_level, dilute_level, dense_level = 20.0, 100.0, 400.0
    _write_intensity_cell(cell, bg_level=bg_level, dilute_level=dilute_level, dense_level=dense_level)

    df = measure_intensity_cell(str(cell), "20220304_L1/10_2")
    assert not df.empty
    c_bg = float(df["C_bg"].iloc[0])
    c_dilute = float(df["C_dilute"].iloc[0])
    c_dense = float(df["C_dense"].iloc[0])

    # After subtraction + clip, mean in bg mask ≈ 0; dense/dilute drop by ~bg_level
    assert c_bg == pytest.approx(0.0, abs=0.5)
    assert c_dilute == pytest.approx(dilute_level - bg_level, abs=1.0)
    assert c_dense == pytest.approx(dense_level - bg_level, abs=1.0)
    # Must not report unsubtracted raw means
    assert c_dense < dense_level - 1.0


def test_measure_intensity_batch_concat(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    _write_intensity_cell(cell)
    manifest = Manifest(
        root=str(tmp_path),
        resolution_3d=[0.2, 0.08, 0.08],
        cells_included=[
            CellRecord(cell_id="20220304_L1/10_2", path=str(cell), stage="L1", valid=True)
        ],
        cells_excluded=[],
        config_snapshot={},
    )
    df = measure_intensity_batch(manifest)
    assert len(df) == 1
    assert "pc" in df.columns
    assert float(df["C_bg"].iloc[0]) == pytest.approx(0.0, abs=0.5)
