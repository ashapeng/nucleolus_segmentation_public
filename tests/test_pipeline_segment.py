import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from skimage import io

from pipeline.inventory import inventory_experiment
from pipeline.segment import batch_gc_segment, run_gc_segment


def _sphere_mask(shape=(8, 24, 24), radius=7):
    mask = np.zeros(shape, dtype=np.uint8)
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 255
    return mask


def _write_synthetic_cell(cell_dir: str):
    os.makedirs(cell_dir, exist_ok=True)
    nucleus = _sphere_mask()
    # Bright GC signal inside nucleus on channel 2
    raw = np.zeros(nucleus.shape + (3,), dtype=np.float32)
    gc_core = _sphere_mask(radius=4)
    for c in range(3):
        raw[..., c] = nucleus.astype(np.float32) * 10
    raw[..., 2] = np.where(gc_core > 0, 500.0, raw[..., 2])
    io.imsave(os.path.join(cell_dir, "Composite_stack.tif"), raw, check_contrast=False)
    io.imsave(os.path.join(cell_dir, "nuclei_mask.tif"), nucleus, check_contrast=False)
    io.imsave(os.path.join(cell_dir, "background_mask.tif"), np.zeros_like(nucleus), check_contrast=False)


def test_run_gc_segment_writes_three_masks(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    _write_synthetic_cell(str(cell))
    paths = run_gc_segment(str(cell))
    assert os.path.isfile(paths.gc)
    assert os.path.isfile(paths.holes)
    assert os.path.isfile(paths.hole_filled)
    gc = io.imread(paths.gc)
    assert gc.dtype == np.uint8
    assert gc.ndim == 3


def test_batch_gc_segment_skips_invalid(tmp_path):
    good = tmp_path / "20220304_L1" / "10_2"
    _write_synthetic_cell(str(good))
    bad = tmp_path / "20220304_L1" / "bad"
    bad.mkdir(parents=True)
    (bad / "Composite_stack.tif").write_bytes(b"not-a-tiff")

    manifest = inventory_experiment(str(tmp_path))
    assert len(manifest.cells_included) == 1
    results = batch_gc_segment(manifest)
    assert len(results) == 1
    cell, paths, err = results[0]
    assert cell.cell_id.endswith("10_2")
    assert paths is not None
    assert err is None
