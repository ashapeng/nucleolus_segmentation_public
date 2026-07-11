import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from skimage import io

from pipeline.inventory import inventory_experiment
from pipeline.measure import measure_shapes


def _sphere_mask(shape=(8, 24, 24), radius=7):
    mask = np.zeros(shape, dtype=np.uint8)
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 255
    return mask


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
