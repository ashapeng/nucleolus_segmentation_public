import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import measure_util as mu


# ---------------------------------------------------------------------------
# extract_stage
# ---------------------------------------------------------------------------

def test_extract_stage_valid():
    assert mu.extract_stage("20220304_L1/10_2") == "L1"
    assert mu.extract_stage("20220304_L2/5_3") == "L2"
    assert mu.extract_stage("batch_L3_run1/cell_001") == "L3"
    assert mu.extract_stage("20220304_L4/10_2") == "L4"


def test_extract_stage_invalid():
    with pytest.raises(ValueError, match="Cannot extract larval stage"):
        mu.extract_stage("20220304/no_stage_here")


# ---------------------------------------------------------------------------
# dilated_mask — Phase 1 regression: 'if_dilated' NameError is fixed
# ---------------------------------------------------------------------------

def test_dilated_mask_3d_no_nameError():
    mask = np.zeros((6, 16, 16), dtype=np.uint8)
    mask[3, 7:9, 7:9] = 1
    out = mu.dilated_mask(mask, radius=1, dilated_3d=True)
    assert out.shape == mask.shape


def test_dilated_mask_slice_by_slice():
    mask = np.zeros((6, 16, 16), dtype=np.uint8)
    mask[3, 7:9, 7:9] = 1
    out = mu.dilated_mask(mask, radius=1, dilated_3d=False, dilate_slice_by_slice=True)
    assert out.shape == mask.shape
    # dilation should grow the mask area
    assert np.count_nonzero(out) >= np.count_nonzero(mask)


# ---------------------------------------------------------------------------
# shape_discriber — resolution is honoured (Phase 1 regression)
# ---------------------------------------------------------------------------

def _make_sphere_mask(shape=(10, 32, 32), radius=5):
    mask = np.zeros(shape, dtype=np.uint8)
    cx, cy, cz = shape[2] // 2, shape[1] // 2, shape[0] // 2
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius ** 2:
                    mask[z, y, x] = 255
    return mask


PARAMS_3D = ["cell_id", "obj_id", "surface_area", "volume",
             "surface_to_volume_ratio", "sphericity", "aspect_ratio", "solidity"]


def test_shape_discriber_empty_mask_returns_zeros():
    mask = np.zeros((6, 16, 16), dtype=np.uint8)
    df = mu.shape_discriber(mask, resolution=[0.2, 0.08, 0.08],
                            cell_id="L1/cell1", measured_parameters=PARAMS_3D)
    assert df["volume"].iloc[0] == 0


def test_shape_discriber_respects_resolution():
    """Volume must scale with voxel size³ (Phase 1 regression: override removed)."""
    mask = _make_sphere_mask()
    res_a = [0.2, 0.08, 0.08]
    res_b = [0.4, 0.16, 0.16]  # 2× in each dim → 8× volume
    df_a = mu.shape_discriber(mask, resolution=res_a, cell_id="L1/c1", measured_parameters=PARAMS_3D)
    df_b = mu.shape_discriber(mask, resolution=res_b, cell_id="L1/c1", measured_parameters=PARAMS_3D)
    vol_a = float(df_a["volume"].iloc[0])
    vol_b = float(df_b["volume"].iloc[0])
    assert vol_a > 0
    # vol_b should be ~8× vol_a (allow 5% tolerance for surface discretisation)
    ratio = vol_b / vol_a
    assert abs(ratio - 8.0) < 0.5, f"Expected ~8×, got {ratio:.3f}"


# ---------------------------------------------------------------------------
# concentration_gc — empty nucleoplasm mask returns NaN, not crash
# ---------------------------------------------------------------------------

def test_concentration_gc_empty_nucleoplasm_returns_nan():
    shape = (5, 16, 16)
    raw_img = np.ones(shape, dtype=np.float32) * 100
    raw_bg = np.ones(shape, dtype=np.float32) * 50
    background_mask = np.zeros(shape, dtype=np.uint8)
    background_mask[2, 0:3, 0:3] = 1
    nucleoplasm_mask = np.zeros(shape, dtype=np.uint8)  # empty
    seg_mask = np.zeros(shape, dtype=np.uint8)
    seg_mask[2, 7:9, 7:9] = 1
    hole_filled = seg_mask.copy()
    nucleus_mask = np.ones(shape, dtype=np.uint8)

    params = ["cell_id", "C_bg", "C_dilute", "C_dense", "pc", "total"]
    df = mu.concentration_gc(raw_img, raw_bg, background_mask, nucleoplasm_mask,
                              seg_mask, hole_filled, nucleus_mask,
                              "L1/cell1", params)
    import math
    assert math.isnan(float(df["pc"].iloc[0]))


# ---------------------------------------------------------------------------
# batch_measure_shape — path separator uses os.sep (Phase 1 regression)
# ---------------------------------------------------------------------------

def test_batch_measure_shape_path_separator(tmp_path):
    """cell_id should use os.sep, not a hardcoded backslash."""
    # Build a minimal directory tree: master/experiment_set/cell/gc.tif
    exp_dir = tmp_path / "20220304_L1"
    cell_dir = exp_dir / "10_2"
    cell_dir.mkdir(parents=True)

    mask = np.zeros((4, 8, 8), dtype=np.uint8)
    from skimage import io
    io.imsave(str(cell_dir / "gc.tif"), mask)

    params = ["cell_id", "obj_id", "surface_area", "volume",
              "surface_to_volume_ratio", "sphericity", "aspect_ratio", "solidity"]
    dfs = mu.batch_measure_shape(str(tmp_path), "gc.tif", params)
    assert len(dfs) == 1
    cell_id = dfs[0]["cell_id"].iloc[0]
    assert "\\" not in cell_id, f"Backslash found in cell_id: {cell_id!r}"
    assert os.sep in cell_id or "/" in cell_id


# ---------------------------------------------------------------------------
# config loader
# ---------------------------------------------------------------------------

def test_config_loader_fallback():
    from config_loader import load_config
    cfg = load_config(path="/nonexistent/path/config.yaml")
    assert "measurement" in cfg
    assert cfg["measurement"]["resolution_3d"] == [0.2, 0.08, 0.08]


# ---------------------------------------------------------------------------
# concentration_gc — semantic column mapping (refactor regression)
# ---------------------------------------------------------------------------

def test_concentration_gc_column_mapping():
    """pc and total must map by name, not by positional Series order."""
    shape = (5, 16, 16)
    raw_img = np.ones(shape, dtype=np.float32) * 200
    raw_bg = np.ones(shape, dtype=np.float32) * 100
    background_mask = np.zeros(shape, dtype=np.uint8)
    background_mask[2, 0:3, 0:3] = 1
    nucleoplasm_mask = np.ones(shape, dtype=np.uint8)
    nucleoplasm_mask[2, 7:9, 7:9] = 0
    seg_mask = np.zeros(shape, dtype=np.uint8)
    seg_mask[2, 7:9, 7:9] = 1
    # Make GC brighter than nucleoplasm in raw so pc > 1
    raw_img[seg_mask > 0] = 400
    hole_filled = seg_mask.copy()
    nucleus_mask = np.ones(shape, dtype=np.uint8)

    params = ["cell_id", "C_bg", "C_dilute", "C_dense", "pc", "total"]
    df = mu.concentration_gc(raw_img, raw_bg, background_mask, nucleoplasm_mask,
                              seg_mask, hole_filled, nucleus_mask,
                              "L1/cell1", params)
    assert float(df["pc"].iloc[0]) > 1.0
    assert float(df["total"].iloc[0]) == 100.0  # mean of raw_bg in nucleus
