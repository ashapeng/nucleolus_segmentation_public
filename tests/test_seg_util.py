import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import seg_util as su


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sphere_mask(shape=(10, 32, 32), radius=8):
    """Return a binary 3D mask containing a single sphere."""
    mask = np.zeros(shape, dtype=np.uint8)
    cx, cy, cz = shape[2] // 2, shape[1] // 2, shape[0] // 2
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius ** 2:
                    mask[z, y, x] = 255
    return mask


# ---------------------------------------------------------------------------
# _percentile_clip_and_normalize
# ---------------------------------------------------------------------------

def test_percentile_clip_normalize_range():
    img = np.random.rand(5, 8, 8).astype(np.float32) * 1000
    out = su._percentile_clip_and_normalize(img)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_percentile_clip_normalize_flat_image():
    img = np.ones((4, 4, 4), dtype=np.float32)
    out = su._percentile_clip_and_normalize(img)
    assert np.all(out == 0)


# ---------------------------------------------------------------------------
# _to_uint8_mask
# ---------------------------------------------------------------------------

def test_to_uint8_mask_values():
    arr = np.array([0, 1, 2, 5], dtype=np.int32)
    out = su._to_uint8_mask(arr)
    assert out.dtype == np.uint8
    assert out[0] == 0
    assert out[1] == 255
    assert out[3] == 255


# ---------------------------------------------------------------------------
# image_2d_seg — Phase 1 regression: 'nucleus' NameError is fixed
# ---------------------------------------------------------------------------

def test_image_2d_seg_no_nucleus_mask():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(5, 16, 16, 1), dtype=np.uint16)
    out = su.image_2d_seg(raw, nucleus_mask=None, sigma_2d=1.0)
    assert out.dtype == np.uint8
    unique = np.unique(out)
    assert set(unique).issubset({0, 255})


def test_image_2d_seg_with_nucleus_mask():
    rng = np.random.default_rng(1)
    raw = rng.integers(0, 256, size=(5, 16, 16, 1), dtype=np.uint16)
    nucleus_mask = np.ones((5, 16, 16, 1), dtype=np.uint8) * 255
    # Should not raise NameError (the 'nucleus' bug fix)
    out = su.image_2d_seg(raw, nucleus_mask=nucleus_mask, sigma_2d=1.0)
    assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# bg_subtraction — 2D / (Y,X,C) regressions (intensity wrapper bug)
# ---------------------------------------------------------------------------

def test_bg_subtraction_2d_single_channel_subtracts():
    """2D (Y,X) must subtract mean positive background (was a silent no-op)."""
    img = np.full((8, 8), 100.0, dtype=np.float32)
    img[0:2, 0:2] = 20.0  # background region
    bg = np.zeros((8, 8), dtype=np.uint8)
    bg[0:2, 0:2] = 255

    out = su.bg_subtraction(img, bg, clip=True)
    assert out.shape == img.shape
    # Foreground 100 - 20 = 80; bg pixels clip to 0
    assert float(out[4, 4]) == pytest.approx(80.0)
    assert float(out[0, 0]) == pytest.approx(0.0)


def test_bg_subtraction_yxc_with_2d_mask():
    """(Y,X,C) with a 2D bg mask must not IndexError and must subtract per channel."""
    img = np.zeros((6, 6, 3), dtype=np.float32)
    img[..., 0] = 50.0
    img[..., 1] = 80.0
    img[..., 2] = 110.0
    img[0:2, 0:2, :] = 10.0
    bg = np.zeros((6, 6), dtype=np.uint8)
    bg[0:2, 0:2] = 255

    out = su.bg_subtraction(img, bg, clip=True)
    assert out.shape == img.shape
    assert float(out[3, 3, 2]) == pytest.approx(100.0)  # 110 - 10
    assert float(out[0, 0, 2]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# global_otsu
# ---------------------------------------------------------------------------

def test_global_otsu_returns_bool():
    rng = np.random.default_rng(2)
    img = rng.random((6, 16, 16)).astype(np.float32)
    mask = np.ones((6, 16, 16), dtype=np.uint8)
    result = su.global_otsu(img, mask, global_thresh_method="ave", mini_size=5)
    assert result.dtype in (bool, np.bool_) or result.max() <= 1


# ---------------------------------------------------------------------------
# gc_segment smoke test
# ---------------------------------------------------------------------------

def test_gc_segment_smoke():
    rng = np.random.default_rng(3)
    raw = rng.random((5, 16, 16, 3)).astype(np.float32)
    nucleus_mask = np.ones((5, 16, 16), dtype=np.uint8)
    final_gc, gc_dark_spot, hole_filled_gc = su.gc_segment(raw, nucleus_mask)
    assert final_gc.shape == (5, 16, 16)
    assert hole_filled_gc.shape == (5, 16, 16)
    assert final_gc.dtype == np.uint8
    assert set(np.unique(final_gc)).issubset({0, 255})


def test_gc_segment_config_override():
    """Explicit sigma argument takes precedence over config."""
    rng = np.random.default_rng(4)
    raw = rng.random((5, 16, 16, 3)).astype(np.float32)
    nucleus_mask = np.ones((5, 16, 16), dtype=np.uint8)
    # Should not raise even with custom config values
    final_gc, _, _ = su.gc_segment(raw, nucleus_mask, sigma=2.0, local_adjust_for_GC=1.0)
    assert final_gc.dtype == np.uint8
