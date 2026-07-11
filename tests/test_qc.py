import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from skimage import io

from pipeline.qc import (
    LOCAL_ADJUST_MAX,
    LOCAL_ADJUST_MIN,
    LOCAL_ADJUST_STEP,
    qc_masks,
    suggest_param_overrides,
)
from pipeline.types import QCReport


def _write_masks(cell_dir, gc, nucleus):
    os.makedirs(cell_dir, exist_ok=True)
    io.imsave(os.path.join(cell_dir, "gc.tif"), gc.astype(np.uint8), check_contrast=False)
    io.imsave(os.path.join(cell_dir, "nuclei_mask.tif"), nucleus.astype(np.uint8), check_contrast=False)


def test_qc_empty_mask_is_red(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    nucleus = np.ones((4, 8, 8), dtype=np.uint8) * 255
    gc = np.zeros_like(nucleus)
    _write_masks(str(cell), gc, nucleus)
    report = qc_masks(str(cell))
    assert report.status == "RED"
    assert report.empty_gc is True


def test_qc_good_fraction_is_green(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    nucleus = np.ones((4, 10, 10), dtype=np.uint8) * 255
    gc = np.zeros_like(nucleus)
    gc[:, 0:3, 0:3] = 255  # 4*9 / 400 = 0.09
    _write_masks(str(cell), gc, nucleus)
    report = qc_masks(str(cell))
    assert report.status == "GREEN"
    assert 0.05 <= report.gc_fraction <= 0.40


def test_qc_outside_nucleus_is_red(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    nucleus = np.zeros((4, 10, 10), dtype=np.uint8)
    nucleus[:, 0:5, 0:5] = 255
    gc = np.zeros_like(nucleus)
    gc[:, 5:10, 5:10] = 255  # entirely outside
    _write_masks(str(cell), gc, nucleus)
    report = qc_masks(str(cell))
    assert report.status == "RED"
    assert report.outside_nucleus_fraction > 0.05


def test_suggest_param_overrides_bounds():
    report = QCReport(
        cell_id="x",
        status="RED",
        gc_fraction=0.01,
        empty_gc=False,
        outside_nucleus_fraction=0.0,
    )
    overrides = suggest_param_overrides(report, attempt=0, current_local_adjust=1.08)
    assert overrides is not None
    assert overrides["local_adjust"] >= LOCAL_ADJUST_MIN
    assert overrides["local_adjust"] <= LOCAL_ADJUST_MAX
    assert suggest_param_overrides(report, attempt=3, current_local_adjust=1.08) is None


def test_suggest_uses_current_local_adjust_not_hardcoded_base():
    """Retries must step from the value that just failed, not a hardcoded 1.08."""
    report = QCReport(
        cell_id="x",
        status="RED",
        gc_fraction=0.01,
        empty_gc=False,
        outside_nucleus_fraction=0.0,
    )
    overrides = suggest_param_overrides(report, attempt=0, current_local_adjust=1.15)
    assert overrides == {"local_adjust": 1.15 - LOCAL_ADJUST_STEP}


def test_suggest_clamps_to_shared_al_bounds():
    report_low = QCReport(
        cell_id="x",
        status="RED",
        gc_fraction=0.0,
        empty_gc=True,
        outside_nucleus_fraction=0.0,
    )
    low = suggest_param_overrides(report_low, attempt=0, current_local_adjust=LOCAL_ADJUST_MIN)
    assert low is None  # already at inclusive floor

    report_high = QCReport(
        cell_id="x",
        status="RED",
        gc_fraction=0.50,
        empty_gc=False,
        outside_nucleus_fraction=0.0,
    )
    high = suggest_param_overrides(report_high, attempt=0, current_local_adjust=LOCAL_ADJUST_MAX)
    assert high is None  # already at exclusive ceiling

    stepped = suggest_param_overrides(report_high, attempt=0, current_local_adjust=1.18)
    assert stepped is not None
    assert stepped["local_adjust"] == LOCAL_ADJUST_MAX


def test_suggest_raises_local_adjust_when_outside_nucleus():
    report = QCReport(
        cell_id="x",
        status="RED",
        gc_fraction=0.20,
        empty_gc=False,
        outside_nucleus_fraction=0.08,
    )
    overrides = suggest_param_overrides(report, attempt=0, current_local_adjust=1.08)
    assert overrides == {"local_adjust": 1.08 + LOCAL_ADJUST_STEP}


def test_active_learning_shares_local_adjust_bounds():
    from ml.active_learning import _SPACE_DEFS

    la = next(d for d in _SPACE_DEFS if d[0] == "local_adjust")
    assert la[1] == LOCAL_ADJUST_MIN
    assert la[2] == LOCAL_ADJUST_MAX
