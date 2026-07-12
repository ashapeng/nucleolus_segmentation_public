import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from skimage import io

from pipeline.qc import (
    LOCAL_ADJUST_MAX,
    LOCAL_ADJUST_MIN,
    LOCAL_ADJUST_STEP,
    SOFT_MID_Z_NO_GC,
    SOFT_MULTI_GC_OBJECTS,
    qc_masks,
    suggest_param_overrides,
)
from pipeline.segment_with_qc import segment_with_qc
from pipeline.types import MaskPaths, QCReport


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
    assert report.n_gc_objects == 0


def test_qc_good_fraction_is_green(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    nucleus = np.ones((4, 10, 10), dtype=np.uint8) * 255
    gc = np.zeros_like(nucleus)
    gc[:, 0:3, 0:3] = 255  # 4*9 / 400 = 0.09
    _write_masks(str(cell), gc, nucleus)
    report = qc_masks(str(cell))
    assert report.status == "GREEN"
    assert 0.05 <= report.gc_fraction <= 0.40
    assert report.n_gc_objects == 1
    assert report.mid_z_index is not None
    assert SOFT_MID_Z_NO_GC not in report.soft_flags


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


def test_qc_low_fraction_amber_band(tmp_path):
    """0.02 ≤ gc_fraction < 0.05 → AMBER (accepted without retry by segment_with_qc)."""
    cell = tmp_path / "20220304_L1" / "10_2"
    nucleus = np.ones((4, 10, 10), dtype=np.uint8) * 255  # 400 voxels
    gc = np.zeros_like(nucleus)
    gc[:, 0:1, 0:3] = 255  # 12 / 400 = 0.03
    _write_masks(str(cell), gc, nucleus)
    report = qc_masks(str(cell))
    assert report.status == "AMBER"
    assert 0.02 <= report.gc_fraction < 0.05
    assert report.outside_nucleus_fraction <= 0.01


def test_qc_high_fraction_amber_band(tmp_path):
    """0.40 < gc_fraction ≤ 0.60 → AMBER."""
    cell = tmp_path / "20220304_L1" / "10_2"
    nucleus = np.ones((4, 10, 10), dtype=np.uint8) * 255  # 400 voxels
    gc = np.zeros_like(nucleus)
    gc[:, 0:5, 0:9] = 255  # 180 / 400 = 0.45
    _write_masks(str(cell), gc, nucleus)
    report = qc_masks(str(cell))
    assert report.status == "AMBER"
    assert 0.40 < report.gc_fraction <= 0.60
    assert report.outside_nucleus_fraction <= 0.01


def test_qc_outside_amber_band(tmp_path):
    """Good fraction + 0.01 < outside ≤ 0.05 → AMBER."""
    cell = tmp_path / "20220304_L1" / "10_2"
    nucleus = np.zeros((4, 10, 10), dtype=np.uint8)
    nucleus[:, 0:8, :] = 255  # 320 voxels
    gc = np.zeros_like(nucleus)
    gc[:, 0:4, 0:4] = 255  # 64 inside
    gc[0, 9, 0] = 255  # 1 outside → 1/65 ≈ 0.0154
    _write_masks(str(cell), gc, nucleus)
    report = qc_masks(str(cell))
    assert report.status == "AMBER"
    assert 0.05 <= report.gc_fraction <= 0.40
    assert 0.01 < report.outside_nucleus_fraction <= 0.05


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
    assert overrides == {"local_adjust": 1.10}


def test_suggest_multi_attempt_schedule_from_nondefault_config():
    """Chained RED retries step by LOCAL_ADJUST_STEP from a non-default seed."""
    report = QCReport(
        cell_id="x",
        status="RED",
        gc_fraction=0.01,
        empty_gc=False,
        outside_nucleus_fraction=0.0,
    )
    current = 1.15
    expected = [1.10, 1.05, 1.00]
    for attempt, want in enumerate(expected):
        overrides = suggest_param_overrides(
            report, attempt=attempt, current_local_adjust=current
        )
        assert overrides == {"local_adjust": want}
        current = overrides["local_adjust"]
    assert suggest_param_overrides(report, attempt=3, current_local_adjust=current) is None


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
    assert overrides == {"local_adjust": 1.13}


def test_suggest_returns_none_for_amber():
    report = QCReport(
        cell_id="x",
        status="AMBER",
        gc_fraction=0.03,
        empty_gc=False,
        outside_nucleus_fraction=0.0,
    )
    assert suggest_param_overrides(report, attempt=0, current_local_adjust=1.08) is None


def test_active_learning_shares_local_adjust_bounds():
    from ml.active_learning import _SPACE_DEFS

    la = next(d for d in _SPACE_DEFS if d[0] == "local_adjust")
    assert la[1] == LOCAL_ADJUST_MIN
    assert la[2] == LOCAL_ADJUST_MAX


def test_segment_with_qc_accepts_amber_without_retry(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    cell.mkdir(parents=True)
    paths = MaskPaths(
        gc=str(cell / "gc.tif"),
        holes=str(cell / "holes.tif"),
        hole_filled=str(cell / "hole_filled.tif"),
    )
    amber = QCReport(
        cell_id="20220304_L1/10_2",
        status="AMBER",
        gc_fraction=0.03,
        empty_gc=False,
        outside_nucleus_fraction=0.0,
        messages=["fraction amber"],
    )
    seg_calls = {"n": 0}

    def _fake_seg(*_a, **_k):
        seg_calls["n"] += 1
        return paths

    with patch("pipeline.segment_with_qc.run_gc_segment", side_effect=_fake_seg), patch(
        "pipeline.segment_with_qc.qc_masks", return_value=amber
    ), patch(
        "pipeline.segment_with_qc.suggest_param_overrides"
    ) as suggest:
        out_paths, reports = segment_with_qc(
            str(cell),
            cell_id="20220304_L1/10_2",
            config={"segmentation": {"gc_segment": {"local_adjust": 1.15}}},
        )

    assert out_paths == paths
    assert len(reports) == 1
    assert reports[0].status == "AMBER"
    assert seg_calls["n"] == 1
    suggest.assert_not_called()


def test_segment_with_qc_retries_red_then_stops(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    cell.mkdir(parents=True)
    paths = MaskPaths(
        gc=str(cell / "gc.tif"),
        holes=str(cell / "holes.tif"),
        hole_filled=str(cell / "hole_filled.tif"),
    )
    red = QCReport(
        cell_id="20220304_L1/10_2",
        status="RED",
        gc_fraction=0.01,
        empty_gc=False,
        outside_nucleus_fraction=0.0,
    )
    seen_overrides = []

    def _fake_seg(*_a, config_overrides=None, **_k):
        seen_overrides.append(config_overrides)
        return paths

    with patch("pipeline.segment_with_qc.run_gc_segment", side_effect=_fake_seg), patch(
        "pipeline.segment_with_qc.qc_masks", return_value=red
    ):
        out_paths, reports = segment_with_qc(
            str(cell),
            cell_id="20220304_L1/10_2",
            max_attempts=3,
            config={"segmentation": {"gc_segment": {"local_adjust": 1.15}}},
        )

    assert out_paths == paths
    assert len(reports) == 3
    assert seen_overrides[0] is None
    assert seen_overrides[1] == {"local_adjust": 1.10}
    assert seen_overrides[2] == {"local_adjust": 1.05}


def test_qc_multi_object_soft_flag_does_not_escalate_green(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    nucleus = np.ones((4, 10, 10), dtype=np.uint8) * 255
    gc = np.zeros_like(nucleus)
    gc[:, 0:2, 0:2] = 255  # blob A
    gc[:, 0:2, 5:7] = 255  # blob B — disconnected
    # 4*4 + 4*4 = 32 / 400 = 0.08 GREEN
    _write_masks(str(cell), gc, nucleus)
    report = qc_masks(str(cell))
    assert report.status == "GREEN"
    assert report.n_gc_objects == 2
    assert SOFT_MULTI_GC_OBJECTS in report.soft_flags
    assert any("disconnected GC objects" in m for m in report.messages)


def test_qc_mid_z_no_gc_soft_flag(tmp_path):
    """GC only off the nucleus mid-Z → soft mid_z_no_gc, traffic light unchanged."""
    cell = tmp_path / "20220304_L1" / "10_2"
    nucleus = np.ones((5, 10, 10), dtype=np.uint8) * 255
    # Mid-Z by area is z=0..4 all equal → argmax → 0; put GC only on z=4
    gc = np.zeros_like(nucleus)
    gc[4, 0:3, 0:3] = 255  # 9 / 500 = 0.018 → RED low fraction actually
    # Need GREEN fraction: 9 is too small. Use larger blob on z=4 only.
    gc[4, 0:5, 0:5] = 255  # 25 / 500 = 0.05 GREEN boundary
    _write_masks(str(cell), gc, nucleus)
    report = qc_masks(str(cell))
    assert report.mid_z_index == 0
    assert SOFT_MID_Z_NO_GC in report.soft_flags
    # Soft flag must not invent a RED on its own when fraction/outside are ok
    assert report.status in ("GREEN", "AMBER")
    assert report.status != "RED" or report.gc_fraction < 0.02  # guard
    assert report.status == "GREEN"
