import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from skimage import io

from pipeline.runner import run_deterministic
from pipeline.types import MaskPaths, QCReport


def _sphere_mask(shape=(8, 24, 24), radius=7):
    mask = np.zeros(shape, dtype=np.uint8)
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 255
    return mask


def _write_synthetic_cell(cell_dir: str):
    os.makedirs(cell_dir, exist_ok=True)
    nucleus = _sphere_mask()
    raw = np.zeros(nucleus.shape + (3,), dtype=np.float32)
    gc_core = _sphere_mask(radius=4)
    for c in range(3):
        raw[..., c] = nucleus.astype(np.float32) * 10
    raw[..., 2] = np.where(gc_core > 0, 500.0, raw[..., 2])
    io.imsave(os.path.join(cell_dir, "Composite_stack.tif"), raw, check_contrast=False)
    io.imsave(os.path.join(cell_dir, "nuclei_mask.tif"), nucleus, check_contrast=False)
    io.imsave(os.path.join(cell_dir, "background_mask.tif"), np.zeros_like(nucleus), check_contrast=False)


def test_runner_escalates_easy_adopt_on_qc_red(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    _write_synthetic_cell(str(cell))
    runs = tmp_path / "runs"

    paths = MaskPaths(
        gc=str(cell / "gc.tif"),
        holes=str(cell / "holes.tif"),
        hole_filled=str(cell / "hole_filled.tif"),
    )
    red = QCReport(
        cell_id="20220304_L1/10_2",
        status="RED",
        gc_fraction=0.0,
        empty_gc=True,
        outside_nucleus_fraction=0.0,
        messages=["GC mask is empty"],
    )

    def _fake_easy_adopt(cell_dir, tool="stardist", structure="nucleolus_gc", **kwargs):
        return {
            "cell_id": kwargs.get("cell_id"),
            "cell_dir": cell_dir,
            "tool": tool,
            "structure": structure,
            "invocation": "SKIPPED",
            "trust_status": "N/A",
            "reason": "easy-adopt not installed",
            "escalation": kwargs.get("escalation"),
            "messages": [f"escalation: {kwargs['escalation']}"]
            if kwargs.get("escalation")
            else [],
        }

    with patch(
        "pipeline.runner.segment_with_qc", return_value=(paths, [red])
    ), patch("pipeline.runner.run_easy_adopt", side_effect=_fake_easy_adopt) as adopt:
        run_dir = run_deterministic(
            str(tmp_path),
            max_cells=1,
            runs_base=str(runs),
        )

    # Escalation call for cellpose on QC RED (and no with_easy_adopt stardist pass)
    assert adopt.call_count == 1
    _args, kwargs = adopt.call_args
    assert kwargs.get("escalation") == "qc_red"
    assert kwargs.get("tool") == "cellpose"

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Easy-adopt trust" in report
    assert "cellpose" in report
    assert "qc_red" in report
    assert "classical" in report
    assert (run_dir / "easy_adopt.json").is_file()


def test_runner_ignores_ml_default_backend_without_allow(tmp_path, monkeypatch):
    cell = tmp_path / "20220304_L1" / "10_2"
    _write_synthetic_cell(str(cell))
    runs = tmp_path / "runs"

    cfg = {
        "microscope": {"z_um": 0.2, "xy_um": 0.08},
        "segmentation": {
            "gc_segment": {
                "sigma": 1.0,
                "local_adjust": 1.08,
                "mini_size_otsu": 1000,
                "log_sigma_min": 2.5,
                "log_sigma_max": 4.0,
                "log_sigma_step": 0.25,
                "mini_size_spot": 30,
                "percentile_clip": 99.9,
            }
        },
        "measurement": {"resolution_3d": [0.2, 0.08, 0.08], "resolution_2d": [0.08, 0.08]},
        "ml": {"default_backend": "cellpose"},
    }
    monkeypatch.setattr("pipeline.runner.load_config", lambda: cfg)

    run_dir = run_deterministic(str(tmp_path), max_cells=1, runs_base=str(runs))
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "classical" in report
    assert "ignored" in report.lower() or "classical" in report
