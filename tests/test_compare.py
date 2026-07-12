"""Tests for cross-run comparison."""

from pathlib import Path

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from pipeline.compare import compare_runs, load_run_summary, write_compare_report
from pipeline.run_store import save_manifest, save_qc_reports
from pipeline.types import CellRecord, Manifest, QCReport


def _seed_run(base, name, *, volumes, qc_rows, intensity=None, included=1):
    run_dir = Path(base) / name
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    manifest = Manifest(
        root="/data",
        resolution_3d=[0.2, 0.08, 0.08],
        cells_included=[
            CellRecord(f"20220304_L1/{i}", "/data", "L1", True) for i in range(included)
        ],
        cells_excluded=[],
        config_snapshot={},
    )
    save_manifest(run_dir, manifest)
    save_qc_reports(run_dir, qc_rows)
    pd.DataFrame(volumes).to_csv(run_dir / "shapes.csv", index=False)
    if intensity is not None:
        pd.DataFrame(intensity).to_csv(run_dir / "intensity.csv", index=False)
    return run_dir


def test_load_run_summary_uses_final_qc_per_cell(tmp_path):
    run_dir = _seed_run(
        tmp_path / "runs",
        "run_final_qc",
        volumes={"cell_id": ["a"], "stage": ["L1"], "volume": [10.0]},
        qc_rows=[
            QCReport("cell/a", "RED", 0.0, True, 0.0, []),
            QCReport("cell/a", "GREEN", 0.1, False, 0.0, []),
            QCReport("cell/b", "AMBER", 0.35, False, 0.0, []),
        ],
        included=2,
    )
    summary = load_run_summary(run_dir)
    assert summary.final_qc_by_cell["cell/a"] == "GREEN"
    assert summary.final_qc_by_cell["cell/b"] == "AMBER"
    assert summary.final_qc_counts == {"GREEN": 1, "AMBER": 1}
    assert summary.qc_attempt_counts["RED"] == 1
    assert summary.mean_volume_by_stage["L1"] == 10.0


def test_compare_runs_detects_status_flip_and_volume_delta(tmp_path):
    runs = tmp_path / "runs"
    run_a = _seed_run(
        runs,
        "run_a",
        volumes={
            "cell_id": ["20220304_L1/0", "20220304_L2/0"],
            "stage": ["L1", "L2"],
            "volume": [10.0, 20.0],
        },
        qc_rows=[
            QCReport("20220304_L1/0", "GREEN", 0.1, False, 0.0, []),
            QCReport("20220304_L2/0", "GREEN", 0.12, False, 0.0, []),
        ],
        intensity={"cell_id": ["20220304_L1/0"], "pc": [1.0]},
        included=2,
    )
    run_b = _seed_run(
        runs,
        "run_b",
        volumes={
            "cell_id": ["20220304_L1/0", "20220304_L2/0"],
            "stage": ["L1", "L2"],
            "volume": [12.0, 30.0],
        },
        qc_rows=[
            QCReport("20220304_L1/0", "GREEN", 0.1, False, 0.0, []),
            QCReport("20220304_L2/0", "RED", 0.0, True, 0.0, []),
        ],
        intensity={"cell_id": ["20220304_L1/0"], "pc": [1.25]},
        included=2,
    )
    payload = compare_runs(run_a, run_b)
    assert payload["inventory"]["delta_included_b_minus_a"] == 0
    flips = payload["qc"]["status_flips"]
    assert flips == [{"cell_id": "20220304_L2/0", "a": "GREEN", "b": "RED"}]
    assert abs(payload["shapes"]["delta_mean_volume_b_minus_a"] - 6.0) < 1e-9
    assert abs(payload["intensity"]["delta_mean_pc_b_minus_a"] - 0.25) < 1e-9

    out = write_compare_report(run_a, run_b, out_path=tmp_path / "diff.md")
    text = out.read_text(encoding="utf-8")
    assert "## Narrative" in text
    assert "20220304_L2/0" in text
    assert "GREEN → RED" in text
    json_path = out.with_suffix(".json")
    assert json_path.is_file()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["qc"]["status_flips"][0]["b"] == "RED"
