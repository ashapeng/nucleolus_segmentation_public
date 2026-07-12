import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.report import write_run_report
from pipeline.run_store import create_run_dir
from pipeline.types import CellRecord, Manifest, QCReport
import pandas as pd


def test_write_run_report_contains_qc_summary(tmp_path):
    run_dir = create_run_dir(base=str(tmp_path / "runs"))
    manifest = Manifest(
        root=str(tmp_path),
        resolution_3d=[0.2, 0.08, 0.08],
        cells_included=[
            CellRecord("20220304_L1/10_2", str(tmp_path), "L1", True),
        ],
        cells_excluded=[],
        config_snapshot={},
    )
    qc = [
        QCReport("20220304_L1/10_2", "GREEN", 0.1, False, 0.0, []),
    ]
    path = write_run_report(run_dir, manifest, qc, mode="no-llm", goal="test")
    text = path.read_text(encoding="utf-8")
    assert "## QC summary" in text
    assert "GREEN: 1" in text


def test_write_run_report_includes_narrative_citing_numbers(tmp_path):
    run_dir = create_run_dir(base=str(tmp_path / "runs"))
    manifest = Manifest(
        root=str(tmp_path),
        resolution_3d=[0.2, 0.08, 0.08],
        cells_included=[
            CellRecord("20220304_L1/10_2", str(tmp_path), "L1", True),
            CellRecord("20220304_L2/10_2", str(tmp_path), "L2", True),
        ],
        cells_excluded=[],
        config_snapshot={},
    )
    qc = [
        QCReport("20220304_L1/10_2", "GREEN", 0.1, False, 0.0, []),
        QCReport("20220304_L2/10_2", "AMBER", 0.35, False, 0.0, []),
    ]
    shapes = pd.DataFrame(
        {
            "cell_id": ["20220304_L1/10_2", "20220304_L2/10_2"],
            "stage": ["L1", "L2"],
            "volume": [5.0, 15.0],
        }
    )
    path = write_run_report(
        run_dir,
        manifest,
        qc,
        shapes_df=shapes,
        mode="no-llm",
        goal="test narrative",
    )
    text = path.read_text(encoding="utf-8")
    assert "## Narrative" in text
    assert "2 included" in text
    assert "1 GREEN" in text
    assert "1 AMBER" in text
    assert "increasing" in text
    assert "5.000" in text
