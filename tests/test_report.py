import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.report import write_run_report
from pipeline.run_store import create_run_dir
from pipeline.types import CellRecord, Manifest, QCReport


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
