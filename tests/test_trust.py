import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.report import write_run_report
from pipeline.run_store import create_run_dir
from pipeline.trust import (
    parse_trust_status,
    run_easy_adopt,
    trust_allows_ml_backend,
)
from pipeline.types import CellRecord, Manifest, TrustReport


def test_easy_adopt_skipped_when_missing(tmp_path):
    result = run_easy_adopt(str(tmp_path), tool="stardist")
    assert result["invocation"] == "SKIPPED"
    assert result["trust_status"] == "N/A"
    assert "not installed" in result["reason"]


def test_parse_trust_status_from_labeled_line():
    assert parse_trust_status("Trust status: RED\nflags: ...") == "RED"
    assert parse_trust_status("overall: AMBER") == "AMBER"
    assert parse_trust_status("verdict = GREEN") == "GREEN"


def test_parse_trust_status_from_json_blob():
    stdout = 'noise\n{"trust_status": "AMBER", "flags": []}\n'
    assert parse_trust_status(stdout) == "AMBER"
    assert parse_trust_status('{"status": "GREEN"}') == "GREEN"


def test_parse_trust_status_unknown_when_absent():
    assert parse_trust_status("ran ok, no traffic light") == "UNKNOWN"
    assert parse_trust_status("") == "UNKNOWN"
    assert parse_trust_status(None) == "UNKNOWN"


def test_run_easy_adopt_parses_completed_stdout(tmp_path):
    class _Completed:
        returncode = 0
        stdout = "Trust Report\nstatus: RED\nIoU looked fine but biology failed\n"
        stderr = ""

    with patch("pipeline.trust.shutil.which", return_value="/usr/bin/easy-adopt"), patch(
        "pipeline.trust.subprocess.run", return_value=_Completed()
    ):
        result = run_easy_adopt(str(tmp_path), tool="cellpose")

    assert result["invocation"] == "COMPLETED"
    assert result["trust_status"] == "RED"
    assert result["tool"] == "cellpose"
    assert result["returncode"] == 0


def test_run_easy_adopt_failed_invocation(tmp_path):
    class _Completed:
        returncode = 2
        stdout = ""
        stderr = "boom"

    with patch("pipeline.trust.shutil.which", return_value="/usr/bin/easy-adopt"), patch(
        "pipeline.trust.subprocess.run", return_value=_Completed()
    ):
        result = run_easy_adopt(str(tmp_path))

    assert result["invocation"] == "FAILED"
    assert result["trust_status"] == "UNKNOWN"


def test_trust_allows_ml_backend_non_red_only():
    assert trust_allows_ml_backend("GREEN") is True
    assert trust_allows_ml_backend("AMBER") is True
    assert trust_allows_ml_backend("RED") is False
    assert trust_allows_ml_backend("UNKNOWN") is False
    assert trust_allows_ml_backend("N/A") is False


def test_write_run_report_includes_trust_section(tmp_path):
    run_dir = create_run_dir(base=str(tmp_path / "runs"))
    manifest = Manifest(
        root=str(tmp_path),
        resolution_3d=[0.2, 0.08, 0.08],
        cells_included=[CellRecord("20220304_L1/10_2", str(tmp_path), "L1", True)],
        cells_excluded=[],
        config_snapshot={},
    )
    trust = [
        TrustReport(
            cell_id="20220304_L1/10_2",
            cell_dir=str(tmp_path),
            tool="stardist",
            structure="nucleolus_gc",
            invocation="COMPLETED",
            trust_status="RED",
            messages=["escalation: qc_red"],
        ).to_dict()
    ]
    path = write_run_report(
        run_dir,
        manifest,
        [],
        trust_reports=trust,
        mode="no-llm",
        goal="test",
    )
    text = path.read_text(encoding="utf-8")
    assert "## Easy-adopt trust" in text
    assert "stardist" in text
    assert "RED" in text
