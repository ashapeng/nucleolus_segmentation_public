"""Tests for deterministic narrative paragraphs."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from pipeline.narrative import (
    build_narrative,
    describe_stage_volume_trend,
    figure_caption,
    stage_volume_means,
)
from pipeline.types import CellRecord, Manifest, QCReport


def _manifest(tmp_path, n_included=1, n_excluded=0):
    included = [
        CellRecord(f"20220304_L1/{i}", str(tmp_path), "L1", True) for i in range(n_included)
    ]
    excluded = [
        CellRecord(f"bad/{i}", str(tmp_path), None, False, missing_files=["gc.tif"])
        for i in range(n_excluded)
    ]
    return Manifest(
        root=str(tmp_path),
        resolution_3d=[0.2, 0.08, 0.08],
        cells_included=included,
        cells_excluded=excluded,
        config_snapshot={},
    )


def test_stage_volume_means_orders_larval_stages():
    df = pd.DataFrame(
        {
            "stage": ["L3", "L1", "L2", "L1"],
            "volume": [30.0, 10.0, 20.0, 12.0],
        }
    )
    assert stage_volume_means(df) == [("L1", 11.0), ("L2", 20.0), ("L3", 30.0)]


def test_describe_stage_volume_trend_increasing():
    trend = describe_stage_volume_trend([("L1", 10.0), ("L2", 20.0), ("L4", 40.0)])
    assert trend is not None
    assert "increasing" in trend
    assert "10.000" in trend
    assert "40.000" in trend


def test_build_narrative_cites_qc_and_csv_numbers(tmp_path):
    manifest = _manifest(tmp_path, n_included=2, n_excluded=1)
    qc = [
        QCReport("20220304_L1/0", "GREEN", 0.1, False, 0.0, []),
        QCReport("20220304_L1/1", "RED", 0.0, True, 0.0, ["empty"]),
    ]
    shapes = pd.DataFrame(
        {
            "cell_id": ["20220304_L1/0", "20220304_L2/0"],
            "stage": ["L1", "L2"],
            "volume": [10.0, 20.0],
        }
    )
    intensity = pd.DataFrame({"cell_id": ["20220304_L1/0"], "pc": [1.5]})
    paragraphs = build_narrative(manifest, qc, shapes, intensity)
    text = " ".join(paragraphs)
    assert "2 included" in text
    assert "1 excluded" in text
    assert "1 GREEN" in text
    assert "1 RED" in text
    assert "`20220304_L1/1`" in text
    assert "10.000" in text
    assert "20.000" in text
    assert "increasing" in text
    assert "1.500" in text


def test_figure_caption_mentions_measurement():
    assert "volume" in figure_caption("volume")
    assert "larval stage" in figure_caption("volume")
