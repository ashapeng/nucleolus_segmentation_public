import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.trust import run_easy_adopt


def test_easy_adopt_skipped_when_missing(tmp_path):
    result = run_easy_adopt(str(tmp_path), tool="stardist")
    assert result["status"] == "SKIPPED"
    assert "not installed" in result["reason"]
