import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from pipeline.trust import resolve_pipeline_backend


def test_resolve_pipeline_backend_classical_default():
    assert resolve_pipeline_backend({}) == "classical"
    assert resolve_pipeline_backend({"ml": {"default_backend": "classical"}}) == "classical"


def test_resolve_pipeline_backend_ignores_ml_without_allow(caplog):
    cfg = {"ml": {"default_backend": "cellpose"}}
    assert resolve_pipeline_backend(cfg, allow_ml_backend=False) == "classical"
    assert "ignored" in caplog.text.lower() or "classical" in caplog.text.lower()


def test_resolve_pipeline_backend_allow_requires_non_red():
    cfg = {"ml": {"default_backend": "nnunet"}}
    with pytest.raises(ValueError, match="Refusing ML backend"):
        resolve_pipeline_backend(cfg, allow_ml_backend=True, trust_status="RED")
    with pytest.raises(ValueError, match="Refusing ML backend"):
        resolve_pipeline_backend(cfg, allow_ml_backend=True, trust_status="UNKNOWN")
    assert (
        resolve_pipeline_backend(cfg, allow_ml_backend=True, trust_status="GREEN")
        == "nnunet"
    )
    assert (
        resolve_pipeline_backend(
            {"ml": {"default_backend": "cellpose"}},
            allow_ml_backend=True,
            trust_status="AMBER",
        )
        == "cellpose"
    )
