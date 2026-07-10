"""Run directory helpers for reproducible pipeline artifacts."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from pipeline.types import Manifest, QCReport


def create_run_dir(base: str = "runs") -> Path:
    """Create ``runs/<UTC timestamp>/`` and standard subfolders."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(base) / run_id
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_manifest(run_dir: Path, manifest: Manifest) -> Path:
    path = Path(run_dir) / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path


def save_qc_reports(run_dir: Path, reports: Iterable[QCReport]) -> Path:
    path = Path(run_dir) / "qc.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for report in reports:
            fh.write(json.dumps(report.to_dict()) + "\n")
    return path


def append_trace(run_dir: Path, event: Dict[str, Any]) -> Path:
    path = Path(run_dir) / "trace.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")
    return path


def save_json(run_dir: Path, name: str, payload: Dict[str, Any]) -> Path:
    path = Path(run_dir) / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
