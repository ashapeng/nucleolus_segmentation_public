"""Segment with bounded QC retries."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from pipeline.qc import qc_masks, suggest_param_overrides
from pipeline.segment import run_gc_segment
from pipeline.types import MaskPaths, QCReport

logger = logging.getLogger(__name__)


def segment_with_qc(
    cell_dir: str,
    cell_id: Optional[str] = None,
    max_attempts: int = 3,
    config: Optional[Dict[str, Any]] = None,
    *,
    backend: Optional[str] = None,
    allow_ml_backend: bool = False,
    trust_status: Optional[str] = None,
) -> Tuple[Optional[MaskPaths], List[QCReport]]:
    """Run GC segmentation, QC, and bounded ``local_adjust`` retries.

    Returns the last successful mask paths (even if final QC is RED) and all QC reports.
    If segmentation itself fails, returns ``(None, reports)``.

    ``backend`` / ``allow_ml_backend`` / ``trust_status`` are forwarded to
    :func:`pipeline.segment.run_gc_segment` (classical by default; ML only when gated).
    """
    reports: List[QCReport] = []
    overrides: Optional[Dict[str, Any]] = None
    paths: Optional[MaskPaths] = None

    for attempt in range(max_attempts):
        try:
            paths = run_gc_segment(
                cell_dir,
                config_overrides=overrides,
                config=config,
                backend=backend,
                allow_ml_backend=allow_ml_backend,
                trust_status=trust_status,
            )
        except Exception:  # noqa: BLE001
            logger.exception("segment_with_qc failed on attempt %s for %s", attempt, cell_dir)
            break

        report = qc_masks(cell_dir, cell_id=cell_id)
        reports.append(report)
        if report.status in ("GREEN", "AMBER"):
            return paths, reports

        overrides = suggest_param_overrides(report, attempt=attempt)
        if overrides is None:
            break

    return paths, reports
