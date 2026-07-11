"""QC heuristics for GC masks (nucleolus_gc structure contract)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from Import_Functions import import_imgs

from pipeline.types import QCReport

_STATUS_RANK = {"GREEN": 0, "AMBER": 1, "RED": 2}

# Shared with ml.active_learning search space — keep QC retries inside the same box.
LOCAL_ADJUST_MIN = 0.90
LOCAL_ADJUST_MAX = 1.20
LOCAL_ADJUST_STEP = 0.05


def _worst(*statuses: str) -> str:
    return max(statuses, key=lambda s: _STATUS_RANK.get(s, 0))


def _clamp_local_adjust(value: float) -> float:
    return float(min(LOCAL_ADJUST_MAX, max(LOCAL_ADJUST_MIN, value)))


def qc_masks(cell_dir: str, cell_id: Optional[str] = None, gc_name: str = "gc.tif") -> QCReport:
    """Score a cell's GC mask against nucleus-relative heuristics."""
    cell_dir = os.path.abspath(cell_dir)
    if cell_id is None:
        experiment_set = os.path.basename(os.path.dirname(cell_dir))
        cell_id = f"{experiment_set}/{os.path.basename(cell_dir)}"

    gc = import_imgs(cell_dir, gc_name, is_mask=True)
    nucleus = import_imgs(cell_dir, "nuclei_mask.tif", is_mask=True)

    gc_pos = gc > 0
    nuc_pos = nucleus > 0
    messages: List[str] = []

    empty_gc = not np.any(gc_pos)
    nuc_count = int(np.count_nonzero(nuc_pos))
    gc_count = int(np.count_nonzero(gc_pos))

    if empty_gc:
        gc_fraction = 0.0
        empty_status = "RED"
        messages.append("GC mask is empty")
    else:
        gc_fraction = float(gc_count / nuc_count) if nuc_count > 0 else 0.0
        if nuc_count == 0:
            empty_status = "RED"
            messages.append("Nucleus mask is empty")
        elif 0.05 <= gc_fraction <= 0.40:
            empty_status = "GREEN"
        elif 0.02 <= gc_fraction < 0.05 or 0.40 < gc_fraction <= 0.60:
            empty_status = "AMBER"
            messages.append(f"GC/nucleus fraction {gc_fraction:.3f} outside preferred 0.05–0.40")
        else:
            empty_status = "RED"
            messages.append(f"GC/nucleus fraction {gc_fraction:.3f} out of range")

    if gc_count == 0:
        outside_fraction = 0.0
        outside_status = "RED" if empty_gc else "GREEN"
    else:
        outside = int(np.count_nonzero(gc_pos & ~nuc_pos))
        outside_fraction = float(outside / gc_count)
        if outside_fraction > 0.05:
            outside_status = "RED"
            messages.append(f"{outside_fraction:.3f} of GC pixels outside nucleus")
        elif outside_fraction > 0.01:
            outside_status = "AMBER"
            messages.append(f"{outside_fraction:.3f} of GC pixels outside nucleus")
        else:
            outside_status = "GREEN"

    status = _worst(empty_status, outside_status)
    return QCReport(
        cell_id=cell_id,
        status=status,
        gc_fraction=gc_fraction,
        empty_gc=empty_gc,
        outside_nucleus_fraction=outside_fraction,
        messages=messages,
    )


def suggest_param_overrides(
    report: QCReport,
    attempt: int,
    *,
    current_local_adjust: float,
) -> Optional[Dict[str, float]]:
    """Suggest a bounded ``local_adjust`` tweak from the value that just failed.

    Steps by ``LOCAL_ADJUST_STEP`` toward a more inclusive (lower) or stricter
    (higher) threshold, clamped to ``[LOCAL_ADJUST_MIN, LOCAL_ADJUST_MAX]``
    (same box as active learning). Returns ``None`` when retries are exhausted
    or the value is already at the relevant bound.
    """
    if attempt >= 3 or report.status == "GREEN":
        return None

    current = float(current_local_adjust)
    if report.empty_gc or report.gc_fraction < 0.05:
        # lower local_adjust → typically more inclusive Otsu local threshold
        proposed = current - LOCAL_ADJUST_STEP
    elif report.gc_fraction > 0.40:
        proposed = current + LOCAL_ADJUST_STEP
    elif report.outside_nucleus_fraction > 0.01:
        proposed = current + LOCAL_ADJUST_STEP
    else:
        return None

    nxt = _clamp_local_adjust(proposed)
    if abs(nxt - current) < 1e-9:
        return None
    return {"local_adjust": nxt}
