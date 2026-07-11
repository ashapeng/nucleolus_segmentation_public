"""QC heuristics for GC masks (nucleolus_gc structure contract)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from Import_Functions import import_imgs

from pipeline.types import QCReport

_STATUS_RANK = {"GREEN": 0, "AMBER": 1, "RED": 2}


def _worst(*statuses: str) -> str:
    return max(statuses, key=lambda s: _STATUS_RANK.get(s, 0))


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


def suggest_param_overrides(report: QCReport, attempt: int) -> Optional[Dict[str, float]]:
    """Suggest bounded ``local_adjust`` tweaks; None when retries are exhausted."""
    if attempt >= 3 or report.status == "GREEN":
        return None
    if report.empty_gc or report.gc_fraction < 0.05:
        delta = -0.05 * (attempt + 1)
        # lower local_adjust → typically more inclusive Otsu local threshold
        return {"local_adjust": max(0.9, 1.08 + delta)}
    if report.gc_fraction > 0.40:
        delta = 0.05 * (attempt + 1)
        return {"local_adjust": min(1.3, 1.08 + delta)}
    if report.outside_nucleus_fraction > 0.01:
        return {"local_adjust": min(1.3, 1.08 + 0.05 * (attempt + 1))}
    return None
