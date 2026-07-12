"""QC heuristics for GC masks (nucleolus_gc structure contract)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from skimage.measure import label

from Import_Functions import import_imgs

from pipeline.types import QCReport

_STATUS_RANK = {"GREEN": 0, "AMBER": 1, "RED": 2}

# Shared with ml.active_learning search space — keep QC retries inside the same box.
LOCAL_ADJUST_MIN = 0.90
LOCAL_ADJUST_MAX = 1.20
LOCAL_ADJUST_STEP = 0.05

# Soft-flag names (informational; do not change traffic-light status by themselves).
SOFT_MULTI_GC_OBJECTS = "multi_gc_objects"
SOFT_MID_Z_NO_GC = "mid_z_no_gc"


def _worst(*statuses: str) -> str:
    return max(statuses, key=lambda s: _STATUS_RANK.get(s, 0))


def _clamp_local_adjust(value: float) -> float:
    """Clamp and round to cents so ±STEP chains stay exact in float."""
    return round(float(min(LOCAL_ADJUST_MAX, max(LOCAL_ADJUST_MIN, value))), 2)


def _count_gc_objects(gc_pos: np.ndarray) -> int:
    """Count 3D connected components in the GC foreground (26-connectivity)."""
    if not np.any(gc_pos):
        return 0
    labeled = label(gc_pos.astype(np.uint8), connectivity=3)
    return int(labeled.max())


def _mid_z_index(nuc_pos: np.ndarray) -> Optional[int]:
    """Nucleus-area mid-Z: Z plane with the most nucleus voxels (tie → lower index)."""
    if nuc_pos.ndim != 3 or not np.any(nuc_pos):
        return None
    per_z = np.count_nonzero(nuc_pos, axis=(1, 2))
    return int(np.argmax(per_z))


def _soft_topology_flags(
    gc_pos: np.ndarray,
    nuc_pos: np.ndarray,
    n_gc_objects: int,
    mid_z_index: Optional[int],
) -> Tuple[List[str], List[str]]:
    """Assistive topology / mid-Z notes — never escalate RED/AMBER on their own."""
    flags: List[str] = []
    messages: List[str] = []

    if n_gc_objects > 1:
        flags.append(SOFT_MULTI_GC_OBJECTS)
        messages.append(
            f"{n_gc_objects} disconnected GC objects (topology note; not a traffic-light driver)"
        )

    if mid_z_index is not None and np.any(gc_pos):
        mid_gc = bool(np.any(gc_pos[mid_z_index] & nuc_pos[mid_z_index]))
        if not mid_gc:
            flags.append(SOFT_MID_Z_NO_GC)
            messages.append(
                f"mid-Z slice z={mid_z_index} has no GC inside nucleus (assistive visual review)"
            )

    return flags, messages


def qc_masks(cell_dir: str, cell_id: Optional[str] = None, gc_name: str = "gc.tif") -> QCReport:
    """Score a cell's GC mask against nucleus-relative heuristics.

    Soft signals (``n_gc_objects``, ``mid_z_index``, ``soft_flags``) are recorded for
    review / future VLM assist; they do not change GREEN/AMBER/RED on their own.
    """
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
    n_gc_objects = _count_gc_objects(gc_pos)
    mid_z_index = _mid_z_index(nuc_pos)

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

    soft_flags, soft_messages = _soft_topology_flags(
        gc_pos, nuc_pos, n_gc_objects, mid_z_index
    )
    messages.extend(soft_messages)

    status = _worst(empty_status, outside_status)
    return QCReport(
        cell_id=cell_id,
        status=status,
        gc_fraction=gc_fraction,
        empty_gc=empty_gc,
        outside_nucleus_fraction=outside_fraction,
        messages=messages,
        n_gc_objects=n_gc_objects,
        mid_z_index=mid_z_index,
        soft_flags=soft_flags,
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
    # AMBER is accepted without retry (same policy as segment_with_qc).
    if attempt >= 3 or report.status in ("GREEN", "AMBER"):
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
