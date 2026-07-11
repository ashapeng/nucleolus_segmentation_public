"""Batch and per-cell GC segmentation wrappers."""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skimage import io

from Import_Functions import import_imgs
from config_loader import load_config
import seg_util as su

from pipeline.trust import resolve_pipeline_backend
from pipeline.types import CellRecord, Manifest, MaskPaths

logger = logging.getLogger(__name__)


def _merge_gc_overrides(config: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    if not overrides:
        return cfg
    gc = cfg.setdefault("segmentation", {}).setdefault("gc_segment", {})
    gc.update(overrides)
    return cfg


def _ensure_mask_triplet(
    gc: np.ndarray,
    holes: Optional[np.ndarray],
    hole_filled: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize classical and ML backend return shapes to three uint8 masks."""
    if holes is None:
        holes = np.zeros_like(gc)
    if hole_filled is None:
        hole_filled = gc.copy()
    return gc, holes, hole_filled


def run_gc_segment(
    cell_dir: str,
    config_overrides: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    backend: Optional[str] = None,
    allow_ml_backend: bool = False,
    trust_status: Optional[str] = None,
) -> MaskPaths:
    """Run ``gc_segment`` on a cell folder and write mask TIFFs.

    Backend selection is gated: unless ``backend`` is passed explicitly, the
    pipeline resolves ``ml.default_backend`` through
    :func:`pipeline.trust.resolve_pipeline_backend` so ML never silently
    replaces classical CV.
    """
    cell_dir = os.path.abspath(cell_dir)
    if config is None:
        config = load_config()
    config = _merge_gc_overrides(config, config_overrides)

    if backend is None:
        backend = resolve_pipeline_backend(
            config,
            allow_ml_backend=allow_ml_backend,
            trust_status=trust_status,
        )

    raw = import_imgs(cell_dir, "Composite_stack.tif", is_mask=False)
    nucleus = import_imgs(cell_dir, "nuclei_mask.tif", is_mask=True)

    if raw.ndim == 3:
        raw = np.expand_dims(raw, axis=-1)

    gc, holes, hole_filled = _ensure_mask_triplet(
        *su.gc_segment(raw, nucleus, config=config, backend=backend)
    )

    paths = MaskPaths(
        gc=os.path.join(cell_dir, "gc.tif"),
        holes=os.path.join(cell_dir, "holes.tif"),
        hole_filled=os.path.join(cell_dir, "hole_filled.tif"),
    )
    io.imsave(paths.gc, gc, check_contrast=False)
    io.imsave(paths.holes, holes, check_contrast=False)
    io.imsave(paths.hole_filled, hole_filled, check_contrast=False)
    return paths


def batch_gc_segment(
    manifest: Manifest,
    config_overrides: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    backend: Optional[str] = None,
    allow_ml_backend: bool = False,
    trust_status: Optional[str] = None,
) -> List[Tuple[CellRecord, Optional[MaskPaths], Optional[str]]]:
    """Segment all included cells; capture per-cell errors without aborting the batch."""
    if config is None:
        config = load_config()
    results: List[Tuple[CellRecord, Optional[MaskPaths], Optional[str]]] = []
    for cell in manifest.cells_included:
        try:
            paths = run_gc_segment(
                cell.path,
                config_overrides=config_overrides,
                config=config,
                backend=backend,
                allow_ml_backend=allow_ml_backend,
                trust_status=trust_status,
            )
            results.append((cell, paths, None))
        except Exception as exc:  # noqa: BLE001 — batch must continue
            logger.exception("Segmentation failed for %s", cell.cell_id)
            results.append((cell, None, str(exc)))
    return results
