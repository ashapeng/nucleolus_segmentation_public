"""Measurement wrappers over measure_util."""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from Import_Functions import import_imgs
from config_loader import load_config
import measure_util as mu
import seg_util as su

from pipeline.types import Manifest

logger = logging.getLogger(__name__)

DEFAULT_SHAPE_PARAMS = [
    "cell_id",
    "obj_id",
    "surface_area",
    "volume",
    "surface_to_volume_ratio",
    "sphericity",
    "aspect_ratio",
    "solidity",
]

DEFAULT_INTENSITY_PARAMS = ["cell_id", "C_bg", "C_dilute", "C_dense", "C_total", "pc"]
DEFAULT_CV_PARAMS = ["cell_id", "cv_r", "cv_g", "cv_b", "qcd_r", "qcd_g", "qcd_b"]


def measure_shapes(
    master_folder: str,
    mask_name: str = "gc.tif",
    shape_parameters: Optional[List[str]] = None,
    config=None,
    cell_dirs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Batch morphometry; returns a single DataFrame with a ``stage`` column when possible.

    If ``cell_dirs`` is provided, only those cell folders are measured (faster for filtered runs).
    """
    if shape_parameters is None:
        shape_parameters = list(DEFAULT_SHAPE_PARAMS)
    if config is None:
        config = load_config()
    resolution_3d = config["measurement"]["resolution_3d"]

    if cell_dirs is not None:
        dfs = []
        root = os.path.abspath(master_folder)
        for cell_dir in cell_dirs:
            cell_dir = os.path.abspath(cell_dir)
            experiment_set = os.path.basename(os.path.dirname(cell_dir))
            cell_name = os.path.basename(cell_dir)
            cell_id = f"{experiment_set}/{cell_name}"
            mask_path = os.path.join(cell_dir, mask_name)
            if not os.path.isfile(mask_path):
                logger.warning("Missing mask for %s", cell_id)
                continue
            mask = import_imgs(cell_dir, mask_name)
            dfs.append(
                mu.shape_discriber(
                    mask, resolution=resolution_3d, cell_id=cell_id, measured_parameters=shape_parameters
                )
            )
    else:
        dfs = mu.batch_measure_shape(
            master_folder, mask_name=mask_name, shape_parameters=shape_parameters, config=config
        )

    if not dfs:
        return pd.DataFrame(columns=shape_parameters + ["stage"])
    out = pd.concat(dfs, ignore_index=True)
    stages = []
    for cell_id in out["cell_id"]:
        try:
            stages.append(mu.extract_stage(str(cell_id).replace("\\", "/")))
        except ValueError:
            stages.append(None)
    out["stage"] = stages
    return out


def measure_intensity_cell(cell_dir: str, cell_id: str) -> pd.DataFrame:
    """Intensity / partition-coefficient metrics for one cell (MIP + ch2 bg-sub path).

    Background subtraction runs on the 2D MIP channel via ``seg_util.bg_subtraction``
    (supports ``(Y,X)``). ``pc`` still uses raw means inside GC vs nucleoplasm.
    """
    raw = import_imgs(cell_dir, "Composite_stack.tif", is_mask=False)
    if raw.ndim == 3:
        raw = np.expand_dims(raw, axis=-1)
    nucleus = import_imgs(cell_dir, "nuclei_mask.tif", is_mask=True)
    background = import_imgs(cell_dir, "background_mask.tif", is_mask=True)
    gc = import_imgs(cell_dir, "gc.tif", is_mask=True)
    hole_filled = import_imgs(cell_dir, "hole_filled.tif", is_mask=True)

    # Max-intensity projection for 2D ROIs (notebook Csat uses largest-Z instead)
    raw_mip = np.max(raw, axis=0)
    nucleus_2d = np.max(nucleus, axis=0)
    background_2d = np.max(background, axis=0) if background.ndim == 3 else background
    gc_2d = np.max(gc, axis=0)
    hole_filled_2d = np.max(hole_filled, axis=0)
    nucleoplasm = ((nucleus_2d > 0) & (hole_filled_2d == 0)).astype(np.uint8) * 255

    channel = raw_mip[..., 2] if raw_mip.ndim == 3 else raw_mip
    bg_sub = su.bg_subtraction(channel, background_2d, clip=True)

    cid = cell_id.replace("\\", "/")
    conc = mu.concentration_gc(
        channel,
        bg_sub,
        background_2d,
        nucleoplasm,
        gc_2d,
        hole_filled_2d,
        nucleus_2d,
        cell_id=cid,
        measured_parameters=list(DEFAULT_INTENSITY_PARAMS),
    )
    try:
        cv = mu.coefficient_of_variances(
            hole_filled, raw, cell_id=cid, measured_parameters=list(DEFAULT_CV_PARAMS)
        )
        return conc.merge(cv, on="cell_id", how="left")
    except Exception:  # noqa: BLE001
        logger.exception("CV measurement failed for %s", cell_id)
        return conc


def measure_intensity_batch(manifest: Manifest) -> pd.DataFrame:
    """Run intensity metrics for all included cells that have GC masks on disk."""
    frames: List[pd.DataFrame] = []
    for cell in manifest.cells_included:
        gc_path = os.path.join(cell.path, "gc.tif")
        hf_path = os.path.join(cell.path, "hole_filled.tif")
        if not (os.path.isfile(gc_path) and os.path.isfile(hf_path)):
            logger.warning("Skipping intensity for %s: masks missing", cell.cell_id)
            continue
        try:
            frames.append(measure_intensity_cell(cell.path, cell.cell_id))
        except Exception:  # noqa: BLE001
            logger.exception("Intensity failed for %s", cell.cell_id)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
