"""Experiment folder inventory and per-cell validation."""

from __future__ import annotations

import os
from typing import List, Optional

from config_loader import load_config
from measure_util import extract_stage

from pipeline.types import CellRecord, Manifest

REQUIRED_FILES = (
    "Composite_stack.tif",
    "nuclei_mask.tif",
    "background_mask.tif",
)


def validate_cell(cell_dir: str, experiment_set: Optional[str] = None) -> CellRecord:
    """Validate that a cell folder has the required input TIFFs."""
    cell_dir = os.path.abspath(cell_dir)
    cell_name = os.path.basename(cell_dir.rstrip(os.sep))
    if experiment_set is None:
        experiment_set = os.path.basename(os.path.dirname(cell_dir))
    cell_id = f"{experiment_set}/{cell_name}"

    missing = [name for name in REQUIRED_FILES if not os.path.isfile(os.path.join(cell_dir, name))]
    notes: List[str] = []
    stage: Optional[str] = None
    try:
        stage = extract_stage(cell_id)
    except ValueError as exc:
        notes.append(str(exc))

    return CellRecord(
        cell_id=cell_id,
        path=cell_dir,
        stage=stage,
        valid=len(missing) == 0,
        missing_files=missing,
        notes=notes,
    )


def inventory_experiment(root: str, config=None) -> Manifest:
    """Walk experiment_set/cell folders and build an inclusion/exclusion manifest."""
    root = os.path.abspath(root)
    if config is None:
        config = load_config()

    included: List[CellRecord] = []
    excluded: List[CellRecord] = []

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Experiment root does not exist: {root}")

    for experiment_set in sorted(os.listdir(root)):
        experiment_dir = os.path.join(root, experiment_set)
        if not os.path.isdir(experiment_dir):
            continue
        for cell_name in sorted(os.listdir(experiment_dir)):
            cell_dir = os.path.join(experiment_dir, cell_name)
            if not os.path.isdir(cell_dir):
                continue
            record = validate_cell(cell_dir, experiment_set=experiment_set)
            if record.valid:
                included.append(record)
            else:
                excluded.append(record)

    return Manifest(
        root=root,
        resolution_3d=list(config["measurement"]["resolution_3d"]),
        cells_included=included,
        cells_excluded=excluded,
        config_snapshot={
            "microscope": config.get("microscope", {}),
            "segmentation": config.get("segmentation", {}),
            "measurement": config.get("measurement", {}),
        },
    )
