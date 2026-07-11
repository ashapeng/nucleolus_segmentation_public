"""
Load (raw_image, nucleus_mask, gc_mask) triplets from the test_image/ directory tree.

Directory layout expected:
    master_folder/
        {date}_{stage}/          e.g. 20220304_L1
            {cell}/              e.g. 10_2
                Composite_stack.tif     (Z, Y, X, 3)  uint16
                nuclei_mask.tif         (Z, Y, X)      uint8
                gc.tif                  (Z, Y, X)      uint8  ← ground truth
                holes.tif               (Z, Y, X)      uint8
                hole_filled.tif         (Z, Y, X)      uint8
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from skimage import io

logger = logging.getLogger(__name__)


@dataclass
class CellSample:
    cell_id: str           # e.g. "20220304_L1/10_2"
    stage: str             # e.g. "L1"
    cell_dir: str          # absolute path to cell directory
    raw: np.ndarray        # (Z, Y, X, 3) uint16  — all channels
    nucleus_mask: np.ndarray        # (Z, Y, X) uint8
    gc_mask: np.ndarray             # (Z, Y, X) uint8  ground-truth GC
    holes_mask: Optional[np.ndarray] = field(default=None)          # (Z, Y, X)
    hole_filled_mask: Optional[np.ndarray] = field(default=None)    # (Z, Y, X)

    @property
    def lpd7(self) -> np.ndarray:
        """LPD-7 channel (index 2) — primary segmentation target."""
        return self.raw[..., 2].astype(np.float32)

    @property
    def rpoa2(self) -> np.ndarray:
        """RPOA-2 channel (index 1) — auxiliary."""
        return self.raw[..., 1].astype(np.float32)

    @property
    def dao5(self) -> np.ndarray:
        """DAO-5 channel (index 0) — auxiliary."""
        return self.raw[..., 0].astype(np.float32)

    @property
    def gc_binary(self) -> np.ndarray:
        """Ground-truth GC mask as bool."""
        return self.gc_mask > 0


def _load_tif(path: str, dtype=None) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    arr = io.imread(path)
    return arr.astype(dtype) if dtype else arr


def load_all_samples(master_folder: str) -> List[CellSample]:
    """Walk master_folder and return one CellSample per cell directory."""
    samples = []
    for exp_set in sorted(os.listdir(master_folder)):
        exp_dir = os.path.join(master_folder, exp_set)
        if not os.path.isdir(exp_dir):
            continue

        import re
        stage_match = re.search(r'\b(L[1-4])\b', exp_set)
        stage = stage_match.group(1) if stage_match else "unknown"

        for cell in sorted(os.listdir(exp_dir)):
            cell_dir = os.path.join(exp_dir, cell)
            if not os.path.isdir(cell_dir):
                continue

            raw = _load_tif(os.path.join(cell_dir, "Composite_stack.tif"))
            nucleus_mask = _load_tif(os.path.join(cell_dir, "nuclei_mask.tif"), dtype=np.uint8)
            gc_mask = _load_tif(os.path.join(cell_dir, "gc.tif"), dtype=np.uint8)

            if raw is None or nucleus_mask is None or gc_mask is None:
                logger.warning("Skipping %s — missing required files", cell_dir)
                continue

            # Ensure raw is 4-D (Z, Y, X, C)
            if raw.ndim == 3:
                raw = raw[..., np.newaxis]

            cell_id = exp_set + os.sep + cell
            samples.append(CellSample(
                cell_id=cell_id,
                stage=stage,
                cell_dir=cell_dir,
                raw=raw,
                nucleus_mask=nucleus_mask,
                gc_mask=gc_mask,
                holes_mask=_load_tif(os.path.join(cell_dir, "holes.tif"), dtype=np.uint8),
                hole_filled_mask=_load_tif(os.path.join(cell_dir, "hole_filled.tif"), dtype=np.uint8),
            ))

    logger.info("Loaded %d cell samples from %s", len(samples), master_folder)
    return samples


def loocv_splits(samples: List[CellSample]):
    """Yield (train_samples, val_sample) pairs for leave-one-out CV."""
    for i, val in enumerate(samples):
        train = [s for j, s in enumerate(samples) if j != i]
        yield train, val
