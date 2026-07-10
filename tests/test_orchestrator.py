import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from skimage import io

from agent.orchestrator import run_goal


def _sphere_mask(shape=(8, 24, 24), radius=7):
    mask = np.zeros(shape, dtype=np.uint8)
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 255
    return mask


def _write_synthetic_cell(cell_dir: str):
    os.makedirs(cell_dir, exist_ok=True)
    nucleus = _sphere_mask()
    raw = np.zeros(nucleus.shape + (3,), dtype=np.float32)
    gc_core = _sphere_mask(radius=4)
    for c in range(3):
        raw[..., c] = nucleus.astype(np.float32) * 10
    raw[..., 2] = np.where(gc_core > 0, 500.0, raw[..., 2])
    io.imsave(os.path.join(cell_dir, "Composite_stack.tif"), raw, check_contrast=False)
    io.imsave(os.path.join(cell_dir, "nuclei_mask.tif"), nucleus, check_contrast=False)
    io.imsave(os.path.join(cell_dir, "background_mask.tif"), np.zeros_like(nucleus), check_contrast=False)


def test_orchestrator_deterministic(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    _write_synthetic_cell(str(cell))
    run_dir = run_goal(
        goal="Segment and report",
        root=str(tmp_path),
        no_llm=True,
        max_cells=1,
        runs_base=str(tmp_path / "runs"),
    )
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "manifest.json").is_file()
