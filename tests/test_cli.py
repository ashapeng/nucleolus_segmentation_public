import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from skimage import io

from pipeline.cli import main


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


def test_cli_inventory(tmp_path, capsys):
    cell = tmp_path / "20220304_L1" / "10_2"
    _write_synthetic_cell(str(cell))
    code = main(["inventory", "--root", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Included: 1" in out


def test_cli_run_no_llm(tmp_path):
    cell = tmp_path / "20220304_L1" / "10_2"
    _write_synthetic_cell(str(cell))
    runs = tmp_path / "runs"
    code = main(
        [
            "run",
            "--root",
            str(tmp_path),
            "--no-llm",
            "--max-cells",
            "1",
            "--runs-base",
            str(runs),
        ]
    )
    assert code == 0
    run_dirs = list(runs.iterdir())
    assert len(run_dirs) == 1
    report = (run_dirs[0] / "report.md").read_text(encoding="utf-8")
    assert "## QC summary" in report
    assert "## Narrative" in report


def test_cli_compare(tmp_path, capsys):
    from pipeline.run_store import save_manifest, save_qc_reports
    from pipeline.types import CellRecord, Manifest, QCReport
    import pandas as pd

    def seed(name, included_volume):
        run_dir = tmp_path / "runs" / name
        run_dir.mkdir(parents=True)
        save_manifest(
            run_dir,
            Manifest(
                root=str(tmp_path),
                resolution_3d=[0.2, 0.08, 0.08],
                cells_included=[CellRecord("20220304_L1/10_2", str(tmp_path), "L1", True)],
                cells_excluded=[],
                config_snapshot={},
            ),
        )
        save_qc_reports(
            run_dir,
            [QCReport("20220304_L1/10_2", "GREEN", 0.1, False, 0.0, [])],
        )
        pd.DataFrame(
            {"cell_id": ["20220304_L1/10_2"], "stage": ["L1"], "volume": [included_volume]}
        ).to_csv(run_dir / "shapes.csv", index=False)
        return run_dir

    run_a = seed("a", 10.0)
    run_b = seed("b", 12.0)
    out = tmp_path / "compare.md"
    code = main(["compare", "--a", str(run_a), "--b", str(run_b), "--out", str(out)])
    assert code == 0
    printed = capsys.readouterr().out.strip().splitlines()[0]
    assert printed == str(out)
    text = out.read_text(encoding="utf-8")
    assert "Cross-run comparison" in text
    assert out.with_suffix(".json").is_file()
