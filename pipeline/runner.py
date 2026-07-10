"""Deterministic end-to-end runner (no LLM)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from pipeline.inventory import inventory_experiment
from pipeline.measure import measure_intensity_batch, measure_shapes
from pipeline.report import plot_and_export, write_run_report
from pipeline.run_store import append_trace, create_run_dir, save_json, save_manifest, save_qc_reports
from pipeline.segment_with_qc import segment_with_qc
from pipeline.types import CellRecord, Manifest, QCReport

logger = logging.getLogger(__name__)


def _filter_manifest(
    manifest: Manifest,
    stage: Optional[str] = None,
    max_cells: Optional[int] = None,
) -> Manifest:
    cells = list(manifest.cells_included)
    if stage:
        cells = [c for c in cells if c.stage == stage]
    if max_cells is not None:
        cells = cells[: max(0, max_cells)]
    return Manifest(
        root=manifest.root,
        resolution_3d=manifest.resolution_3d,
        cells_included=cells,
        cells_excluded=manifest.cells_excluded,
        config_snapshot=manifest.config_snapshot,
    )


def run_deterministic(
    root: str,
    *,
    stage: Optional[str] = None,
    max_cells: Optional[int] = None,
    with_intensity: bool = False,
    with_easy_adopt: bool = False,
    runs_base: str = "runs",
    goal: str = "",
) -> Path:
    """Inventory → segment+QC → measure → report. Returns the run directory path."""
    run_dir = create_run_dir(runs_base)
    append_trace(run_dir, {"event": "start", "root": root, "goal": goal, "mode": "no-llm"})

    manifest = inventory_experiment(root)
    manifest = _filter_manifest(manifest, stage=stage, max_cells=max_cells)
    save_manifest(run_dir, manifest)
    append_trace(
        run_dir,
        {
            "event": "inventory",
            "included": len(manifest.cells_included),
            "excluded": len(manifest.cells_excluded),
        },
    )

    qc_reports: List[QCReport] = []
    measured_cells: List[CellRecord] = []
    notes: List[str] = []
    trust_payloads = []

    for cell in manifest.cells_included:
        paths, reports = segment_with_qc(cell.path, cell_id=cell.cell_id)
        qc_reports.extend(reports)
        final = reports[-1] if reports else None
        append_trace(
            run_dir,
            {
                "event": "segment_qc",
                "cell_id": cell.cell_id,
                "status": final.status if final else "FAILED",
                "attempts": len(reports),
                "paths": paths.to_dict() if paths else None,
            },
        )
        if final and final.status in ("GREEN", "AMBER") and paths is not None:
            measured_cells.append(cell)
        else:
            notes.append(f"Skipped measurement for {cell.cell_id} (QC RED or segment failure)")

        if with_easy_adopt:
            from pipeline.trust import run_easy_adopt

            trust = run_easy_adopt(cell.path)
            trust_payloads.append(trust)
            append_trace(run_dir, {"event": "easy_adopt", "cell_id": cell.cell_id, "trust": trust})

    save_qc_reports(run_dir, qc_reports)
    if trust_payloads:
        save_json(run_dir, "easy_adopt.json", {"results": trust_payloads})

    shapes_df = pd.DataFrame()
    intensity_df = pd.DataFrame()
    if measured_cells:
        measure_manifest = Manifest(
            root=manifest.root,
            resolution_3d=manifest.resolution_3d,
            cells_included=measured_cells,
            cells_excluded=manifest.cells_excluded,
            config_snapshot=manifest.config_snapshot,
        )
        shapes_df = measure_shapes(
            manifest.root,
            cell_dirs=[c.path for c in measured_cells],
        )
        shapes_df.to_csv(run_dir / "shapes.csv", index=False)
        plot_and_export(shapes_df, run_dir)

        if with_intensity:
            intensity_df = measure_intensity_batch(measure_manifest)
            if not intensity_df.empty:
                intensity_df.to_csv(run_dir / "intensity.csv", index=False)

    report_path = write_run_report(
        run_dir,
        manifest,
        qc_reports,
        shapes_df=shapes_df,
        intensity_df=intensity_df,
        mode="no-llm",
        goal=goal,
        extra_notes=notes,
    )
    append_trace(run_dir, {"event": "done", "report": str(report_path)})
    logger.info("Run complete: %s", report_path)
    return run_dir
