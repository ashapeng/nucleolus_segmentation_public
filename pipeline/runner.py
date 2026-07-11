"""Deterministic end-to-end runner (no LLM)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config_loader import load_config
from pipeline.inventory import inventory_experiment
from pipeline.measure import measure_intensity_batch, measure_shapes
from pipeline.report import plot_and_export, write_run_report
from pipeline.run_store import append_trace, create_run_dir, save_json, save_manifest, save_qc_reports
from pipeline.segment_with_qc import segment_with_qc
from pipeline.trust import resolve_pipeline_backend, run_easy_adopt
from pipeline.types import CellRecord, Manifest, QCReport

logger = logging.getLogger(__name__)

_BACKEND_TO_EASY_ADOPT_TOOL = {
    "cellpose": "cellpose",
    "nnunet": "nnunet",
    "classical": "stardist",
}


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


def _gate_segment_backend(
    config: dict,
    manifest: Manifest,
    *,
    allow_ml_backend: bool,
    trust_payloads: list,
    notes: List[str],
) -> tuple:
    """Resolve run-level segment backend; optionally probe Easy-adopt before ML."""
    requested = ((config.get("ml") or {}).get("default_backend") or "classical")
    requested = str(requested).strip().lower() or "classical"

    if not allow_ml_backend or requested == "classical":
        backend = resolve_pipeline_backend(config, allow_ml_backend=False)
        if requested != "classical" and not allow_ml_backend:
            notes.append(
                f"ml.default_backend={requested!r} ignored; pipeline used classical "
                "(pass --allow-ml-backend with non-RED Easy-adopt trust to opt in)"
            )
        return backend, None

    if not manifest.cells_included:
        notes.append("No cells to probe for ML backend gate; using classical")
        return "classical", None

    probe = manifest.cells_included[0]
    tool = _BACKEND_TO_EASY_ADOPT_TOOL.get(requested, requested)
    probe_trust = run_easy_adopt(
        probe.path,
        tool=tool,
        cell_id=probe.cell_id,
        escalation="ml_backend_gate",
    )
    trust_payloads.append(probe_trust)
    trust_status = probe_trust.get("trust_status")
    try:
        backend = resolve_pipeline_backend(
            config,
            allow_ml_backend=True,
            trust_status=trust_status,
        )
    except ValueError as exc:
        notes.append(str(exc))
        notes.append(f"Falling back to classical after ML gate for {probe.cell_id}")
        return "classical", trust_status

    notes.append(
        f"ML backend {backend!r} allowed after Easy-adopt trust={trust_status} "
        f"on probe cell {probe.cell_id} (tool={tool})"
    )
    return backend, trust_status


def run_deterministic(
    root: str,
    *,
    stage: Optional[str] = None,
    max_cells: Optional[int] = None,
    with_intensity: bool = False,
    with_easy_adopt: bool = False,
    allow_ml_backend: bool = False,
    runs_base: str = "runs",
    goal: str = "",
) -> Path:
    """Inventory → segment+QC → measure → report. Returns the run directory path."""
    run_dir = create_run_dir(runs_base)
    append_trace(
        run_dir,
        {
            "event": "start",
            "root": root,
            "goal": goal,
            "mode": "no-llm",
            "allow_ml_backend": allow_ml_backend,
        },
    )

    config = load_config()
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
    trust_payloads: list = []

    segment_backend, gate_trust_status = _gate_segment_backend(
        config,
        manifest,
        allow_ml_backend=allow_ml_backend,
        trust_payloads=trust_payloads,
        notes=notes,
    )
    append_trace(
        run_dir,
        {
            "event": "backend_gate",
            "segment_backend": segment_backend,
            "gate_trust_status": gate_trust_status,
            "allow_ml_backend": allow_ml_backend,
        },
    )

    for cell in manifest.cells_included:
        paths, reports = segment_with_qc(
            cell.path,
            cell_id=cell.cell_id,
            config=config,
            backend=segment_backend,
        )
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
                "backend": segment_backend,
            },
        )
        if final and final.status in ("GREEN", "AMBER") and paths is not None:
            measured_cells.append(cell)
        else:
            notes.append(f"Skipped measurement for {cell.cell_id} (QC RED or segment failure)")

        if with_easy_adopt:
            trust = run_easy_adopt(
                cell.path,
                tool="stardist",
                cell_id=cell.cell_id,
            )
            trust_payloads.append(trust)
            append_trace(
                run_dir,
                {"event": "easy_adopt", "cell_id": cell.cell_id, "trust": trust},
            )

        # (c) QC-RED escalate → Easy-adopt on cellpose (informational; never swaps masks)
        if final is not None and final.status == "RED":
            esc = run_easy_adopt(
                cell.path,
                tool="cellpose",
                cell_id=cell.cell_id,
                escalation="qc_red",
            )
            trust_payloads.append(esc)
            notes.append(
                f"QC RED for {cell.cell_id}: Easy-adopt escalation tool=cellpose "
                f"invocation={esc.get('invocation')} trust={esc.get('trust_status')}"
            )
            append_trace(
                run_dir,
                {"event": "easy_adopt_escalation", "cell_id": cell.cell_id, "trust": esc},
            )

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
        trust_reports=trust_payloads,
        mode="no-llm",
        goal=goal,
        extra_notes=notes,
        segment_backend=segment_backend,
    )
    append_trace(run_dir, {"event": "done", "report": str(report_path)})
    logger.info("Run complete: %s", report_path)
    return run_dir
