"""Tool schemas and dispatch for the nucleolus pipeline agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from pipeline.inventory import inventory_experiment
from pipeline.measure import measure_intensity_batch, measure_shapes
from pipeline.qc import qc_masks
from pipeline.report import plot_and_export, write_run_report
from pipeline.run_store import append_trace, create_run_dir, save_json, save_manifest, save_qc_reports
from pipeline.segment import run_gc_segment
from pipeline.segment_with_qc import segment_with_qc
from pipeline.trust import run_easy_adopt
from pipeline.types import Manifest, QCReport


@dataclass
class RunContext:
    root: str
    runs_base: str = "runs"
    run_dir: Optional[Path] = None
    manifest: Optional[Manifest] = None
    qc_reports: List[QCReport] = field(default_factory=list)
    trust_reports: List[Dict[str, Any]] = field(default_factory=list)
    shapes_df: Optional[pd.DataFrame] = None
    intensity_df: Optional[pd.DataFrame] = None
    notes: List[str] = field(default_factory=list)
    goal: str = ""
    allow_ml_backend: bool = False
    segment_backend: str = "classical"


TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "inventory_experiment",
            "description": "Inventory and validate cell folders under an experiment root.",
            "parameters": {
                "type": "object",
                "properties": {"root": {"type": "string"}},
                "required": ["root"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "segment_with_qc",
            "description": "Segment one cell with bounded QC retries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_dir": {"type": "string"},
                    "cell_id": {"type": "string"},
                    "max_attempts": {"type": "integer"},
                },
                "required": ["cell_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_gc_segment",
            "description": "Run classical GC segmentation once without QC retries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_dir": {"type": "string"},
                    "config_overrides": {"type": "object"},
                },
                "required": ["cell_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "qc_masks",
            "description": "QC an existing GC mask against nucleus heuristics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_dir": {"type": "string"},
                    "cell_id": {"type": "string"},
                },
                "required": ["cell_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_shapes",
            "description": "Batch morphometry over the experiment root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "master_folder": {"type": "string"},
                    "mask_name": {"type": "string"},
                },
                "required": ["master_folder"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_intensity_batch",
            "description": "Intensity metrics for cells in the current manifest that have masks.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_easy_adopt",
            "description": (
                "Optional Easy-adopt trust report (informational; do not auto-swap "
                "segmenter on RED/UNKNOWN). Parses Trust Report GREEN|AMBER|RED when present."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_dir": {"type": "string"},
                    "tool": {"type": "string"},
                    "structure": {"type": "string"},
                },
                "required": ["cell_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_run",
            "description": "Write report.md and finish the run.",
            "parameters": {
                "type": "object",
                "properties": {"mode": {"type": "string"}},
            },
        },
    },
]


def _ensure_run_dir(ctx: RunContext) -> Path:
    if ctx.run_dir is None:
        ctx.run_dir = create_run_dir(ctx.runs_base)
        append_trace(ctx.run_dir, {"event": "start", "root": ctx.root, "goal": ctx.goal})
    return ctx.run_dir


def execute_tool(name: str, arguments: Dict[str, Any], context: RunContext) -> Dict[str, Any]:
    """Dispatch a tool call and return a JSON-serializable result."""
    run_dir = _ensure_run_dir(context)
    append_trace(run_dir, {"event": "tool_call", "name": name, "arguments": arguments})

    if name == "inventory_experiment":
        root = arguments.get("root", context.root)
        context.manifest = inventory_experiment(root)
        save_manifest(run_dir, context.manifest)
        result = {
            "included": len(context.manifest.cells_included),
            "excluded": len(context.manifest.cells_excluded),
            "cells": [c.to_dict() for c in context.manifest.cells_included],
        }
    elif name == "segment_with_qc":
        paths, reports = segment_with_qc(
            arguments["cell_dir"],
            cell_id=arguments.get("cell_id"),
            max_attempts=int(arguments.get("max_attempts", 3)),
        )
        context.qc_reports.extend(reports)
        result = {
            "paths": paths.to_dict() if paths else None,
            "qc": [r.to_dict() for r in reports],
            "final_status": reports[-1].status if reports else "FAILED",
        }
    elif name == "run_gc_segment":
        paths = run_gc_segment(arguments["cell_dir"], config_overrides=arguments.get("config_overrides"))
        result = paths.to_dict()
    elif name == "qc_masks":
        report = qc_masks(arguments["cell_dir"], cell_id=arguments.get("cell_id"))
        context.qc_reports.append(report)
        result = report.to_dict()
    elif name == "measure_shapes":
        folder = arguments.get("master_folder", context.root)
        df = measure_shapes(folder, mask_name=arguments.get("mask_name", "gc.tif"))
        context.shapes_df = df
        df.to_csv(run_dir / "shapes.csv", index=False)
        plot_and_export(df, run_dir)
        result = {"rows": len(df), "columns": list(df.columns)}
    elif name == "measure_intensity_batch":
        if context.manifest is None:
            raise ValueError("Call inventory_experiment before measure_intensity_batch")
        df = measure_intensity_batch(context.manifest)
        context.intensity_df = df
        if not df.empty:
            df.to_csv(run_dir / "intensity.csv", index=False)
        result = {"rows": len(df)}
    elif name == "run_easy_adopt":
        trust = run_easy_adopt(
            arguments["cell_dir"],
            tool=arguments.get("tool", "stardist"),
            structure=arguments.get("structure", "nucleolus_gc"),
            cell_id=arguments.get("cell_id"),
            escalation=arguments.get("escalation"),
        )
        context.trust_reports.append(trust)
        save_json(run_dir, "easy_adopt_last.json", trust)
        result = trust
    elif name == "finalize_run":
        if context.manifest is None:
            context.manifest = inventory_experiment(context.root)
        save_qc_reports(run_dir, context.qc_reports)
        if context.trust_reports:
            save_json(run_dir, "easy_adopt.json", {"results": context.trust_reports})
        report = write_run_report(
            run_dir,
            context.manifest,
            context.qc_reports,
            shapes_df=context.shapes_df,
            intensity_df=context.intensity_df,
            trust_reports=context.trust_reports,
            mode=arguments.get("mode", "llm"),
            goal=context.goal,
            extra_notes=context.notes,
            segment_backend=context.segment_backend,
        )
        result = {"report": str(report), "run_dir": str(run_dir)}
    else:
        raise ValueError(f"Unknown tool: {name}")

    append_trace(run_dir, {"event": "tool_result", "name": name, "result": result})
    return result
