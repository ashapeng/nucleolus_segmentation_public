"""Cross-run comparison of pipeline artifacts under ``runs/<id>/``."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from pipeline.narrative import STAGE_ORDER, stage_volume_means


@dataclass
class RunSummary:
    """Compact stats loaded from an existing run directory."""

    run_id: str
    path: str
    root: Optional[str] = None
    n_included: int = 0
    n_excluded: int = 0
    qc_attempt_counts: Dict[str, int] = field(default_factory=dict)
    final_qc_by_cell: Dict[str, str] = field(default_factory=dict)
    final_qc_counts: Dict[str, int] = field(default_factory=dict)
    mean_volume_by_stage: Dict[str, float] = field(default_factory=dict)
    mean_volume_overall: Optional[float] = None
    n_shape_rows: int = 0
    mean_pc: Optional[float] = None
    n_intensity_rows: int = 0
    missing_artifacts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _final_qc_by_cell(qc_path: Path) -> Tuple[Dict[str, str], Counter]:
    """Last QC status per cell_id; also return attempt-level status counts."""
    attempt_counts: Counter = Counter()
    final: Dict[str, str] = {}
    if not qc_path.is_file():
        return final, attempt_counts
    with qc_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            status = str(row.get("status", "UNKNOWN"))
            attempt_counts[status] += 1
            cell_id = row.get("cell_id")
            if cell_id:
                final[str(cell_id)] = status
    return final, attempt_counts


def load_run_summary(run_dir: Path | str) -> RunSummary:
    """Load inventory/QC/shape/intensity summaries from a run directory."""
    run_dir = Path(run_dir)
    summary = RunSummary(run_id=run_dir.name, path=str(run_dir.resolve()))

    manifest = _read_json(run_dir / "manifest.json")
    if manifest is None:
        summary.missing_artifacts.append("manifest.json")
    else:
        summary.root = manifest.get("root")
        summary.n_included = len(manifest.get("cells_included") or [])
        summary.n_excluded = len(manifest.get("cells_excluded") or [])

    qc_path = run_dir / "qc.jsonl"
    if not qc_path.is_file():
        summary.missing_artifacts.append("qc.jsonl")
    else:
        final, attempts = _final_qc_by_cell(qc_path)
        summary.final_qc_by_cell = final
        summary.qc_attempt_counts = dict(attempts)
        summary.final_qc_counts = dict(Counter(final.values()))

    shapes_path = run_dir / "shapes.csv"
    if shapes_path.is_file():
        shapes_df = pd.read_csv(shapes_path)
        summary.n_shape_rows = len(shapes_df)
        if not shapes_df.empty and "volume" in shapes_df.columns:
            summary.mean_volume_overall = float(shapes_df["volume"].mean())
            summary.mean_volume_by_stage = {
                stage: mean for stage, mean in stage_volume_means(shapes_df)
            }
    else:
        summary.missing_artifacts.append("shapes.csv")

    intensity_path = run_dir / "intensity.csv"
    if intensity_path.is_file():
        intensity_df = pd.read_csv(intensity_path)
        summary.n_intensity_rows = len(intensity_df)
        if not intensity_df.empty and "pc" in intensity_df.columns:
            summary.mean_pc = float(intensity_df["pc"].mean())
    else:
        summary.missing_artifacts.append("intensity.csv")

    return summary


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return b - a


def compare_runs(run_a: Path | str, run_b: Path | str) -> Dict[str, Any]:
    """Compare two run directories; returns a JSON-serializable diff payload."""
    a = load_run_summary(run_a)
    b = load_run_summary(run_b)

    cells_a = set(a.final_qc_by_cell)
    cells_b = set(b.final_qc_by_cell)
    shared = sorted(cells_a & cells_b)
    status_flips = []
    for cell_id in shared:
        sa, sb = a.final_qc_by_cell[cell_id], b.final_qc_by_cell[cell_id]
        if sa != sb:
            status_flips.append({"cell_id": cell_id, "a": sa, "b": sb})

    stage_deltas = []
    stages = [s for s in STAGE_ORDER if s in a.mean_volume_by_stage or s in b.mean_volume_by_stage]
    extra = sorted(
        set(a.mean_volume_by_stage) | set(b.mean_volume_by_stage) - set(STAGE_ORDER)
    )
    for stage in list(stages) + extra:
        va = a.mean_volume_by_stage.get(stage)
        vb = b.mean_volume_by_stage.get(stage)
        stage_deltas.append(
            {
                "stage": stage,
                "mean_volume_a": va,
                "mean_volume_b": vb,
                "delta_b_minus_a": _delta(va, vb),
            }
        )

    return {
        "run_a": a.to_dict(),
        "run_b": b.to_dict(),
        "inventory": {
            "n_included_a": a.n_included,
            "n_included_b": b.n_included,
            "delta_included_b_minus_a": b.n_included - a.n_included,
            "n_excluded_a": a.n_excluded,
            "n_excluded_b": b.n_excluded,
            "delta_excluded_b_minus_a": b.n_excluded - a.n_excluded,
        },
        "qc": {
            "final_counts_a": a.final_qc_counts,
            "final_counts_b": b.final_qc_counts,
            "cells_only_in_a": sorted(cells_a - cells_b),
            "cells_only_in_b": sorted(cells_b - cells_a),
            "status_flips": status_flips,
        },
        "shapes": {
            "n_rows_a": a.n_shape_rows,
            "n_rows_b": b.n_shape_rows,
            "mean_volume_a": a.mean_volume_overall,
            "mean_volume_b": b.mean_volume_overall,
            "delta_mean_volume_b_minus_a": _delta(a.mean_volume_overall, b.mean_volume_overall),
            "stage_deltas": stage_deltas,
        },
        "intensity": {
            "n_rows_a": a.n_intensity_rows,
            "n_rows_b": b.n_intensity_rows,
            "mean_pc_a": a.mean_pc,
            "mean_pc_b": b.mean_pc,
            "delta_mean_pc_b_minus_a": _delta(a.mean_pc, b.mean_pc),
        },
    }


def _fmt_opt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def write_compare_report(
    run_a: Path | str,
    run_b: Path | str,
    out_path: Optional[Path | str] = None,
) -> Path:
    """Write a markdown cross-run comparison report; also saves ``compare.json`` beside it."""
    run_a = Path(run_a)
    run_b = Path(run_b)
    payload = compare_runs(run_a, run_b)

    if out_path is None:
        out_path = run_b / f"compare_vs_{run_a.name}.md"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    inv = payload["inventory"]
    qc = payload["qc"]
    shapes = payload["shapes"]
    intensity = payload["intensity"]

    lines = [
        f"# Cross-run comparison — `{run_a.name}` vs `{run_b.name}`",
        "",
        f"- **Run A:** `{run_a}`",
        f"- **Run B:** `{run_b}`",
        f"- **Machine-readable:** `{json_path.name}`",
        "",
        "## Inventory",
        "",
        f"- Included cells: A={inv['n_included_a']}, B={inv['n_included_b']} "
        f"(Δ B−A = {inv['delta_included_b_minus_a']})",
        f"- Excluded cells: A={inv['n_excluded_a']}, B={inv['n_excluded_b']} "
        f"(Δ B−A = {inv['delta_excluded_b_minus_a']})",
        "",
        "## QC (final status per cell)",
        "",
        f"- Final counts A: {qc['final_counts_a'] or '{}'}",
        f"- Final counts B: {qc['final_counts_b'] or '{}'}",
    ]
    if qc["cells_only_in_a"]:
        lines.append("- Cells only in A: " + ", ".join(f"`{c}`" for c in qc["cells_only_in_a"]))
    if qc["cells_only_in_b"]:
        lines.append("- Cells only in B: " + ", ".join(f"`{c}`" for c in qc["cells_only_in_b"]))
    if qc["status_flips"]:
        lines.append("- Status flips (A → B):")
        for flip in qc["status_flips"]:
            lines.append(f"  - `{flip['cell_id']}`: {flip['a']} → {flip['b']}")
    else:
        lines.append("- No shared-cell QC status flips.")
    lines.append("")

    lines.extend(
        [
            "## Shapes",
            "",
            f"- Shape rows: A={shapes['n_rows_a']}, B={shapes['n_rows_b']}",
            f"- Overall mean volume: A={_fmt_opt(shapes['mean_volume_a'])}, "
            f"B={_fmt_opt(shapes['mean_volume_b'])} "
            f"(Δ B−A = {_fmt_opt(shapes['delta_mean_volume_b_minus_a'])})",
        ]
    )
    if shapes["stage_deltas"]:
        lines.append("- Mean volume by stage:")
        for row in shapes["stage_deltas"]:
            lines.append(
                f"  - {row['stage']}: A={_fmt_opt(row['mean_volume_a'])}, "
                f"B={_fmt_opt(row['mean_volume_b'])}, "
                f"Δ={_fmt_opt(row['delta_b_minus_a'])}"
            )
    lines.append("")

    lines.extend(
        [
            "## Intensity",
            "",
            f"- Intensity rows: A={intensity['n_rows_a']}, B={intensity['n_rows_b']}",
            f"- Mean pc: A={_fmt_opt(intensity['mean_pc_a'])}, B={_fmt_opt(intensity['mean_pc_b'])} "
            f"(Δ B−A = {_fmt_opt(intensity['delta_mean_pc_b_minus_a'])})",
            "",
            "## Narrative",
            "",
        ]
    )

    # Short grounded comparison narrative
    narrative_bits = [
        f"Compared run `{run_a.name}` to `{run_b.name}`: included cells changed by "
        f"{inv['delta_included_b_minus_a']} (A {inv['n_included_a']} → B {inv['n_included_b']})."
    ]
    if shapes["delta_mean_volume_b_minus_a"] is not None:
        narrative_bits.append(
            f"Overall mean GC volume changed by {_fmt_opt(shapes['delta_mean_volume_b_minus_a'])} "
            f"(A {_fmt_opt(shapes['mean_volume_a'])} → B {_fmt_opt(shapes['mean_volume_b'])})."
        )
    if qc["status_flips"]:
        narrative_bits.append(
            f"{len(qc['status_flips'])} shared cell(s) changed final QC status between runs."
        )
    else:
        narrative_bits.append("Shared cells kept the same final QC status between runs.")
    if intensity["delta_mean_pc_b_minus_a"] is not None:
        narrative_bits.append(
            f"Mean partition coefficient (pc) changed by "
            f"{_fmt_opt(intensity['delta_mean_pc_b_minus_a'])} "
            f"(A {_fmt_opt(intensity['mean_pc_a'])} → B {_fmt_opt(intensity['mean_pc_b'])})."
        )
    lines.append(" ".join(narrative_bits))
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
