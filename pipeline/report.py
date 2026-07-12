"""Deterministic run reports and CSV/figure export helpers."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

import pandas as pd

from pipeline.narrative import build_narrative, figure_caption
from pipeline.trust import summarize_trust_reports
from pipeline.types import Manifest, QCReport

logger = logging.getLogger(__name__)


def _boxplot_with_stage_labels(ax, data, stages):
    """Matplotlib ≥3.9 uses tick_labels; older releases used labels."""
    try:
        return ax.boxplot(data, tick_labels=stages)
    except TypeError:
        return ax.boxplot(data, labels=stages)


def plot_and_export(df: pd.DataFrame, out_dir: Path, measurement: str = "volume") -> List[Path]:
    """Save CSV always; save a boxplot when stage + measurement columns exist."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: List[Path] = []
    csv_path = out_dir / "shapes.csv"
    df.to_csv(csv_path, index=False)
    artifacts.append(csv_path)

    if not df.empty and "stage" in df.columns and measurement in df.columns:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            stages = [s for s in ["L1", "L2", "L3", "L4"] if s in set(df["stage"].dropna())]
            data = [df.loc[df["stage"] == s, measurement].dropna().values for s in stages]
            if stages and any(len(d) for d in data):
                _boxplot_with_stage_labels(ax, data, stages)
                ax.set_ylabel(measurement)
                ax.set_title(f"{measurement} by larval stage")
                fig_path = out_dir / "figures" / f"{measurement}_by_stage.png"
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_path, bbox_inches="tight")
                artifacts.append(fig_path)
                caption_path = fig_path.with_suffix(".caption.txt")
                caption_path.write_text(figure_caption(measurement) + "\n", encoding="utf-8")
                artifacts.append(caption_path)
            plt.close(fig)
        except Exception:
            logger.exception("Failed to export %s boxplot under %s", measurement, out_dir)
    return artifacts


def write_run_report(
    run_dir: Path,
    manifest: Manifest,
    qc_reports: Iterable[QCReport],
    shapes_df: Optional[pd.DataFrame] = None,
    intensity_df: Optional[pd.DataFrame] = None,
    trust_reports: Optional[List[Dict[str, Any]]] = None,
    mode: str = "no-llm",
    goal: str = "",
    extra_notes: Optional[List[str]] = None,
    segment_backend: str = "classical",
) -> Path:
    """Write a deterministic markdown report for the run."""
    run_dir = Path(run_dir)
    reports = list(qc_reports)
    counts = Counter(r.status for r in reports)
    red_cells = [r.cell_id for r in reports if r.status == "RED"]

    lines = [
        f"# Nucleolus pipeline run — `{run_dir.name}`",
        "",
        f"- **Mode:** {mode}",
        f"- **Root:** `{manifest.root}`",
        f"- **Goal:** {goal or '(deterministic full run)'}",
        f"- **Resolution (z,y,x µm):** {manifest.resolution_3d}",
        f"- **Segment backend:** `{segment_backend}`",
        "",
        "## Narrative",
        "",
    ]
    for paragraph in build_narrative(manifest, reports, shapes_df, intensity_df):
        lines.append(paragraph)
        lines.append("")

    lines.extend(
        [
            "## Inventory",
            "",
            f"- Included cells: **{len(manifest.cells_included)}**",
            f"- Excluded cells: **{len(manifest.cells_excluded)}**",
            "",
            "## QC summary",
            "",
            f"- GREEN: {counts.get('GREEN', 0)}",
            f"- AMBER: {counts.get('AMBER', 0)}",
            f"- RED: {counts.get('RED', 0)}",
            "",
        ]
    )
    if red_cells:
        lines.append("RED cells:")
        lines.extend(f"- `{c}`" for c in red_cells)
        lines.append("")

    lines.extend(summarize_trust_reports(trust_reports or []))

    lines.extend(["## Shape summary", ""])
    if shapes_df is not None and not shapes_df.empty and "volume" in shapes_df.columns:
        if "stage" in shapes_df.columns:
            grouped = shapes_df.groupby("stage")["volume"].mean()
            for stage, mean_vol in grouped.items():
                lines.append(f"- Mean volume ({stage}): {mean_vol:.3f}")
        else:
            lines.append(f"- Mean volume: {shapes_df['volume'].mean():.3f}")
    else:
        lines.append("- No shape measurements.")
    lines.append("")

    lines.extend(["## Intensity highlights", ""])
    if intensity_df is not None and not intensity_df.empty:
        lines.append(f"- Intensity rows: {len(intensity_df)}")
        if "pc" in intensity_df.columns:
            lines.append(f"- Mean partition coefficient (pc): {intensity_df['pc'].mean():.3f}")
    else:
        lines.append("- No intensity measurements.")
    lines.append("")

    if extra_notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {n}" for n in extra_notes)
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            "- `manifest.json`",
            "- `qc.jsonl`",
            "- `easy_adopt.json` (if trust evaluated)",
            "- `shapes.csv` (if measured)",
            "- `intensity.csv` (if measured)",
            "- `trace.jsonl`",
            "- `figures/`",
            "",
        ]
    )

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
