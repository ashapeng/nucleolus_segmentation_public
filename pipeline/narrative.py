"""Deterministic narrative paragraphs grounded only in run stats (cite numbers)."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from pipeline.types import Manifest, QCReport

STAGE_ORDER: Sequence[str] = ("L1", "L2", "L3", "L4")


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def stage_volume_means(shapes_df: pd.DataFrame) -> List[tuple]:
    """Return ``[(stage, mean_volume), ...]`` in L1–L4 order when present."""
    if shapes_df is None or shapes_df.empty:
        return []
    if "stage" not in shapes_df.columns or "volume" not in shapes_df.columns:
        return []
    means = shapes_df.groupby("stage", dropna=True)["volume"].mean()
    out: List[tuple] = []
    for stage in STAGE_ORDER:
        if stage in means.index:
            out.append((stage, float(means.loc[stage])))
    for stage, val in means.items():
        if stage not in STAGE_ORDER:
            out.append((str(stage), float(val)))
    return out


def describe_stage_volume_trend(stage_means: Sequence[tuple]) -> Optional[str]:
    """One sentence on L1→L4 mean GC volume direction; None if <2 stages."""
    if len(stage_means) < 2:
        return None
    parts = [f"{stage} {_fmt(mean)}" for stage, mean in stage_means]
    values = [mean for _, mean in stage_means]
    first, last = values[0], values[-1]
    if last > first * 1.05:
        direction = "increasing"
    elif last < first * 0.95:
        direction = "decreasing"
    else:
        direction = "roughly stable"
    return (
        f"Mean GC volume by stage ({', '.join(parts)}) is {direction} "
        f"from {stage_means[0][0]} to {stage_means[-1][0]}."
    )


def figure_caption(measurement: str = "volume") -> str:
    """Caption for the standard stage boxplot artifact."""
    return (
        f"Box plot of GC {measurement} by larval stage (L1–L4) for cells measured in this run."
    )


def build_narrative(
    manifest: Manifest,
    qc_reports: Iterable[QCReport],
    shapes_df: Optional[pd.DataFrame] = None,
    intensity_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Return narrative paragraphs that only cite numbers from the provided inputs."""
    reports = list(qc_reports)
    counts = Counter(r.status for r in reports)
    n_included = len(manifest.cells_included)
    n_excluded = len(manifest.cells_excluded)
    n_green = counts.get("GREEN", 0)
    n_amber = counts.get("AMBER", 0)
    n_red = counts.get("RED", 0)
    n_qc = len(reports)

    paragraphs: List[str] = []
    paragraphs.append(
        f"This run inventoried {n_included} included cell(s) and {n_excluded} excluded "
        f"cell(s) under `{manifest.root}`."
    )

    if n_qc == 0:
        paragraphs.append("No QC evaluations were recorded for this run.")
    else:
        qc_sentence = (
            f"QC evaluated {n_qc} attempt record(s): {n_green} GREEN, {n_amber} AMBER, "
            f"and {n_red} RED."
        )
        if n_red:
            red_ids = sorted({r.cell_id for r in reports if r.status == "RED" and r.cell_id})
            shown = ", ".join(f"`{c}`" for c in red_ids[:5])
            more = f" (+{len(red_ids) - 5} more)" if len(red_ids) > 5 else ""
            qc_sentence += f" RED cell id(s): {shown}{more}."
        paragraphs.append(qc_sentence)

    if shapes_df is not None and not shapes_df.empty and "volume" in shapes_df.columns:
        n_rows = len(shapes_df)
        overall = float(shapes_df["volume"].mean())
        stage_means = stage_volume_means(shapes_df)
        shape_bits = [
            f"Shape measurements cover {n_rows} object row(s) with overall mean GC volume "
            f"{_fmt(overall)}."
        ]
        trend = describe_stage_volume_trend(stage_means)
        if trend:
            shape_bits.append(trend)
        elif stage_means:
            stage, mean = stage_means[0]
            shape_bits.append(f"Only stage {stage} is present (mean volume {_fmt(mean)}).")
        paragraphs.append(" ".join(shape_bits))
    else:
        paragraphs.append("No shape measurements were available to summarize.")

    if intensity_df is not None and not intensity_df.empty:
        n_int = len(intensity_df)
        if "pc" in intensity_df.columns:
            mean_pc = float(intensity_df["pc"].mean())
            paragraphs.append(
                f"Intensity metrics include {n_int} row(s); mean partition coefficient "
                f"(pc) is {_fmt(mean_pc)}."
            )
        else:
            paragraphs.append(f"Intensity metrics include {n_int} row(s).")
    else:
        paragraphs.append("No intensity measurements were available to summarize.")

    return paragraphs
