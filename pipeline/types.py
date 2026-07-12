"""Shared dataclasses for pipeline inventory, QC, and run artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CellRecord:
    cell_id: str  # e.g. "20220304_L1/10_2"
    path: str
    stage: Optional[str]
    valid: bool
    missing_files: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Manifest:
    root: str
    resolution_3d: List[float]
    cells_included: List[CellRecord]
    cells_excluded: List[CellRecord]
    config_snapshot: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "resolution_3d": self.resolution_3d,
            "cells_included": [c.to_dict() for c in self.cells_included],
            "cells_excluded": [c.to_dict() for c in self.cells_excluded],
            "config_snapshot": self.config_snapshot,
        }


@dataclass
class QCReport:
    cell_id: str
    status: str  # GREEN | AMBER | RED
    gc_fraction: float
    empty_gc: bool
    outside_nucleus_fraction: float
    messages: List[str] = field(default_factory=list)
    # Soft signals only — do not drive RED retries / local_adjust by themselves.
    n_gc_objects: int = 0
    mid_z_index: Optional[int] = None
    soft_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MaskPaths:
    gc: str
    holes: str
    hole_filled: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrustReport:
    """Easy-adopt invocation result plus parsed Trust Report traffic light.

    ``invocation`` is process-level: SKIPPED | COMPLETED | FAILED.
    ``trust_status`` is biological/tool trust: GREEN | AMBER | RED | UNKNOWN | N/A.
    """

    cell_dir: str
    tool: str
    structure: str
    invocation: str
    trust_status: str
    cell_id: Optional[str] = None
    reason: Optional[str] = None
    returncode: Optional[int] = None
    stdout_tail: Optional[str] = None
    stderr_tail: Optional[str] = None
    messages: List[str] = field(default_factory=list)
    escalation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
