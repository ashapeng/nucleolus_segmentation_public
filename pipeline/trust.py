"""Optional Easy-adopt trust evaluation wrapper and ML backend gate helpers."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from pipeline.types import TrustReport

logger = logging.getLogger(__name__)

_TRUST_COLORS = ("GREEN", "AMBER", "RED")
_LABELED_STATUS_RE = re.compile(
    r"(?i)\b(?:trust[\s_-]?status|trust[\s_-]?report|overall|verdict|status)\b"
    r"\s*[:=]\s*(GREEN|AMBER|RED)\b"
)
_JSON_STATUS_KEYS = ("trust_status", "trustStatus", "verdict", "status", "overall")


def parse_trust_status(stdout: Optional[str]) -> str:
    """Extract Easy-adopt Trust Report color from CLI stdout.

    Prefers labeled lines / JSON fields over bare color tokens. Returns
    ``UNKNOWN`` when nothing parseable is found.
    """
    if not stdout:
        return "UNKNOWN"
    text = stdout.strip()
    if not text:
        return "UNKNOWN"

    labeled = _LABELED_STATUS_RE.findall(text)
    if labeled:
        return labeled[-1].upper()

    for blob in _iter_json_objects(text):
        color = _color_from_mapping(blob)
        if color is not None:
            return color

    # Last resort: a standalone traffic-light token on its own line.
    for line in text.splitlines():
        token = line.strip().upper()
        if token in _TRUST_COLORS:
            return token

    return "UNKNOWN"


def _iter_json_objects(text: str):
    """Yield JSON objects embedded in stdout (whole text or per-line)."""
    candidates = [text] + text.splitlines()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate.startswith("{"):
            continue
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            yield loaded


def _color_from_mapping(payload: Dict[str, Any]) -> Optional[str]:
    for key in _JSON_STATUS_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.upper() in _TRUST_COLORS:
            return value.upper()
    return None


def trust_allows_ml_backend(trust_status: Optional[str]) -> bool:
    """True when Easy-adopt trust is non-RED (GREEN or AMBER)."""
    return (trust_status or "").upper() in ("GREEN", "AMBER")


def resolve_pipeline_backend(
    config: Optional[Dict[str, Any]],
    *,
    allow_ml_backend: bool = False,
    trust_status: Optional[str] = None,
) -> str:
    """Resolve which ``gc_segment`` backend the pipeline may use.

    Classical is always allowed. An ML backend from ``ml.default_backend`` is
    only returned when ``allow_ml_backend`` is set **and** Easy-adopt trust is
    non-RED. Otherwise the requested ML backend is ignored (logged) and
    classical is forced — never a silent swap.
    """
    requested = "classical"
    if config:
        requested = (config.get("ml") or {}).get("default_backend") or "classical"
    requested = str(requested).strip().lower() or "classical"
    if requested == "classical":
        return "classical"

    if not allow_ml_backend:
        logger.warning(
            "ml.default_backend=%r ignored; pipeline uses classical. "
            "Pass --allow-ml-backend with a non-RED Easy-adopt trust report to opt in.",
            requested,
        )
        return "classical"

    if not trust_allows_ml_backend(trust_status):
        raise ValueError(
            f"Refusing ML backend {requested!r}: trust_status={trust_status!r} "
            "(need GREEN or AMBER from Easy-adopt; RED/UNKNOWN/N/A block the swap)"
        )
    if requested not in ("nnunet", "cellpose"):
        raise ValueError(
            f"Unknown ml.default_backend {requested!r}; expected classical|nnunet|cellpose"
        )
    return requested


def run_easy_adopt(
    cell_dir: str,
    tool: str = "stardist",
    structure: str = "nucleolus_gc",
    *,
    cell_id: Optional[str] = None,
    escalation: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Easy-adopt if installed; otherwise return a SKIPPED status payload.

    Return value is a ``TrustReport.to_dict()`` with both ``invocation`` and
    parsed ``trust_status`` (never confuses process exit with Trust color).
    """
    cell_dir = os.path.abspath(cell_dir)
    messages: List[str] = []
    if escalation:
        messages.append(f"escalation: {escalation}")

    exe = shutil.which("easy-adopt")
    if exe is None:
        try:
            import easy_adopt  # noqa: F401
        except ImportError:
            report = TrustReport(
                cell_id=cell_id,
                cell_dir=cell_dir,
                tool=tool,
                structure=structure,
                invocation="SKIPPED",
                trust_status="N/A",
                reason="easy-adopt not installed",
                messages=messages,
                escalation=escalation,
            )
            return report.to_dict()
        messages.append(
            "easy_adopt importable but 'easy-adopt' not on PATH; trying python -m"
        )
        cmd = [
            os.environ.get("PYTHON", "python3"),
            "-m",
            "easy_adopt",
            "--tool",
            tool,
            "--structure",
            structure,
            "--cell",
            cell_dir,
        ]
    else:
        cmd = [exe, "--tool", tool, "--structure", structure, "--cell", cell_dir]

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        logger.exception("easy-adopt invocation failed")
        report = TrustReport(
            cell_id=cell_id,
            cell_dir=cell_dir,
            tool=tool,
            structure=structure,
            invocation="FAILED",
            trust_status="UNKNOWN",
            reason=str(exc),
            messages=messages,
            escalation=escalation,
        )
        return report.to_dict()

    invocation = "COMPLETED" if completed.returncode == 0 else "FAILED"
    stdout_tail = (completed.stdout or "")[-4000:]
    stderr_tail = (completed.stderr or "")[-2000:]
    combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    parsed = parse_trust_status(combined if invocation == "FAILED" else completed.stdout)
    if invocation == "COMPLETED":
        trust_status = parsed
    else:
        trust_status = parsed if parsed in _TRUST_COLORS else "UNKNOWN"

    report = TrustReport(
        cell_id=cell_id,
        cell_dir=cell_dir,
        tool=tool,
        structure=structure,
        invocation=invocation,
        trust_status=trust_status,
        returncode=completed.returncode,
        stdout_tail=stdout_tail or None,
        stderr_tail=stderr_tail or None,
        messages=messages,
        escalation=escalation,
        reason=None if invocation == "COMPLETED" else (stderr_tail or "non-zero exit"),
    )
    return report.to_dict()


def summarize_trust_reports(trust_reports: List[Dict[str, Any]]) -> List[str]:
    """Markdown bullet lines for a run report trust section."""
    if not trust_reports:
        return []
    lines = ["## Easy-adopt trust", ""]
    for item in trust_reports:
        cell = item.get("cell_id") or item.get("cell_dir") or "?"
        tool = item.get("tool", "?")
        inv = item.get("invocation", "?")
        color = item.get("trust_status", "UNKNOWN")
        esc = item.get("escalation")
        suffix = f" (escalation: {esc})" if esc else ""
        reason = item.get("reason")
        extra = f" — {reason}" if reason else ""
        lines.append(
            f"- `{cell}` tool=`{tool}` invocation={inv} trust={color}{suffix}{extra}"
        )
    lines.append("")
    return lines
