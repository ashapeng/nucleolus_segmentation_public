"""Optional Easy-adopt trust evaluation wrapper."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any, Dict

logger = logging.getLogger(__name__)


def run_easy_adopt(
    cell_dir: str,
    tool: str = "stardist",
    structure: str = "nucleolus_gc",
) -> Dict[str, Any]:
    """Run Easy-adopt if installed; otherwise return a SKIPPED status payload."""
    exe = shutil.which("easy-adopt")
    if exe is None:
        try:
            import easy_adopt  # noqa: F401
        except ImportError:
            return {
                "status": "SKIPPED",
                "reason": "easy-adopt not installed",
                "cell_dir": cell_dir,
                "tool": tool,
                "structure": structure,
            }

    cmd = ["easy-adopt", "--tool", tool, "--structure", structure, "--cell", cell_dir]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "status": "COMPLETED" if completed.returncode == 0 else "FAILED",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-2000:],
            "cell_dir": cell_dir,
            "tool": tool,
            "structure": structure,
        }
    except OSError as exc:
        logger.exception("easy-adopt invocation failed")
        return {
            "status": "FAILED",
            "reason": str(exc),
            "cell_dir": cell_dir,
            "tool": tool,
            "structure": structure,
        }
