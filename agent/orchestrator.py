"""Orchestrator: deterministic plan or optional OpenAI-compatible tool loop."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOL_SPECS, RunContext, execute_tool
from pipeline.runner import run_deterministic

logger = logging.getLogger(__name__)


def run_goal(
    goal: str,
    root: str,
    *,
    no_llm: bool = False,
    stage: Optional[str] = None,
    max_cells: Optional[int] = None,
    with_intensity: bool = False,
    with_easy_adopt: bool = False,
    allow_ml_backend: bool = False,
    runs_base: str = "runs",
    max_steps: int = 40,
) -> Path:
    """Run a natural-language goal, or fall back to the deterministic plan."""
    if no_llm or not _has_api_key():
        if not no_llm and not _has_api_key():
            logger.warning("No API key found; using deterministic --no-llm plan")
        return run_deterministic(
            root,
            stage=stage,
            max_cells=max_cells,
            with_intensity=with_intensity,
            with_easy_adopt=with_easy_adopt,
            allow_ml_backend=allow_ml_backend,
            runs_base=runs_base,
            goal=goal,
        )
    return _run_llm(
        goal=goal,
        root=root,
        stage=stage,
        max_cells=max_cells,
        with_intensity=with_intensity,
        with_easy_adopt=with_easy_adopt,
        allow_ml_backend=allow_ml_backend,
        runs_base=runs_base,
        max_steps=max_steps,
    )


def _has_api_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def _run_llm(
    *,
    goal: str,
    root: str,
    stage: Optional[str],
    max_cells: Optional[int],
    with_intensity: bool,
    with_easy_adopt: bool,
    allow_ml_backend: bool,
    runs_base: str,
    max_steps: int,
) -> Path:
    """Minimal OpenAI-compatible tool loop. Requires OPENAI_API_KEY."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package required for LLM mode. pip install openai or pass --no-llm"
        ) from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Pass --no-llm for deterministic runs.")

    client = OpenAI()
    ctx = RunContext(
        root=root,
        runs_base=runs_base,
        goal=goal,
        allow_ml_backend=allow_ml_backend,
    )
    user = (
        f"Goal: {goal}\nRoot: {root}\n"
        f"Stage filter: {stage or 'all'}\nMax cells: {max_cells or 'all'}\n"
        f"With intensity: {with_intensity}\nWith easy-adopt: {with_easy_adopt}\n"
        f"Allow ML backend: {allow_ml_backend}\n"
        "Inventory first, segment_with_qc each included cell (respect max_cells), "
        "measure_shapes, optionally intensity/easy-adopt, then finalize_run. "
        "Never swap to nnunet/cellpose unless Easy-adopt trust is non-RED and "
        "allow_ml_backend is true."
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    run_dir: Optional[Path] = None
    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=os.environ.get("NUCLEOLUS_AGENT_MODEL", "gpt-4o-mini"),
            messages=messages,
            tools=TOOL_SPECS,
        )
        msg = response.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in (msg.tool_calls or [])
                ]
                or None,
            }
        )
        if not msg.tool_calls:
            break
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            if tc.function.name == "inventory_experiment" and max_cells is not None:
                # still inventory all; agent should respect max_cells when segmenting
                pass
            result = execute_tool(tc.function.name, args, ctx)
            run_dir = ctx.run_dir
            if stage and tc.function.name == "inventory_experiment" and ctx.manifest:
                ctx.manifest.cells_included = [c for c in ctx.manifest.cells_included if c.stage == stage]
                if max_cells is not None:
                    ctx.manifest.cells_included = ctx.manifest.cells_included[:max_cells]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )
            if tc.function.name == "finalize_run":
                return Path(result["run_dir"])

    # Ensure a report exists even if the model forgot finalize_run
    result = execute_tool("finalize_run", {"mode": "llm"}, ctx)
    return Path(result["run_dir"])
